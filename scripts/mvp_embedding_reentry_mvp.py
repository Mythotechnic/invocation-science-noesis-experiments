import argparse, os, json, yaml
import numpy as np
import torch
import scipy.ndimage as ndi

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def gaussian_smooth(X, sigma=1.0):
    return ndi.gaussian_filter1d(X, sigma=sigma, axis=0, mode="nearest")


def compute_curvature(X, sigma=1.0):
    """
    X: [T, d] trajectory of pooled embeddings.
    Returns kappa[t] of length T (nan at endpoints).
    """
    if X.shape[0] < 3:
        return np.full(X.shape[0], np.nan, dtype=float)

    Xs = gaussian_smooth(X, sigma=sigma)
    x1 = (Xs[2:] - Xs[:-2]) / 2.0
    x2 = Xs[2:] - 2 * Xs[1:-1] + Xs[:-2]

    num = np.linalg.norm(x2, axis=1)
    den = (1.0 + np.linalg.norm(x1, axis=1) ** 2) ** 1.5
    kappa = num / (den + 1e-12)

    return np.concatenate(([np.nan], kappa, [np.nan]))


def compute_echo_strength(X, k_max=3):
    """
    Echo strength from cosine similarity bands of the trajectory.
    """
    if X.shape[0] < 2:
        return float("nan")
    S = cosine_similarity(X)
    vals = []
    for k in range(1, min(k_max + 1, X.shape[0])):
        vals.append(np.mean(np.diag(S, k)))
    return float(np.mean(vals)) if vals else float("nan")


def finite_time_lyapunov(X_base, X_pert):
    """
    Simple finite-time Lyapunov estimate between two trajectories.
    """
    T = min(len(X_base), len(X_pert))
    if T < 2:
        return float("nan"), None

    deltas = np.linalg.norm(X_pert[:T] - X_base[:T], axis=1)
    deltas = np.maximum(deltas, 1e-12)
    times = np.arange(T)
    slope = np.polyfit(times, np.log(deltas), 1)[0]
    return float(slope), deltas


class ReentryController:
    """
    Minimal embedding-level re-entry:
    pooled (d) -> repeated inputs_embeds [1, L, d].
    """
    def __init__(self, model, seq_len=4, device="cpu"):
        self.model = model
        self.seq_len = seq_len
        self.device = device
        self.d = model.config.n_embd

    def pooled_to_inputs_embeds(self, pooled_vec):
        v = torch.tensor(pooled_vec, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, d]
        rep = v.unsqueeze(1).repeat(1, self.seq_len, 1)  # [1, L, d]
        return rep


def run_cycle(model, tokenizer, controller, prompt, cycles=4, perturb_eps=None, device="cpu"):
    """
    Run NOESIS-style recursive re-entry:
      - first pass on text input_ids,
      - subsequent passes on inputs_embeds constructed from pooled embeddings.
    Returns trajectory X [cycles, d].
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    pooled_list = []

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        pooled0 = out.hidden_states[-1].mean(dim=1).squeeze(0).cpu().numpy()
        pooled_list.append(pooled0)

    if perturb_eps is not None:
        with torch.no_grad():
            wte = model.transformer.wte(input_ids)
            noise = torch.randn_like(wte) * perturb_eps
            inputs_embeds = wte + noise
            outp = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
            pooled_list[0] = outp.hidden_states[-1].mean(dim=1).squeeze(0).cpu().numpy()

    for t in range(1, cycles):
        pooled_prev = pooled_list[-1]
        inputs_embeds = controller.pooled_to_inputs_embeds(pooled_prev)
        with torch.no_grad():
            out = model(inputs_embeds=inputs_embeds, output_hidden_states=True)
            pooled = out.hidden_states[-1].mean(dim=1).squeeze(0).cpu().numpy()
            pooled_list.append(pooled)

    X = np.stack(pooled_list, axis=0)
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model_name"]
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    cycles_list = cfg["cycles"]
    runs_per_condition = cfg["runs_per_condition"]
    sigma = cfg["smoothing"]["sigma"]
    prompts_file = cfg["tasks"]["prompts_file"]

    os.makedirs(args.outdir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    controller = ReentryController(model, seq_len=cfg["pooled_to_inputs"]["reentry_seq_len"], device=device)

    with open(prompts_file, "r", encoding="utf-8") as f:
        prompt_spec = json.load(f)

    prompts = []
    for k in prompt_spec:
        prompts.extend(prompt_spec[k])

    summary = []

    for T in cycles_list:
        for run in range(runs_per_condition):
            for item in prompts:
                prompt_id = item["id"]
                prompt = item["prompt"]

                if T == 0:
                    X = run_cycle(model, tokenizer, controller, prompt, cycles=1, perturb_eps=None, device=device)
                else:
                    X = run_cycle(model, tokenizer, controller, prompt, cycles=T, perturb_eps=None, device=device)

                kappa = compute_curvature(X, sigma=sigma)
                echo = compute_echo_strength(X)

                # crude PCA energy as a scalar summary
                if X.shape[0] > 1:
                    pca_dim = min(cfg["pca_dim"], max(1, X.shape[0] - 1))
                    pca = PCA(n_components=pca_dim)
                    PC = pca.fit_transform(X)
                    energy = float(np.sum(PC ** 2))
                else:
                    energy = 0.0

                rec = {
                    "T": T,
                    "run": run,
                    "prompt_id": prompt_id,
                    "kappa_mean": float(np.nanmean(kappa)),
                    "echo": float(echo),
                    "energy": energy,
                    "traj_len": int(X.shape[0]),
                    "d": int(X.shape[1]),
                }

                fname = os.path.join(args.outdir, f"res_T{T}_run{run}_{prompt_id}.npz")
                np.savez_compressed(fname, X=X, kappa=kappa, echo=echo, energy=energy, meta=rec)

                summary.append(rec)

    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved summary and per-run results to", args.outdir)


if __name__ == "__main__":
    main()
