import argparse, os, json, math
import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import gaussian_filter1d


def pooled_from_hidden(hidden_states):
    last = hidden_states[-1].squeeze(0)  # [seqlen, d]
    return last.mean(dim=0).cpu().numpy()


def smooth(X, sigma):
    return gaussian_filter1d(X, sigma=sigma, axis=0, mode="nearest")


def compute_curvature(X, sigma=1.0):
    if X.shape[0] < 3:
        return np.full(X.shape[0], np.nan)
    Xs = smooth(X, sigma)
    x1 = (Xs[2:] - Xs[:-2]) / 2.0
    x2 = Xs[2:] - 2 * Xs[1:-1] + Xs[:-2]
    num = np.linalg.norm(x2, axis=1)
    den = (1.0 + np.linalg.norm(x1, axis=1) ** 2) ** 1.5
    kappa = num / (den + 1e-12)
    return np.concatenate(([np.nan], kappa, [np.nan]))


class Controller:
    def __init__(self, model, decoder_path, seq_len, device):
        self.model = model
        self.seq_len = seq_len
        self.device = device
        self.d = model.config.n_embd
        self.decoder = None
        if decoder_path and os.path.exists(decoder_path):
            from torch import nn
            class Dec(nn.Module):
                def __init__(self, d, seq_len):
                    super().__init__()
                    self.fc = nn.Sequential(
                        nn.Linear(d, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, seq_len * d),
                    )
                    self.seq_len = seq_len
                    self.d = d
                def forward(self, x):
                    out = self.fc(x)
                    return out.view(-1, self.seq_len, self.d)
            self.decoder = Dec(self.d, self.seq_len).to(device)
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
            self.decoder.eval()

    def pooled_to_inputs_embeds(self, pooled):
        v = torch.tensor(pooled, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.decoder is not None:
            with torch.no_grad():
                out = self.decoder(v)
            return out
        # fixed: replicate pooled as a short sequence
        return v.unsqueeze(1).repeat(1, self.seq_len, 1)


def run_cycle(model, tokenizer, controller, prompt, cycles, device, perturb_eps=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    pooled_list = []
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        pooled0 = pooled_from_hidden(out.hidden_states)

    if perturb_eps is not None:
        with torch.no_grad():
            wte = model.transformer.wte(input_ids)
            noise = torch.randn_like(wte) * perturb_eps
            outp = model(inputs_embeds=wte + noise, attention_mask=attention_mask, output_hidden_states=True)
            pooled0 = pooled_from_hidden(outp.hidden_states)

    pooled_list.append(pooled0)

    for _ in range(1, max(cycles, 1)):
        pooled_prev = pooled_list[-1]
        inputs_embeds = controller.pooled_to_inputs_embeds(pooled_prev)
        with torch.no_grad():
            out = model(inputs_embeds=inputs_embeds, output_hidden_states=True)
            pooled = pooled_from_hidden(out.hidden_states)
        pooled_list.append(pooled)

    return np.stack(pooled_list, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_name = cfg["model_name"]
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    decoder_path = "models/decoder.pt" if cfg["pooled_to_inputs"]["mode"] == "trained" else None
    controller = Controller(
        model,
        decoder_path=decoder_path,
        seq_len=cfg["pooled_to_inputs"]["reentry_seq_len"],
        device=device,
    )

    tasks = json.load(open("tasks/prompts.json"))
    prompts = [item for group in tasks.values() for item in group]

    os.makedirs(args.outdir, exist_ok=True)
    summary = []

    for T in cfg["cycles"]:
        for seed in cfg.get("seeds", [42]):
            np.random.seed(seed)
            torch.manual_seed(seed)
            for item in prompts:
                X = run_cycle(model, tokenizer, controller, item["prompt"], cycles=max(1, T), device=device)
                kappa = compute_curvature(X, sigma=cfg["smoothing"]["sigma"])
                if X.shape[0] > 1:
                    S = cosine_similarity(X)
                    bands = []
                    for k in range(1, min(4, X.shape[0])):
                        bands.append(np.mean(np.diag(S, k=k)))
                    echo = float(np.mean(bands))
                else:
                    echo = float("nan")
                pca_dim = min(cfg["pca_dim"], max(1, X.shape[0] - 1))
                if X.shape[0] > 1:
                    pca = PCA(n_components=pca_dim)
                    PC = pca.fit_transform(X)
                    energy = float(np.sum(PC ** 2))
                else:
                    energy = 0.0
                rec = {
                    "T": T,
                    "seed": seed,
                    "prompt_id": item["id"],
                    "kappa_mean": float(np.nanmean(kappa)),
                    "echo": echo,
                    "energy": energy,
                    "len": int(X.shape[0]),
                }
                fname = os.path.join(args.outdir, f"res_T{T}_s{seed}_{item['id']}.npz")
                np.savez_compressed(fname, X=X, kappa=kappa, echo=echo, energy=energy, meta=rec)
                summary.append(rec)

    json.dump(summary, open(os.path.join(args.outdir, "summary.json"), "w"), indent=2)
    print("saved summary to", os.path.join(args.outdir, "summary.json"))


if __name__ == "__main__":
    main()
