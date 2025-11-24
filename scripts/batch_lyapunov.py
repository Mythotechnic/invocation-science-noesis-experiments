import argparse, os, json
import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import cosine_similarity

from mvp_embedding_reentry_full import Controller, pooled_from_hidden, run_cycle  # type: ignore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resultsdir", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_name = cfg["model_name"]
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    controller = Controller(
        model,
        decoder_path=None,
        seq_len=cfg["pooled_to_inputs"]["reentry_seq_len"],
        device=device,
    )

    tasks = json.load(open("tasks/prompts.json"))
    prompts = [item["prompt"] for group in tasks.values() for item in group]

    os.makedirs(args.resultsdir, exist_ok=True)

    all_out = {}
    for T in cfg["cycles"]:
        slopes = []
        for prompt in prompts:
            # baseline
            X_base = run_cycle(model, tokenizer, controller, prompt, cycles=max(1, T), device=device, perturb_eps=None)
            # perturbed
            X_pert = run_cycle(model, tokenizer, controller, prompt, cycles=max(1, T), device=device, perturb_eps=cfg["perturb_eps"])
            L = min(len(X_base), len(X_pert))
            deltas = np.linalg.norm(X_pert[:L] - X_base[:L], axis=1)
            deltas = np.maximum(deltas, 1e-12)
            t = np.arange(L)
            slope = np.polyfit(t, np.log(deltas), 1)[0]
            slopes.append(float(slope))
        all_out[T] = slopes
        json.dump({"T": T, "slopes": slopes}, open(os.path.join(args.resultsdir, f"lyap_T{T}.json"), "w"), indent=2)

    json.dump(all_out, open(os.path.join(args.resultsdir, "lyap_all.json"), "w"), indent=2)
    print("saved Lyapunov summaries")


if __name__ == "__main__":
    main()
