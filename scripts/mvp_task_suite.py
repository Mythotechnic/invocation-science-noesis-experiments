import argparse, json, os
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import tempfile, subprocess


def run_model_prompt(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def eval_code_snippet(snippet: str, timeout: int = 3):
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(snippet)
            path = f.name
        proc = subprocess.run(["python", path], capture_output=True, text=True, timeout=timeout)
        return proc.returncode == 0, proc.stdout + proc.stderr
    except Exception as e:
        return False, str(e)


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

    tasks = json.load(open("tasks/prompts.json"))
    results = []
    for group_name, group in tasks.items():
        for item in group:
            out = run_model_prompt(model, tokenizer, item["prompt"], device=device)
            rec = {
                "group": group_name,
                "id": item["id"],
                "prompt": item["prompt"],
                "output": out,
            }
            if group_name == "code":
                ok, exec_out = eval_code_snippet(out)
                rec["exec_ok"] = ok
                rec["exec_output"] = exec_out
            results.append(rec)

    os.makedirs(args.resultsdir, exist_ok=True)
    json.dump(results, open(os.path.join(args.resultsdir, "task_results.json"), "w"), indent=2)
    print("saved task_results.json")


if __name__ == "__main__":
    main()
