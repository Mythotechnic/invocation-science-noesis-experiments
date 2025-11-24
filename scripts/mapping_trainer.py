import argparse, os, json, random
import torch
from torch import nn, optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml

class Decoder(nn.Module):
    def __init__(self, d_model: int, seq_len: int = 4, hidden: int = 1024):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, seq_len * d_model),
        )

    def forward(self, x):
        out = self.fc(x)
        return out.view(-1, self.seq_len, self.d_model)


def collect_pairs(model, tokenizer, prompts, device):
    pairs = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            last = out.hidden_states[-1].squeeze(0)  # [seqlen, d]
            pooled = last.mean(dim=0).cpu().numpy()
            seq_len = min(4, last.size(0))
            target = last[-seq_len:].cpu().numpy()
        pairs.append((pooled, target))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="models/decoder.pt")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_name = cfg["model_name"]
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    tasks = json.load(open("tasks/prompts.json"))
    prompts = [item["prompt"] for group in tasks.values() for item in group]
    prompts = prompts[:20]

    pairs = collect_pairs(model, tokenizer, prompts, device)
    d_model = model.config.n_embd
    decoder = Decoder(d_model, seq_len=cfg["pooled_to_inputs"]["reentry_seq_len"]).to(device)
    opt = optim.Adam(decoder.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(3):
        random.shuffle(pairs)
        total = 0.0
        for pooled, target in pairs:
            pooled_t = torch.tensor(pooled, dtype=torch.float32, device=device).unsqueeze(0)
            target_t = torch.tensor(target, dtype=torch.float32, device=device).unsqueeze(0)
            pred = decoder(pooled_t)
            loss = loss_fn(pred, target_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch} loss {total / max(len(pairs),1):.6f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(decoder.state_dict(), args.out)
    print("saved decoder to", args.out)


if __name__ == "__main__":
    main()
