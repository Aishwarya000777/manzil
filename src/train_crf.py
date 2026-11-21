"""Train Manzil's feature-hashed first-order linear-chain CRF baseline."""
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

FEATURE_DIM = 4096

def token_features(tokens, i):
    word = tokens[i]
    lower = word.lower()
    shape = "".join("A" if c.isupper() else "a" if c.islower() else "0" if c.isdigit() else c for c in word)
    f = [f"w={lower}", f"shape={shape}", f"len={min(len(word),15)}",
         f"p1={lower[:1]}", f"p2={lower[:2]}", f"p3={lower[:3]}",
         f"s1={lower[-1:]}", f"s2={lower[-2:]}", f"s3={lower[-3:]}"]
    f += ["BOS"] if i == 0 else [f"prev={tokens[i-1].lower()}", f"prev2={tokens[i-1].lower()[:3]}"]
    f += ["EOS"] if i == len(tokens)-1 else [f"next={tokens[i+1].lower()}", f"next2={tokens[i+1].lower()[:3]}"]
    if i: f.append(f"bp={tokens[i-1].lower()}_{lower}")
    if i < len(tokens)-1: f.append(f"bn={lower}_{tokens[i+1].lower()}")
    return f

def encode(tokens):
    x = np.zeros((len(tokens), FEATURE_DIM), dtype=np.float32)
    for i in range(len(tokens)):
        for f in token_features(tokens, i):
            j = int(hashlib.md5(f.encode()).hexdigest()[:8], 16) % FEATURE_DIM
            x[i, j] += 1
    return torch.from_numpy(x)

class LinearChainCRF(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.em = nn.Parameter(torch.zeros(FEATURE_DIM, n))
        self.trans = nn.Parameter(torch.zeros(n, n))
        self.start = nn.Parameter(torch.zeros(n))
        self.end = nn.Parameter(torch.zeros(n))

    def loss(self, x, y):
        e = x @ self.em
        a = self.start + e[0]
        for t in range(1, len(y)):
            a = torch.logsumexp(a[:, None] + self.trans + e[t][None, :], dim=0)
        z = torch.logsumexp(a + self.end, dim=0)
        s = self.start[y[0]] + e[0, y[0]]
        for t in range(1, len(y)): s = s + self.trans[y[t-1], y[t]] + e[t, y[t]]
        return z - s - self.end[y[-1]] + self.end[y[-1]]  # log-partition minus gold path below

    def gold_nll(self, x, y):
        e = x @ self.em
        a = self.start + e[0]
        for t in range(1, len(y)):
            a = torch.logsumexp(a[:, None] + self.trans + e[t][None, :], dim=0)
        z = torch.logsumexp(a + self.end, dim=0)
        s = self.start[y[0]] + e[0, y[0]]
        for t in range(1, len(y)): s = s + self.trans[y[t-1], y[t]] + e[t, y[t]]
        s = s + self.end[y[-1]]
        return z - s

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/manzil_v2_token_level.jsonl")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=429)
    p.add_argument("--out", default="models/manzil_linear_crf.pt")
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rows = [json.loads(x) for x in open(a.data, encoding="utf-8") if x.strip()]
    tags = sorted({t for r in rows for t in r["bio_tags"]})
    ids = {t:i for i,t in enumerate(tags)}
    tr = [r for r in rows if r.get("split") == "train"]
    va = [r for r in rows if r.get("split") == "validation"]
    if not tr or not va: raise ValueError("Expected train and validation splits.")
    m = LinearChainCRF(len(tags)); opt = Adam(m.parameters(), lr=a.lr, weight_decay=1e-4)
    best, state = float("inf"), None
    for ep in range(a.epochs):
        random.shuffle(tr); m.train(); loss_sum = 0.0
        for r in tr:
            x = encode(r["token_sequence"]); y = torch.tensor([ids[t] for t in r["bio_tags"]])
            opt.zero_grad(); loss = m.gold_nll(x,y); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
            loss_sum += float(loss.detach())
        m.eval()
        with torch.no_grad():
            vl = sum(float(m.gold_nll(encode(r["token_sequence"]),torch.tensor([ids[t] for t in r["bio_tags"]]))) for r in va)/len(va)
        print(f"epoch={ep+1} train_loss={loss_sum/len(tr):.4f} val_loss={vl:.4f}")
        if vl < best: best, state = vl, {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    if state is None: raise RuntimeError("No checkpoint selected")
    m.load_state_dict(state); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":m.state_dict(),"tags":tags,"feature_dim":FEATURE_DIM,"feature_version":"manzil-hash-v1"},out)
    print("Saved:",out)

if __name__ == "__main__": main()
