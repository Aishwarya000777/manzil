"""Retrain the Manzil linear-chain CRF using cleaned BIO tags.

Compatibility: state_dict key names, FEATURE_DIM, and feature hashing are kept
compatible with backend.model_service.LinearChainCRF. Do not change the backend
prediction interface for this experiment.
"""
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

torch.set_num_threads(1)
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

def legal_masks(tags):
    n=len(tags); sm=torch.ones(n,dtype=torch.bool); tm=torch.ones(n,n,dtype=torch.bool)
    for j,t in enumerate(tags):
        if t.startswith("I-"):
            sm[j]=False
            lab=t[2:]
            for i,prev in enumerate(tags):
                if prev not in ("B-"+lab,"I-"+lab): tm[i,j]=False
    return sm,tm

class LinearChainCRF(nn.Module):
    def __init__(self, tags):
        super().__init__(); self.em=nn.Parameter(torch.zeros(FEATURE_DIM,len(tags)))
        self.trans=nn.Parameter(torch.zeros(len(tags),len(tags)))
        self.start=nn.Parameter(torch.zeros(len(tags))); self.end=nn.Parameter(torch.zeros(len(tags)))
        sm,tm=legal_masks(tags); self.register_buffer("start_mask",sm); self.register_buffer("trans_mask",tm)
    def masked(self):
        s=self.start.masked_fill(~self.start_mask,-1e4)
        t=self.trans.masked_fill(~self.trans_mask,-1e4)
        return s,t
    def gold_nll(self,x,y):
        e=x@self.em; start,trans=self.masked(); a=start+e[0]
        for k in range(1,len(y)):
            a=torch.logsumexp(a[:,None]+trans+e[k][None,:],dim=0)
        z=torch.logsumexp(a+self.end,dim=0)
        score=start[y[0]]+e[0,y[0]]
        for k in range(1,len(y)):
            score=score+trans[y[k-1],y[k]]+e[k,y[k]]
        score=score+self.end[y[-1]]
        return z-score
    def decode(self,x):
        e=x@self.em; start,trans=self.masked(); a=start+e[0]; bp=[]
        for k in range(1,len(e)):
            v=a[:,None]+trans; best,idx=v.max(0); a=best+e[k]; bp.append(idx)
        last=int((a+self.end).argmax()); out=[last]
        for idx in bp[::-1]: last=int(idx[last]); out.append(last)
        return out[::-1]

def load_rows(path):
    rows=[json.loads(s) for s in open(path,encoding="utf-8") if s.strip()]
    for r in rows:
        if len(r["token_sequence"]) != len(r["bio_tags"]): raise ValueError(f"Token/tag length mismatch: {r['id']}")
        for i,t in enumerate(r["bio_tags"]):
            if t.startswith("I-") and (i==0 or r["bio_tags"][i-1] not in ("B-"+t[2:],"I-"+t[2:])):
                raise ValueError(f"Malformed BIO in {r['id']}: {r['bio_tags']}")
    return rows

def span_f1(model, rows, tags):
    gold_by={};pred_by={}
    for r in rows:
        with torch.no_grad(): pred=[tags[i] for i in model.decode(encode(r['token_sequence']))]
        def get_spans(seq):
            out=[];st=None;lab=None
            for i,t in enumerate(seq+['O']):
                if t=='O' or t.startswith('B-') or (t.startswith('I-') and lab!=t[2:]):
                    if st is not None: out.append((lab,st,i))
                    st=None;lab=None
                if t.startswith('B-'): st=i;lab=t[2:]
                elif t.startswith('I-') and st is None: st=i;lab=t[2:]
            return set(out)
        gold_by[r['id']]=get_spans(r['bio_tags']);pred_by[r['id']]=get_spans(pred)
    labels=sorted({x[0] for ss in gold_by.values() for x in ss})
    fs=[]; tp_all=fp_all=fn_all=0
    for lab in labels:
        g={(k,*x) for k,v in gold_by.items() for x in v if x[0]==lab}
        p={(k,*x) for k,v in pred_by.items() for x in v if x[0]==lab}
        tp=len(g&p);fp=len(p-g);fn=len(g-p)
        pr=tp/(tp+fp) if tp+fp else 0;rc=tp/(tp+fn) if tp+fn else 0
        fs.append(2*pr*rc/(pr+rc) if pr+rc else 0)
        tp_all+=tp;fp_all+=fp;fn_all+=fn
    micro_p=tp_all/(tp_all+fp_all) if tp_all+fp_all else 0
    micro_r=tp_all/(tp_all+fn_all) if tp_all+fn_all else 0
    micro=2*micro_p*micro_r/(micro_p+micro_r) if micro_p+micro_r else 0
    return (sum(fs)/len(fs) if fs else 0),micro

def train(data, out, epochs=30, lr=0.01, seed=429, rare_repeat=4):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows=load_rows(data); tags=sorted({t for r in rows for t in r["bio_tags"]}); ids={t:i for i,t in enumerate(tags)}
    tr=[r for r in rows if r.get("split")=="train"]; va=[r for r in rows if r.get("split")=="validation"]
    test=[r for r in rows if r.get("split")=="test"]
    missing=set(tags)-{t for r in tr for t in r['bio_tags']}
    if missing: raise ValueError(f"Training split misses labels: {sorted(missing)}")
    pool=[]
    for r in tr:
        labels={t.split('-',1)[1] for t in r['bio_tags'] if t!='O'}
        repeat=rare_repeat if labels & {'DIR','CONTEXT'} else 1
        pool.extend([r]*repeat)
    model=LinearChainCRF(tags); opt=Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    best=-1.0; best_state=None; best_epoch=0; patience=8; stale=0
    for ep in range(epochs):
        random.shuffle(pool); model.train(); loss_sum=0.0
        for r in pool:
            x=encode(r['token_sequence']); y=torch.tensor([ids[t] for t in r['bio_tags']],dtype=torch.long)
            opt.zero_grad(); loss=model.gold_nll(x,y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); loss_sum+=float(loss.detach())
        model.eval()
        with torch.no_grad():
            val=sum(float(model.gold_nll(encode(r['token_sequence']),torch.tensor([ids[t] for t in r['bio_tags']]))) for r in va)/len(va)
        macro_span,micro_span=span_f1(model,va,tags)
        print(f"epoch={ep+1:02d} train_nll={loss_sum/len(pool):.4f} val_nll={val:.4f} val_span_macro_f1={macro_span:.4f} val_span_micro_f1={micro_span:.4f}")
        if macro_span > best+1e-5:
            best=macro_span; best_epoch=ep+1; stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k not in ('start_mask','trans_mask')}
        else:
            stale+=1
            if stale>=patience: break
    if best_state is None: raise RuntimeError('No best checkpoint captured')
    model.load_state_dict(best_state,strict=False); model.eval()
    with torch.no_grad():
        model.start.copy_(model.start.masked_fill(~model.start_mask,-1e4))
        model.trans.copy_(model.trans.masked_fill(~model.trans_mask,-1e4))
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items() if k not in ('start_mask','trans_mask')},
                'tags':tags,'feature_dim':FEATURE_DIM,'feature_version':'manzil-hash-v1-bio-constrained',
                'selected_epoch':best_epoch,'validation_macro_span_f1':best,'training_rows':len(tr),'training_utterances_after_oversampling':len(pool)},out)
    print(f"Saved checkpoint: {out} (best epoch {best_epoch}; validation macro span F1 {best:.4f}; train={len(tr)}, val={len(va)}, test={len(test)})")

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/manzil_v2_token_level_clean_augmented.jsonl');p.add_argument('--epochs',type=int,default=30);p.add_argument('--lr',type=float,default=.01);p.add_argument('--rare-repeat',type=int,default=8);p.add_argument('--seed',type=int,default=429);p.add_argument('--out',default='models/manzil_linear_crf.pt');a=p.parse_args()
    train(a.data,a.out,a.epochs,a.lr,a.seed,a.rare_repeat)
