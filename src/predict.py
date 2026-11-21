"""Run Manzil intent classification and CRF NER prediction on one query."""
import argparse, gzip, hashlib, io, json
import joblib, numpy as np, torch
from torch import nn

FEATURE_DIM=4096

def token_features(tokens,i):
    w=tokens[i]; lo=w.lower()
    shape="".join("A" if c.isupper() else "a" if c.islower() else "0" if c.isdigit() else c for c in w)
    f=[f"w={lo}",f"shape={shape}",f"len={min(len(w),15)}",f"p1={lo[:1]}",f"p2={lo[:2]}",f"p3={lo[:3]}",f"s1={lo[-1:]}",f"s2={lo[-2:]}",f"s3={lo[-3:]}"]
    f += ["BOS"] if i==0 else [f"prev={tokens[i-1].lower()}",f"prev2={tokens[i-1].lower()[:3]}"]
    f += ["EOS"] if i==len(tokens)-1 else [f"next={tokens[i+1].lower()}",f"next2={tokens[i+1].lower()[:3]}"]
    if i: f.append(f"bp={tokens[i-1].lower()}_{lo}")
    if i<len(tokens)-1: f.append(f"bn={lo}_{tokens[i+1].lower()}")
    return f

def encode(tokens):
    x=np.zeros((len(tokens),FEATURE_DIM),dtype=np.float32)
    for i in range(len(tokens)):
        for f in token_features(tokens,i):
            j=int(hashlib.md5(f.encode()).hexdigest()[:8],16)%FEATURE_DIM; x[i,j]+=1
    return torch.from_numpy(x)

class LinearChainCRF(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.em=nn.Parameter(torch.zeros(FEATURE_DIM,n)); self.trans=nn.Parameter(torch.zeros(n,n))
        self.start=nn.Parameter(torch.zeros(n)); self.end=nn.Parameter(torch.zeros(n))
    def decode(self,x):
        e=x@self.em; score=self.start+e[0]; b=[]
        for t in range(1,len(e)):
            v,ix=(score[:,None]+self.trans).max(0); score=v+e[t]; b.append(ix)
        last=int((score+self.end).argmax()); path=[last]
        for bp in reversed(b): last=int(bp[last]); path.append(last)
        return list(reversed(path))

def load_gz(path, loader):
    raw=open(path,"rb").read()
    if path.endswith(".gz"): raw=gzip.decompress(raw)
    return loader(io.BytesIO(raw))

def spans(query,tokens,tags):
    offsets=[]; cur=0
    for t in tokens:
        s=query.find(t,cur)
        if s<0: s=cur
        offsets.append((s,s+len(t))); cur=s+len(t)
    out=[]; start=None; label=None
    for i in range(len(tags)+1):
        tag=tags[i] if i<len(tags) else "O"
        pref,lab=(tag[:2],tag[2:]) if "-" in tag else ("O",None)
        if start is not None and (pref=="B-" or pref=="O" or lab!=label):
            s,e=offsets[start][0],offsets[i-1][1]
            out.append({"text":query[s:e],"label":label,"start":s,"end":e})
            start=label=None
        if pref=="B-": start,label=i,lab
        elif pref=="I-" and start is None: start,label=i,lab
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--crf-model",default="models/manzil_linear_crf.pt.gz")
    p.add_argument("--intent-model",default="models/intent_char_tfidf_logreg.joblib.gz")
    a=p.parse_args()
    tokens=a.query.split()
    if not tokens: raise SystemExit("Query is empty.")
    ck=load_gz(a.crf_model,lambda b:torch.load(b,map_location="cpu",weights_only=False))
    tags=ck["tags"]; model=LinearChainCRF(len(tags)); model.load_state_dict(ck["state_dict"]); model.eval()
    with torch.no_grad(): pred=[tags[i] for i in model.decode(encode(tokens))]
    obj=load_gz(a.intent_model,joblib.load)
    intent=obj["classifier"].predict(obj["vectorizer"].transform([a.query]))[0]
    print(json.dumps({"query":a.query,"intent":str(intent),"tokens":tokens,"bio_tags":pred,"entities":spans(a.query,tokens,pred)},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
