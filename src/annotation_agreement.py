#!/usr/bin/env python3
"""Compute inter-annotator agreement from exported Annotator_A/B CSV sheets."""
import argparse, json
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score

FIELDS=["Annotator intent","Ambiguity","Resolution status"]

def spans(value):
    if pd.isna(value) or not str(value).strip(): return set()
    return {(str(x["text"]),str(x["label"]),int(x["start"]),int(x["end"])) for x in json.loads(value)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--a",required=True); p.add_argument("--b",required=True)
    a=p.parse_args(); x=pd.read_csv(a.a).set_index("ID"); y=pd.read_csv(a.b).set_index("ID")
    ids=sorted(set(x.index)&set(y.index)); print("Matched records:",len(ids))
    for f in FIELDS:
        xa=x.loc[ids,f].fillna("").astype(str); ya=y.loc[ids,f].fillna("").astype(str)
        keep=(xa!="")&(ya!="")
        if keep.any():
            print(f"{f}: n={int(keep.sum())}, agreement={accuracy_score(xa[keep],ya[keep]):.4f}, kappa={cohen_kappa_score(xa[keep],ya[keep]):.4f}")
    tp=fp=fn=exact=n=0
    for i in ids:
        av=x.loc[i].get("Entities JSON"); bv=y.loc[i].get("Entities JSON")
        if pd.isna(av) or pd.isna(bv) or not str(av).strip() or not str(bv).strip(): continue
        sa,sb=spans(av),spans(bv); tp+=len(sa&sb); fp+=len(sb-sa); fn+=len(sa-sb); exact+=int(sa==sb); n+=1
    pr=tp/(tp+fp) if tp+fp else 0; re=tp/(tp+fn) if tp+fn else 0
    f1=2*pr*re/(pr+re) if pr+re else 0
    print(f"Entity spans: n={n}, exact-set agreement={exact/n if n else 0:.4f}, P={pr:.4f}, R={re:.4f}, F1={f1:.4f}")
    print("Agreement measures consistency, not correctness. Adjudicate disagreements before designating labels as final.")

if __name__=="__main__": main()
