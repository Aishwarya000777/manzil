#!/usr/bin/env python3
"""Validate Manzil token/entity offsets, BIO legality, duplicates and split-family isolation."""
import argparse
import json
import collections
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/manzil_v2_token_level_clean_augmented.jsonl')
    ap.add_argument('--out',default='data/ner_dataset_audit_runtime.json')
    a=ap.parse_args()
    rows=[json.loads(x) for x in open(a.data,encoding='utf-8') if x.strip()]
    bad_bio=[]; bad_offsets=[]; bad_entities=[]; bad_lengths=[]
    norm=collections.defaultdict(list); fam=collections.defaultdict(set)
    for r in rows:
        q=r['query']; ts=r['token_sequence']; tags=r['bio_tags']; toks=r['tokens']
        if len(ts)!=len(tags) or len(ts)!=len(toks):
            bad_lengths.append(r['id']); continue
        for i,t in enumerate(tags):
            if t.startswith('I-') and (i==0 or tags[i-1] not in ('B-'+t[2:],'I-'+t[2:])):
                bad_bio.append((r['id'],i,t))
        for t,o,tag in zip(ts,toks,tags):
            if q[o['start']:o['end']]!=t or o['tag']!=tag: bad_offsets.append(r['id'])
        for e in r.get('entities',[]):
            if q[e['start']:e['end']]!=e['text']: bad_entities.append((r['id'],e))
        norm[' '.join(q.casefold().split())].append((r['id'],r['split']))
        fam[r['scenario_family']].add(r['split'])
    duplicates={k:v for k,v in norm.items() if len(v)>1}
    family_overlap={k:sorted(v) for k,v in fam.items() if len(v)>1}
    labels={}
    for split in ('train','validation','test'):
        c=collections.Counter()
        for r in rows:
            if r['split']==split: c.update(t[2:] for t in r['bio_tags'] if t!='O')
        labels[split]=dict(sorted(c.items()))
    result={'rows':len(rows),'split_counts':dict(collections.Counter(r['split'] for r in rows)),
            'invalid_bio_transitions':len(bad_bio),'token_offset_or_tag_errors':len(bad_offsets),
            'entity_offset_errors':len(bad_entities),'token_length_errors':len(bad_lengths),
            'normalized_exact_duplicate_groups':len(duplicates),'family_cross_split_overlaps':len(family_overlap),
            'entity_counts_by_split':labels}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if any((bad_bio,bad_offsets,bad_entities,bad_lengths,duplicates,family_overlap)):
        raise SystemExit(2)

if __name__=='__main__':
    main()
