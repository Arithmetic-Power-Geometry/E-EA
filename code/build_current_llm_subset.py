#!/usr/bin/env python3
"""Build a deterministic, category-stratified BBQ subset for contemporary LLM evaluation.

The subset preserves adjacent ambiguous/disambiguated pairs and records all metadata
needed to score conventional BBQ behavior and the E-EA proxies. No model inference is
performed by this script.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def read_jsonl(p: Path):
    with p.open(encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--pairs', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20260811)
    ap.add_argument('--out', type=Path, default=None)
    args=ap.parse_args(); root=args.root.resolve()
    out=args.out or root/'current_llm'/'bbq_current_llm_subset.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    meta=pd.read_csv(root/'data'/'additional_metadata.csv')
    meta['question_index']=meta['question_index'].astype(str)
    meta_idx={(r.category,str(r.question_index),int(r.example_id)):r for r in meta.itertuples(index=False)}
    cats=[]
    for f in sorted((root/'data'/'BBQ_data').glob('*.jsonl')):
        rr=read_jsonl(f)
        pairs=[]
        for k,(a,d) in enumerate(zip(rr[::2],rr[1::2])):
            if a['context_condition']!='ambig' or d['context_condition']!='disambig':
                raise ValueError(f'Unexpected pair order in {f} at pair {k}')
            key=(a['category'],str(a['question_index']),int(a['example_id']))
            mr=meta_idx.get(key)
            if mr is None or pd.isna(mr.target_loc):
                continue
            pairs.append((a,d,int(mr.target_loc)))
        cats.append((rr[0]['category'],pairs))
    ncat=len(cats)
    base=args.pairs//ncat; rem=args.pairs%ncat
    rng=np.random.default_rng(args.seed)
    selected=[]
    for ci,(cat,pairs) in enumerate(cats):
        n=min(len(pairs),base+(1 if ci<rem else 0))
        idx=np.sort(rng.choice(len(pairs),size=n,replace=False))
        for j in idx:
            a,d,target_loc=pairs[int(j)]
            # Added evidence is typically the disambiguated suffix; retain full context if not prefix-compatible.
            added=d['context'][len(a['context']):].strip() if d['context'].startswith(a['context']) else d['context']
            selected.append({
                'pair_uid':f"{cat}:{a['example_id']}:{d['example_id']}",
                'category':cat,'question_index':str(a['question_index']),
                'question_polarity':a['question_polarity'],'target_loc':target_loc,
                'ambig_example_id':int(a['example_id']),'disambig_example_id':int(d['example_id']),
                'ambig_context':a['context'],'disambig_context':d['context'],'added_evidence':added,
                'question':a['question'],'ans0':a['ans0'],'ans1':a['ans1'],'ans2':a['ans2'],
                'ambig_label':int(a['label']),'disambig_label':int(d['label']),
                'answer_info':a['answer_info'],'additional_metadata':a.get('additional_metadata',{}),
            })
    if len(selected)!=min(args.pairs,sum(len(x[1]) for x in cats)):
        raise AssertionError(f'Unexpected subset size {len(selected)}')
    with out.open('w',encoding='utf-8') as f:
        for r in selected: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    counts=pd.Series([r['category'] for r in selected]).value_counts().sort_index()
    manifest={'seed':args.seed,'requested_pairs':args.pairs,'actual_pairs':len(selected),
              'categories':counts.to_dict(),'source':'BBQ supplied project snapshot'}
    (out.parent/'subset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
