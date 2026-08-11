#!/usr/bin/env python3
"""Run all Experiment-2 models declared in current_llm/model_registry.json."""
import argparse, json, subprocess, sys
from pathlib import Path

def slug(s):
    return ''.join(c.lower() if c.isalnum() else '_' for c in s).strip('_')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--registry',default='current_llm/model_registry.json'); ap.add_argument('--subset',default='current_llm/bbq_current_llm_subset.jsonl'); ap.add_argument('--outdir',default='outputs'); ap.add_argument('--only',nargs='*'); a=ap.parse_args()
    reg=json.load(open(a.registry,encoding='utf-8')); Path(a.outdir).mkdir(parents=True,exist_ok=True)
    for m in reg['models']:
        if a.only and m['name'] not in a.only: continue
        cmd=[sys.executable,'code/run_current_llm.py','--subset',a.subset,'--model-name',m['name'],'--model-id',m['model_id'],'--base-url',m['base_url'],'--api-key-env',m.get('api_key_env','OPENAI_API_KEY'),'--out',str(Path(a.outdir)/(slug(m['name'])+'.jsonl')),'--replicates',str(reg['sampling']['replicates']),'--temperature',str(reg['sampling']['temperature']),'--max-tokens',str(reg['sampling']['max_tokens'])]
        print('RUN:', ' '.join(cmd)); subprocess.run(cmd,check=True)
if __name__=='__main__': main()
