#!/usr/bin/env python3
"""Run, validate, analyze, and render all publication outputs for Experiment 2."""
import argparse, subprocess, sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--models",nargs="*",default=["deepseek","mistral","qwen"])
    ap.add_argument("--limit",type=int,help="10 for smoke; omit for full frozen 1,000-pair run.")
    ap.add_argument("--outdir",type=Path,default=Path("outputs/current_llm"))
    ap.add_argument("--results",type=Path,default=Path("results/experiment2"))
    args=ap.parse_args()

    run=[sys.executable,"code/run_all_current_llm.py","--only",*args.models,"--outdir",str(args.outdir)]
    if args.limit is not None: run+=["--limit",str(args.limit)]
    subprocess.run(run,check=True)

    files=sorted(str(p) for p in args.outdir.glob("*.jsonl") if not p.name.endswith(".errors.jsonl"))
    if not files: raise SystemExit("No Experiment-2 evidence files found.")

    val=[sys.executable,"code/validate_current_llm.py","--inputs",*files]
    if args.limit is not None: val+=["--limit",str(args.limit)]
    subprocess.run(val,check=True)

    subprocess.run([sys.executable,"code/analyze_current_llm.py","--inputs",*files,
                    "--outdir",str(args.results),"--strict"],check=True)
    print(f"Experiment 2 complete. Tables/figures: {args.results}")

if __name__=="__main__":
    main()
