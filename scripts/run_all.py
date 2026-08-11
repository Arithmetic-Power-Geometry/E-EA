#!/usr/bin/env python3
"""Run both experiments and generate the complete manuscript-ready output bundle."""
import argparse, subprocess, sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment2-limit",type=int,help="10 for smoke; omit for full Experiment 2.")
    ap.add_argument("--models",nargs="*",default=["deepseek","mistral","qwen"])
    args=ap.parse_args()

    subprocess.run([sys.executable,"scripts/run_experiment1.py"],check=True)

    cmd=[sys.executable,"scripts/run_experiment2.py","--models",*args.models]
    if args.experiment2_limit is not None:
        cmd+=["--limit",str(args.experiment2_limit)]
    subprocess.run(cmd,check=True)

    subprocess.run([
        sys.executable,"code/build_combined_outputs.py",
        "--experiment1-root","runtime/experiment1",
        "--experiment2-root","results/experiment2",
        "--outdir","results/combined"
    ],check=True)
    subprocess.run([sys.executable,"code/build_output_index.py"],check=True)
    print("All experiments complete.")
    print("Experiment 1: runtime/experiment1/results/")
    print("Experiment 2: results/experiment2/")
    print("Combined manuscript-ready outputs: results/combined/")

if __name__=="__main__":
    main()
