#!/usr/bin/env python3
"""Download/prepare BBQ and reproduce the completed retrospective Experiment 1."""
import argparse, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, help="Optional local official BBQ checkout")
    ap.add_argument("--root", type=Path, default=Path("runtime/experiment1"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    prep = [sys.executable, "code/prepare_bbq.py", "--dest", str(args.root)]
    if args.source:
        prep += ["--source", str(args.source)]
    subprocess.run(prep, check=True)
    subprocess.run([
        sys.executable, "code/eea_bbq.py",
        "--root", str(args.root), "--bootstrap", str(args.bootstrap)
    ], check=True)
    print(f"Experiment 1 complete: {args.root/'results'}")

if __name__ == "__main__":
    main()
