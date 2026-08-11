#!/usr/bin/env python3
"""Acquire/prepare the official BBQ repository in the layout required by Experiment 1.

By default this clones https://github.com/nyu-mll/BBQ.git.
Use --source to point to an already downloaded official BBQ checkout.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess
from pathlib import Path

OFFICIAL = "https://github.com/nyu-mll/BBQ.git"

def copytree_files(src: Path, dst: Path, pattern="*"):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob(pattern):
        if p.is_file():
            shutil.copy2(p, dst / p.name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=Path("runtime/experiment1"))
    ap.add_argument("--source", type=Path)
    ap.add_argument("--repo-url", default=OFFICIAL)
    args = ap.parse_args()

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if args.source:
        source = args.source.resolve()
    else:
        source = dest / "_official_BBQ"
        if not source.exists():
            subprocess.run(["git", "clone", "--depth", "1", args.repo_url, str(source)], check=True)

    required = [source/"data", source/"results", source/"supplemental"/"additional_metadata.csv"]
    if not all(p.exists() for p in required):
        raise SystemExit(f"Source does not look like the official BBQ repository: {source}")

    target = dest / "data"
    copytree_files(source/"data", target/"BBQ_data", "*.jsonl")
    shutil.copytree(source/"results"/"UnifiedQA", target/"BBQ_published_results"/"UnifiedQA", dirs_exist_ok=True)
    shutil.copytree(source/"results"/"RoBERTa_and_DeBERTaV3",
                    target/"BBQ_published_results"/"RoBERTa_and_DeBERTaV3", dirs_exist_ok=True)
    shutil.copy2(source/"supplemental"/"additional_metadata.csv", target/"additional_metadata.csv")
    if (source/"LICENSE").exists():
        shutil.copy2(source/"LICENSE", target/"BBQ_LICENSE")

    commit = None
    if (source / '.git').exists():
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            pass

    meta = {"source": str(source), "official_repository": args.repo_url, "commit": commit}
    (dest/"BBQ_SOURCE.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Prepared Experiment-1 data under {target}")

if __name__ == "__main__":
    main()
