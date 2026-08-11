#!/usr/bin/env python3
"""Create a concise index of generated publication outputs."""
import argparse
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment1-root",type=Path,default=Path("runtime/experiment1"))
    ap.add_argument("--experiment2-root",type=Path,default=Path("results/experiment2"))
    ap.add_argument("--combined-root",type=Path,default=Path("results/combined"))
    ap.add_argument("--out",type=Path,default=Path("results/OUTPUT_INDEX.md"))
    a=ap.parse_args()
    a.out.parent.mkdir(parents=True,exist_ok=True)

    groups=[
        ("Experiment 1 tables",a.experiment1_root/"results/tables"),
        ("Experiment 1 figures",a.experiment1_root/"results/figures"),
        ("Experiment 2 tables",a.experiment2_root/"tables"),
        ("Experiment 2 figures",a.experiment2_root/"figures"),
        ("Combined tables",a.combined_root/"tables"),
        ("Combined figures",a.combined_root/"figures"),
    ]
    lines=["# Generated manuscript-ready outputs",""]
    for title,p in groups:
        lines += [f"## {title}",""]
        if p.exists():
            for f in sorted(x for x in p.iterdir() if x.is_file()):
                lines.append(f"- `{f.as_posix()}`")
        else:
            lines.append("- Not generated yet.")
        lines.append("")
    a.out.write_text("\n".join(lines),encoding="utf-8")
    print(a.out)

if __name__=="__main__":
    main()
