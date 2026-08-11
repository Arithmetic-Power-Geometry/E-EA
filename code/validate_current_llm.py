#!/usr/bin/env python3
"""Validate Experiment-2 output coverage and parseability before analysis."""
import argparse, json
from pathlib import Path
from collections import Counter

def read_jsonl(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--subset", default="current_llm/bbq_current_llm_subset.jsonl")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min-parse-rate", type=float, default=0.99)
    args = ap.parse_args()

    subset = read_jsonl(args.subset)
    if args.limit is not None:
        subset = subset[:args.limit]
    expected_pairs = {r["pair_uid"] for r in subset}
    failed = False

    for p in args.inputs:
        rows = read_jsonl(p)
        model = next((r.get("model_name") for r in rows if r.get("model_name")), Path(p).stem)
        expected = len(expected_pairs) * args.replicates
        keys = [(r.get("pair_uid"), int(r.get("replicate", -1))) for r in rows
                if r.get("pair_uid") in expected_pairs]
        dup = sum(v - 1 for v in Counter(keys).values() if v > 1)
        errors = sum(bool(r.get("error")) for r in rows)
        in_scope = [r for r in rows if r.get("pair_uid") in expected_pairs and not r.get("error")]
        fields = ("ambig_pred", "disambig_pred", "update_initial_pred", "update_final_pred")
        parseable = sum(all(r.get(k) in (0, 1, 2) for k in fields) for r in in_scope)
        rate = parseable / expected if expected else 0.0
        unknown_pairs = sum(r.get("pair_uid") not in expected_pairs for r in rows)
        missing = max(0, expected - len(keys))
        print(
            f"{model}: records={len(keys)}/{expected}; missing={missing}; "
            f"errors={errors}; duplicates={dup}; out_of_scope={unknown_pairs}; "
            f"fully_parseable={parseable}; parse_rate={rate:.3%}"
        )
        if len(keys) != expected or errors or dup or unknown_pairs or rate < args.min_parse_rate:
            failed = True

    if failed:
        raise SystemExit("VALIDATION FAILED: do not use these outputs as manuscript evidence.")
    print("VALIDATION PASSED: output coverage and parseability meet the declared gate.")

if __name__ == "__main__":
    main()
