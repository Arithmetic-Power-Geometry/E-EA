#!/usr/bin/env python3
"""Static reproducibility audit for required repository artifacts and frozen subset."""
import hashlib, json
from pathlib import Path

REQ = [
    "README.md", "LICENSE", "CITATION.cff", "requirements.txt", "VERIFICATION.md",
    "code/eea_bbq.py", "code/prepare_bbq.py", "code/run_current_llm.py",
    "code/run_all_current_llm.py", "code/validate_current_llm.py",
    "code/analyze_current_llm.py", "code/build_combined_outputs.py", "code/build_output_index.py",
    "current_llm/bbq_current_llm_subset.jsonl",
    "current_llm/subset_manifest.json", "current_llm/model_registry.json",
    "current_llm/prompts.json", "docs/DATA.md", "docs/MODELS.md",
    "tests/test_current_llm.py", "docs/QWEN_ENDPOINT.md", ".github/workflows/ci.yml",
    ".github/workflows/experiment1.yml", ".github/workflows/experiment2.yml",
    ".github/workflows/run_all.yml"
]
EXPECTED_SUBSET_SHA256 = "3709c95538b98c87b837441626da2c585d034b53deff39526f6685ea35782187"

def main():
    root = Path(__file__).resolve().parents[1]
    missing = [x for x in REQ if not (root/x).exists()]
    if missing:
        raise SystemExit("Missing required files: " + ", ".join(missing))
    subset = root/"current_llm/bbq_current_llm_subset.jsonl"
    sha = hashlib.sha256(subset.read_bytes()).hexdigest()
    if sha != EXPECTED_SUBSET_SHA256:
        raise SystemExit(f"Frozen subset checksum mismatch: {sha}")
    rows = [json.loads(x) for x in subset.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) != 1000 or len({r["pair_uid"] for r in rows}) != 1000:
        raise SystemExit("Frozen subset must contain exactly 1,000 unique pair_uid values.")
    print("Repository audit passed: required files present and frozen subset verified.")

if __name__ == "__main__":
    main()
