#!/usr/bin/env python3
"""Run one or all models declared in the frozen Experiment-2 registry."""
import argparse, json, os, subprocess, sys
from pathlib import Path

def slug(s):
    return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="current_llm/model_registry.json")
    ap.add_argument("--subset", default="current_llm/bbq_current_llm_subset.jsonl")
    ap.add_argument("--prompts", default="current_llm/prompts.json")
    ap.add_argument("--outdir", default="outputs/current_llm")
    ap.add_argument("--only", nargs="*", help="Model key(s) or exact model name(s)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    reg = json.load(open(args.registry, encoding="utf-8"))
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    selected = []
    for m in reg["models"]:
        if args.only and m.get("key") not in args.only and m["name"] not in args.only:
            continue
        selected.append(m)

    if not selected:
        raise SystemExit("No models selected.")

    s = reg["sampling"]
    for m in selected:
        base_url = os.getenv(m.get("base_url_env", ""), "") if m.get("base_url_env") else ""
        base_url = base_url or m["base_url"]
        if m.get("key") == "qwen" and base_url.startswith(("http://127.0.0.1", "http://localhost")):
            print("Qwen is configured for a local OpenAI-compatible endpoint.")
        cmd = [
            sys.executable, "code/run_current_llm.py",
            "--subset", args.subset,
            "--prompts", args.prompts,
            "--model-name", m["name"],
            "--model-id", m["model_id"],
            "--base-url", base_url,
            "--api-key-env", m.get("api_key_env", "OPENAI_API_KEY"),
            "--request-overrides-json", json.dumps(m.get("request_overrides", {})),
            "--out", str(Path(args.outdir) / (slug(m["name"]) + ".jsonl")),
            "--replicates", str(s["replicates"]),
            "--temperature", str(s["temperature"]),
            "--max-tokens", str(s["max_tokens"]),
            "--timeout", str(s.get("timeout_seconds", 120)),
            "--retries", str(s.get("retries", 5)),
            "--retry-backoff", str(s.get("retry_backoff_seconds", 2.0)),
            "--workers", str(m.get("workers", 4)),
        ]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        print("RUN:", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
