#!/usr/bin/env python3
"""Resumable E-EA contemporary-LLM inference via OpenAI-compatible chat endpoints.

Each pair/replicate produces three API calls:
1. ambiguous single-turn answer,
2. disambiguated single-turn answer,
3. evidence-update continuation whose first assistant turn is the model's own
   ambiguous response from (1).

Only successful records are written to the evidence JSONL. Exhausted API failures
are written to a separate *.errors.jsonl file. Independent pair/replicate tasks may
run concurrently; within each task, the two-turn dependency is preserved exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LETTER_TO_LABEL = {"A": 0, "B": 1, "C": 2}


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def parse_label(text):
    m = re.search(r"(?i)(?:^|\b)([ABC])(?:\b|[.)])", str(text).strip())
    return LETTER_TO_LABEL.get(m.group(1).upper()) if m else None


def load_prompts(path: Path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def render_single(template: str, context: str, row: dict) -> str:
    return template.format(
        context=context, question=row["question"],
        ans0=row["ans0"], ans1=row["ans1"], ans2=row["ans2"]
    )


def call_chat(base_url, api_key, model, messages, temperature, max_tokens,
              timeout=120, retries=5, retry_backoff=2.0, request_overrides=None):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model, "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens, "stream": False
    }
    if request_overrides:
        payload.update(request_overrides)

    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            content = raw["choices"][0]["message"]["content"]
            return content, raw
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
            last = exc
            retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in {
                408, 409, 425, 429, 500, 502, 503, 504
            }
            if attempt >= retries or not retryable:
                break
            wait = retry_backoff * (2 ** attempt) + random.random() * 0.25
            time.sleep(wait)
    raise RuntimeError(f"API request failed after retries: {last!r}")


def successful_keys(path: Path):
    if not path.exists():
        return set()
    keys = set()
    for r in read_jsonl(path):
        if not r.get("error") and all(r.get(k) in (0, 1, 2) for k in
            ("ambig_pred", "disambig_pred", "update_initial_pred", "update_final_pred")):
            keys.add((r["pair_uid"], int(r["replicate"])))
    return keys


def infer_one(row, rep, args, prompts, key, request_overrides):
    common = {
        "pair_uid": row["pair_uid"], "category": row["category"],
        "question_polarity": row["question_polarity"], "target_loc": row["target_loc"],
        "ambig_label": row["ambig_label"], "disambig_label": row["disambig_label"],
        "model_name": args.model_name, "model_id": args.model_id,
        "replicate": rep, "temperature": args.temperature
    }
    try:
        amb_prompt = render_single(prompts["single_turn_template"], row["ambig_context"], row)
        dis_prompt = render_single(prompts["single_turn_template"], row["disambig_context"], row)

        amb_text, _ = call_chat(
            args.base_url, key, args.model_id,
            [{"role": "user", "content": amb_prompt}],
            args.temperature, args.max_tokens, args.timeout, args.retries,
            args.retry_backoff, request_overrides
        )
        dis_text, _ = call_chat(
            args.base_url, key, args.model_id,
            [{"role": "user", "content": dis_prompt}],
            args.temperature, args.max_tokens, args.timeout, args.retries,
            args.retry_backoff, request_overrides
        )
        follow_text = prompts["two_turn_followup_template"].format(added_evidence=row["added_evidence"])
        upd_text, _ = call_chat(
            args.base_url, key, args.model_id,
            [
                {"role": "user", "content": amb_prompt},
                {"role": "assistant", "content": amb_text},
                {"role": "user", "content": follow_text},
            ],
            args.temperature, args.max_tokens, args.timeout, args.retries,
            args.retry_backoff, request_overrides
        )

        return True, {
            **common,
            "ambig_text": amb_text, "ambig_pred": parse_label(amb_text),
            "disambig_text": dis_text, "disambig_pred": parse_label(dis_text),
            "update_initial_text": amb_text, "update_initial_pred": parse_label(amb_text),
            "update_final_text": upd_text, "update_final_pred": parse_label(upd_text)
        }
    except Exception as exc:
        return False, {**common, "error": repr(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, default=Path("current_llm/prompts.json"))
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--request-overrides-json", default="{}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--retry-backoff", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    rows = read_jsonl(args.subset)
    if args.limit is not None:
        rows = rows[:args.limit]
    prompts = load_prompts(args.prompts)
    request_overrides = json.loads(args.request_overrides_json)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    error_path = args.out.with_suffix(".errors.jsonl")
    done = successful_keys(args.out)
    key = os.getenv(args.api_key_env, "")
    local = args.base_url.startswith(("http://localhost", "http://127.0.0.1"))
    if not local and not key:
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")

    pending = [(row, rep) for row in rows for rep in range(args.replicates)
               if (row["pair_uid"], rep) not in done]
    total = len(pending)
    if not total:
        print(f"{args.model_name}: all requested records are already complete.")
        return

    completed = 0
    with args.out.open("a", encoding="utf-8") as goodf, error_path.open("a", encoding="utf-8") as errf:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(infer_one, row, rep, args, prompts, key, request_overrides):
                (row["pair_uid"], rep) for row, rep in pending
            }
            for fut in as_completed(futures):
                ok, rec = fut.result()
                target = goodf if ok else errf
                target.write(json.dumps(rec, ensure_ascii=False) + "\n")
                target.flush()
                completed += 1
                if completed % 75 == 0 or completed == total:
                    print(f"{args.model_name}: {completed}/{total} pending records processed")

if __name__ == "__main__":
    main()
