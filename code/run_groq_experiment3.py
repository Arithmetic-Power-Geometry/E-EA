import json
import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

MODEL = "openai/gpt-oss-20b"
BASE_URL = "https://api.groq.com/openai/v1"
API_KEY = os.environ["GROQ_API_KEY"]

SUBSET = Path("current_llm/groq_99_pair_subset.jsonl")
PROMPTS = Path("current_llm/prompts.json")

OUTDIR = Path("outputs/experiment3")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "groq_gpt_oss_20b.jsonl"
ERR = OUTDIR / "groq_gpt_oss_20b.errors.jsonl"

REPLICATES = 3
TOTAL = 99 * REPLICATES

LETTER_TO_LABEL = {"A": 0, "B": 1, "C": 2}


def parse_label(text):
    m = re.search(
        r"(?i)(?:^|\b)([ABC])(?:\b|[.)])",
        str(text).strip()
    )
    return LETTER_TO_LABEL.get(m.group(1).upper()) if m else None


def call_model(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 16,
        "reasoning_effort": "low",
        "stream": False
    }

    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY
        }
    )

    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            return data["choices"][0]["message"]["content"]

        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 3 + attempt * 2
                print(
                    f"RATE LIMIT: waiting {wait}s",
                    flush=True
                )
                time.sleep(wait)
                continue
            raise

    raise RuntimeError("Groq request failed after retries")


rows = [
    json.loads(x)
    for x in SUBSET.read_text(encoding="utf-8").splitlines()
    if x.strip()
]

prompts = json.loads(PROMPTS.read_text(encoding="utf-8"))

single = prompts["single_turn_template"]
follow = prompts["two_turn_followup_template"]

done = set()

if OUT.exists():
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            done.add((r["pair_uid"], r["replicate"]))

counter = 0

for pair_no, row in enumerate(rows, 1):

    for rep in range(REPLICATES):

        counter += 1
        key = (row["pair_uid"], rep)

        if key in done:
            print(
                f"[{counter:03d}/{TOTAL}] SKIP completed | "
                f"{row['category']} | {row['pair_uid']} | rep={rep}",
                flush=True
            )
            continue

        print(
            f"[{counter:03d}/{TOTAL}] RUNNING | "
            f"category={row['category']} | "
            f"pair={row['pair_uid']} | "
            f"replicate={rep}",
            flush=True
        )

        try:

            amb_prompt = single.format(
                context=row["ambig_context"],
                question=row["question"],
                ans0=row["ans0"],
                ans1=row["ans1"],
                ans2=row["ans2"]
            )

            dis_prompt = single.format(
                context=row["disambig_context"],
                question=row["question"],
                ans0=row["ans0"],
                ans1=row["ans1"],
                ans2=row["ans2"]
            )

            amb_text = call_model([
                {"role": "user", "content": amb_prompt}
            ])

            print(
                f"[{counter:03d}/{TOTAL}] ambiguous complete",
                flush=True
            )

            dis_text = call_model([
                {"role": "user", "content": dis_prompt}
            ])

            print(
                f"[{counter:03d}/{TOTAL}] disambiguated complete",
                flush=True
            )

            follow_text = follow.format(
                added_evidence=row["added_evidence"]
            )

            update_text = call_model([
                {"role": "user", "content": amb_prompt},
                {"role": "assistant", "content": amb_text},
                {"role": "user", "content": follow_text}
            ])

            print(
                f"[{counter:03d}/{TOTAL}] evidence-update complete",
                flush=True
            )

            rec = {
                "pair_uid": row["pair_uid"],
                "category": row["category"],
                "question_polarity": row["question_polarity"],
                "target_loc": row["target_loc"],
                "ambig_label": row["ambig_label"],
                "disambig_label": row["disambig_label"],

                "model_name": "Groq GPT-OSS-20B",
                "model_id": MODEL,

                "replicate": rep,
                "temperature": 0.7,

                "ambig_text": amb_text,
                "ambig_pred": parse_label(amb_text),

                "disambig_text": dis_text,
                "disambig_pred": parse_label(dis_text),

                "update_initial_text": amb_text,
                "update_initial_pred": parse_label(amb_text),

                "update_final_text": update_text,
                "update_final_pred": parse_label(update_text)
            }

            with OUT.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(rec, ensure_ascii=False) + "\n"
                )

            print(
                f"[{counter:03d}/{TOTAL}] SAVED ✓",
                flush=True
            )

        except Exception as exc:

            with ERR.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps({
                        "pair_uid": row["pair_uid"],
                        "category": row["category"],
                        "replicate": rep,
                        "error": repr(exc)
                    }) + "\n"
                )

            print(
                f"[{counter:03d}/{TOTAL}] ERROR: {exc}",
                flush=True
            )

print("======================================", flush=True)
print("EXPERIMENT 3 INFERENCE COMPLETE", flush=True)
print(f"Expected records: {TOTAL}", flush=True)
print("======================================", flush=True)
