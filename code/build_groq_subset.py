from pathlib import Path
import json
from collections import defaultdict

src = Path("current_llm/bbq_current_llm_subset.jsonl")
out = Path("current_llm/groq_99_pair_subset.jsonl")

rows = [
    json.loads(x)
    for x in src.read_text(encoding="utf-8").splitlines()
    if x.strip()
]

by_cat = defaultdict(list)

for row in rows:
    by_cat[row["category"]].append(row)

selected = []

for category in sorted(by_cat):
    selected.extend(by_cat[category][:9])

assert len(selected) == 99
assert len(set(r["category"] for r in selected)) == 11

with out.open("w", encoding="utf-8") as f:
    for row in selected:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Experiment 3 subset created")
print("99 pairs = 9 pairs x 11 BBQ categories")
