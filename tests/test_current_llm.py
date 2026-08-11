import importlib.util, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT/rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

runner = load("runner", "code/run_current_llm.py")

def test_parse_label_plain():
    assert runner.parse_label("A") == 0
    assert runner.parse_label("B.") == 1
    assert runner.parse_label("C)") == 2

def test_parse_label_rejects_unparseable():
    assert runner.parse_label("I cannot decide") is None

def test_prompt_templates_exist_and_render():
    p = json.load(open(ROOT/"current_llm/prompts.json", encoding="utf-8"))
    row = {"question":"Q?","ans0":"x","ans1":"y","ans2":"z"}
    s = runner.render_single(p["single_turn_template"], "ctx", row)
    assert "Context: ctx" in s and "A. x" in s and "Return only A, B, or C." in s

def test_registry_has_three_frozen_models():
    r = json.load(open(ROOT/"current_llm/model_registry.json", encoding="utf-8"))
    assert [m["key"] for m in r["models"]] == ["deepseek", "mistral", "qwen"]
    assert r["sampling"]["pairs"] == 1000
    assert r["sampling"]["replicates"] == 3

def test_subset_integrity():
    rows = [json.loads(x) for x in (ROOT/"current_llm/bbq_current_llm_subset.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 1000
    assert len({r["pair_uid"] for r in rows}) == 1000

def test_two_turn_prompt_is_frozen():
    p = json.load(open(ROOT/"current_llm/prompts.json", encoding="utf-8"))
    assert "{added_evidence}" in p["two_turn_followup_template"]
    assert "same question again" in p["two_turn_followup_template"]
