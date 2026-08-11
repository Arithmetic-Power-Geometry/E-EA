import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('runner', ROOT/'code'/'run_current_llm.py')
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)

def test_parse_label():
    assert runner.parse_label('A')==0
    assert runner.parse_label('B.')==1
    assert runner.parse_label('Answer: C')==2
    assert runner.parse_label('unknown') is None

def test_subset_exists_and_nonempty():
    p=ROOT/'current_llm'/'bbq_current_llm_subset.jsonl'
    assert p.exists() and p.stat().st_size>1000
