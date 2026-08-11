# E-EA: reproducibility software

Software for **Ethical Answerability in Language Models: A Structural and Counterfactual Experience Architecture of Bias and Fairness**.

## What is empirical and what is prospective
Experiment 1 is the completed retrospective analysis reported in the manuscript. The contemporary-LLM extension is a **prospective reproducibility protocol**: this repository provides the frozen subset and executable pipeline, but no unexecuted model outputs are presented as evidence.

## Repository contents
- `code/eea_bbq.py` - Experiment 1 analysis.
- `code/build_current_llm_subset.py` - reproduces the frozen stratified subset from the full BBQ release.
- `code/run_current_llm.py` - resumable OpenAI-compatible inference runner.
- `code/run_all_current_llm.py` - runs all models declared in the registry.
- `code/validate_current_llm.py` - coverage/error/parseability gate; failed outputs must not be used as evidence.
- `code/analyze_current_llm.py` - E-EA/BBQ analysis, category bootstrap CIs, response-field JSD, and figures.
- `current_llm/bbq_current_llm_subset.jsonl` - frozen 1,000-pair Experiment-2 subset.
- `current_llm/model_registry.json` - model identifiers and sampling settings frozen 2026-08-11.
- `current_llm/subset_manifest.json` - subset construction metadata.
- `current_llm/prompts.json` - frozen prompt templates used by the prospective protocol.
- `docs/DATA.md` - BBQ provenance, licensing, exact Experiment-1 layout, and subset reconstruction.
- `docs/MODELS.md` - frozen contemporary-model identifiers and official source pages.
- `CITATION.cff` - software citation metadata.
- `third_party/` - BBQ CC BY 4.0 license and attribution notice.
- `tests/` - unit tests.

## Installation and tests
Python 3.10+ is recommended.
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pytest -q
```

## Experiment 1 data
The full BBQ release and published prediction files are not duplicated here. Obtain the official BBQ repository/data associated with Parrish et al. (2022), then arrange the analysis root as:
```text
<root>/
  data/
    BBQ_data/*.jsonl
    BBQ_published_results/
      UnifiedQA/preds_*.jsonl
      RoBERTa_and_DeBERTaV3/df_bbq.csv
    additional_metadata.csv
```
Run `python code/eea_bbq.py --help` for the current CLI. The frozen Experiment-2 subset is already included, so prospective current-model inference does not require rebuilding the subset.

Official BBQ repository: https://github.com/nyu-mll/BBQ

Official BBQ paper: Parrish et al. (2022), DOI 10.18653/v1/2022.findings-acl.165. BBQ data are distributed under CC BY 4.0; the derived frozen subset retains that attribution/licensing requirement. See `docs/DATA.md` and `third_party/` for the exact provenance and license notice.

## Prospective contemporary-LLM protocol
The registry freezes three model identifiers: DeepSeek-V4-Flash (`deepseek-v4-flash`), Mistral Small 4 v26.03 (`mistral-small-2603`), and Qwen3.5-4B (`Qwen/Qwen3.5-4B`). API credentials and/or compatible local accelerator infrastructure are **not** bundled.

For API models, export the required key named in `current_llm/model_registry.json`. For Qwen, start an OpenAI-compatible local server (for example vLLM) at the registry URL. Then:
```bash
python code/run_all_current_llm.py
python code/validate_current_llm.py --inputs outputs/*.jsonl
python code/analyze_current_llm.py --inputs outputs/*.jsonl --outdir results/current_llm
```
The validator expects 1,000 pairs x 3 replicates per model and rejects errors, duplicates, unknown pair IDs, or parseability below the declared threshold.

## Reproducibility safeguards
- Frozen subset seed: `20260811`.
- Retrospective/bootstrap seed: `20260810`.
- Raw responses are retained in JSONL.
- The model's own first-turn response is preserved in the two-turn trajectory.
- No synthetic answer is injected into the conversation.
- Failed/incomplete runs are rejected by the validation gate before analysis.
- Contemporary results should be added to the manuscript only after real outputs pass validation.

## License
Code in this repository is released under the MIT License. The derived BBQ subset remains subject to the source dataset's CC BY 4.0 terms and attribution requirements.


## Frozen prompt templates
The exact single-turn and two-turn prompt templates are separately recorded in `current_llm/prompts.json` for auditability. `code/run_current_llm.py` implements the same text and preserves the model's own first-turn answer in the multi-turn trajectory.

## Citation
A machine-readable `CITATION.cff` is included. Please also cite BBQ (Parrish et al., 2022) when using the benchmark-derived subset.
