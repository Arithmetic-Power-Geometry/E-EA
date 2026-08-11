# E-EA — Ethical Experience Architecture

Reproducibility software for the manuscript **Ethical Answerability in Language Models: A Structural and Counterfactual Experience Architecture of Bias and Fairness**.

The repository runs two declared studies on the same BBQ benchmark family and automatically generates the CSV tables, LaTeX tables, PDF figures, PNG figures, statistical summaries, and combined historical-versus-contemporary outputs needed for manuscript revision.

## Experiment status

**Experiment 1 — completed retrospective validation.**  
Uses the complete public BBQ release and the model prediction files published with BBQ.

**Experiment 2 — prospective contemporary-LLM validation.**  
Uses a frozen, stratified 1,000-pair BBQ subset, three repeated generations per pair/model, single-turn ambiguous/disambiguated evaluation, and a genuine two-turn evidence-update trajectory. Real outputs become evidence only after the validation gate passes.

No synthetic output is stored in this repository or treated as manuscript evidence.

## One-click GitHub route

After uploading this repository:

1. Open **Actions**.
2. Run **CI**.
3. Run **Experiment 1 - retrospective BBQ**.
4. Configure Experiment-2 credentials under **Settings → Secrets and variables → Actions**:
   - secret `DEEPSEEK_API_KEY`
   - secret `MISTRAL_API_KEY`
   - secret `QWEN_API_KEY`
   - variable `QWEN_BASE_URL` pointing to an OpenAI-compatible endpoint serving `Qwen/Qwen3.5-4B`
5. Run **Experiment 2 - contemporary LLMs** with `scope=smoke`.
6. When the smoke run passes, rerun with `scope=full`.
7. For the complete historical + contemporary manuscript bundle, run **Run all experiments** with `scope=full`.

Generated outputs are downloadable from the workflow artifacts.

## Local route

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
python code/audit_repository.py
```

### Experiment 1

```bash
python scripts/run_experiment1.py
```

Output:

```text
runtime/experiment1/results/
├── tables/
│   ├── dataset_inventory.csv
│   ├── bbq_scores.csv
│   ├── eea_by_category.csv
│   ├── eea_aggregate.csv
│   ├── eea_bootstrap_ci.csv
│   ├── association_tests.csv
│   ├── dataset_inventory.tex
│   └── eea_aggregate.tex
└── figures/
    ├── fig_condition_profiles.pdf/.png
    ├── fig_bias_vs_eea.pdf/.png
    ├── fig_category_heatmap.pdf/.png
    └── fig_architecture_distance.pdf/.png
```

### Experiment 2

First configure the API keys and Qwen endpoint described in `docs/GITHUB_ACTIONS.md`.

Smoke run:

```bash
python scripts/run_experiment2.py --limit 10
```

Full frozen run:

```bash
python scripts/run_experiment2.py
```

Output:

```text
results/experiment2/
├── tables/
│   ├── current_llm_by_category.csv
│   ├── current_llm_aggregate.csv
│   ├── current_llm_bootstrap_ci.csv
│   ├── current_llm_association_tests.csv
│   ├── current_llm_architecture_distance.csv
│   ├── current_llm_aggregate.tex
│   ├── current_llm_association_tests.tex
│   └── current_llm_architecture_distance.tex
├── figures/
│   ├── fig_current_llm_profiles.pdf/.png
│   ├── fig_current_llm_bias_vs_eea.pdf/.png
│   ├── fig_current_llm_category_heatmap.pdf/.png
│   ├── fig_current_llm_architecture_distance.pdf/.png
│   └── fig_current_llm_response_field_jsd.pdf/.png
└── publication_output_manifest.json
```

### Run everything and build combined manuscript outputs

```bash
python scripts/run_all.py
```

This creates:

```text
results/combined/
├── tables/
│   ├── combined_model_summary.csv
│   ├── combined_model_summary.tex
│   ├── combined_architecture_distance.csv
│   └── combined_architecture_distance.tex
├── figures/
│   ├── fig_combined_profiles.pdf/.png
│   ├── fig_combined_bias_vs_eea.pdf/.png
│   └── fig_combined_architecture_distance.pdf/.png
└── publication_output_manifest.json
```

It also creates `results/OUTPUT_INDEX.md`, which lists every generated table and figure.

## Frozen Experiment-2 design

- Benchmark family: BBQ
- Matched pairs: 1,000
- Sampling seed: `20260811`
- Model families:
  - DeepSeek-V4-Flash
  - Mistral Small 4 v26.03
  - Qwen3.5-4B
- Repeated generations: 3
- Bootstrap replicates: 2,000
- Bootstrap seed: `20260810`
- Exact prompts: `current_llm/prompts.json`
- Frozen model/configuration registry: `current_llm/model_registry.json`

The model's own ambiguous answer is preserved as the assistant turn before disambiguating evidence is introduced in the two-turn trajectory.

## Validation safeguard

`code/validate_current_llm.py` blocks analysis if the run is incomplete, duplicated, out-of-scope, contains recorded API failures, or falls below the declared parseability threshold. `scripts/run_experiment2.py` invokes this gate automatically before publication outputs are generated.

## Data

The full BBQ archive is **not duplicated in this repository**. `code/prepare_bbq.py` retrieves the official BBQ repository for Experiment 1 and records the exact source commit. The frozen Experiment-2 subset is included directly for reproducibility.

See `docs/DATA.md`, `docs/MODELS.md`, and `third_party/`.

## Licensing

Original E-EA code and documentation: MIT License.  
BBQ-derived material remains subject to the original BBQ licensing and attribution requirements.

## Citation

See `CITATION.cff`.
