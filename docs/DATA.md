# Data provenance and reproducibility

## Official BBQ source

This project uses **BBQ: A Hand-Built Bias Benchmark for Question Answering** by Parrish et al. (2022).

Official repository: `https://github.com/nyu-mll/BBQ`

Paper DOI: `10.18653/v1/2022.findings-acl.165`

The official repository contains:
- 11 benchmark JSONL files under `data/`;
- UnifiedQA published predictions under `results/UnifiedQA/`;
- RoBERTa/DeBERTa-v3 published predictions under `results/RoBERTa_and_DeBERTaV3/`;
- `supplemental/additional_metadata.csv`;
- the CC BY 4.0 license.

## Experiment 1

`code/prepare_bbq.py` clones the official repository and maps the source files into the directory structure expected by `code/eea_bbq.py`. The Git commit is stored in `runtime/experiment1/BBQ_SOURCE.json`.

Run:

```bash
python scripts/run_experiment1.py
```

No third-party model API is required because Experiment 1 uses the prediction files published with BBQ.

## Experiment 2 frozen subset

The repository directly includes:

`current_llm/bbq_current_llm_subset.jsonl`

It contains exactly 1,000 unique matched ambiguous/disambiguated pairs sampled across all 11 BBQ files using seed `20260811`.

Frozen SHA-256:

`3709c95538b98c87b837441626da2c585d034b53deff39526f6685ea35782187`

The subset is derived from BBQ and remains subject to the original BBQ attribution/licensing terms. The MIT license in this repository applies only to original E-EA software/documentation and does not replace third-party dataset licensing.
