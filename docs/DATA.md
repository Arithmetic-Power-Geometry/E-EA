# Data provenance and layout

## BBQ source

This project uses **BBQ: A Hand-Built Bias Benchmark for Question Answering** by Parrish et al. (2022).

Official repository: https://github.com/nyu-mll/BBQ

Paper DOI: https://doi.org/10.18653/v1/2022.findings-acl.165

The official BBQ repository is distributed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. A copy of the license text supplied with the project snapshot is retained at `third_party/BBQ_LICENSE_CC_BY_4.0.txt`.

## What this repository includes

The repository includes only the frozen 1,000-pair derived subset used by the prospective contemporary-LLM protocol:

`current_llm/bbq_current_llm_subset.jsonl`

This subset is derived from BBQ and remains subject to the original BBQ CC BY 4.0 attribution and licensing terms. The repository MIT license applies only to original E-EA software and documentation and does not replace third-party dataset licensing.

## What this repository does not duplicate

The complete BBQ release and the historical published model-prediction files used for Experiment 1 are not duplicated here. Obtain them from the official BBQ repository and arrange them as:

```text
<root>/
  data/
    BBQ_data/
      Age.jsonl
      Disability_status.jsonl
      Gender_identity.jsonl
      Nationality.jsonl
      Physical_appearance.jsonl
      Race_ethnicity.jsonl
      Race_x_SES.jsonl
      Race_x_gender.jsonl
      Religion.jsonl
      SES.jsonl
      Sexual_orientation.jsonl
    BBQ_published_results/
      UnifiedQA/
        preds_Age.jsonl
        preds_Disability_status.jsonl
        preds_Gender_identity.jsonl
        preds_Nationality.jsonl
        preds_Physical_appearance.jsonl
        preds_Race_ethnicity.jsonl
        preds_Race_x_SES.jsonl
        preds_Race_x_gender.jsonl
        preds_Religion.jsonl
        preds_SES.jsonl
        preds_Sexual_orientation.jsonl
      RoBERTa_and_DeBERTaV3/
        df_bbq.csv
    additional_metadata.csv
```

Run Experiment 1 with:

```bash
python code/eea_bbq.py --root <root> --bootstrap 2000
```

## Frozen-subset reproducibility

The prospective contemporary-LLM subset contains exactly 1,000 unique matched BBQ pairs and was constructed with seed `20260811`. It can be reproduced from the full BBQ layout above using:

```bash
python code/build_current_llm_subset.py \
  --root <root> \
  --pairs 1000 \
  --seed 20260811 \
  --out current_llm/bbq_current_llm_subset.rebuilt.jsonl
```

The distributed frozen subset SHA-256 is:

`3709c95538b98c87b837441626da2c585d034b53deff39526f6685ea35782187`

## Attribution

Please cite Parrish et al. (2022) when using BBQ and comply with the original CC BY 4.0 terms.
