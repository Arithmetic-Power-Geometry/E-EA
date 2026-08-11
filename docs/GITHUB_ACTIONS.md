# Running E-EA with GitHub Actions

## Repository secrets

Create secrets under:

`Settings -> Secrets and variables -> Actions -> Secrets`

Required for the full prospective Experiment 2:

- `DEEPSEEK_API_KEY`
- `MISTRAL_API_KEY`
- `QWEN_API_KEY`

Do not place API keys in source files, workflow YAML, issues, or commits.

## Repository variable

Create under:

`Settings -> Secrets and variables -> Actions -> Variables`

- `QWEN_BASE_URL`

It must be the base URL of an OpenAI-compatible hosted inference endpoint serving the frozen model `Qwen/Qwen3.5-4B`. The standard GitHub-hosted runner is used only as the experiment orchestrator; it does not have the GPU resources expected for local Qwen inference.

## Workflows

### CI

Runs on pushes/pull requests and can also be started manually. It compiles Python, runs tests, and verifies the frozen subset checksum.

### Experiment 1 - retrospective BBQ

Manual, no LLM credentials required. It clones the official BBQ repository, records the exact source commit, runs the historical analysis, and uploads result tables/figures as a workflow artifact.

### Experiment 2 - contemporary LLMs

Manual. Choose one model or all three and choose:

- `smoke`: first 10 frozen pairs
- `full`: all 1,000 frozen pairs

The workflow validates outputs before analysis. When `model=all`, a final combined analysis is produced after all three model jobs pass.

### Run all experiments

Manual convenience workflow. It runs Experiment 1 and the three prospective contemporary-model jobs in parallel where possible, validates the contemporary evidence, then automatically generates Experiment-1 outputs, Experiment-2 outputs, and the combined manuscript-ready tables/figures. Use `scope=smoke` before `scope=full`.

## Artifacts

GitHub Actions artifacts retain the generated raw output and/or result files after a job finishes, so they can be downloaded from the workflow-run page. Raw API keys are not included in artifacts.
