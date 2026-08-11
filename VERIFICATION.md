# Verification record

Verification date: 2026-08-11

Before packaging, the repository was checked as follows:

- Python source compilation: passed.
- Unit tests: 9/9 passed.
- Repository audit: passed.
- All four GitHub Actions workflow YAML files parsed successfully.
- The frozen Experiment-2 subset checksum and 1,000 unique pair IDs passed the repository audit.
- The Experiment-2 analyzer was exercised with temporary non-evidence test records outside the repository and generated all declared CSV, LaTeX, PDF, PNG, statistical, and manifest outputs.
- The combined historical/contemporary output builder was exercised using reproduced Experiment-1 outputs plus temporary non-evidence Experiment-2 test records and generated all declared combined tables and figures.
- Temporary test records and generated test results are not included in this repository and are not manuscript evidence.
- Real Experiment-2 results become manuscript evidence only after the frozen inference run completes and `validate_current_llm.py` passes.

Experiment 1 uses the completed retrospective analysis pipeline and automatically retrieves/prepares the official BBQ release when executed.
