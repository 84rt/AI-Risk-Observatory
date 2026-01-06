# Pipeline

This directory houses the pipeline source, scripts, and tests. Full docs live in `context/pipeline_docs/` (architecture, guides, and status notes).

## Layout
- `src/` — core pipeline code
- `scripts/` — helper/maintenance CLIs (diagnostics, downloads, setup)
- `tests/` — test entrypoints and fixtures

## Data locations
Pipeline reads/writes under the repo-level `data/` tree:
- Inputs/reference: `data/reference/`
- Generated: `data/raw/`, `data/processed/`, `data/results/`, `data/logs/`, `data/db/`

Regenerate data by running the pipeline (e.g., `python run_pipeline.py --companies ../data/reference/companies_template.csv`).
