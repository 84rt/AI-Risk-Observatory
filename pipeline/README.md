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

## Golden set Phase 1 (CNI appendix)
- Download + preprocess: `python scripts/golden_set_phase1.py --all --years 2024 2023`  
  - Downloads → `data/raw/{ixbrl,pdfs}/{year}/`  
  - Manifest → `data/runs/{run_id}/ingestion.json`  
  - Preprocessed text → `data/processed/{run_id}/documents.parquet`
- Verify coverage: `python scripts/golden_set_phase1.py --verify --run-id <run_id>`
- Annotate: open `notebooks/golden_set_annotation.ipynb`, set `RUN_ID`, add labels, then append to `data/annotations/human/annotations.parquet` (JSONL also written).
- Load annotations into SQLite: `python scripts/load_golden_to_db.py --run-id <run_id>` (writes to `mentions` and `risk_classifications`).
