# CLAUDE.md

Minimal guidance for Claude Code when working in `pipeline/`. The current focus is implementing the golden set plan in `context/golden_set_implementation_plan.md`.

## Guardrails

- Never commit real API keys. Use placeholders only (e.g., `GEMINI_API_KEY=your_gemini_api_key_here`, `COMPANIES_HOUSE_API_KEY=your_companies_house_api_key_here`). Real keys live in `.env.local` or `.env` (both gitignored).
- Keep edits scoped to `pipeline/` unless explicitly asked otherwise.

## Quick start

```bash
cd pipeline
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.template .env  # then add placeholder keys
```

Run pipeline (examples):

- `python run_pipeline.py --companies data/companies_template.csv`
- `python run_pipeline.py --year 2024`
- `python run_pipeline.py --skip-download`
- Enable debug logs: `python run_pipeline.py --log-level DEBUG`

## Key files (pipeline/)

- `run_pipeline.py` – CLI entry point
- `src/pipeline.py` – orchestration
- `src/preprocessor.py` / `src/chunker.py` – text filtering and chunking
- `src/llm_classifier.py` – Gemini classification
- `src/database.py` / `src/aggregator.py` – storage and rollups
- Data dirs: `../data/raw`, `../data/processed/preprocessed/`, `../data/db/airo.db`

## File organization (pipeline/)

```text
pipeline/
├── run_pipeline.py                 # CLI entry point
├── requirements.txt                # Python dependencies
├── logs/pipeline.log               # Pipeline log output
├── src/                            # Core modules
│   ├── config.py                   # Settings from .env files
│   ├── pipeline.py                 # Main orchestrator
│   ├── xbrl_filings_client.py      # filings.xbrl.org API client
│   ├── companies_house.py          # Companies House API client
│   ├── ixbrl_extractor.py          # iXBRL/XHTML extraction + text cleaning
│   ├── pdf_extractor.py            # PDF extraction (fallback)
│   ├── preprocessor.py             # Text filtering strategies
│   ├── chunker.py                  # Text chunking logic
│   ├── llm_classifier.py           # Gemini classification
│   ├── database.py                 # SQLAlchemy models
│   └── aggregator.py               # Firm-level aggregation
└── ../data/
    ├── raw/                        # Downloaded iXBRL and PDFs
    ├── processed/preprocessed/     # Markdown extracts
    ├── results/                    # Classification outputs and samples
    ├── annotations/                # Exports for review
    └── db/airo.db                  # SQLite database
```

## Working notes for golden set work

- Source plan: `context/golden_set_implementation_plan.md`
- Prefer small, testable steps; document notable changes.
- If you add new dimensions or outputs, update models, prompts, and aggregation together.
- Phase 1 helpers:
  - `scripts/golden_set_phase1.py` (ingest + preprocess + verify into `data/raw` and `data/processed/{run_id}`)
  - `notebooks/golden_set_annotation.ipynb` (human labels → `data/annotations/human/annotations.parquet|jsonl`)
  - `scripts/load_golden_to_db.py` (load annotations into SQLite `mentions`/`risk_classifications`)
