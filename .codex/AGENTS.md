## Source of Truth
Refer to `context/golden_set_implementation_plan.md` for the current plan and data layout.

## Quick Start
```bash
cd pipeline
source venv/bin/activate
pip install -r requirements.txt
```

## Current Data State
- Raw iXBRL files: `data/raw/ixbrl/{year}/`
- Processed outputs: `data/processed/<run_id>/`
- Database: `data/db/airo.db`

## Key Scripts
- `scripts/golden_set_phase1.py` — download, preprocess, verify
- `scripts/chunk_markdown.py` — generate AI-mention chunks
- `tests/qa_manager.py` — run QA checks


- Note: If you use Fish shell, activate with `source pipeline/venv/bin/activate.fish` (this should not cause issues).
- The root-level `.env.local` file is used for environment configuration and secrets shared by all pipeline code.