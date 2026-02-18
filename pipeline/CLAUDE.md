# CLAUDE.md

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

## Gemini Batch API Gotchas
- `gemini-3-flash-preview` uses thinking tokens by default, and they count against `max_output_tokens`. Use at least `2048` for simple schemas (thinking alone averages ~650 tokens). The phase 2 classifiers already use 2048; the substantiveness testbed initially used 1024 and hit 30% truncation errors.

## Guardrails
- Never commit real API keys. Use `.env` (gitignored).
