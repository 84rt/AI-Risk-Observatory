QA Suite

Purpose
- Focused, offline checks that validate document quality and extraction sanity.
- Designed to catch obvious regressions (empty output, broken spacing, bad ratios).

How to run
- From repo root: `pipeline/venv/bin/python -m pytest pipeline/tests/qa_suite`

Guidelines
- Keep tests deterministic and offline (no network).
- Prefer a single representative raw iXBRL file found under `data/raw/ixbrl`.
- Use conservative thresholds so tests flag real regressions, not minor variance.
