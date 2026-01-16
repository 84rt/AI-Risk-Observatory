Tests Overview

QA suite (offline, deterministic):
- Location: pipeline/tests/qa_suite
- Run: pipeline/venv/bin/python -m pytest pipeline/tests/qa_suite

Integration/manual tests:
- Network/API, model comparisons, and download checks stay in pipeline/tests
- Run individually as needed
