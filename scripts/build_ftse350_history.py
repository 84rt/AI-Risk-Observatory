#!/usr/bin/env python3
"""CLI wrapper for the FTSE 350 historical membership collector."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from src.ftse350_history import main


if __name__ == "__main__":
    raise SystemExit(main())
