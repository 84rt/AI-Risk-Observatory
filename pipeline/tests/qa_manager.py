#!/usr/bin/env python3
"""QA manager for preprocessing and chunking stages."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from tests import qa_chunking, qa_preprocessing  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QA checks for pipeline stages")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID for data/processed/<run_id>/",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["preprocess", "chunking", "all"],
        help="Pipeline stage to validate (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write QA reports",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=None,
        help="Optional explicit chunks.jsonl path",
    )
    return parser.parse_args()


def _safe_run(step_name: str, func, *args, **kwargs) -> None:
    try:
        func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("QA step '%s' failed: %s", step_name, exc)


def run(stage: str, run_id: str, output_dir: Optional[Path], chunks_path: Optional[Path]) -> None:
    if stage in ("preprocess", "all"):
        _safe_run(
            "preprocess",
            qa_preprocessing.run_qa,
            run_id=run_id,
            output_dir=output_dir,
        )
    if stage in ("chunking", "all"):
        _safe_run(
            "chunking",
            qa_chunking.run_qa,
            run_id=run_id,
            chunks_path=chunks_path,
            output_dir=output_dir,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    run(
        stage=args.stage,
        run_id=args.run_id,
        output_dir=args.output_dir,
        chunks_path=args.chunks_path,
    )


if __name__ == "__main__":
    main()
