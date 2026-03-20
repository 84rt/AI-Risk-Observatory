#!/usr/bin/env python3
"""Prepare data/processed/<run_id>/documents_manifest.json for local FR markdown runs.

This is the FR-direct equivalent of a preprocessing step: it does not transform
markdown, it simply writes the manifest expected by pipeline/scripts/chunk_markdown.py
and the downstream batch orchestrator.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = (
    REPO_ROOT / "data" / "reference" / "batch_1000_definitive_main_market" / "reports_1000.csv"
)
DEFAULT_QUEUE = (
    REPO_ROOT / "data" / "reference" / "batch_1000_definitive_main_market" / "processing_queue_ready.json"
)
DEFAULT_RUN_ID = "fr-phase1-definitive-main-market-1000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-csv",
        type=Path,
        default=DEFAULT_REPORTS,
        help="Reports CSV with one local-ready row per report.",
    )
    parser.add_argument(
        "--queue-json",
        type=Path,
        default=DEFAULT_QUEUE,
        help="Source queue JSON path to record in the manifest.",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help=f"Processed run ID (default: {DEFAULT_RUN_ID})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing documents_manifest.json for this run ID.",
    )
    return parser.parse_args()


def make_absolute(path_str: str) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def main() -> None:
    args = parse_args()

    if not args.reports_csv.exists():
        raise FileNotFoundError(f"Reports CSV not found: {args.reports_csv}")
    if not args.queue_json.exists():
        raise FileNotFoundError(f"Queue JSON not found: {args.queue_json}")

    processed_dir = REPO_ROOT / "data" / "processed" / args.run_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = processed_dir / "documents_manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"Manifest already exists at {manifest_path}. Re-run with --force to overwrite."
        )

    documents: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, int]] = set()

    with args.reports_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("local_status") != "local_markdown_ready":
                raise RuntimeError(
                    "Reports CSV contains non-local-ready rows. "
                    "Use a definitive ready-now reports CSV for this helper."
                )

            lei = (row.get("lei") or "").strip()
            fr_pk = (row.get("fr_pk") or "").strip()
            company_name = (row.get("company_name") or "").strip()
            markdown_path = make_absolute((row.get("markdown_path") or "").strip())
            fiscal_year = int(row["fiscal_year"])

            if not Path(markdown_path).exists():
                raise FileNotFoundError(f"Missing markdown file at {markdown_path}")

            pair = (lei, fiscal_year)
            if pair in seen_pairs:
                raise RuntimeError(f"Duplicate (lei, fiscal_year) pair in reports CSV: {pair}")
            seen_pairs.add(pair)

            documents.append(
                {
                    "document_id": f"fr-{fr_pk}",
                    "company_id": lei,
                    "company_number": None,
                    "company_name": company_name,
                    "ticker": company_name,
                    "lei": lei,
                    "cni_sector": row.get("cni_sector_primary", ""),
                    "isic_sector": row.get("isic_name", ""),
                    "market_segment": row.get("market_segment", ""),
                    "market_segment_refined": row.get("market_segment_refined", ""),
                    "year": fiscal_year,
                    "run_id": args.run_id,
                    "status": "downloaded",
                    "source_format": "markdown",
                    "preprocess_strategy": "fr_direct_markdown",
                    "markdown_path": markdown_path,
                    "source": "financialreports_queue",
                    "source_pk": fr_pk,
                    "source_title": row.get("fr_title", ""),
                    "publication_date": row.get("publication_date", ""),
                    "publication_datetime": row.get("publication_datetime", ""),
                }
            )

    payload = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_queue": str(args.queue_json.relative_to(REPO_ROOT)),
        "documents": documents,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {manifest_path}")
    print(f"Documents: {len(documents)}")


if __name__ == "__main__":
    main()
