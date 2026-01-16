#!/usr/bin/env python3
"""Load company list and ingestion manifest into SQLite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.company_utils import load_companies_csv
from src.config import get_settings
from src.database import Database
from src.identifiers import make_company_id, make_document_id


def load_manifest(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("records", [])


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    default_companies = settings.data_dir / "reference" / "golden_set_companies.csv"
    parser = argparse.ArgumentParser(description="Load golden set data into SQLite")
    parser.add_argument(
        "--companies",
        type=Path,
        default=default_companies,
        help=f"Path to companies CSV (default: {default_companies})",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID to locate data/runs/<run_id>/ingestion.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Explicit path to ingestion.json (overrides --run-id)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    db = Database()

    companies = load_companies_csv(args.companies)

    manifest_path = args.manifest
    if not manifest_path and args.run_id:
        manifest_path = settings.data_dir / "runs" / args.run_id / "ingestion.json"

    manifest_records: list[dict] = []
    if manifest_path and manifest_path.exists():
        manifest_records = load_manifest(manifest_path)
    elif manifest_path:
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    session = db.get_session()
    try:
        for company in companies:
            db.upsert_company(session, company)

        for record in manifest_records:
            company_number = record.get("company_number")
            company_name = record.get("company_name") or ""
            lei = record.get("lei")
            sector = record.get("cni_sector") or record.get("sector") or "Unknown"
            year = record.get("year")
            fmt = record.get("format") or "unknown"

            company_id = record.get("company_id") or make_company_id(
                company_number, lei, company_name
            )
            document_id = make_document_id(
                company_number, lei, company_name, year, sector, fmt
            )

            db.upsert_document(
                session,
                {
                    "document_id": document_id,
                    "company_id": company_id,
                    "company_name": company_name,
                    "company_number": company_number,
                    "lei": lei,
                    "ticker": record.get("ticker"),
                    "sector": sector,
                    "report_year": year,
                    "source_format": record.get("format"),
                    "raw_path": record.get("raw_path"),
                    "checksum_sha256": record.get("checksum_sha256"),
                    "source": record.get("source"),
                    "status": record.get("status"),
                    "error": record.get("error"),
                    "run_id": record.get("run_id"),
                },
            )

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    print(f"✅ Loaded {len(companies)} companies into SQLite")
    if manifest_records:
        print(f"✅ Loaded {len(manifest_records)} ingestion records into SQLite")


if __name__ == "__main__":
    main()
