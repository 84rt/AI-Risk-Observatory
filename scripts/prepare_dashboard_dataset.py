#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare dashboard-ready data files from a processed annotation run."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to final merged annotations JSONL.",
    )
    parser.add_argument(
        "--documents-manifest",
        type=Path,
        required=True,
        help="Path to documents_manifest.json for the processed run.",
    )
    parser.add_argument(
        "--companies-csv",
        type=Path,
        required=True,
        help="Path to selected companies CSV for the cohort.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for dashboard-ready files.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def normalize_market_segment(row: dict[str, str]) -> str:
    legacy = (row.get("market_segment") or "").strip()
    refined = (row.get("market_segment_refined") or "").strip()
    if legacy == "FTSE 350":
        return legacy
    if refined:
        return refined
    return legacy or "Other"


def build_document_months(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text())
    docs = manifest.get("documents", [])
    mapping: dict[str, str] = {}
    for doc in docs:
        publication_date = (doc.get("publication_date") or "").strip()
        if len(publication_date) < 7:
            continue
        month = publication_date[:7]
        document_id = (doc.get("document_id") or "").strip()
        company_name = (doc.get("company_name") or "").strip()
        year = doc.get("year")
        if document_id:
            mapping[document_id] = month
        if company_name and year is not None:
            mapping[f"{company_name}|||{year}"] = month
    return mapping


def build_companies_csv(companies_path: Path, output_path: Path) -> None:
    with companies_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    fieldnames = [
        "company_name",
        "lei",
        "market_segment",
        "market_segment_refined",
        "cni_sector",
        "isic_sector_name",
        "source_type",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "company_name": (row.get("company_name") or "").strip(),
                    "lei": (row.get("lei") or "").strip(),
                    "market_segment": normalize_market_segment(row),
                    "market_segment_refined": (row.get("market_segment_refined") or "").strip(),
                    "cni_sector": (row.get("cni_sector_primary") or row.get("cni_sector") or "").strip() or "Unknown",
                    "isic_sector_name": (row.get("isic_name") or row.get("isic_sector_name") or "").strip() or "Unknown",
                    "source_type": "Definitive Main Market 1000",
                }
            )


def main() -> None:
    args = parse_args()
    annotations_path = resolve_path(args.annotations)
    manifest_path = resolve_path(args.documents_manifest)
    companies_path = resolve_path(args.companies_csv)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(annotations_path, output_dir / "annotations.jsonl")

    document_months = build_document_months(manifest_path)
    (output_dir / "document_months.json").write_text(
        json.dumps(document_months, indent=2, sort_keys=True) + "\n"
    )

    build_companies_csv(companies_path, output_dir / "golden_set_companies.csv")

    summary = {
        "annotations_source": str(annotations_path.relative_to(REPO_ROOT)),
        "documents_manifest_source": str(manifest_path.relative_to(REPO_ROOT)),
        "companies_source": str(companies_path.relative_to(REPO_ROOT)),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "document_month_keys": len(document_months),
    }
    (output_dir / "dashboard_export_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(f"Wrote {output_dir / 'annotations.jsonl'}")
    print(f"Wrote {output_dir / 'document_months.json'}")
    print(f"Wrote {output_dir / 'golden_set_companies.csv'}")
    print(f"Wrote {output_dir / 'dashboard_export_summary.json'}")


if __name__ == "__main__":
    main()
