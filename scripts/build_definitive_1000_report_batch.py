#!/usr/bin/env python3
"""Build the definitive 1000-report ready-now batch with Main Market swaps.

Strategy:
- Start from the FTSE350 fiscal-year cohort in data/reference/batch_1000_ftse350.
- Keep the 195 companies that already have all five local markdown files.
- Drop the 5 incomplete companies.
- Replace them with 5 non-selected Main Market companies that already have
  5/5 local markdown coverage for fiscal years 2021-2025.

Outputs:
- data/reference/batch_1000_definitive_main_market/companies_200.csv
- data/reference/batch_1000_definitive_main_market/reports_1000.csv
- data/reference/batch_1000_definitive_main_market/processing_queue_ready.json
- data/reference/batch_1000_definitive_main_market/swap_summary.csv
- data/reference/batch_1000_definitive_main_market/summary.json
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_BATCH_DIR = REPO_ROOT / "data" / "reference" / "batch_1000_ftse350"
OUTPUT_DIR = REPO_ROOT / "data" / "reference" / "batch_1000_definitive_main_market"
SOURCE_COMPANIES = BASE_BATCH_DIR / "companies_200.csv"
SOURCE_REPORTS = BASE_BATCH_DIR / "reports_1000.csv"
TARGET_YEARS = (2021, 2022, 2023, 2024, 2025)
TARGET_YEAR_SET = set(TARGET_YEARS)
SWAP_COUNT = 5


def load_base_builder() -> Any:
    builder_path = REPO_ROOT / "scripts" / "build_ftse350_1000_report_batch.py"
    spec = importlib.util.spec_from_file_location("ftse350_batch_builder", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load builder module from {builder_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_int(row: dict[str, str], key: str) -> int:
    return int(row.get(key) or 0)


def build_replacement_pool(builder: Any, excluded_leis: set[str]) -> list[dict[str, object]]:
    markdown_index = builder.build_markdown_index()
    fr_index = builder.build_fr_index()
    target_by_lei, _ = builder.load_target_rows()

    pool: list[dict[str, object]] = []
    for lei, rows in target_by_lei.items():
        if lei in excluded_leis:
            continue
        rows = sorted(rows, key=lambda item: int(item["fiscal_year"]))
        if {int(row["fiscal_year"]) for row in rows} != TARGET_YEAR_SET:
            continue
        if rows[0].get("market_segment_refined") != "Main Market":
            continue

        enriched_rows: list[dict[str, object]] = []
        for row in rows:
            enriched = builder.enrich_report_row(row, fr_index, markdown_index)
            if enriched is None:
                enriched_rows = []
                break
            enriched_rows.append(enriched)

        if len(enriched_rows) != len(TARGET_YEARS):
            continue

        local_ready = sum(1 for row in enriched_rows if row["local_status"] == "local_markdown_ready")
        if local_ready != len(TARGET_YEARS):
            continue

        status_counts = Counter(str(row["fr_status"]) for row in enriched_rows)
        pool.append(
            {
                "lei": lei,
                "company_name": rows[0]["company_name"],
                "market_segment": rows[0]["market_segment"],
                "market_segment_refined": rows[0]["market_segment_refined"],
                "cni_sector_primary": rows[0]["cni_sector_primary"],
                "isic_name": rows[0]["isic_name"],
                "local_markdown_ready_count": local_ready,
                "md_available_count": status_counts["md_available"],
                "fr_no_status_count": status_counts["fr_no_status"],
                "fr_pending_count": status_counts["fr_pending"],
                "fr_failed_count": status_counts["fr_failed"],
                "ready_score": sum(int(row["report_score"]) for row in enriched_rows),
                "report_rows": enriched_rows,
            }
        )

    pool.sort(
        key=lambda row: (
            -int(row["md_available_count"]),
            int(row["fr_failed_count"]),
            int(row["fr_pending_count"]),
            int(row["fr_no_status_count"]),
            str(row["company_name"]).lower(),
            str(row["lei"]),
        )
    )
    return pool


def main() -> None:
    builder = load_base_builder()

    company_rows = read_csv(SOURCE_COMPANIES)
    report_rows = read_csv(SOURCE_REPORTS)
    reports_by_lei: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in report_rows:
        reports_by_lei[row["lei"]].append(row)

    dropped_companies = [
        row for row in company_rows if as_int(row, "local_markdown_ready_count") < len(TARGET_YEARS)
    ]
    if len(dropped_companies) != SWAP_COUNT:
        raise RuntimeError(f"Expected {SWAP_COUNT} incomplete companies, found {len(dropped_companies)}")

    kept_companies = [
        row for row in company_rows if as_int(row, "local_markdown_ready_count") == len(TARGET_YEARS)
    ]
    if len(kept_companies) != len(company_rows) - SWAP_COUNT:
        raise RuntimeError("Unexpected kept/dropped company split")

    excluded_leis = {row["lei"] for row in company_rows}
    replacements = build_replacement_pool(builder, excluded_leis)[:SWAP_COUNT]
    if len(replacements) != SWAP_COUNT:
        raise RuntimeError(f"Need {SWAP_COUNT} replacement companies, found {len(replacements)}")

    definitive_company_rows: list[dict[str, object]] = []
    definitive_report_rows: list[dict[str, object]] = []
    definitive_queue: list[dict[str, object]] = []

    kept_companies.sort(key=lambda row: as_int(row, "selection_rank"))

    for company in kept_companies:
        rank = as_int(company, "selection_rank")
        definitive_company_rows.append(dict(company))
        for report in sorted(reports_by_lei[company["lei"]], key=lambda item: int(item["fiscal_year"])):
            report_copy = dict(report)
            report_copy["selection_rank"] = str(rank)
            definitive_report_rows.append(report_copy)
            definitive_queue.append(
                {
                    "company_name": report_copy["company_name"],
                    "lei": report_copy["lei"],
                    "year": int(report_copy["fiscal_year"]),
                    "cni_sector": report_copy["cni_sector_primary"],
                    "isic_sector": report_copy["isic_name"],
                    "file_path": report_copy["markdown_path"],
                    "pk": report_copy["fr_pk"],
                    "source_title": report_copy["fr_title"],
                    "publication_date": report_copy["publication_date"],
                }
            )

    dropped_companies.sort(key=lambda row: as_int(row, "selection_rank"))
    swap_rows: list[dict[str, object]] = []

    for dropped, replacement in zip(dropped_companies, replacements, strict=True):
        rank = as_int(dropped, "selection_rank")
        definitive_company_rows.append(
            {
                "selection_rank": str(rank),
                "lei": replacement["lei"],
                "company_name": replacement["company_name"],
                "market_segment": replacement["market_segment"],
                "market_segment_refined": replacement["market_segment_refined"],
                "cni_sector_primary": replacement["cni_sector_primary"],
                "isic_name": replacement["isic_name"],
                "local_markdown_ready_count": str(replacement["local_markdown_ready_count"]),
                "md_available_count": str(replacement["md_available_count"]),
                "fr_no_status_count": str(replacement["fr_no_status_count"]),
                "fr_pending_count": str(replacement["fr_pending_count"]),
                "fr_failed_count": str(replacement["fr_failed_count"]),
                "ready_score": str(replacement["ready_score"]),
            }
        )

        for report in replacement["report_rows"]:
            report_copy = dict(report)
            report_copy["selection_rank"] = str(rank)
            definitive_report_rows.append(report_copy)
            definitive_queue.append(
                {
                    "company_name": report_copy["company_name"],
                    "lei": report_copy["lei"],
                    "year": int(report_copy["fiscal_year"]),
                    "cni_sector": report_copy["cni_sector_primary"],
                    "isic_sector": report_copy["isic_name"],
                    "file_path": report_copy["markdown_path"],
                    "pk": report_copy["fr_pk"],
                    "source_title": report_copy["fr_title"],
                    "publication_date": report_copy["publication_date"],
                }
            )

        missing_reports = sorted(
            [row for row in reports_by_lei[dropped["lei"]] if row["local_status"] != "local_markdown_ready"],
            key=lambda item: int(item["fiscal_year"]),
        )
        swap_rows.append(
            {
                "selection_rank": str(rank),
                "dropped_company_name": dropped["company_name"],
                "dropped_lei": dropped["lei"],
                "dropped_local_markdown_ready_count": dropped["local_markdown_ready_count"],
                "dropped_missing_fiscal_years": "|".join(row["fiscal_year"] for row in missing_reports),
                "dropped_missing_report_pks": "|".join(row["fr_pk"] for row in missing_reports),
                "replacement_company_name": replacement["company_name"],
                "replacement_lei": replacement["lei"],
                "replacement_market_segment": replacement["market_segment"],
                "replacement_market_segment_refined": replacement["market_segment_refined"],
                "replacement_local_markdown_ready_count": str(replacement["local_markdown_ready_count"]),
                "replacement_md_available_count": str(replacement["md_available_count"]),
            }
        )

    definitive_company_rows.sort(key=lambda row: as_int(row, "selection_rank"))
    definitive_report_rows.sort(key=lambda row: (as_int(row, "selection_rank"), int(row["fiscal_year"])))
    definitive_queue.sort(key=lambda row: (str(row["company_name"]).lower(), int(row["year"])))

    if len(definitive_company_rows) != 200:
        raise RuntimeError(f"Expected 200 definitive companies, found {len(definitive_company_rows)}")
    if len(definitive_report_rows) != 1000:
        raise RuntimeError(f"Expected 1000 definitive reports, found {len(definitive_report_rows)}")
    if any(row["local_status"] != "local_markdown_ready" for row in definitive_report_rows):
        raise RuntimeError("Definitive report set is not fully local-ready")
    if len(definitive_queue) != 1000:
        raise RuntimeError(f"Expected 1000 queue items, found {len(definitive_queue)}")

    summary = {
        "selection_rule": {
            "base_batch": "data/reference/batch_1000_ftse350",
            "kept_complete_ftse350_companies": len(kept_companies),
            "dropped_incomplete_companies": len(dropped_companies),
            "replacement_pool_rule": [
                "outside current selected 200",
                "market_segment_refined == Main Market",
                "five fiscal years present: 2021-2025",
                "all five reports local_markdown_ready",
                "prefer md_available_count desc",
                "then fr_failed_count asc",
                "then fr_pending_count asc",
                "then fr_no_status_count asc",
                "then company_name asc",
            ],
        },
        "selected": {
            "company_count": len(definitive_company_rows),
            "report_count": len(definitive_report_rows),
            "ready_queue_count": len(definitive_queue),
            "company_local_markdown_ready_distribution": Counter(
                as_int(row, "local_markdown_ready_count") for row in definitive_company_rows
            ),
            "report_counts_by_local_status": Counter(row["local_status"] for row in definitive_report_rows),
            "report_counts_by_market_segment": Counter(row["market_segment"] for row in definitive_report_rows),
            "report_counts_by_market_segment_refined": Counter(
                row["market_segment_refined"] for row in definitive_report_rows
            ),
        },
        "swaps": swap_rows,
        "output_files": {
            "companies_csv": "data/reference/batch_1000_definitive_main_market/companies_200.csv",
            "reports_csv": "data/reference/batch_1000_definitive_main_market/reports_1000.csv",
            "processing_queue_ready_json": "data/reference/batch_1000_definitive_main_market/processing_queue_ready.json",
            "swap_summary_csv": "data/reference/batch_1000_definitive_main_market/swap_summary.csv",
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / "companies_200.csv",
        definitive_company_rows,
        [
            "selection_rank",
            "lei",
            "company_name",
            "market_segment",
            "market_segment_refined",
            "cni_sector_primary",
            "isic_name",
            "local_markdown_ready_count",
            "md_available_count",
            "fr_no_status_count",
            "fr_pending_count",
            "fr_failed_count",
            "ready_score",
        ],
    )
    write_csv(
        OUTPUT_DIR / "reports_1000.csv",
        definitive_report_rows,
        [
            "selection_rank",
            "lei",
            "company_name",
            "market_segment",
            "market_segment_refined",
            "cni_sector_primary",
            "isic_name",
            "fiscal_year",
            "publication_date",
            "publication_month",
            "publication_datetime",
            "release_year",
            "fr_pk",
            "fr_filing_type",
            "fr_title",
            "fr_status",
            "local_status",
            "report_score",
            "markdown_path",
            "markdown_chars",
            "ch_made_up_date",
            "ch_submission_date",
        ],
    )
    (OUTPUT_DIR / "processing_queue_ready.json").write_text(
        json.dumps(definitive_queue, indent=2), encoding="utf-8"
    )
    write_csv(
        OUTPUT_DIR / "swap_summary.csv",
        swap_rows,
        [
            "selection_rank",
            "dropped_company_name",
            "dropped_lei",
            "dropped_local_markdown_ready_count",
            "dropped_missing_fiscal_years",
            "dropped_missing_report_pks",
            "replacement_company_name",
            "replacement_lei",
            "replacement_market_segment",
            "replacement_market_segment_refined",
            "replacement_local_markdown_ready_count",
            "replacement_md_available_count",
        ],
    )
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {OUTPUT_DIR / 'companies_200.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'reports_1000.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'processing_queue_ready.json'}")
    print(f"Wrote {OUTPUT_DIR / 'swap_summary.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'summary.json'}")
    print()
    print(f"Kept complete FTSE350 companies: {len(kept_companies)}")
    print(f"Replacements added: {len(replacements)}")
    print(f"Definitive ready-now queue items: {len(definitive_queue)}")


if __name__ == "__main__":
    main()
