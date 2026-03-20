#!/usr/bin/env python3
"""Build the clean FTSE350 200-company / 1000-report batch manifests.

Source of truth:
- data/reference/target_manifest.csv for canonical company x fiscal_year rows.
- data/FR_dataset/manifest.csv for exact FR publication timestamps and titles.
- local markdown directories for current batch readiness.

Eligibility:
- market_segment == "FTSE 350"
- has exactly one target-manifest row for each fiscal year 2021-2025
- no target year is marked not_in_fr
- every target year has an FR pk and FR release metadata

Selection:
- choose the 200 most batch-ready eligible companies from the 208 eligible
- prioritize more local markdown already present
- break ties by fewer failed / pending / no-status reports, then alphabetical

Outputs:
- data/reference/batch_1000_ftse350/companies_200.csv
- data/reference/batch_1000_ftse350/reports_1000.csv
- data/reference/batch_1000_ftse350/processing_queue_ready.json
- data/reference/batch_1000_ftse350/reports_missing_local.csv
- data/reference/batch_1000_ftse350/companies_reserve.csv
- data/reference/batch_1000_ftse350/summary.json
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
TARGET_MANIFEST = DATA_DIR / "reference" / "target_manifest.csv"
FR_MANIFEST = DATA_DIR / "FR_dataset" / "manifest.csv"
OUTPUT_DIR = DATA_DIR / "reference" / "batch_1000_ftse350"

TARGET_YEARS = (2021, 2022, 2023, 2024, 2025)
TARGET_YEAR_SET = set(TARGET_YEARS)
TARGET_COMPANIES = 200

MARKDOWN_SEARCH_DIRS = (
    DATA_DIR / "FinancialReports_downloaded" / "markdown",
    DATA_DIR / "FR_dataset" / "markdown",
    DATA_DIR / "FR_clean" / "markdown",
    DATA_DIR / "FR_clean_from_frd" / "markdown",
    DATA_DIR / "FR_2026-02-05" / "markdown",
    DATA_DIR / "FR-2021-to-2023" / "markdown",
    DATA_DIR / "FR-UK-2021-2023-test-2" / "markdown",
    DATA_DIR / "full annual 2021-2023 batch test 6" / "markdown",
    DATA_DIR / "full annual 2024-2026 batch test 7" / "markdown",
    DATA_DIR / "FR_batch_test_2021" / "markdown",
    DATA_DIR / "FR_consolidated",
)

REPORT_STATUS_SCORE = {
    "local_markdown_ready": 3,
    "md_available": 2,
    "fr_no_status": 1,
    "fr_pending": 0,
    "fr_failed": -2,
}


@dataclass
class CompanySummary:
    lei: str
    company_name: str
    market_segment: str
    market_segment_refined: str
    cni_sector_primary: str
    isic_name: str
    ready_score: int
    local_markdown_ready_count: int
    md_available_count: int
    fr_no_status_count: int
    fr_pending_count: int
    fr_failed_count: int
    selection_rank: int = 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def path_for_output(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_markdown_index() -> dict[str, Path]:
    index: dict[str, Path] = {}

    for directory in MARKDOWN_SEARCH_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.md")):
            index.setdefault(path.stem, path)

    return index


def build_fr_index() -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in read_csv(FR_MANIFEST):
        pk = (row.get("pk") or "").strip()
        if pk:
            index[pk] = row
    return index


def load_target_rows() -> tuple[dict[str, list[dict[str, object]]], Counter]:
    by_lei: dict[str, list[dict[str, object]]] = defaultdict(list)
    counters: Counter = Counter()

    for row in read_csv(TARGET_MANIFEST):
        try:
            fiscal_year = int(row["fiscal_year"])
        except (TypeError, ValueError):
            counters["bad_fiscal_year"] += 1
            continue
        if fiscal_year not in TARGET_YEAR_SET:
            continue

        row["fiscal_year"] = fiscal_year
        by_lei[row["lei"]].append(row)
        counters["target_rows"] += 1
        if row.get("market_segment") == "FTSE 350":
            counters["ftse350_rows"] += 1

    return by_lei, counters


def enrich_report_row(
    row: dict[str, object],
    fr_index: dict[str, dict[str, str]],
    markdown_index: dict[str, Path],
) -> dict[str, object] | None:
    fr_pk = str(row.get("fr_pk") or "").strip()
    if not fr_pk:
        return None

    fr_meta = fr_index.get(fr_pk)
    if not fr_meta:
        return None

    release_datetime = (fr_meta.get("release_datetime") or "").strip()
    if not release_datetime:
        return None

    markdown_path = markdown_index.get(fr_pk)
    markdown_rel = path_for_output(markdown_path) if markdown_path else ""
    markdown_chars = 0
    if markdown_path:
        markdown_chars = len(markdown_path.read_text(encoding="utf-8"))

    publication_date = release_datetime[:10]
    publication_month = publication_date[:7] if len(publication_date) >= 7 else ""
    local_status = "local_markdown_ready" if markdown_path else str(row["fr_status"])
    report_score = REPORT_STATUS_SCORE.get(local_status, REPORT_STATUS_SCORE.get(str(row["fr_status"]), -3))

    return {
        "lei": row["lei"],
        "company_name": row["company_name"],
        "market_segment": row["market_segment"],
        "market_segment_refined": row["market_segment_refined"],
        "cni_sector_primary": row["cni_sector_primary"],
        "isic_name": row["isic_name"],
        "fiscal_year": row["fiscal_year"],
        "publication_date": publication_date,
        "publication_month": publication_month,
        "publication_datetime": release_datetime,
        "release_year": fr_meta.get("release_year", ""),
        "fr_pk": fr_pk,
        "fr_filing_type": row.get("fr_filing_type", ""),
        "fr_title": fr_meta.get("title", ""),
        "fr_status": row["fr_status"],
        "local_status": local_status,
        "report_score": report_score,
        "markdown_path": markdown_rel,
        "markdown_chars": markdown_chars,
        "ch_made_up_date": row.get("ch_made_up_date", ""),
        "ch_submission_date": row.get("ch_submission_date", ""),
    }


def build_company_pool() -> tuple[list[CompanySummary], list[dict[str, object]], dict[str, str]]:
    markdown_index = build_markdown_index()
    fr_index = build_fr_index()
    target_by_lei, counters = load_target_rows()

    eligible: list[CompanySummary] = []
    report_rows_by_lei: list[dict[str, object]] = []
    report_map: dict[str, list[dict[str, object]]] = {}
    ineligible_counts: Counter = Counter()

    for lei, rows in target_by_lei.items():
        rows = sorted(rows, key=lambda item: int(item["fiscal_year"]))
        first = rows[0]

        if first.get("market_segment") != "FTSE 350":
            continue
        counters["ftse350_companies"] += 1

        years = {int(row["fiscal_year"]) for row in rows}
        if years != TARGET_YEAR_SET:
            ineligible_counts["missing_target_year_rows"] += 1
            continue

        if any(str(row.get("fr_status") or "") == "not_in_fr" for row in rows):
            ineligible_counts["not_in_fr_gap"] += 1
            continue

        enriched_rows: list[dict[str, object]] = []
        missing_meta = False
        for row in rows:
            enriched = enrich_report_row(row, fr_index, markdown_index)
            if enriched is None:
                missing_meta = True
                break
            enriched_rows.append(enriched)

        if missing_meta:
            ineligible_counts["missing_fr_metadata"] += 1
            continue

        status_counts = Counter(str(row["fr_status"]) for row in enriched_rows)
        local_ready = sum(1 for row in enriched_rows if row["local_status"] == "local_markdown_ready")
        ready_score = sum(int(row["report_score"]) for row in enriched_rows)

        summary = CompanySummary(
            lei=lei,
            company_name=str(first["company_name"]),
            market_segment=str(first["market_segment"]),
            market_segment_refined=str(first["market_segment_refined"]),
            cni_sector_primary=str(first["cni_sector_primary"]),
            isic_name=str(first["isic_name"]),
            ready_score=ready_score,
            local_markdown_ready_count=local_ready,
            md_available_count=status_counts["md_available"],
            fr_no_status_count=status_counts["fr_no_status"],
            fr_pending_count=status_counts["fr_pending"],
            fr_failed_count=status_counts["fr_failed"],
        )
        eligible.append(summary)
        report_map[lei] = enriched_rows

    eligible.sort(
        key=lambda item: (
            -item.local_markdown_ready_count,
            item.fr_failed_count,
            item.fr_pending_count,
            item.fr_no_status_count,
            item.company_name.lower(),
            item.lei,
        )
    )

    selection_notes = {
        "ftse350_companies_total": counters["ftse350_companies"],
        "eligible_companies": len(eligible),
        "ineligible_missing_target_year_rows": ineligible_counts["missing_target_year_rows"],
        "ineligible_not_in_fr_gap": ineligible_counts["not_in_fr_gap"],
        "ineligible_missing_fr_metadata": ineligible_counts["missing_fr_metadata"],
        "markdown_index_files": len(markdown_index),
    }

    for summary in eligible:
        report_rows_by_lei.extend(report_map[summary.lei])

    return eligible, report_rows_by_lei, selection_notes


def main() -> None:
    eligible_companies, report_rows_by_lei, selection_notes = build_company_pool()
    if len(eligible_companies) < TARGET_COMPANIES:
        raise RuntimeError(
            f"Need {TARGET_COMPANIES} eligible companies, found {len(eligible_companies)}"
        )

    report_rows_lookup: dict[tuple[str, int], dict[str, object]] = {}
    for row in report_rows_by_lei:
        key = (str(row["lei"]), int(row["fiscal_year"]))
        report_rows_lookup[key] = row

    selected_companies = eligible_companies[:TARGET_COMPANIES]
    reserve_companies = eligible_companies[TARGET_COMPANIES:]

    selected_company_rows: list[dict[str, object]] = []
    selected_report_rows: list[dict[str, object]] = []
    ready_queue: list[dict[str, object]] = []
    missing_local_rows: list[dict[str, object]] = []

    for rank, company in enumerate(selected_companies, start=1):
        company.selection_rank = rank
        selected_company_rows.append(
            {
                "selection_rank": rank,
                "lei": company.lei,
                "company_name": company.company_name,
                "market_segment": company.market_segment,
                "market_segment_refined": company.market_segment_refined,
                "cni_sector_primary": company.cni_sector_primary,
                "isic_name": company.isic_name,
                "local_markdown_ready_count": company.local_markdown_ready_count,
                "md_available_count": company.md_available_count,
                "fr_no_status_count": company.fr_no_status_count,
                "fr_pending_count": company.fr_pending_count,
                "fr_failed_count": company.fr_failed_count,
                "ready_score": company.ready_score,
            }
        )

        for fiscal_year in TARGET_YEARS:
            report = dict(report_rows_lookup[(company.lei, fiscal_year)])
            report["selection_rank"] = rank
            selected_report_rows.append(report)

            if report["local_status"] == "local_markdown_ready":
                ready_queue.append(
                    {
                        "company_name": report["company_name"],
                        "lei": report["lei"],
                        "year": report["fiscal_year"],
                        "cni_sector": report["cni_sector_primary"],
                        "isic_sector": report["isic_name"],
                        "file_path": report["markdown_path"],
                        "pk": report["fr_pk"],
                        "source_title": report["fr_title"],
                        "publication_date": report["publication_date"],
                    }
                )
            else:
                missing_local_rows.append(report)

    reserve_company_rows = []
    for rank, company in enumerate(reserve_companies, start=1):
        reserve_company_rows.append(
            {
                "reserve_rank": rank,
                "lei": company.lei,
                "company_name": company.company_name,
                "market_segment": company.market_segment,
                "market_segment_refined": company.market_segment_refined,
                "cni_sector_primary": company.cni_sector_primary,
                "isic_name": company.isic_name,
                "local_markdown_ready_count": company.local_markdown_ready_count,
                "md_available_count": company.md_available_count,
                "fr_no_status_count": company.fr_no_status_count,
                "fr_pending_count": company.fr_pending_count,
                "fr_failed_count": company.fr_failed_count,
                "ready_score": company.ready_score,
            }
        )

    selected_report_rows.sort(
        key=lambda row: (int(row["selection_rank"]), int(row["fiscal_year"]))
    )
    ready_queue.sort(key=lambda row: (str(row["company_name"]).lower(), int(row["year"])))
    missing_local_rows.sort(
        key=lambda row: (int(row["selection_rank"]), int(row["fiscal_year"]))
    )

    report_status_counts = Counter(str(row["fr_status"]) for row in selected_report_rows)
    local_status_counts = Counter(str(row["local_status"]) for row in selected_report_rows)
    missing_local_status_counts = Counter(str(row["fr_status"]) for row in missing_local_rows)
    publication_date_missing = sum(1 for row in selected_report_rows if not row["publication_date"])

    summary = {
        "selection_rule": {
            "market_segment": "FTSE 350",
            "target_years": list(TARGET_YEARS),
            "target_companies": TARGET_COMPANIES,
            "selection_order": [
                "local_markdown_ready_count desc",
                "fr_failed_count asc",
                "fr_pending_count asc",
                "fr_no_status_count asc",
                "company_name asc",
                "lei asc",
            ],
        },
        "eligibility": selection_notes,
        "selected": {
            "company_count": len(selected_companies),
            "report_count": len(selected_report_rows),
            "reserve_company_count": len(reserve_companies),
            "publication_date_missing_count": publication_date_missing,
            "company_local_markdown_ready_distribution": Counter(
                row["local_markdown_ready_count"] for row in selected_company_rows
            ),
            "report_counts_by_fr_status": report_status_counts,
            "report_counts_by_local_status": local_status_counts,
            "missing_local_report_count": len(missing_local_rows),
            "missing_local_counts_by_fr_status": missing_local_status_counts,
            "ready_queue_count": len(ready_queue),
        },
        "output_files": {
            "companies_csv": "data/reference/batch_1000_ftse350/companies_200.csv",
            "reports_csv": "data/reference/batch_1000_ftse350/reports_1000.csv",
            "processing_queue_ready_json": "data/reference/batch_1000_ftse350/processing_queue_ready.json",
            "reports_missing_local_csv": "data/reference/batch_1000_ftse350/reports_missing_local.csv",
            "companies_reserve_csv": "data/reference/batch_1000_ftse350/companies_reserve.csv",
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / "companies_200.csv",
        selected_company_rows,
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
        selected_report_rows,
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
        json.dumps(ready_queue, indent=2), encoding="utf-8"
    )
    write_csv(
        OUTPUT_DIR / "reports_missing_local.csv",
        missing_local_rows,
        [
            "selection_rank",
            "lei",
            "company_name",
            "fiscal_year",
            "publication_date",
            "publication_datetime",
            "fr_pk",
            "fr_title",
            "fr_status",
            "local_status",
            "ch_made_up_date",
            "ch_submission_date",
        ],
    )
    write_csv(
        OUTPUT_DIR / "companies_reserve.csv",
        reserve_company_rows,
        [
            "reserve_rank",
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
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {OUTPUT_DIR / 'companies_200.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'reports_1000.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'processing_queue_ready.json'}")
    print(f"Wrote {OUTPUT_DIR / 'reports_missing_local.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'companies_reserve.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'summary.json'}")
    print()
    print(f"Eligible FTSE350 companies with 2021-2025 FR coverage: {len(eligible_companies)}")
    print(f"Selected companies: {len(selected_companies)}")
    print(f"Selected reports: {len(selected_report_rows)}")
    print(f"Ready-now queue items: {len(ready_queue)}")
    print(f"Selected reports missing local markdown: {len(missing_local_rows)}")


if __name__ == "__main__":
    main()
