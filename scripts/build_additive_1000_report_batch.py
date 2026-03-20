#!/usr/bin/env python3
"""Build the additive 850-report batch for the 1000-report expansion target.

Selection rule:
- Anchor on the already processed 50-company / 150-report cohort.
- Expand those 50 companies from 3 years (2022-2024) to 5 years (2021-2025).
- Add 150 new companies chosen from firms with full local markdown coverage for
  2021-2025, prioritizing stronger market tiers and stable alphabetical order.

Outputs:
- data/reference/batch_1000_expansion/companies_200.csv
- data/reference/batch_1000_expansion/reports_850.csv
- data/reference/batch_1000_expansion/summary.json

Token estimate methodology:
- Use the exact batch-input footprint from the prior 150-report run.
- Scale by report count and by the markdown-size ratio of the new selected
  batch versus the prior 150-report cohort.
- Convert characters to tokens with a transparent 4 chars/token estimate.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

TARGET_MANIFEST = DATA_DIR / "reference" / "target_manifest.csv"
EXISTING_RUN_MANIFEST = (
    DATA_DIR / "processed" / "fr-phase1-20260217-143317-150" / "documents_manifest.json"
)
OUTPUT_DIR = DATA_DIR / "reference" / "batch_1000_expansion"

TARGET_YEARS = (2021, 2022, 2023, 2024, 2025)
TARGET_TOTAL_COMPANIES = 200
EXISTING_COMPANIES = 50
NEW_COMPANIES = TARGET_TOTAL_COMPANIES - EXISTING_COMPANIES

MARKDOWN_DIRS = (
    DATA_DIR / "FinancialReports_downloaded" / "markdown",
    DATA_DIR / "FR_consolidated",
)

RAW_IXBRL_DIRS = (
    DATA_DIR / "raw" / "ixbrl",
    DATA_DIR / "archive_2026_01_16" / "raw" / "ixbrl",
)

RAW_PDF_DIRS = (
    DATA_DIR / "raw" / "pdfs",
    DATA_DIR / "archive_2026_01_16" / "raw" / "pdfs",
)

MARKET_RANK = {
    "FTSE 350": 0,
    "Main Market": 1,
    "Premium": 2,
    "AIM": 3,
    "Other": 4,
    "AQSE": 5,
}

# Prior 150-report run batch footprints.
PRIOR_REPORT_COUNT = 150
PRIOR_PHASE1_REQUESTS = 1538
PRIOR_PHASE1_CHARS = 11_337_097
PRIOR_PHASE2_COUNTS = {
    "adoption_type": 638,
    "risk": 378,
    "vendor": 124,
}
PRIOR_PHASE2_CHARS = {
    "adoption_type": 4_218_807,
    "risk": 3_722_282,
    "vendor": 742_239,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_markdown_path(fr_pk: str) -> Path | None:
    if not fr_pk:
        return None
    for base in MARKDOWN_DIRS:
        path = base / f"{fr_pk}.md"
        if path.exists():
            return path
    return None


def has_raw_ixbrl(lei: str, fiscal_year: int) -> bool:
    for base in RAW_IXBRL_DIRS:
        year_dir = base / str(fiscal_year)
        if not year_dir.exists():
            continue
        if any(year_dir.glob(f"{lei}_*")):
            return True
    return False


def has_raw_pdf(lei: str, fiscal_year: int) -> bool:
    for base in RAW_PDF_DIRS:
        year_dir = base / str(fiscal_year)
        if not year_dir.exists():
            continue
        if any(year_dir.glob(f"*{lei}*")):
            return True
    return False


def load_existing_run() -> tuple[list[dict], set[tuple[str, int]], set[str]]:
    payload = json.loads(EXISTING_RUN_MANIFEST.read_text(encoding="utf-8"))
    docs = payload["documents"]
    pairs = {(doc["lei"], int(doc["year"])) for doc in docs}
    leis = {doc["lei"] for doc in docs}
    return docs, pairs, leis


def build_manifest_index() -> tuple[dict[tuple[str, int], dict], dict[str, dict]]:
    rows = read_csv(TARGET_MANIFEST)
    by_lei_year: dict[tuple[str, int], dict] = {}
    company_meta: dict[str, dict] = {}

    for row in rows:
        try:
            fiscal_year = int(row["fiscal_year"])
        except (TypeError, ValueError):
            continue
        if fiscal_year not in TARGET_YEARS:
            continue

        fr_pk = (row.get("fr_pk") or "").strip()
        markdown_path = find_markdown_path(fr_pk)

        enriched = dict(row)
        enriched["fiscal_year"] = fiscal_year
        enriched["fr_pk"] = fr_pk
        enriched["markdown_path"] = str(markdown_path) if markdown_path else ""
        enriched["md_exists_actual"] = bool(markdown_path)

        key = (row["lei"], fiscal_year)
        by_lei_year[key] = enriched
        company_meta[row["lei"]] = enriched

    return by_lei_year, company_meta


def summarize_companies(
    by_lei_year: dict[tuple[str, int], dict],
    company_meta: dict[str, dict],
    existing_leis: set[str],
) -> list[dict]:
    companies: list[dict] = []

    for lei, meta in company_meta.items():
        counts = Counter()
        for fiscal_year in TARGET_YEARS:
            row = by_lei_year.get((lei, fiscal_year))
            if row is None:
                counts["missing_row"] += 1
                continue

            if row["md_exists_actual"]:
                counts["md_exists"] += 1
            else:
                status = row["fr_status"]
                if status == "md_available":
                    counts["md_available_missing_local"] += 1
                else:
                    counts[status] += 1

        companies.append(
            {
                "lei": lei,
                "company_name": meta["company_name"],
                "market_segment": meta["market_segment"],
                "market_segment_refined": meta["market_segment_refined"],
                "cni_sector_primary": meta["cni_sector_primary"],
                "is_existing50": lei in existing_leis,
                "md_exists": counts["md_exists"],
                "md_available_missing_local": counts["md_available_missing_local"],
                "fr_pending": counts["fr_pending"],
                "fr_no_status": counts["fr_no_status"],
                "fr_failed": counts["fr_failed"],
                "fr_skipped": counts["fr_skipped"],
                "not_in_fr": counts["not_in_fr"],
                "missing_row": counts["missing_row"],
            }
        )

    return companies


def select_companies(companies: list[dict]) -> tuple[list[dict], list[dict]]:
    existing = [row for row in companies if row["is_existing50"]]
    if len(existing) != EXISTING_COMPANIES:
        raise RuntimeError(f"Expected {EXISTING_COMPANIES} existing companies, found {len(existing)}")

    new_candidates = [
        row for row in companies if not row["is_existing50"] and row["md_exists"] == len(TARGET_YEARS)
    ]
    new_candidates.sort(
        key=lambda row: (
            MARKET_RANK.get(row["market_segment"], 99),
            row["company_name"].lower(),
            row["lei"],
        )
    )
    new_selected = new_candidates[:NEW_COMPANIES]
    if len(new_selected) != NEW_COMPANIES:
        raise RuntimeError(f"Expected {NEW_COMPANIES} new companies, found {len(new_selected)}")

    for row in existing:
        row["selection_bucket"] = "existing_50"
        row["selection_rank"] = ""
    for idx, row in enumerate(new_selected, start=1):
        row["selection_bucket"] = "new_150"
        row["selection_rank"] = idx

    return existing, new_selected


def build_report_rows(
    selected_companies: list[dict],
    existing_pairs: set[tuple[str, int]],
    by_lei_year: dict[tuple[str, int], dict],
) -> list[dict]:
    rows: list[dict] = []

    for company in sorted(selected_companies, key=lambda row: row["company_name"].lower()):
        lei = company["lei"]
        if company["selection_bucket"] == "existing_50":
            years = [year for year in TARGET_YEARS if (lei, year) not in existing_pairs]
            report_bucket = "existing_backfill"
        else:
            years = list(TARGET_YEARS)
            report_bucket = "new_company"

        for fiscal_year in years:
            manifest_row = by_lei_year.get((lei, fiscal_year))
            fr_status = "no_manifest_row"
            storage_status = "no_manifest_row"
            fr_pk = ""
            markdown_path = ""
            markdown_chars = ""

            if manifest_row is not None:
                fr_status = manifest_row["fr_status"]
                fr_pk = manifest_row["fr_pk"]
                markdown_path = manifest_row["markdown_path"]
                if manifest_row["md_exists_actual"]:
                    storage_status = "markdown_ready"
                    markdown_chars = len(Path(markdown_path).read_text(encoding="utf-8"))
                elif fr_status == "md_available":
                    storage_status = "md_available_missing_local"
                else:
                    storage_status = fr_status

            rows.append(
                {
                    "selection_bucket": report_bucket,
                    "lei": lei,
                    "company_name": company["company_name"],
                    "market_segment": company["market_segment"],
                    "market_segment_refined": company["market_segment_refined"],
                    "cni_sector_primary": company["cni_sector_primary"],
                    "fiscal_year": fiscal_year,
                    "fr_status": fr_status,
                    "storage_status": storage_status,
                    "fr_pk": fr_pk,
                    "markdown_path": markdown_path,
                    "markdown_chars": markdown_chars,
                    "local_raw_ixbrl_exists": has_raw_ixbrl(lei, fiscal_year),
                    "local_raw_pdf_exists": has_raw_pdf(lei, fiscal_year),
                }
            )

    return rows


def estimate_tokens(report_rows: list[dict], existing_docs: list[dict]) -> dict:
    existing_sizes = [
        len(Path(doc["markdown_path"]).read_text(encoding="utf-8"))
        for doc in existing_docs
    ]
    ready_sizes = [int(row["markdown_chars"]) for row in report_rows if row["storage_status"] == "markdown_ready"]

    size_scale = statistics.mean(ready_sizes) / statistics.mean(existing_sizes)
    report_scale = len(report_rows) / PRIOR_REPORT_COUNT

    phase1_requests = round(PRIOR_PHASE1_REQUESTS * report_scale * size_scale)
    phase1_chars = PRIOR_PHASE1_CHARS * report_scale * size_scale

    phase2 = {}
    for classifier_name, prior_count in PRIOR_PHASE2_COUNTS.items():
        requests_est = round(prior_count * report_scale * size_scale)
        chars_est = PRIOR_PHASE2_CHARS[classifier_name] * report_scale * size_scale
        phase2[classifier_name] = {
            "requests_estimate": requests_est,
            "input_chars_estimate": round(chars_est),
            "input_tokens_estimate": round(chars_est / 4),
        }

    phase2_chars = sum(item["input_chars_estimate"] for item in phase2.values())
    total_chars = round(phase1_chars) + phase2_chars

    ready_report_count = sum(1 for row in report_rows if row["storage_status"] == "markdown_ready")
    ready_report_scale = ready_report_count / PRIOR_REPORT_COUNT
    ready_only_chars = round(
        PRIOR_PHASE1_CHARS * ready_report_scale * size_scale
        + sum(PRIOR_PHASE2_CHARS.values()) * ready_report_scale * size_scale
    )

    return {
        "method": (
            "Scaled from the exact 150-report batch-input footprint using the selected "
            "batch's markdown-size ratio versus the prior 150-report cohort."
        ),
        "assumptions": {
            "report_years": list(TARGET_YEARS),
            "prior_phase1_requests": PRIOR_PHASE1_REQUESTS,
            "prior_phase2_counts": PRIOR_PHASE2_COUNTS,
            "char_to_token_ratio": 4.0,
            "size_scale_vs_prior_150": round(size_scale, 6),
        },
        "phase1": {
            "requests_estimate": phase1_requests,
            "input_chars_estimate": round(phase1_chars),
            "input_tokens_estimate": round(phase1_chars / 4),
        },
        "phase2": phase2,
        "totals": {
            "input_chars_estimate": total_chars,
            "input_tokens_estimate": round(total_chars / 4),
            "input_tokens_estimate_low_15pct": round(total_chars / 4 * 0.85),
            "input_tokens_estimate_high_15pct": round(total_chars / 4 * 1.15),
            "ready_only_report_count": ready_report_count,
            "ready_only_input_tokens_estimate": round(ready_only_chars / 4),
        },
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    existing_docs, existing_pairs, existing_leis = load_existing_run()
    by_lei_year, company_meta = build_manifest_index()
    companies = summarize_companies(by_lei_year, company_meta, existing_leis)
    existing_companies, new_companies = select_companies(companies)

    selected_companies = sorted(
        existing_companies + new_companies,
        key=lambda row: row["company_name"].lower(),
    )
    report_rows = build_report_rows(selected_companies, existing_pairs, by_lei_year)
    token_summary = estimate_tokens(report_rows, existing_docs)

    company_rows = []
    for row in selected_companies:
        company_rows.append(
            {
                "selection_bucket": row["selection_bucket"],
                "selection_rank": row["selection_rank"],
                "lei": row["lei"],
                "company_name": row["company_name"],
                "market_segment": row["market_segment"],
                "market_segment_refined": row["market_segment_refined"],
                "cni_sector_primary": row["cni_sector_primary"],
                "md_exists_2021_2025": row["md_exists"],
                "md_available_missing_local_2021_2025": row["md_available_missing_local"],
                "fr_pending_2021_2025": row["fr_pending"],
                "fr_no_status_2021_2025": row["fr_no_status"],
                "fr_failed_2021_2025": row["fr_failed"],
                "fr_skipped_2021_2025": row["fr_skipped"],
                "not_in_fr_2021_2025": row["not_in_fr"],
                "missing_row_2021_2025": row["missing_row"],
            }
        )

    summary = {
        "selection_rule": {
            "existing_anchor_companies": EXISTING_COMPANIES,
            "new_companies_selected": NEW_COMPANIES,
            "new_company_rule": (
                "Choose companies with full local markdown coverage for 2021-2025, "
                "prioritizing FTSE 350 then stronger market tiers, then alphabetical order."
            ),
            "target_years": list(TARGET_YEARS),
            "additive_reports": len(report_rows),
        },
        "report_storage": {
            "counts_by_storage_status": Counter(row["storage_status"] for row in report_rows),
            "counts_by_selection_bucket": Counter(row["selection_bucket"] for row in report_rows),
            "counts_by_new_company_market_segment": Counter(row["market_segment"] for row in new_companies),
            "non_markdown_rows": [
                {
                    "company_name": row["company_name"],
                    "lei": row["lei"],
                    "fiscal_year": row["fiscal_year"],
                    "selection_bucket": row["selection_bucket"],
                    "storage_status": row["storage_status"],
                    "fr_status": row["fr_status"],
                    "local_raw_ixbrl_exists": row["local_raw_ixbrl_exists"],
                    "local_raw_pdf_exists": row["local_raw_pdf_exists"],
                }
                for row in report_rows
                if row["storage_status"] != "markdown_ready"
            ],
        },
        "token_estimate": token_summary,
        "output_files": {
            "companies_csv": str((OUTPUT_DIR / "companies_200.csv").relative_to(REPO_ROOT)),
            "reports_csv": str((OUTPUT_DIR / "reports_850.csv").relative_to(REPO_ROOT)),
            "summary_json": str((OUTPUT_DIR / "summary.json").relative_to(REPO_ROOT)),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / "companies_200.csv",
        company_rows,
        [
            "selection_bucket",
            "selection_rank",
            "lei",
            "company_name",
            "market_segment",
            "market_segment_refined",
            "cni_sector_primary",
            "md_exists_2021_2025",
            "md_available_missing_local_2021_2025",
            "fr_pending_2021_2025",
            "fr_no_status_2021_2025",
            "fr_failed_2021_2025",
            "fr_skipped_2021_2025",
            "not_in_fr_2021_2025",
            "missing_row_2021_2025",
        ],
    )
    write_csv(
        OUTPUT_DIR / "reports_850.csv",
        report_rows,
        [
            "selection_bucket",
            "lei",
            "company_name",
            "market_segment",
            "market_segment_refined",
            "cni_sector_primary",
            "fiscal_year",
            "fr_status",
            "storage_status",
            "fr_pk",
            "markdown_path",
            "markdown_chars",
            "local_raw_ixbrl_exists",
            "local_raw_pdf_exists",
        ],
    )
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {OUTPUT_DIR / 'companies_200.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'reports_850.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'summary.json'}")
    print()
    print(f"Additive reports: {len(report_rows)}")
    print(f"Markdown-ready now: {sum(1 for row in report_rows if row['storage_status'] == 'markdown_ready')}")
    print(f"Token estimate (all 850): {summary['token_estimate']['totals']['input_tokens_estimate']:,}")


if __name__ == "__main__":
    main()
