#!/usr/bin/env python3
"""Audit FR company accessibility from a CH-backed baseline.

This audit intentionally avoids using the curated `data/FR_dataset/companies.csv`
as evidence of API accessibility. Instead it:

1. Starts from a Companies House-backed baseline (`ch_coverage.csv`).
2. Queries the FR `/companies/` API for each baseline company by name.
3. Checks whether the API returns an exact LEI match for that company.
4. Compares the results to the raw bulk UK list in `data/FR list of UK companies.csv`.

Outputs are written under `data/reference/fr_api_company_access_audit/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env.local", override=False)

API_BASE = "https://api.financialreports.eu/api"
DEFAULT_BASELINE = REPO_ROOT / "data" / "FR_dataset" / "ch_coverage.csv"
DEFAULT_BULK = REPO_ROOT / "data" / "FR list of UK companies.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "reference" / "fr_api_company_access_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--bulk", type=Path, default=DEFAULT_BULK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def normalize_name(value: str) -> str:
    value = (value or "").lower()
    value = value.replace("&", " and ")
    value = re.sub(r"\b(plc|limited|ltd|holdings|group)\b", " ", value)
    value = re.sub(r"[^a-z0-9]+", "", value)
    return value


def load_ch_baseline(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open()))
    filtered = [row for row in rows if row.get("ch_found") == "True"]
    filtered.sort(key=lambda row: (row.get("name", ""), row.get("lei", "")))
    return filtered


def load_bulk_uk(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    leis = {row.get("lei", "").strip().upper() for row in rows if row.get("lei", "").strip()}
    return rows, leis


def read_existing_results(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open()))
    return {row["baseline_lei"]: row for row in rows if row.get("baseline_lei")}


def api_headers() -> dict[str, str]:
    api_key = os.environ.get("FR_API_KEY", "")
    if not api_key:
        raise RuntimeError("FR_API_KEY not found in environment or .env.local")
    return {"x-api-key": api_key, "Accept": "application/json"}


def query_company_search(
    session: requests.Session,
    *,
    name: str,
    page_size: int,
    timeout: float,
) -> tuple[int | None, dict[str, Any] | None, str]:
    params = {"search": name, "page_size": page_size}
    try:
        response = session.get(
            f"{API_BASE}/companies/",
            params=params,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return None, None, f"request_error:{type(exc).__name__}"

    try:
        payload = response.json()
    except ValueError:
        payload = None

    return response.status_code, payload, ""


def audit_row(
    row: dict[str, str],
    *,
    session: requests.Session,
    page_size: int,
    timeout: float,
) -> dict[str, str]:
    baseline_lei = row["lei"].strip().upper()
    baseline_name = row["name"].strip()
    baseline_norm = normalize_name(baseline_name)

    status_code, payload, error = query_company_search(
        session,
        name=baseline_name,
        page_size=page_size,
        timeout=timeout,
    )

    results: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        results = payload.get("results") or []

    matched_rank = ""
    matched_id = ""
    matched_name = ""
    matched_lei = ""
    exact_lei_match = "False"
    exact_name_match = "False"

    for index, candidate in enumerate(results, start=1):
        candidate_lei = (candidate.get("lei") or "").strip().upper()
        candidate_name = (candidate.get("name") or "").strip()
        if candidate_lei == baseline_lei:
            matched_rank = str(index)
            matched_id = str(candidate.get("id") or "")
            matched_name = candidate_name
            matched_lei = candidate_lei
            exact_lei_match = "True"
            if normalize_name(candidate_name) == baseline_norm:
                exact_name_match = "True"
            break

    top_result = results[0] if results else {}

    return {
        "baseline_lei": baseline_lei,
        "baseline_name": baseline_name,
        "baseline_company_number": row.get("company_number", ""),
        "baseline_jurisdiction": row.get("jurisdiction", ""),
        "baseline_market_segment": row.get("market_segment", ""),
        "api_query_name": baseline_name,
        "api_status_code": str(status_code or ""),
        "api_error": error,
        "api_result_count": str(len(results)),
        "exact_lei_match": exact_lei_match,
        "exact_name_match": exact_name_match,
        "matched_result_rank": matched_rank,
        "matched_result_id": matched_id,
        "matched_result_name": matched_name,
        "matched_result_lei": matched_lei,
        "top_result_id": str(top_result.get("id") or ""),
        "top_result_name": str(top_result.get("name") or ""),
        "top_result_lei": str(top_result.get("lei") or ""),
        "top_result_country": str(top_result.get("country_code") or ""),
        "top_result_market_segment": str(top_result.get("market_segment") or ""),
        "top_result_listed_exchanges": str(top_result.get("listed_exchanges") or ""),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    *,
    baseline_rows: list[dict[str, str]],
    result_rows: list[dict[str, str]],
    bulk_uk_leis: set[str],
) -> dict[str, Any]:
    baseline_leis = {row["lei"].strip().upper() for row in baseline_rows}
    exact_lei = {row["baseline_lei"] for row in result_rows if row["exact_lei_match"] == "True"}
    http_ok = {row["baseline_lei"] for row in result_rows if row["api_status_code"] == "200"}
    top_hit = {
        row["baseline_lei"]
        for row in result_rows
        if row["matched_result_rank"] == "1" and row["exact_lei_match"] == "True"
    }
    status_counts = Counter(row["api_status_code"] for row in result_rows)
    error_counts = Counter(row["api_error"] for row in result_rows if row["api_error"])

    return {
        "baseline_company_count": len(baseline_rows),
        "baseline_unique_lei_count": len(baseline_leis),
        "api_http_200_count": len(http_ok),
        "api_exact_lei_match_count": len(exact_lei),
        "api_top_result_exact_lei_match_count": len(top_hit),
        "api_no_exact_lei_match_count": len(baseline_leis - exact_lei),
        "bulk_uk_unique_lei_count": len(bulk_uk_leis),
        "baseline_in_bulk_uk_count": len(baseline_leis & bulk_uk_leis),
        "baseline_missing_from_bulk_uk_count": len(baseline_leis - bulk_uk_leis),
        "bulk_uk_not_in_baseline_count": len(bulk_uk_leis - baseline_leis),
        "api_status_code_counts": dict(sorted(status_counts.items())),
        "api_error_counts": dict(sorted(error_counts.items())),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = load_ch_baseline(args.baseline)
    bulk_rows, bulk_uk_leis = load_bulk_uk(args.bulk)

    if args.limit is not None:
        baseline_rows = baseline_rows[: args.limit]

    results_path = args.out_dir / "api_company_search_results.csv"
    baseline_snapshot_path = args.out_dir / "baseline_ch_companies.csv"
    summary_path = args.out_dir / "summary.json"
    missing_api_path = args.out_dir / "baseline_missing_from_api.csv"
    extra_bulk_path = args.out_dir / "bulk_uk_not_in_ch_baseline.csv"

    write_csv(baseline_snapshot_path, baseline_rows)

    existing = {} if args.refresh else read_existing_results(results_path)
    result_rows: list[dict[str, str]] = []

    session = requests.Session()
    session.headers.update(api_headers())

    for index, row in enumerate(baseline_rows, start=1):
        lei = row["lei"].strip().upper()
        cached = existing.get(lei)
        if cached:
            result_rows.append(cached)
            continue

        audited = audit_row(
            row,
            session=session,
            page_size=args.page_size,
            timeout=args.timeout,
        )
        result_rows.append(audited)
        if index % 25 == 0:
            print(f"Processed {index}/{len(baseline_rows)} companies...")
            write_csv(results_path, result_rows)
        time.sleep(args.sleep)

    write_csv(results_path, result_rows)

    summary = build_summary(
        baseline_rows=baseline_rows,
        result_rows=result_rows,
        bulk_uk_leis=bulk_uk_leis,
    )
    summary_path.write_text(json.dumps(summary, indent=2))

    missing_from_api = [
        row for row in result_rows if row["exact_lei_match"] != "True"
    ]
    write_csv(missing_api_path, missing_from_api)

    baseline_leis = {row["lei"].strip().upper() for row in baseline_rows}
    bulk_uk_not_in_baseline = [
        row for row in bulk_rows if (row.get("lei", "").strip().upper() not in baseline_leis)
    ]
    write_csv(extra_bulk_path, bulk_uk_not_in_baseline)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {results_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
