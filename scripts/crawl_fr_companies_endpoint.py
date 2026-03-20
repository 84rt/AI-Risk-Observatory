#!/usr/bin/env python3
"""Crawl the raw FR /companies/ endpoint and compare it to local baselines."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env.local", override=False)

API_BASE = "https://api.financialreports.eu/api/companies/"
DEFAULT_BASELINE = REPO_ROOT / "data" / "FR_dataset" / "ch_coverage.csv"
DEFAULT_BULK = REPO_ROOT / "data" / "FR list of UK companies.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "reference" / "fr_api_company_access_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--bulk", type=Path, default=DEFAULT_BULK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def api_headers() -> dict[str, str]:
    api_key = os.environ.get("FR_API_KEY", "")
    if not api_key:
        raise RuntimeError("FR_API_KEY not found in environment or .env.local")
    return {"x-api-key": api_key, "Accept": "application/json"}


def load_ch_baseline(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open()))
    return [row for row in rows if row.get("ch_found") == "True"]


def load_bulk_uk(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def crawl_companies(*, session: requests.Session, page_size: int, timeout: float) -> tuple[list[dict[str, Any]], int]:
    page = 1
    all_rows: list[dict[str, Any]] = []
    expected_count = -1
    while True:
        response = session.get(API_BASE, params={"page": page, "page_size": page_size}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if expected_count < 0:
            expected_count = int(payload.get("count", 0))
        results = payload.get("results") or []
        all_rows.extend(results)
        if not payload.get("next"):
            break
        page += 1
    return all_rows, expected_count


def probe_filters(*, session: requests.Session, timeout: float) -> dict[str, Any]:
    probes = [
        {"page_size": 1, "country_code": "ZZ"},
        {"page_size": 1, "country": "NOT_A_COUNTRY"},
        {"page_size": 1, "listed_exchanges": "NOT_A_REAL_EXCHANGE"},
        {"page_size": 1, "foo": "bar"},
    ]
    rows = []
    for params in probes:
        response = session.get(API_BASE, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        first = (payload.get("results") or [{}])[0]
        rows.append(
            {
                "params": params,
                "count": payload.get("count"),
                "first_name": first.get("name", ""),
                "first_country_code": first.get("country_code", ""),
            }
        )
    return {"probes": rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = load_ch_baseline(args.baseline)
    bulk_rows = load_bulk_uk(args.bulk)

    session = requests.Session()
    session.headers.update(api_headers())

    endpoint_rows, expected_count = crawl_companies(
        session=session,
        page_size=args.page_size,
        timeout=args.timeout,
    )
    filter_probe = probe_filters(session=session, timeout=args.timeout)

    baseline_leis = {row["lei"].strip().upper() for row in baseline_rows if row.get("lei")}
    bulk_leis = {row["lei"].strip().upper() for row in bulk_rows if row.get("lei", "").strip()}
    endpoint_leis = {str(row.get("lei") or "").strip().upper() for row in endpoint_rows if str(row.get("lei") or "").strip()}

    summary = {
        "endpoint_expected_count": expected_count,
        "endpoint_rows_fetched": len(endpoint_rows),
        "endpoint_unique_lei_count": len(endpoint_leis),
        "baseline_ch_company_count": len(baseline_rows),
        "baseline_ch_unique_lei_count": len(baseline_leis),
        "bulk_uk_rows": len(bulk_rows),
        "bulk_uk_unique_lei_count": len(bulk_leis),
        "baseline_in_endpoint_count": len(baseline_leis & endpoint_leis),
        "baseline_missing_from_endpoint_count": len(baseline_leis - endpoint_leis),
        "endpoint_not_in_baseline_count": len(endpoint_leis - baseline_leis),
        "bulk_uk_in_endpoint_count": len(bulk_leis & endpoint_leis),
        "bulk_uk_missing_from_endpoint_count": len(bulk_leis - endpoint_leis),
        "endpoint_not_in_bulk_uk_count": len(endpoint_leis - bulk_leis),
        "filter_probe": filter_probe,
    }

    endpoint_csv = args.out_dir / "api_companies_endpoint.csv"
    summary_json = args.out_dir / "api_companies_endpoint_summary.json"
    baseline_missing_csv = args.out_dir / "baseline_missing_from_endpoint.csv"
    endpoint_not_bulk_csv = args.out_dir / "endpoint_not_in_bulk_uk.csv"

    write_csv(endpoint_csv, endpoint_rows)
    summary_json.write_text(json.dumps(summary, indent=2))

    baseline_missing_rows = [
        row for row in baseline_rows if row["lei"].strip().upper() not in endpoint_leis
    ]
    write_csv(baseline_missing_csv, baseline_missing_rows)

    endpoint_not_bulk_rows = [
        row for row in endpoint_rows if str(row.get("lei") or "").strip().upper() not in bulk_leis
    ]
    write_csv(endpoint_not_bulk_csv, endpoint_not_bulk_rows)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {endpoint_csv.relative_to(REPO_ROOT)}")
    print(f"Wrote {summary_json.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
