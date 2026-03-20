#!/usr/bin/env python3
"""Build a dated FR UK waterfall audit snapshot."""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


csv.field_size_limit(sys.maxsize)

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env.local", override=False)

LSE_PATH = REPO_ROOT / "data" / "LSE_List_of_all_companies.csv"
AQSE_PATH = REPO_ROOT / "data" / "aqse.csv"
CH_COVERAGE_PATH = REPO_ROOT / "data" / "FR_dataset" / "ch_coverage.csv"
FR_BULK_UK_PATH = REPO_ROOT / "data" / "FR list of UK companies.csv"
TARGET_MANIFEST_PATH = REPO_ROOT / "data" / "reference" / "target_manifest.csv"
FY_QA_PATH = REPO_ROOT / "data" / "reference" / "fy_qa_review.csv"

COMPANIES_API = "https://api.financialreports.eu/api/companies/"
FILINGS_API = "https://api.financialreports.eu/api/filings/"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_name(value: str) -> str:
    value = (value or "").upper().replace("&", " AND ")
    value = re.sub(
        r"\b(PLC|LIMITED|LTD|GROUP|HOLDINGS|ORD|LD|N\.V\.|NV|INC|SHARES|NONVOTING|THE)\b",
        " ",
        value,
    )
    value = re.sub(r"[^A-Z0-9]+", "", value)
    return value


def load_csv(path: Path, *, encoding: str = "utf-8") -> list[dict[str, str]]:
    with path.open(encoding=encoding) as handle:
        return list(csv.DictReader(handle))


def api_headers() -> dict[str, str]:
    api_key = os.environ.get("FR_API_KEY", "")
    if not api_key:
        raise RuntimeError("FR_API_KEY not found in environment or .env.local")
    return {"x-api-key": api_key, "Accept": "application/json"}


def session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update(api_headers())
    return sess


def crawl_companies_api(sess: requests.Session, *, page_size: int = 100, timeout: float = 30.0) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    page = 1
    expected_count = -1
    while True:
        response = sess.get(COMPANIES_API, params={"page": page, "page_size": page_size}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if expected_count < 0:
            expected_count = int(payload.get("count", 0))
        results = payload.get("results") or []
        rows.extend(results)
        if not payload.get("next"):
            break
        page += 1
    return rows, expected_count


def probe_endpoint(
    sess: requests.Session,
    *,
    url: str,
    probes: list[dict[str, Any]],
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for params in probes:
        observed_at = now_utc()
        try:
            response = sess.get(url, params=params, timeout=timeout)
            data = response.json()
            first = (data.get("results") or [{}])[0]
            out.append(
                {
                    "observed_at": observed_at,
                    "params": params,
                    "status_code": response.status_code,
                    "count": data.get("count"),
                    "first_id": first.get("id"),
                    "first_name": first.get("name") or first.get("title"),
                    "first_country_code": first.get("country_code"),
                    "first_processing_status": first.get("processing_status"),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            out.append(
                {
                    "observed_at": observed_at,
                    "params": params,
                    "status_code": "",
                    "count": "",
                    "first_id": "",
                    "first_name": "",
                    "first_country_code": "",
                    "first_processing_status": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return out


def listing_summary() -> dict[str, Any]:
    lse = load_csv(LSE_PATH, encoding="utf-8-sig")
    aqse = load_csv(AQSE_PATH, encoding="utf-8-sig")
    lse_uk = [row for row in lse if row.get("Country of Incorporation", "").strip() == "United Kingdom"]
    lse_names = {normalize_name(row["Company Name"]) for row in lse if normalize_name(row["Company Name"])}
    aqse_names = {normalize_name(row["Instrument"]) for row in aqse if normalize_name(row["Instrument"])}
    return {
        "observed_at": now_utc(),
        "lse_rows": len(lse),
        "lse_uk_incorporated_rows": len(lse_uk),
        "lse_market_counts": dict(Counter((row.get("Market", "").strip() or "<blank>") for row in lse)),
        "aqse_rows": len(aqse),
        "lse_unique_normalized_names": len(lse_names),
        "aqse_unique_normalized_names": len(aqse_names),
        "lse_aqse_overlap_normalized_names": len(lse_names & aqse_names),
        "lse_aqse_union_normalized_names": len(lse_names | aqse_names),
    }


def ch_summary() -> dict[str, Any]:
    rows = load_csv(CH_COVERAGE_PATH)
    ch_true = [row for row in rows if row.get("ch_found") == "True"]
    return {
        "observed_at": now_utc(),
        "rows": len(rows),
        "ch_found_true": len(ch_true),
        "jurisdiction_counts_all": dict(Counter(row.get("jurisdiction", "") for row in rows)),
        "jurisdiction_counts_ch_found": dict(Counter(row.get("jurisdiction", "") for row in ch_true)),
        "market_segment_counts_all": dict(Counter(row.get("market_segment", "") for row in rows)),
        "market_segment_counts_ch_found": dict(Counter(row.get("market_segment", "") for row in ch_true)),
    }


def bulk_uk_summary() -> dict[str, Any]:
    rows = load_csv(FR_BULK_UK_PATH, encoding="utf-8-sig")
    leis = {row.get("lei", "").strip().upper() for row in rows if row.get("lei", "").strip()}
    return {
        "observed_at": now_utc(),
        "rows": len(rows),
        "unique_nonempty_lei": len(leis),
        "blank_lei_rows": sum(1 for row in rows if not row.get("lei", "").strip()),
    }


def report_coverage_summary(ch_leis: set[str]) -> dict[str, Any]:
    rows = []
    with TARGET_MANIFEST_PATH.open() as handle:
        for row in csv.DictReader(handle):
            if row["fiscal_year"] in {"2021", "2022", "2023", "2024", "2025"} and row["lei"] in ch_leis:
                rows.append(row)
    companies_with_manifest = {row["lei"] for row in rows}
    companies_with_fr = {row["lei"] for row in rows if (row.get("fr_pk") or "").strip()}
    companies_with_md = {row["lei"] for row in rows if row.get("fr_status") == "md_available"}
    return {
        "observed_at": now_utc(),
        "baseline_company_count": len(ch_leis),
        "manifest_rows_2021_2025": len(rows),
        "companies_with_any_manifest_row": len(companies_with_manifest),
        "companies_with_any_fr_pk": len(companies_with_fr),
        "companies_with_any_markdown": len(companies_with_md),
        "companies_with_zero_fr_pk": len(ch_leis - companies_with_fr),
        "companies_with_zero_markdown": len(ch_leis - companies_with_md),
        "fr_status_counts": dict(Counter(row["fr_status"] for row in rows)),
        "rows_with_nonempty_fr_pk": sum(1 for row in rows if (row.get("fr_pk") or "").strip()),
        "rows_with_markdown": sum(1 for row in rows if row.get("fr_status") == "md_available"),
    }


def fiscal_year_summary(ch_leis: set[str]) -> dict[str, Any]:
    rows = [row for row in load_csv(FY_QA_PATH) if row.get("lei") in ch_leis]
    return {
        "observed_at": now_utc(),
        "qa_rows_for_baseline": len(rows),
        "confidence_counts": dict(Counter(row["confidence"] for row in rows)),
        "manifest_detected_mismatch_count": sum(1 for row in rows if row["manifest_fy"] != row["detected_fy"]),
        "anomaly_type_counts": dict(Counter(row["anomaly_type"] for row in rows)),
    }


def companies_endpoint_summary(
    endpoint_rows: list[dict[str, Any]],
    expected_count: int,
    ch_leis: set[str],
    bulk_leis: set[str],
) -> dict[str, Any]:
    endpoint_leis = {str(row.get("lei") or "").strip().upper() for row in endpoint_rows if str(row.get("lei") or "").strip()}
    return {
        "observed_at": now_utc(),
        "endpoint_expected_count": expected_count,
        "endpoint_rows_fetched": len(endpoint_rows),
        "endpoint_unique_nonempty_lei": len(endpoint_leis),
        "endpoint_blank_lei_rows": sum(1 for row in endpoint_rows if not str(row.get("lei") or "").strip()),
        "endpoint_country_code_counts": dict(Counter(str(row.get("country_code") or "") for row in endpoint_rows)),
        "ch_baseline_overlap": len(ch_leis & endpoint_leis),
        "ch_baseline_missing_from_endpoint": len(ch_leis - endpoint_leis),
        "bulk_uk_overlap": len(bulk_leis & endpoint_leis),
        "bulk_uk_missing_from_endpoint": len(bulk_leis - endpoint_leis),
        "endpoint_not_in_bulk_uk": len(endpoint_leis - bulk_leis),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = REPO_ROOT / "data" / "reference" / "fr_uk_waterfall_audit" / snapshot_date
    out_dir.mkdir(parents=True, exist_ok=True)

    lse_summary = listing_summary()
    ch_cov_summary = ch_summary()
    bulk_summary = bulk_uk_summary()

    ch_rows = load_csv(CH_COVERAGE_PATH)
    ch_leis = {row["lei"].strip().upper() for row in ch_rows if row.get("ch_found") == "True" and row.get("lei")}
    bulk_rows = load_csv(FR_BULK_UK_PATH, encoding="utf-8-sig")
    bulk_leis = {row.get("lei", "").strip().upper() for row in bulk_rows if row.get("lei", "").strip()}

    sess = session()
    endpoint_rows, expected_count = crawl_companies_api(sess)
    companies_probes = probe_endpoint(
        sess,
        url=COMPANIES_API,
        probes=[
            {"page_size": 1},
            {"page_size": 1, "countries": "GB"},
            {"page_size": 1, "countries": "ZZ"},
            {"page_size": 1, "listed_exchanges": "NOT_A_REAL_EXCHANGE"},
            {"page_size": 1, "country_code": "ZZ"},
        ],
    )
    filings_probes = probe_endpoint(
        sess,
        url=FILINGS_API,
        probes=[
            {"page_size": 1},
            {"page_size": 1, "countries": "GB"},
            {"page_size": 1, "countries": "ZZ"},
            {"page_size": 1, "types": "AR"},
        ],
    )

    companies_summary = companies_endpoint_summary(endpoint_rows, expected_count, ch_leis, bulk_leis)
    reports_summary = report_coverage_summary(ch_leis)
    fy_summary = fiscal_year_summary(ch_leis)

    endpoint_leis = {str(row.get("lei") or "").strip().upper() for row in endpoint_rows if str(row.get("lei") or "").strip()}
    bulk_missing_from_endpoint = [
        row for row in bulk_rows if row.get("lei", "").strip().upper() not in endpoint_leis
    ]
    endpoint_not_in_bulk = [
        row for row in endpoint_rows if str(row.get("lei") or "").strip().upper() not in bulk_leis
    ]

    report_gap_rows = []
    with TARGET_MANIFEST_PATH.open() as handle:
        for row in csv.DictReader(handle):
            if row["fiscal_year"] not in {"2021", "2022", "2023", "2024", "2025"}:
                continue
            if row["lei"] not in ch_leis:
                continue
            if row.get("fr_status") == "md_available":
                continue
            report_gap_rows.append(row)

    fy_rows_for_baseline = [
        row for row in load_csv(FY_QA_PATH) if row.get("lei") in ch_leis
    ]

    write_json(out_dir / "listing_summary.json", lse_summary)
    write_json(out_dir / "ch_coverage_summary.json", ch_cov_summary)
    write_json(out_dir / "fr_bulk_uk_summary.json", bulk_summary)
    write_json(out_dir / "fr_companies_endpoint_summary.json", companies_summary)
    write_json(out_dir / "fr_companies_probe_results.json", companies_probes)
    write_json(out_dir / "fr_filings_probe_results.json", filings_probes)
    write_json(out_dir / "report_coverage_summary.json", reports_summary)
    write_json(out_dir / "fiscal_year_summary.json", fy_summary)
    write_csv(out_dir / "fr_companies_endpoint.csv", endpoint_rows)
    write_csv(out_dir / "bulk_uk_missing_from_endpoint.csv", bulk_missing_from_endpoint)
    write_csv(out_dir / "endpoint_not_in_bulk_uk.csv", endpoint_not_in_bulk)
    write_csv(out_dir / "report_gap_rows.csv", report_gap_rows)
    write_csv(out_dir / "fy_qa_rows_for_baseline.csv", fy_rows_for_baseline)

    manifest = {
        "generated_at": now_utc(),
        "snapshot_dir": str(out_dir.relative_to(REPO_ROOT)),
        "inputs": {
            "lse_snapshot": str(LSE_PATH.relative_to(REPO_ROOT)),
            "aqse_snapshot": str(AQSE_PATH.relative_to(REPO_ROOT)),
            "ch_coverage": str(CH_COVERAGE_PATH.relative_to(REPO_ROOT)),
            "fr_bulk_uk": str(FR_BULK_UK_PATH.relative_to(REPO_ROOT)),
            "target_manifest": str(TARGET_MANIFEST_PATH.relative_to(REPO_ROOT)),
            "fy_qa_review": str(FY_QA_PATH.relative_to(REPO_ROOT)),
        },
    }
    write_json(out_dir / "snapshot_manifest.json", manifest)

    print(json.dumps({"snapshot_dir": str(out_dir.relative_to(REPO_ROOT))}, indent=2))


if __name__ == "__main__":
    main()
