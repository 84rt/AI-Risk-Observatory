#!/usr/bin/env python3
"""Enrich data/uk_annual_reports_export/metadata.csv with legacy company metadata.

This is the tracked repo copy of the enrichment helper. It backfills the older
company-level fields expected by downstream classifier workflows.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_METADATA = REPO_ROOT / "data" / "uk_annual_reports_export" / "metadata.csv"
FULL_DB_COMPANIES = REPO_ROOT / "data" / "FULL_DB" / "companies.csv"
AUDIT_COMPANIES = (
    REPO_ROOT / "data" / "reference" / "fr_api_company_access_audit" / "api_companies_endpoint.csv"
)
COMPANY_CNI = REPO_ROOT / "data" / "reference" / "company_cni_sectors.csv"
LSE_LIST = REPO_ROOT / "data" / "LSE_List_of_all_companies.csv"
LSE_REPORTS_UNIVERSE = REPO_ROOT / "data" / "reference" / "lse_company_reports_universe.csv"

NEW_FIELDS = [
    "market_segment",
    "market_segment_refined",
    "cni_sector_primary",
    "cni_sector",
    "isic_code",
    "isic_name",
    "isic_sector",
]


EXACT_RULES: dict[str, str] = {
    "1920": "Energy",
    "2540": "Defence",
    "3040": "Defence",
    "5122": "Space",
    "6310": "Data Infrastructure",
    "6311": "Data Infrastructure",
    "8424": "Emergency Services",
}

PREFIX_3_RULES: dict[str, str] = {
    "091": "Energy",
}

PREFIX_2_RULES: dict[str, str] = {
    "01": "Food",
    "03": "Food",
    "05": "Energy",
    "06": "Energy",
    "10": "Food",
    "11": "Food",
    "20": "Chemicals",
    "21": "Health",
    "35": "Energy",
    "36": "Water",
    "37": "Water",
    "38": "Water",
    "49": "Transport",
    "50": "Transport",
    "51": "Transport",
    "60": "Communications",
    "61": "Communications",
    "64": "Finance",
    "65": "Finance",
    "66": "Finance",
    "84": "Government",
    "86": "Health",
}

ISIC_NAME_OVERRIDES: dict[str, str] = {
    "2420": "Manufacture of basic precious and other non-ferrous metals",
}

LSE_MARKET_CODE_TO_REFINED: dict[str, str] = {
    "AMSM": "AIM",
    "ASQ1": "AIM",
    "ASX1": "AIM",
    "SET1": "Main Market",
    "SET3": "Main Market",
    "SSMM": "Main Market",
    "SSQ3": "Main Market",
    "SSX3": "Main Market",
    "STMM": "Main Market",
}


def normalize_name(value: str) -> str:
    return " ".join((value or "").upper().split())


def normalize_isic_code(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return raw.zfill(4)


def lookup_cni_sector(isic_code: str) -> str:
    code = normalize_isic_code(isic_code)
    if not code:
        return ""
    if code in EXACT_RULES:
        return EXACT_RULES[code]
    prefix3 = code[:3]
    if prefix3 in PREFIX_3_RULES:
        return PREFIX_3_RULES[prefix3]
    return PREFIX_2_RULES.get(code[:2], "Other")


def normalize_market_label(raw: str) -> str:
    label = " ".join((raw or "").upper().split())
    if not label:
        return ""
    if "AIM" in label:
        return "AIM"
    if "AQSE" in label or "AQUIS" in label:
        return "AQSE"
    if "MAIN MARKET" in label:
        return "Main Market"
    return "Other"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_market_lookup_by_name() -> dict[str, str]:
    result: dict[str, str] = {}
    for row in load_rows(LSE_LIST):
        name = normalize_name(row.get("Company Name", ""))
        market = normalize_market_label(row.get("Market", ""))
        if name and market and name not in result:
            result[name] = market
    return result


def build_market_lookup_by_ticker() -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in load_rows(LSE_REPORTS_UNIVERSE):
        ticker = (row.get("epic") or "").strip().upper()
        code = (row.get("market_code") or "").strip().upper()
        market = LSE_MARKET_CODE_TO_REFINED.get(code, "")
        if ticker and market:
            grouped[ticker].append(market)

    resolved: dict[str, str] = {}
    for ticker, markets in grouped.items():
        unique = sorted(set(markets))
        if len(unique) == 1:
            resolved[ticker] = unique[0]
    return resolved


def main() -> int:
    csv.field_size_limit(sys.maxsize)

    full_by_lei = {
        (row.get("lei") or "").strip(): row
        for row in load_rows(FULL_DB_COMPANIES)
        if (row.get("lei") or "").strip()
    }
    audit_rows = load_rows(AUDIT_COMPANIES)
    audit_by_lei = {
        (row.get("lei") or "").strip(): row
        for row in audit_rows
        if (row.get("lei") or "").strip()
    }
    audit_by_name = {
        normalize_name(row.get("name", "")): row
        for row in audit_rows
        if normalize_name(row.get("name", ""))
    }
    isic_by_code: dict[str, dict[str, str]] = {}
    for row in load_rows(COMPANY_CNI):
        code = normalize_isic_code(row.get("isic_code", ""))
        if code and code not in isic_by_code:
            isic_by_code[code] = row
    for row in full_by_lei.values():
        code = normalize_isic_code(row.get("isic_code", ""))
        if code and code not in isic_by_code:
            isic_by_code[code] = {
                "isic_name": row.get("isic_name", ""),
            }

    market_by_name = build_market_lookup_by_name()
    market_by_ticker = build_market_lookup_by_ticker()

    with EXPORT_METADATA.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        original_fields = reader.fieldnames or []
        rows = list(reader)

    output_fields = list(original_fields)
    for field in NEW_FIELDS:
        if field not in output_fields:
            output_fields.append(field)

    stats = Counter()
    enriched_rows: list[dict[str, str]] = []

    for row in rows:
        lei = (row.get("company_lei") or "").strip()
        name_key = normalize_name(row.get("company_name", ""))
        ticker = (row.get("company_ticker") or "").strip().upper()

        full = full_by_lei.get(lei)
        audit = audit_by_lei.get(lei) or audit_by_name.get(name_key)

        if full:
            market_segment = (full.get("market_segment") or "").strip()
            market_segment_refined = (full.get("market_segment_refined") or market_segment).strip()
            cni_sector_primary = (full.get("cni_sector_primary") or "").strip()
            isic_code = normalize_isic_code(full.get("isic_code", ""))
            isic_name = (full.get("isic_name") or "").strip()
            stats["rows_from_full_db"] += 1
        else:
            audit_code = normalize_isic_code((audit or {}).get("sub_industry_code", ""))
            isic_ref = isic_by_code.get(audit_code, {})
            isic_code = audit_code
            isic_name = (
                (isic_ref.get("isic_name") or "").strip()
                or ISIC_NAME_OVERRIDES.get(audit_code, "")
            )
            cni_sector_primary = lookup_cni_sector(audit_code)

            market_segment_refined = (
                market_by_name.get(name_key)
                or market_by_ticker.get(ticker, "")
                or "Other"
            )
            market_segment = market_segment_refined

            if audit:
                stats["rows_with_audit_fallback"] += 1
            else:
                stats["rows_without_any_company_match"] += 1

            if market_by_name.get(name_key):
                stats["rows_market_from_lse_list"] += 1
            elif market_by_ticker.get(ticker):
                stats["rows_market_from_lse_ticker"] += 1
            else:
                stats["rows_market_default_other"] += 1

        row["market_segment"] = market_segment
        row["market_segment_refined"] = market_segment_refined
        row["cni_sector_primary"] = cni_sector_primary
        row["cni_sector"] = cni_sector_primary
        row["isic_code"] = isic_code
        row["isic_name"] = isic_name
        row["isic_sector"] = isic_name

        if row["market_segment"]:
            stats["rows_with_market_segment"] += 1
        if row["cni_sector_primary"]:
            stats["rows_with_cni"] += 1
        if row["isic_code"]:
            stats["rows_with_isic_code"] += 1
        if row["isic_name"]:
            stats["rows_with_isic_name"] += 1

        enriched_rows.append(row)

    tmp_path = EXPORT_METADATA.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(enriched_rows)
    tmp_path.replace(EXPORT_METADATA)

    print(f"Updated {EXPORT_METADATA}")
    print(f"Rows: {len(enriched_rows)}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
