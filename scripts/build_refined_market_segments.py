#!/usr/bin/env python3
"""Build refined market-segment labels and optionally backfill consolidated metadata.

Outputs a company-level mapping with the newer exchange-oriented categories:
  - Main Market
  - AIM
  - AQSE
  - Other

Precedence:
  1. Live AQSE list
  2. Live LSE issuer list
  3. Latest match in the LSE issuer archive (2023-2025)
  4. Existing repo mapping collapsed to the new categories

The script also updates data/FR_consolidated/metadata.csv with:
  - market_segment_depricated
  - market_segment_refined
  - market_segment_refined_source
  - market_segment_refined_authoritative
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

MARKET_SEGMENTS_CSV = DATA_DIR / "reference" / "market_segments.csv"
MARKET_SEGMENTS_MANUAL_CSV = DATA_DIR / "reference" / "market_segments_manual.csv"
REFINED_OUTPUT_CSV = DATA_DIR / "reference" / "market_segments_refined.csv"
COMPANIES_CSV = DATA_DIR / "FR_dataset" / "companies.csv"
LSE_LIVE_CSV = DATA_DIR / "LSE_List_of_all_companies.csv"
AQSE_CSV = DATA_DIR / "aqse.csv"
LSE_ARCHIVE_DIR = DATA_DIR / "issuer archive"
LSE_UNIVERSE_CSV = DATA_DIR / "reference" / "lse_company_reports_universe.csv"
METADATA_CSV = DATA_DIR / "FR_consolidated" / "metadata.csv"

AIM_CODES = {"ASQ1", "ASX1", "AMSM"}
MAIN_CODES = {"SET1", "SET2", "SET3", "STMM", "SSMM", "SSQ3"}

MONTH_ORDER = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-path",
        default=str(METADATA_CSV),
        help="Metadata CSV to backfill with refined market columns.",
    )
    parser.add_argument(
        "--no-update-metadata",
        action="store_true",
        help="Write the refined mapping CSV but do not touch metadata.csv.",
    )
    return parser.parse_args()


def canon_name(value: str) -> str:
    value = (value or "").upper()
    substitutions = {
        "&": " AND ",
        "ABRDN": "ABERDEEN",
        " CO ": " COMPANY ",
        " CO'S ": " COMPANIES ",
        " INV TST ": " INVESTMENT TRUST ",
        " INV TRUST ": " INVESTMENT TRUST ",
        " INV ": " INVESTMENT ",
        " TST ": " TRUST ",
        " GLBL ": " GLOBAL ",
        " UTIL ": " UTILITIES ",
        " INF ": " INFRASTRUCTURE ",
        " RES ": " RESOURCES ",
        " CAP AND INC ": " CAPITAL AND INCOME ",
        " CAP ": " CAPITAL ",
        " INC ": " INCOME ",
        " INTL ": " INTERNATIONAL ",
        " PHARMA ": " PHARMACEUTICALS ",
        " AUTO TRADER ": " AUTOTRADER ",
        " BRITISH LAND CO ": " BRITISH LAND COMPANY ",
    }
    value = f" {value} "
    for old, new in substitutions.items():
        value = value.replace(old, new)
    value = re.sub(r"\bPUBLIC LIMITED COMPANY\b", " ", value)
    value = re.sub(r"\bP\.?L\.?C\.?\b", " ", value)
    value = re.sub(r"\bLIMITED\b", " ", value)
    value = re.sub(r"\bLTD\.?\b", " ", value)
    value = re.sub(r"\bTHE\b", " ", value)
    value = re.sub(r"[^A-Z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def load_csv(path: Path, *, encoding: str = "utf-8-sig") -> list[dict[str, str]]:
    with path.open(encoding=encoding, newline="") as handle:
        return list(csv.DictReader(handle))


def load_lse_archive_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    header_idx = None
    for idx, row in enumerate(rows):
        cleaned = [cell.strip() for cell in row if cell.strip()]
        if cleaned[:2] == ["Admission Date", "Company Name"]:
            header_idx = idx
            break

    if header_idx is None:
        return []

    header = [cell.strip() for cell in rows[header_idx] if cell.strip()]
    parsed: list[dict[str, str]] = []
    for row in rows[header_idx + 1 :]:
        values = [cell.strip() for cell in row if cell.strip()]
        if len(values) < len(header):
            continue
        parsed.append(dict(zip(header, values[: len(header)])))
    return parsed


def collapse_segment(segment: str) -> str:
    if segment in {"Premium", "Main Market", "FTSE 350"}:
        return "Main Market"
    if segment in {"AIM", "AQSE", "Other"}:
        return segment
    return "Other"


def load_base_mapping() -> list[dict[str, str]]:
    base_rows = load_csv(MARKET_SEGMENTS_CSV, encoding="utf-8")
    manual_rows = load_csv(MARKET_SEGMENTS_MANUAL_CSV, encoding="utf-8")
    manual_by_lei = {row["lei"]: row["market_segment_manual"] for row in manual_rows}

    companies_by_lei = {row["lei"]: row for row in load_csv(COMPANIES_CSV, encoding="utf-8")}
    lse_codes = {
        row["epic"].strip(): row["market_code"].strip()
        for row in load_csv(LSE_UNIVERSE_CSV, encoding="utf-8")
    }
    manual_leis = set(manual_by_lei)

    refined_rows: list[dict[str, str]] = []
    for row in base_rows:
        lei = row["lei"]
        ticker = (row["ticker"] or "").strip()
        company = companies_by_lei.get(lei, {})
        listed = company.get("listed_exchanges", row.get("listed_exchanges", ""))
        indices = company.get("stock_indices", row.get("stock_indices", ""))

        legacy = manual_by_lei.get(lei, row["market_segment_v2"])
        collapsed = collapse_segment(legacy)

        if lei in manual_leis:
            fallback_source = "legacy_manual_review"
        elif "FTSE AIM All-Share" in indices:
            fallback_source = "legacy_ftse_aim_all_share"
        elif "FTSE 100" in indices or "FTSE 250" in indices:
            fallback_source = "legacy_ftse_350"
        elif "Aquis" in listed:
            fallback_source = "legacy_listed_exchanges_aquis"
        elif lse_codes.get(ticker) in AIM_CODES:
            fallback_source = "legacy_lse_market_code_aim"
        elif lse_codes.get(ticker) in MAIN_CODES:
            fallback_source = "legacy_lse_market_code_main"
        else:
            fallback_source = "legacy_existing_mapping"

        refined_rows.append(
            {
                "lei": lei,
                "name": row["name"],
                "ticker": ticker,
                "market_segment_legacy": legacy,
                "market_segment_refined": collapsed,
                "market_segment_refined_source": fallback_source,
                "market_segment_refined_authoritative": "false",
            }
        )

    return refined_rows


def load_authoritative_sources() -> tuple[set[str], set[str], dict[str, str], dict[str, tuple[int, int, str]]]:
    aqse_rows = load_csv(AQSE_CSV)
    aqse_names = {canon_name(row["Instrument"]) for row in aqse_rows}
    aqse_symbols = {row["Symbol"].strip().upper() for row in aqse_rows if row["Symbol"].strip()}

    lse_live_rows = load_csv(LSE_LIVE_CSV)
    lse_live = {
        canon_name(row["Company Name"]): row["Market"].strip()
        for row in lse_live_rows
        if row.get("Company Name", "").strip()
    }

    archive_latest: dict[str, tuple[int, int, str]] = {}
    for path in sorted(LSE_ARCHIVE_DIR.rglob("*.csv")):
        if path.name == ".DS_Store" or "Notes & Disclaimer" in path.name:
            continue

        year_match = re.search(r"(\d{4})", str(path.parent))
        month_key = path.name.split("-")[0].upper()
        if not year_match or month_key not in MONTH_ORDER:
            continue

        year = int(year_match.group(1))
        month = MONTH_ORDER[month_key]

        for row in load_lse_archive_csv(path):
            name = canon_name(row.get("Company Name", ""))
            market = row.get("Market", "").strip()
            if not name or not market:
                continue

            existing = archive_latest.get(name)
            if existing is None or (year, month) > (existing[0], existing[1]):
                archive_latest[name] = (year, month, market)

    return aqse_names, aqse_symbols, lse_live, archive_latest


def apply_authoritative_overrides(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    aqse_names, aqse_symbols, lse_live, archive_latest = load_authoritative_sources()

    updated: list[dict[str, str]] = []
    for row in rows:
        name_key = canon_name(row["name"])
        ticker = row["ticker"].upper()

        refined = row["market_segment_refined"]
        source = row["market_segment_refined_source"]
        authoritative = row["market_segment_refined_authoritative"]

        if name_key in aqse_names or ticker in aqse_symbols:
            refined = "AQSE"
            source = "aqse_live"
            authoritative = "true"
        elif name_key in lse_live:
            market = lse_live[name_key]
            if market == "AIM":
                refined = "AIM"
                source = "lse_live"
                authoritative = "true"
            elif market == "MAIN MARKET":
                refined = "Main Market"
                source = "lse_live"
                authoritative = "true"
        elif name_key in archive_latest:
            year, month, market = archive_latest[name_key]
            if market == "AIM":
                refined = "AIM"
                source = f"lse_archive_{year:04d}-{month:02d}"
                authoritative = "true"
            elif market == "MAIN MARKET":
                refined = "Main Market"
                source = f"lse_archive_{year:04d}-{month:02d}"
                authoritative = "true"

        updated.append(
            {
                **row,
                "market_segment_refined": refined,
                "market_segment_refined_source": source,
                "market_segment_refined_authoritative": authoritative,
            }
        )

    return updated


def write_refined_mapping(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "lei",
        "name",
        "ticker",
        "market_segment_legacy",
        "market_segment_depricated",
        "market_segment_refined",
        "market_segment_refined_source",
        "market_segment_refined_authoritative",
    ]
    with REFINED_OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_metadata(path: Path, refined_rows: list[dict[str, str]]) -> None:
    with path.open(newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))

    if not metadata_rows:
        return

    refined_by_lei = {row["lei"]: row for row in refined_rows}
    fieldnames = list(metadata_rows[0].keys())
    if "market_segment" in fieldnames:
        fieldnames[fieldnames.index("market_segment")] = "market_segment_depricated"
    elif "market_segment_depricated" not in fieldnames:
        fieldnames.append("market_segment_depricated")
    new_fields = [
        "market_segment_refined",
        "market_segment_refined_source",
        "market_segment_refined_authoritative",
    ]
    for field in new_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    for row in metadata_rows:
        refined = refined_by_lei.get(row["lei"])
        row["market_segment_depricated"] = row.pop("market_segment", "")
        if refined:
            row["market_segment_refined"] = refined["market_segment_refined"]
            row["market_segment_refined_source"] = refined["market_segment_refined_source"]
            row["market_segment_refined_authoritative"] = refined[
                "market_segment_refined_authoritative"
            ]
        else:
            row["market_segment_refined"] = ""
            row["market_segment_refined_source"] = ""
            row["market_segment_refined_authoritative"] = ""

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)


def print_summary(rows: list[dict[str, str]]) -> None:
    seg_counts = Counter(row["market_segment_refined"] for row in rows)
    source_counts = Counter(row["market_segment_refined_source"] for row in rows)
    auth_counts = Counter(row["market_segment_refined_authoritative"] for row in rows)

    print("Refined segment counts:")
    for key, count in seg_counts.most_common():
        print(f"  {key:<12} {count:>5}")

    print("\nAuthoritative coverage:")
    for key, count in auth_counts.items():
        print(f"  {key:<12} {count:>5}")

    print("\nTop sources:")
    for key, count in source_counts.most_common(12):
        print(f"  {key:<30} {count:>5}")


def main() -> None:
    args = parse_args()
    refined_rows = apply_authoritative_overrides(load_base_mapping())
    write_refined_mapping(refined_rows)
    print_summary(refined_rows)
    print(f"\nWrote refined mapping: {REFINED_OUTPUT_CSV}")

    if not args.no_update_metadata:
        metadata_path = Path(args.metadata_path)
        update_metadata(metadata_path, refined_rows)
        print(f"Updated metadata: {metadata_path}")


if __name__ == "__main__":
    main()
