#!/usr/bin/env python3
"""Build a normalized FTSE 350 master company list from current and historical inputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
import re
from html import unescape
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_HISTORY_CSV = REPO_ROOT / "data" / "reference" / "ftse350_history" / "ftse350_2021_2025_long.csv"
DEFAULT_LOCAL_CSV = REPO_ROOT / "ftse350_constituents.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "reference" / "ftse350_history" / "ftse350_company_master.csv"
DEFAULT_SUMMARY_JSON = REPO_ROOT / "data" / "reference" / "ftse350_history" / "ftse350_company_master_summary.json"
YEARS = ["2021", "2022", "2023", "2024", "2025"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-csv", default=str(DEFAULT_HISTORY_CSV))
    parser.add_argument("--local-csv", default=str(DEFAULT_LOCAL_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def clean_cell_text(value: str) -> str:
    text = unescape(value or "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_company_name(name: str) -> str:
    text = clean_cell_text(name)
    text = text.translate(
        str.maketrans(
            {
                "’": "'",
                "“": '"',
                "”": '"',
                "–": "-",
                "—": "-",
            }
        )
    )
    text = re.sub(r"\s+\([^)]*\)$", "", text)
    text = re.sub(r"\bordinary shares?\b.*$", "", text, flags=re.I)
    text = re.sub(r"\b(ord|ords)\b.*$", "", text, flags=re.I)
    text = re.sub(r"[^A-Za-z0-9&+'/-]+", " ", text)
    text = text.lower()
    text = re.sub(r"\b(public limited company|plc)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_ticker(value: str | None) -> str:
    if not value:
        return ""
    text = clean_cell_text(value).upper()
    text = re.sub(r"^(LON|LSE)\s*:\s*", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^A-Z0-9.\-]", "", text)
    return text


def entity_key(*, ticker: str, normalized_name: str) -> str:
    if ticker:
        return f"ticker:{ticker}"
    return f"name:{normalized_name}"


def choose_latest_name(year_to_name: dict[str, str]) -> str:
    if not year_to_name:
        return ""
    latest_year = max(year_to_name)
    return year_to_name[latest_year]


def build_master_rows(
    history_rows: list[dict[str, str]],
    local_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    entities: dict[str, dict[str, object]] = {}

    for row in history_rows:
        ticker = normalize_ticker(row.get("ticker_epic"))
        normalized_name = row.get("normalized_company_name") or normalize_company_name(row.get("company_name", ""))
        key = entity_key(ticker=ticker, normalized_name=normalized_name)
        entity = entities.setdefault(
            key,
            {
                "entity_key": key,
                "ticker": ticker,
                "normalized_company_name": normalized_name,
                "history_rows": [],
                "history_names": set(),
                "history_raw_names": set(),
                "history_years": set(),
                "history_segments": set(),
                "history_name_by_year": {},
                "segment_by_year": {},
                "local_row": None,
            },
        )
        entity["history_rows"].append(row)
        entity["history_names"].add(row["company_name"])
        entity["history_raw_names"].add(row["company_name_raw"])
        entity["history_years"].add(row["year"])
        entity["history_segments"].add(row["segment"])
        entity["history_name_by_year"][row["year"]] = row["company_name"]
        entity["segment_by_year"][row["year"]] = row["segment"]

    for row in local_rows:
        ticker = normalize_ticker(row.get("Ticker"))
        normalized_name = normalize_company_name(row.get("Company", ""))
        key = entity_key(ticker=ticker, normalized_name=normalized_name)
        entity = entities.setdefault(
            key,
            {
                "entity_key": key,
                "ticker": ticker,
                "normalized_company_name": normalized_name,
                "history_rows": [],
                "history_names": set(),
                "history_raw_names": set(),
                "history_years": set(),
                "history_segments": set(),
                "history_name_by_year": {},
                "segment_by_year": {},
                "local_row": None,
            },
        )
        entity["local_row"] = row

    master_rows: list[dict[str, str]] = []
    source_status_counter: Counter[str] = Counter()

    for key, entity in sorted(entities.items()):
        local_row = entity["local_row"]
        history_years = sorted(entity["history_years"])
        latest_historical_name = choose_latest_name(entity["history_name_by_year"])
        canonical_name = local_row["Company"] if local_row else latest_historical_name
        canonical_ticker = normalize_ticker(local_row["Ticker"]) if local_row else entity["ticker"]
        source_status = (
            "both" if local_row and history_years else
            "local_only" if local_row else
            "history_only"
        )
        source_status_counter[source_status] += 1

        output_row = {
            "entity_key": key,
            "canonical_company_name": canonical_name,
            "canonical_ticker": canonical_ticker,
            "normalized_company_name": entity["normalized_company_name"],
            "source_status": source_status,
            "in_local_constituents": "True" if local_row else "False",
            "in_history_2021_2025": "True" if history_years else "False",
            "history_years_present": "|".join(history_years),
            "history_years_count": str(len(history_years)),
            "first_history_year": min(history_years) if history_years else "",
            "last_history_year": max(history_years) if history_years else "",
            "historical_name_variants": " | ".join(sorted(entity["history_names"])),
            "historical_raw_name_variants": " | ".join(sorted(entity["history_raw_names"])),
            "latest_historical_company_name": latest_historical_name,
            "historical_segments_seen": "|".join(sorted(entity["history_segments"])),
            "current_local_company_name": local_row["Company"] if local_row else "",
            "current_local_ticker": normalize_ticker(local_row["Ticker"]) if local_row else "",
            "current_local_sector": local_row["Sector"] if local_row else "",
            "current_local_index": local_row["Index"] if local_row else "",
        }

        for year in YEARS:
            output_row[f"present_{year}"] = "True" if year in entity["history_years"] else "False"
            output_row[f"segment_{year}"] = entity["segment_by_year"].get(year, "")
            output_row[f"name_{year}"] = entity["history_name_by_year"].get(year, "")

        master_rows.append(output_row)

    master_rows.sort(
        key=lambda row: (
            row["canonical_company_name"].lower(),
            row["canonical_ticker"],
        )
    )

    summary = {
        "history_input_rows": len(history_rows),
        "local_input_rows": len(local_rows),
        "master_company_rows": len(master_rows),
        "source_status_counts": dict(sorted(source_status_counter.items())),
        "history_only_rows": source_status_counter["history_only"],
        "local_only_rows": source_status_counter["local_only"],
        "both_rows": source_status_counter["both"],
    }
    return master_rows, summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    history_rows = read_csv(Path(args.history_csv))
    local_rows = read_csv(Path(args.local_csv))
    master_rows, summary = build_master_rows(history_rows, local_rows)
    write_csv(Path(args.output_csv), master_rows)
    write_json(Path(args.summary_json), summary)
    print(
        f"Wrote {len(master_rows)} normalized company rows to {args.output_csv} "
        f"({summary['both_rows']} both, {summary['local_only_rows']} local-only, "
        f"{summary['history_only_rows']} history-only)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
