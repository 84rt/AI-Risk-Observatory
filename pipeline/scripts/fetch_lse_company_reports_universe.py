#!/usr/bin/env python3
"""Build an external LSE company-report universe from londonstockexchange.com.

This script scrapes the public "instrument-result" pages with:
  filterBy=CompanyReports&filterClause=1
and exports a flat dataset that can be used as an external company universe
reference when FR API exchange filters are unreliable.

Output:
  - CSV (all rows)
  - JSON summary (counts by market code, initials, dedup keys)

Notes:
  - LSE pages list instruments, not perfectly normalized issuers.
  - We keep raw instrument-level rows and also report unique EPIC/ISIN counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_CSV = REPO_ROOT / "data" / "reference" / "lse_company_reports_universe.csv"
DEFAULT_OUT_SUMMARY = REPO_ROOT / "data" / "reference" / "lse_company_reports_universe.summary.json"

BASE_URL = "https://www.londonstockexchange.com/exchange/instrument-result.html"
# Matches javascript: UpdateOpener('NAME', 'ISIN|COUNTRY|CCY|MKT|SEDOL|EPIC')
ROW_PATTERN = re.compile(r"UpdateOpener\('(?P<name>.*?)',\s*'(?P<payload>.*?)'\);", re.S)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--initials",
        default="A-Z,0",
        help="Initial buckets to crawl. Default: A-Z,0",
    )
    p.add_argument(
        "--rate-limit-sec",
        type=float,
        default=0.2,
        help="Sleep between requests (default: 0.2)",
    )
    p.add_argument(
        "--timeout-sec",
        type=int,
        default=30,
        help="HTTP timeout in seconds (default: 30)",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per page on request errors (default: 3)",
    )
    p.add_argument(
        "--max-pages-per-initial",
        type=int,
        default=500,
        help="Safety cap for pages crawled per initial (default: 500)",
    )
    p.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUT_CSV),
        help=f"Output CSV path (default: {DEFAULT_OUT_CSV})",
    )
    p.add_argument(
        "--summary-json",
        default=str(DEFAULT_OUT_SUMMARY),
        help=f"Output summary JSON path (default: {DEFAULT_OUT_SUMMARY})",
    )
    return p.parse_args()


def parse_initials(spec: str) -> list[str]:
    spec = (spec or "").strip().upper()
    if not spec:
        return list(string.ascii_uppercase) + ["0"]

    out: list[str] = []
    for token in [t.strip() for t in spec.split(",") if t.strip()]:
        if token == "A-Z":
            out.extend(list(string.ascii_uppercase))
        elif token in string.ascii_uppercase or token == "0":
            out.append(token)
        elif len(token) == 3 and token[1] == "-" and token[0] in string.ascii_uppercase and token[2] in string.ascii_uppercase:
            start, end = token[0], token[2]
            if start <= end:
                out.extend([chr(c) for c in range(ord(start), ord(end) + 1)])
        else:
            raise ValueError(f"Unsupported initials token: {token}")

    seen = set()
    deduped: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def build_url(initial: str, page: int) -> str:
    return (
        f"{BASE_URL}?filterBy=CompanyReports&filterClause=1"
        f"&initial={initial}&page={page}"
    )


def fetch_html(session: requests.Session, initial: str, page: int, timeout_sec: int, max_retries: int, rate_limit_sec: float) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(build_url(initial, page), timeout=timeout_sec)
            resp.raise_for_status()
            if rate_limit_sec > 0:
                time.sleep(rate_limit_sec)
            return resp.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2.0, attempt * 0.3))
                continue
            break
    raise RuntimeError(f"Failed to fetch initial={initial} page={page}: {last_error}")


def normalize_payload(payload: str) -> list[str]:
    compact = " ".join((payload or "").split())
    parts = [p.strip() for p in compact.split("|")]
    if len(parts) < 6:
        parts.extend([""] * (6 - len(parts)))
    return parts[:6]


def parse_rows(html: str, initial: str, page: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for m in ROW_PATTERN.finditer(html):
        name = m.group("name").strip()
        isin, country, currency, market_code, sedol, epic = normalize_payload(m.group("payload"))
        rows.append(
            {
                "initial": initial,
                "page": str(page),
                "instrument_name": name,
                "epic": epic,
                "isin": isin,
                "sedol": sedol,
                "market_code": market_code,
                "currency": currency,
                "country": country,
                "source_url": build_url(initial, page),
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = [
        "initial",
        "page",
        "instrument_name",
        "epic",
        "isin",
        "sedol",
        "market_code",
        "currency",
        "country",
        "source_url",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    args = parse_args()
    initials = parse_initials(args.initials)
    out_csv = Path(args.output_csv)
    out_summary = Path(args.summary_json)

    all_rows: list[dict[str, str]] = []
    pages_by_initial: dict[str, int] = {}

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; AIRiskObservatory/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )

    for idx, initial in enumerate(initials, start=1):
        page = 1
        non_empty_pages = 0
        rows_for_initial = 0
        seen_page_signatures: set[tuple[tuple[str, str, str], ...]] = set()
        while page <= args.max_pages_per_initial:
            html = fetch_html(session, initial, page, args.timeout_sec, args.max_retries, args.rate_limit_sec)
            page_rows = parse_rows(html, initial, page)
            if not page_rows:
                break

            signature = tuple(
                (r["epic"], r["isin"], r["instrument_name"])
                for r in page_rows
            )
            if signature in seen_page_signatures:
                # Defensive break if site starts repeating old pages.
                break
            seen_page_signatures.add(signature)

            all_rows.extend(page_rows)
            non_empty_pages += 1
            rows_for_initial += len(page_rows)
            page += 1

        pages_by_initial[initial] = non_empty_pages
        print(
            f"[{idx}/{len(initials)}] initial={initial} pages={non_empty_pages} rows={rows_for_initial}",
            flush=True,
        )

    write_csv(out_csv, all_rows)

    unique_epic = {r["epic"] for r in all_rows if r["epic"]}
    unique_isin = {r["isin"] for r in all_rows if r["isin"]}
    market_counts = Counter(r["market_code"] for r in all_rows if r["market_code"])
    initial_counts = Counter(r["initial"] for r in all_rows)

    summary = {
        "source": "https://www.londonstockexchange.com/exchange/instrument-result.html?filterBy=CompanyReports&filterClause=1",
        "rows": len(all_rows),
        "unique_epic": len(unique_epic),
        "unique_isin": len(unique_isin),
        "initials": initials,
        "pages_by_initial": pages_by_initial,
        "counts_by_initial": dict(sorted(initial_counts.items())),
        "counts_by_market_code": dict(sorted(market_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "notes": [
            "Rows are instrument-level from LSE search, not issuer-normalized entities.",
            "Use EPIC or ISIN as join keys when reconciling with FR companies.",
        ],
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWrote CSV: {out_csv}")
    print(f"Wrote summary: {out_summary}")
    print(f"Rows: {len(all_rows)} | unique EPIC: {len(unique_epic)} | unique ISIN: {len(unique_isin)}")


if __name__ == "__main__":
    main()
