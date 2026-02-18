#!/usr/bin/env python3
"""Download FinancialReports markdown files by filing ID.

By default, this script reads `data/FinancialReports_downloaded/metadata.csv`,
finds filing IDs (`pk`) that are not present in
`data/FinancialReports_downloaded/markdown`, and downloads them from:

    https://api.financialreports.eu/filings/{id}/markdown/

You can also pass explicit IDs via --ids.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Iterable

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = REPO_ROOT / "data" / "FinancialReports_downloaded" / "metadata.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "FinancialReports_downloaded" / "markdown"
API_BASE = "https://api.financialreports.eu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Path to metadata.csv (default: data/FinancialReports_downloaded/metadata.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for <filing_id>.md files",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Optional explicit filing IDs. If omitted, downloads missing IDs from metadata.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of IDs to download.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.25,
        help="Pause between API calls (default: 0.25).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if local file exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show IDs that would be downloaded without making API calls.",
    )
    return parser.parse_args()


def load_ids_from_metadata(metadata_path: Path) -> list[str]:
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row["pk"].strip() for row in reader if row.get("pk", "").strip()]


def missing_ids_from_disk(ids: Iterable[str], output_dir: Path) -> list[str]:
    missing: list[str] = []
    for filing_id in ids:
        if not (output_dir / f"{filing_id}.md").exists():
            missing.append(filing_id)
    return missing


def extract_markdown(resp: requests.Response) -> str:
    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "json" not in content_type:
        return resp.text

    body = resp.json()
    if isinstance(body, dict):
        for key in ("markdown", "content", "text"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(body, ensure_ascii=False, indent=2)
    if isinstance(body, str):
        return body
    return json.dumps(body, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env.local", override=True)
    api_key = os.environ.get("FR_API_KEY")
    if not api_key:
        raise SystemExit("FR_API_KEY is missing. Set it in .env.local.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.ids:
        candidate_ids = [str(x).strip() for x in args.ids if str(x).strip()]
    else:
        all_ids = load_ids_from_metadata(args.metadata)
        candidate_ids = sorted(set(missing_ids_from_disk(all_ids, args.output_dir)))

    if not args.overwrite:
        candidate_ids = missing_ids_from_disk(candidate_ids, args.output_dir)

    if args.limit is not None:
        candidate_ids = candidate_ids[: args.limit]

    print(f"IDs selected: {len(candidate_ids)}")
    if candidate_ids:
        print("First IDs:", ", ".join(candidate_ids[:10]))

    if args.dry_run:
        return 0

    headers = {"x-api-key": api_key, "Accept": "application/json"}
    ok = 0
    not_found = 0
    forbidden = 0
    failed = 0

    for i, filing_id in enumerate(candidate_ids, start=1):
        url = f"{API_BASE}/filings/{filing_id}/markdown/"
        try:
            resp = requests.get(url, headers=headers, timeout=60)
        except requests.RequestException as exc:
            failed += 1
            print(f"[{i}/{len(candidate_ids)}] {filing_id}: request error: {exc}")
            continue

        if resp.status_code == 200:
            text = extract_markdown(resp)
            if not text.strip():
                failed += 1
                print(f"[{i}/{len(candidate_ids)}] {filing_id}: empty response body")
            else:
                out_path = args.output_dir / f"{filing_id}.md"
                out_path.write_text(text, encoding="utf-8")
                ok += 1
                print(f"[{i}/{len(candidate_ids)}] {filing_id}: saved")
        elif resp.status_code == 404:
            not_found += 1
            print(f"[{i}/{len(candidate_ids)}] {filing_id}: 404 (not processed)")
        elif resp.status_code == 403:
            forbidden += 1
            print(f"[{i}/{len(candidate_ids)}] {filing_id}: 403 (access denied)")
        else:
            failed += 1
            print(f"[{i}/{len(candidate_ids)}] {filing_id}: {resp.status_code}")

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    print(
        f"Done. saved={ok} not_found={not_found} forbidden={forbidden} failed={failed}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
