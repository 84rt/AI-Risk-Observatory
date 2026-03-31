#!/usr/bin/env python3
"""Audit FR coverage against the CH annual-report universe using one FR search per LEI.

Workflow:
  1. Build the expected CH universe from ch_period_of_accounts.csv.
  2. Build the local markdown universe from FR_consolidated + gap-fill corpora.
  3. Compute missing (lei, fiscal_year) pairs.
  4. Query FR once per missing LEI for annual reports (cached on disk).
  5. Match returned FR filings to missing fiscal years locally by date window.
  6. Emit:
       - coverage_report.csv
       - download_queue.csv
       - fr_missing_filings_report.csv
  7. Optionally download markdown for matched FR filing IDs we do not have locally.

This is the fast path for answering:
  "Which CH-confirmed annual reports do we already have locally, which are in FR,
   and which are missing from FR?"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

FR_API_BASE = "https://api.financialreports.eu"
FR_ANNUAL_REPORT_CODE = "10-K"
FR_MATCH_WINDOW_DAYS = 270

CH_PERIOD_OF_ACCOUNTS = DATA_ROOT / "FR_dataset" / "ch_period_of_accounts.csv"
FR_CONSOLIDATED_METADATA = DATA_ROOT / "FR_consolidated" / "metadata.csv"
GAP_MAIN_MANIFEST = DATA_ROOT / "ch_gap_fill" / "gap_manifest.csv"
GAP_2021_MANIFEST = DATA_ROOT / "ch_gap_fill_2021" / "gap_manifest.csv"
GAP_FY2020_MANIFEST = DATA_ROOT / "ch_gap_fill_fy2020" / "gap_manifest.csv"
GAP_2021_DIRECT_MANIFEST = DATA_ROOT / "ch_gap_fill_2021_direct" / "gap_manifest.csv"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "fr_coverage_audit"

HAVE_MARKDOWN_STATUSES = {"fr_recovered", "ch_processed"}

COVERAGE_FIELDS = [
    "lei",
    "company_name",
    "ch_company_number",
    "fiscal_year",
    "made_up_date",
    "submission_date",
    "local_status",
    "local_source",
    "local_path",
    "fr_match_status",
    "fr_pk",
    "fr_filing_date",
    "fr_processing_status",
    "fr_title",
    "action",
]

DOWNLOAD_FIELDS = [
    "fr_pk",
    "lei",
    "company_name",
    "fiscal_year",
    "fr_filing_date",
    "fr_processing_status",
    "fr_title",
]

MISSING_FIELDS = [
    "lei",
    "company_name",
    "ch_company_number",
    "fiscal_year",
    "made_up_date",
    "submission_date",
    "fr_match_status",
    "fr_pk",
    "fr_filing_date",
    "fr_processing_status",
    "fr_title",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=CH_PERIOD_OF_ACCOUNTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-year", type=int, default=2020)
    parser.add_argument("--max-year", type=int, default=2025)
    parser.add_argument("--release-year", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--requests-per-second", type=float, default=45.0)
    parser.add_argument("--limit-leis", type=int, default=None)
    parser.add_argument("--download-markdown", action="store_true")
    parser.add_argument("--overwrite-search-cache", action="store_true")
    parser.add_argument("--overwrite-downloads", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _pair_key(lei: str, fiscal_year: str | int) -> tuple[str, str]:
    return (str(lei).strip(), str(fiscal_year).strip())


def _parse_iso_date(raw: str) -> date | None:
    raw = str(raw or "").strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _resolve_repo_relative_path(raw: str) -> Path | None:
    raw = str(raw or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    if raw.startswith("../data/"):
        return REPO_ROOT / raw[3:]
    if raw.startswith("data/"):
        return REPO_ROOT / raw
    return REPO_ROOT / raw


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_ch_universe(
    input_csv_path: Path,
    *,
    min_year: int,
    max_year: int,
    release_year: int | None,
) -> dict[tuple[str, str], dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    with input_csv_path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            lei = str(raw.get("lei") or "").strip()
            fiscal_year = str(raw.get("fiscal_year") or "").strip()
            if not lei or not fiscal_year:
                continue
            try:
                fiscal_year_int = int(fiscal_year)
            except ValueError:
                continue

            submission_date = str(raw.get("submission_date") or "").strip()
            if release_year is not None:
                release_year_str = str(raw.get("release_year") or "").strip() or submission_date[:4]
                if release_year_str != str(release_year):
                    continue
            elif not (min_year <= fiscal_year_int <= max_year):
                continue

            ch_company_number = str(
                raw.get("ch_company_number") or raw.get("company_number") or ""
            ).strip()
            if ch_company_number.isdigit():
                ch_company_number = ch_company_number.zfill(8)

            groups[_pair_key(lei, fiscal_year)].append({
                "lei": lei,
                "company_name": str(raw.get("company_name") or raw.get("name") or "").strip(),
                "ch_company_number": ch_company_number,
                "fiscal_year": fiscal_year,
                "made_up_date": str(raw.get("made_up_date") or "").strip(),
                "submission_date": submission_date,
                "_source_filing_type": str(raw.get("filing_type") or "").strip(),
            })

    result: dict[tuple[str, str], dict[str, str]] = {}
    for key, group in groups.items():
        aa = [row for row in group if row.get("_source_filing_type") == "AA"]
        pool = aa if aa else group
        best = max(pool, key=lambda row: row.get("submission_date", ""))
        best = dict(best)
        best.pop("_source_filing_type", None)
        result[key] = best
    return result


def load_local_markdown_pairs() -> dict[tuple[str, str], dict[str, str]]:
    pairs: dict[tuple[str, str], dict[str, str]] = {}

    if FR_CONSOLIDATED_METADATA.exists():
        with FR_CONSOLIDATED_METADATA.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lei = str(row.get("lei") or "").strip()
                fiscal_year = str(row.get("fiscal_year") or "").strip()
                src_path = _resolve_repo_relative_path(row.get("src_path", ""))
                if not lei or not fiscal_year or not src_path or not src_path.exists():
                    continue
                pairs[_pair_key(lei, fiscal_year)] = {
                    "local_status": "have_markdown",
                    "local_source": "fr_consolidated",
                    "local_path": str(src_path),
                    "local_fr_pk": str(row.get("pk") or "").strip(),
                }

    for manifest_path, default_source in (
        (GAP_MAIN_MANIFEST, "ch_gap_fill"),
        (GAP_2021_MANIFEST, "ch_gap_fill_2021"),
        (GAP_FY2020_MANIFEST, "ch_gap_fill_fy2020"),
    ):
        if not manifest_path.exists():
            continue
        with manifest_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") not in HAVE_MARKDOWN_STATUSES:
                    continue
                lei = str(row.get("lei") or "").strip()
                fiscal_year = str(row.get("fiscal_year") or "").strip()
                markdown_path = _resolve_repo_relative_path(row.get("markdown_path", ""))
                if not lei or not fiscal_year or not markdown_path or not markdown_path.exists():
                    continue
                if _pair_key(lei, fiscal_year) in pairs:
                    continue
                pairs[_pair_key(lei, fiscal_year)] = {
                    "local_status": "have_markdown",
                    "local_source": default_source,
                    "local_path": str(markdown_path),
                    "local_fr_pk": str(row.get("pk") or "").strip(),
                }

    if GAP_2021_DIRECT_MANIFEST.exists():
        with GAP_2021_DIRECT_MANIFEST.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ch_processed":
                    continue
                lei = str(row.get("lei") or "").strip()
                fiscal_year = str(row.get("fiscal_year") or "").strip()
                markdown_path = _resolve_repo_relative_path(row.get("markdown_path", ""))
                if not lei or not fiscal_year or not markdown_path or not markdown_path.exists():
                    continue
                if _pair_key(lei, fiscal_year) in pairs:
                    continue
                pairs[_pair_key(lei, fiscal_year)] = {
                    "local_status": "have_markdown",
                    "local_source": "ch_gap_fill_2021_direct",
                    "local_path": str(markdown_path),
                    "local_fr_pk": "",
                }
    return pairs


class RateLimiter:
    def __init__(self, requests_per_second: float):
        self.interval = 0.0 if requests_per_second <= 0 else 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            if now < self.next_allowed:
                sleep_for = self.next_allowed - now
                self.next_allowed += self.interval
            else:
                sleep_for = 0.0
                self.next_allowed = now + self.interval
        if sleep_for > 0:
            time.sleep(sleep_for)


def _status_rank(status: str) -> int:
    status = str(status or "").lower()
    if not status:
        return 0
    if any(token in status for token in ("pending", "queued", "running", "processing")):
        return 1
    if any(token in status for token in ("failed", "error")):
        return 2
    return 0


def _extract_markdown(resp: requests.Response) -> str:
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
    return resp.text


def _fetch_fr_filings_for_lei(
    lei: str,
    *,
    api_key: str,
    limiter: RateLimiter,
    timeout: int = 60,
) -> dict[str, Any]:
    headers = {"x-api-key": api_key, "Accept": "application/json"}
    params = {
        "company__lei": lei,
        "filing_type__code": FR_ANNUAL_REPORT_CODE,
        "page_size": 100,
    }
    url = f"{FR_API_BASE}/filings/"

    filings: list[dict[str, str]] = []
    request_count = 0
    while url:
        limiter.wait()
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        request_count += 1
        if resp.status_code == 429:
            return {"lei": lei, "error": "http_429", "filings": [], "request_count": request_count}
        resp.raise_for_status()
        body = resp.json()
        results = body.get("results", []) if isinstance(body, dict) else []
        for filing in results:
            filing_id = str(filing.get("id") or "").strip()
            if not filing_id:
                continue
            filings.append({
                "pk": filing_id,
                "filing_date": str(filing.get("filing_date") or "").strip(),
                "processing_status": str(filing.get("processing_status") or "").strip(),
                "title": str(filing.get("title") or "").strip(),
            })
        next_url = body.get("next") if isinstance(body, dict) else None
        url = str(next_url).strip() if next_url else ""
        if url.startswith("/"):
            url = f"{FR_API_BASE}{url}"
        params = None

    return {"lei": lei, "error": "", "filings": filings, "request_count": request_count}


def load_search_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            lei = str(record.get("lei") or "").strip()
            if lei:
                cache[lei] = record
    return cache


def append_search_cache(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _match_rows_to_filings(
    rows: list[dict[str, str]],
    filings: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    matches: dict[tuple[str, str], dict[str, str]] = {}
    used_ids: set[str] = set()

    sorted_rows = sorted(rows, key=lambda row: (row.get("made_up_date", ""), row["fiscal_year"]))
    for row in sorted_rows:
        made_up = _parse_iso_date(row.get("made_up_date", ""))
        if made_up is None:
            continue
        submission = _parse_iso_date(row.get("submission_date", ""))

        candidates: list[tuple[tuple[int, int, int, str], dict[str, str]]] = []
        for filing in filings:
            filing_id = filing["pk"]
            if filing_id in used_ids:
                continue
            filing_date = _parse_iso_date(filing.get("filing_date", ""))
            if filing_date is None:
                continue
            delta_days = (filing_date - made_up).days
            if not (0 <= delta_days <= FR_MATCH_WINDOW_DAYS):
                continue
            submission_delta = abs((filing_date - submission).days) if submission else delta_days
            score = (
                _status_rank(filing.get("processing_status", "")),
                submission_delta,
                delta_days,
                filing_id,
            )
            candidates.append((score, filing))

        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0])
        filing = candidates[0][1]
        used_ids.add(filing["pk"])
        matches[_pair_key(row["lei"], row["fiscal_year"])] = filing

    return matches


def build_coverage_rows(
    ch_rows: dict[tuple[str, str], dict[str, str]],
    local_pairs: dict[tuple[str, str], dict[str, str]],
    search_cache: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    coverage_rows: list[dict[str, str]] = []
    download_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []

    missing_by_lei: dict[str, list[dict[str, str]]] = defaultdict(list)
    for key, row in ch_rows.items():
        if key not in local_pairs:
            missing_by_lei[row["lei"]].append(row)

    matches_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for lei, rows in missing_by_lei.items():
        record = search_cache.get(lei, {})
        filings = record.get("filings", []) if not record.get("error") else []
        matches_by_pair.update(_match_rows_to_filings(rows, filings))

    seen_download_ids: set[str] = set()
    for key, row in sorted(ch_rows.items(), key=lambda item: (item[1]["fiscal_year"], item[1]["lei"])):
        local = local_pairs.get(key)
        base = {
            "lei": row["lei"],
            "company_name": row["company_name"],
            "ch_company_number": row["ch_company_number"],
            "fiscal_year": row["fiscal_year"],
            "made_up_date": row["made_up_date"],
            "submission_date": row["submission_date"],
            "local_status": "",
            "local_source": "",
            "local_path": "",
            "fr_match_status": "",
            "fr_pk": "",
            "fr_filing_date": "",
            "fr_processing_status": "",
            "fr_title": "",
            "action": "",
        }

        if local:
            base.update({
                "local_status": local["local_status"],
                "local_source": local["local_source"],
                "local_path": local["local_path"],
                "fr_pk": local.get("local_fr_pk", ""),
                "action": "already_have_local_md",
            })
            coverage_rows.append(base)
            continue

        record = search_cache.get(row["lei"])
        if record and record.get("error"):
            base.update({
                "fr_match_status": "search_error",
                "action": "search_error",
            })
            coverage_rows.append(base)
            missing_rows.append({
                field: base.get(field, "") for field in MISSING_FIELDS
            })
            continue

        match = matches_by_pair.get(key)
        if not match:
            base.update({
                "fr_match_status": "missing_from_fr",
                "action": "missing_from_fr",
            })
            coverage_rows.append(base)
            missing_rows.append({
                field: base.get(field, "") for field in MISSING_FIELDS
            })
            continue

        base.update({
            "fr_match_status": "matched_in_fr",
            "fr_pk": match["pk"],
            "fr_filing_date": match.get("filing_date", ""),
            "fr_processing_status": match.get("processing_status", ""),
            "fr_title": match.get("title", ""),
            "action": "download_from_fr",
        })
        coverage_rows.append(base)
        if match["pk"] not in seen_download_ids:
            seen_download_ids.add(match["pk"])
            download_rows.append({
                "fr_pk": match["pk"],
                "lei": row["lei"],
                "company_name": row["company_name"],
                "fiscal_year": row["fiscal_year"],
                "fr_filing_date": match.get("filing_date", ""),
                "fr_processing_status": match.get("processing_status", ""),
                "fr_title": match.get("title", ""),
            })

    return coverage_rows, download_rows, missing_rows


def download_markdown_files(
    queue_rows: list[dict[str, str]],
    *,
    output_dir: Path,
    api_key: str,
    workers: int,
    requests_per_second: float,
    overwrite: bool,
) -> Counter[str]:
    if not queue_rows:
        return Counter()

    md_dir = output_dir / "fr_recovered" / "markdown"
    md_dir.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(requests_per_second)
    headers = {"x-api-key": api_key, "Accept": "application/json"}

    def _one(row: dict[str, str]) -> tuple[str, str]:
        filing_id = row["fr_pk"]
        out_path = md_dir / f"{filing_id}.md"
        if out_path.exists() and not overwrite:
            return filing_id, "exists"
        limiter.wait()
        resp = requests.get(f"{FR_API_BASE}/filings/{filing_id}/markdown/", headers=headers, timeout=60)
        if resp.status_code == 429:
            return filing_id, "http_429"
        if resp.status_code != 200:
            return filing_id, f"http_{resp.status_code}"
        text = _extract_markdown(resp)
        if not text.strip():
            return filing_id, "empty_body"
        out_path.write_text(text, encoding="utf-8")
        return filing_id, "saved"

    counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_one, row) for row in queue_rows]
        with tqdm(total=len(futures), desc="download_fr", unit="file") as pbar:
            for future in as_completed(futures):
                _, status = future.result()
                counts[status] += 1
                pbar.set_postfix(saved=counts["saved"], exists=counts["exists"], errors=len(queue_rows) - counts["saved"] - counts["exists"])
                pbar.update(1)
    return counts


def main() -> int:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env.local", override=True)
    load_dotenv(REPO_ROOT / ".env", override=False)
    api_key = os.environ.get("FR_API_KEY", "")
    if (not args.dry_run) and (not api_key):
        raise SystemExit("FR_API_KEY not set in .env.local/.env")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    search_cache_path = args.output_dir / "lei_search_cache.jsonl"
    coverage_path = args.output_dir / "coverage_report.csv"
    download_queue_path = args.output_dir / "download_queue.csv"
    missing_path = args.output_dir / "fr_missing_filings_report.csv"

    ch_rows = load_ch_universe(
        args.input_csv,
        min_year=args.min_year,
        max_year=args.max_year,
        release_year=args.release_year,
    )
    local_pairs = load_local_markdown_pairs()
    missing_pairs = {key: row for key, row in ch_rows.items() if key not in local_pairs}
    missing_leis = sorted({row["lei"] for row in missing_pairs.values()})
    if args.limit_leis is not None:
        missing_leis = missing_leis[:args.limit_leis]
        allowed = set(missing_leis)
        missing_pairs = {key: row for key, row in missing_pairs.items() if row["lei"] in allowed}
        ch_rows = {key: row for key, row in ch_rows.items() if (key in local_pairs) or (row["lei"] in allowed)}

    print(f"CH expected pairs: {len(ch_rows)}")
    print(f"Already have local markdown: {sum(1 for key in ch_rows if key in local_pairs)}")
    print(f"Missing pairs to audit: {len(missing_pairs)}")
    print(f"Unique LEIs to search in FR: {len(missing_leis)}")

    if args.dry_run:
        return 0

    if args.overwrite_search_cache and search_cache_path.exists():
        search_cache_path.unlink()
    search_cache = load_search_cache(search_cache_path)
    leis_to_search = [lei for lei in missing_leis if lei not in search_cache]
    print(f"LEIs already cached: {len(missing_leis) - len(leis_to_search)}")
    print(f"LEIs to search now: {len(leis_to_search)}")

    limiter = RateLimiter(args.requests_per_second)
    total_requests = 0
    cache_lock = threading.Lock()

    def _search_one(lei: str) -> dict[str, Any]:
        try:
            return _fetch_fr_filings_for_lei(lei, api_key=api_key, limiter=limiter)
        except requests.RequestException as exc:
            return {"lei": lei, "error": f"request_error: {exc}", "filings": [], "request_count": 0}

    if leis_to_search:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_search_one, lei): lei for lei in leis_to_search}
            with tqdm(total=len(futures), desc="search_fr_by_lei", unit="lei") as pbar:
                for future in as_completed(futures):
                    record = future.result()
                    lei = record["lei"]
                    with cache_lock:
                        search_cache[lei] = record
                        append_search_cache(search_cache_path, record)
                        total_requests += int(record.get("request_count", 0))
                    pbar.set_postfix(requests=total_requests, cached=len(search_cache))
                    pbar.update(1)

    coverage_rows, download_rows, missing_rows = build_coverage_rows(ch_rows, local_pairs, search_cache)
    _write_csv(coverage_path, COVERAGE_FIELDS, coverage_rows)
    _write_csv(download_queue_path, DOWNLOAD_FIELDS, download_rows)
    _write_csv(missing_path, MISSING_FIELDS, missing_rows)

    action_counts = Counter(row["action"] for row in coverage_rows)
    print("Coverage summary:", dict(sorted(action_counts.items())))
    print(f"Wrote {coverage_path}")
    print(f"Wrote {download_queue_path}")
    print(f"Wrote {missing_path}")

    if args.download_markdown:
        counts = download_markdown_files(
            download_rows,
            output_dir=args.output_dir,
            api_key=api_key,
            workers=args.workers,
            requests_per_second=args.requests_per_second,
            overwrite=args.overwrite_downloads,
        )
        print("Download summary:", dict(sorted(counts.items())))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
