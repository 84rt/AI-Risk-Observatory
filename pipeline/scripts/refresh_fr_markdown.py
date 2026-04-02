#!/usr/bin/env python3
"""Download FR markdown for all UK annual reports we are missing locally.

Three phases (each resumable):

  Phase 1 — sweep
    GET /filings/?countries=GB&types=10-K,10-K-ESEF
                &release_datetime_from=2020-01-01T00:00:00Z
                &page_size=100&ordering=release_datetime
    Paginates through all ~11,000+ results and saves the index to
    data/fr_refresh/fr_index.jsonl  (one JSON object per filing, one line).
    Safe to re-run — skips pages already saved.

  Phase 2 — diff
    Compares fr_index.jsonl against the local corpus.
    Writes data/fr_refresh/download_queue.csv  (filings to fetch).

  Phase 3 — download
    GET /filings/{id}/markdown/  for each queued filing.
    Saves markdown to data/fr_refresh/markdown/{id}.md
    Tracks progress in data/fr_refresh/download_log.jsonl (resumable).

Local corpus sources checked (highest priority first):
  FR_consolidated/metadata.csv         pk column
  ch_gap_fill/gap_manifest.csv         pk column  (fr_recovered only)
  ch_gap_fill_2021/gap_manifest.csv    pk column
  ch_gap_fill_fy2020/gap_manifest.csv  pk column

Only filings with processing_status=COMPLETED are downloaded.

Usage:
    python scripts/refresh_fr_markdown.py --phase sweep
    python scripts/refresh_fr_markdown.py --phase diff
    python scripts/refresh_fr_markdown.py --phase download
    python scripts/refresh_fr_markdown.py          # runs all three
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

FR_CONSOLIDATED_META  = DATA_ROOT / "FR_consolidated" / "metadata.csv"
GAP_MAIN_MANIFEST     = DATA_ROOT / "ch_gap_fill" / "gap_manifest.csv"
GAP_2021_MANIFEST     = DATA_ROOT / "ch_gap_fill_2021" / "gap_manifest.csv"
GAP_FY2020_MANIFEST   = DATA_ROOT / "ch_gap_fill_fy2020" / "gap_manifest.csv"

REFRESH_DIR    = DATA_ROOT / "fr_refresh"
FR_INDEX       = REFRESH_DIR / "fr_index.jsonl"
DOWNLOAD_QUEUE = REFRESH_DIR / "download_queue.csv"
DOWNLOAD_LOG   = REFRESH_DIR / "download_log.jsonl"
MARKDOWN_DIR   = REFRESH_DIR / "markdown"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FR_API_BASE         = "https://api.financialreports.eu"
COUNTRIES           = "GB"
TYPES               = "10-K,10-K-ESEF"
RELEASE_FROM        = "2020-01-01T00:00:00Z"
PAGE_SIZE           = 100
SWEEP_RPS           = 5.0    # conservative — FR has burst limits
DOWNLOAD_RPS        = 3.0    # markdown downloads are heavier

HAVE_MARKDOWN_STATUSES = {"fr_recovered", "ch_processed"}

QUEUE_FIELDS = ["id", "lei", "company_name", "release_datetime", "filing_type_code", "title"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _headers(api_key: str) -> dict:
    return {"x-api-key": api_key, "Accept": "application/json"}


def _sleep(rps: float) -> None:
    if rps > 0:
        time.sleep(1.0 / rps)


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Local corpus — which FR PKs do we already have?
# ---------------------------------------------------------------------------

def load_local_pks() -> set[str]:
    """Return the set of FR filing IDs (pks) already present in the local corpus."""
    pks: set[str] = set()

    if FR_CONSOLIDATED_META.exists():
        for r in csv.DictReader(FR_CONSOLIDATED_META.open(newline="", encoding="utf-8")):
            pk = str(r.get("pk") or "").strip()
            if pk:
                pks.add(pk)

    for manifest_path in (GAP_MAIN_MANIFEST, GAP_2021_MANIFEST, GAP_FY2020_MANIFEST):
        if not manifest_path.exists():
            continue
        for r in csv.DictReader(manifest_path.open(newline="", encoding="utf-8")):
            if r.get("status") not in HAVE_MARKDOWN_STATUSES:
                continue
            pk = str(r.get("pk") or r.get("fr_pk") or "").strip()
            if pk:
                pks.add(pk)

    return pks


# ---------------------------------------------------------------------------
# Phase 1 — Sweep
# ---------------------------------------------------------------------------

def phase_sweep(api_key: str) -> None:
    """Paginate through all UK 10-K/10-K-ESEF filings and save index."""
    headers = _headers(api_key)

    # Work out which pages we already have
    existing = _load_jsonl(FR_INDEX)
    saved_ids = {str(r["id"]) for r in existing}
    print(f"  {len(existing)} filings already in index.")

    # First call to get total count and start URL
    params = {
        "countries":             COUNTRIES,
        "types":                 TYPES,
        "release_datetime_from": RELEASE_FROM,
        "page_size":             PAGE_SIZE,
        "ordering":              "release_datetime",
    }
    url: str | None = f"{FR_API_BASE}/filings/"
    total_fetched = len(existing)
    total_count   = None

    with tqdm(desc="Sweep", unit="filing") as bar:
        while url:
            _sleep(SWEEP_RPS)
            resp = requests.get(url, headers=headers, params=params, timeout=60)
            if resp.status_code == 429:
                retry = int(resp.headers.get("Retry-After", 10))
                print(f"\n  Rate limited — sleeping {retry}s …", flush=True)
                time.sleep(retry)
                continue
            resp.raise_for_status()

            body = resp.json()
            if total_count is None:
                total_count = body.get("count", 0)
                bar.total = total_count
                bar.refresh()

            for filing in body.get("results", []):
                fid = str(filing.get("id") or "")
                if fid in saved_ids:
                    bar.update(1)
                    continue
                co = filing.get("company") or {}
                record = {
                    "id":               fid,
                    "lei":              str(co.get("lei") or ""),
                    "company_id":       str(co.get("id") or ""),
                    "company_name":     str(co.get("name") or ""),
                    "release_datetime": str(filing.get("release_datetime") or ""),
                    "filing_type_code": str((filing.get("filing_type") or {}).get("code") or ""),
                    "processing_status":str(filing.get("processing_status") or ""),
                    "title":            str(filing.get("title") or ""),
                    "file_extension":   str(filing.get("file_extension") or ""),
                }
                _append_jsonl(FR_INDEX, record)
                saved_ids.add(fid)
                total_fetched += 1
                bar.update(1)

            next_url = body.get("next")
            url    = str(next_url).strip() if next_url else None
            params = None  # subsequent URLs are absolute and self-contained

    print(f"  Sweep complete. {total_fetched} filings in index (FR total: {total_count}).")


# ---------------------------------------------------------------------------
# Phase 2 — Diff
# ---------------------------------------------------------------------------

def phase_diff() -> None:
    """Compare the index against the local corpus and write download_queue.csv."""
    index = _load_jsonl(FR_INDEX)
    if not index:
        sys.exit("ERROR: fr_index.jsonl is empty — run --phase sweep first.")

    local_pks = load_local_pks()
    print(f"  {len(local_pks)} filings already in local corpus.")
    print(f"  {len(index)} filings in FR index.")

    queue = []
    skipped_failed  = 0
    skipped_have    = 0

    for r in index:
        fid    = str(r.get("id") or "")
        status = str(r.get("processing_status") or "")

        if fid in local_pks:
            skipped_have += 1
            continue
        if status != "COMPLETED":
            skipped_failed += 1
            continue

        queue.append({k: r.get(k, "") for k in QUEUE_FIELDS})

    _write_csv(DOWNLOAD_QUEUE, QUEUE_FIELDS, queue)

    print(f"  Already have locally:     {skipped_have}")
    print(f"  Not COMPLETED (skip):     {skipped_failed}")
    print(f"  To download:              {len(queue)}")
    print(f"  Written → {DOWNLOAD_QUEUE}")


# ---------------------------------------------------------------------------
# Phase 3 — Download
# ---------------------------------------------------------------------------

def phase_download(api_key: str) -> None:
    """Download markdown for each filing in download_queue.csv."""
    if not DOWNLOAD_QUEUE.exists():
        sys.exit("ERROR: download_queue.csv not found — run --phase diff first.")

    queue = list(csv.DictReader(DOWNLOAD_QUEUE.open(newline="", encoding="utf-8")))
    if not queue:
        print("  Nothing to download.")
        return

    # Resume: skip already downloaded
    done_log = _load_jsonl(DOWNLOAD_LOG)
    done_ids = {r["id"] for r in done_log if r.get("status") == "ok"}
    print(f"  {len(done_ids)} already downloaded. {len(queue) - len(done_ids)} remaining.")

    headers = _headers(api_key)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

    to_fetch = [r for r in queue if r["id"] not in done_ids]
    errors = 0

    for row in tqdm(to_fetch, desc="Download", unit="filing"):
        fid = row["id"]
        url = f"{FR_API_BASE}/filings/{fid}/markdown/"

        _sleep(DOWNLOAD_RPS)
        try:
            resp = requests.get(url, headers=headers, timeout=120)
        except requests.RequestException as exc:
            _append_jsonl(DOWNLOAD_LOG, {"id": fid, "status": "error", "error": str(exc)})
            errors += 1
            continue

        if resp.status_code == 429:
            retry = int(resp.headers.get("Retry-After", 30))
            print(f"\n  Rate limited — sleeping {retry}s …", flush=True)
            time.sleep(retry)
            # Retry once
            _sleep(DOWNLOAD_RPS)
            resp = requests.get(url, headers=headers, timeout=120)

        if resp.status_code == 404:
            _append_jsonl(DOWNLOAD_LOG, {"id": fid, "status": "not_found"})
            continue

        if not resp.ok:
            _append_jsonl(DOWNLOAD_LOG, {"id": fid, "status": f"http_{resp.status_code}"})
            errors += 1
            continue

        # Extract markdown text
        ct = (resp.headers.get("Content-Type") or "").lower()
        if "json" in ct:
            body = resp.json()
            if isinstance(body, dict):
                text = body.get("markdown") or body.get("content") or body.get("text") or ""
            else:
                text = ""
        else:
            text = resp.text

        if not text.strip():
            _append_jsonl(DOWNLOAD_LOG, {"id": fid, "status": "empty"})
            continue

        out_path = MARKDOWN_DIR / f"{fid}.md"
        out_path.write_text(text, encoding="utf-8")
        _append_jsonl(DOWNLOAD_LOG, {
            "id":               fid,
            "lei":              row.get("lei", ""),
            "company_name":     row.get("company_name", ""),
            "release_datetime": row.get("release_datetime", ""),
            "filing_type_code": row.get("filing_type_code", ""),
            "title":            row.get("title", ""),
            "status":           "ok",
            "bytes":            len(text.encode("utf-8")),
        })

    done_final = len(_load_jsonl(DOWNLOAD_LOG))
    print(f"\n  Done. {len(to_fetch) - errors} downloaded, {errors} errors.")
    print(f"  Markdown files → {MARKDOWN_DIR}")
    print(f"  Log            → {DOWNLOAD_LOG}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["sweep", "diff", "download", "all"], default="all",
                   help="Which phase to run (default: all)")
    p.add_argument("--sweep-rps", type=float, default=SWEEP_RPS,
                   help=f"Sweep requests per second (default: {SWEEP_RPS})")
    p.add_argument("--download-rps", type=float, default=DOWNLOAD_RPS,
                   help=f"Download requests per second (default: {DOWNLOAD_RPS})")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv(REPO_ROOT / ".env.local")
    api_key = os.environ.get("FR_API_KEY", "").strip()
    if not api_key:
        sys.exit("ERROR: FR_API_KEY not set in .env.local")

    REFRESH_DIR.mkdir(parents=True, exist_ok=True)

    run_all = args.phase == "all"

    if run_all or args.phase == "sweep":
        print("\n── Phase 1: Sweep ──────────────────────────────────────")
        phase_sweep(api_key)

    if run_all or args.phase == "diff":
        print("\n── Phase 2: Diff ───────────────────────────────────────")
        phase_diff()

    if run_all or args.phase == "download":
        print("\n── Phase 3: Download ───────────────────────────────────")
        phase_download(api_key)


if __name__ == "__main__":
    main()
