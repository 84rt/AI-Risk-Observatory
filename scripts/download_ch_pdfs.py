#!/usr/bin/env python3
"""Download annual report PDFs from Companies House for reports missing from FR.

Reads data/reference/fr_missing_annual_reports.csv and downloads the PDF for each
(company, submission_year) pair from the CH filing history API.

Output structure:
  data/ch_pdfs/{company_number}/{company_number}_{made_up_date}.pdf

Usage:
  python scripts/download_ch_pdfs.py              # full run (resumable)
  python scripts/download_ch_pdfs.py --test       # first 10 companies only
  python scripts/download_ch_pdfs.py --no-resume  # ignore checkpoint, start fresh

Requires:
  CH_API_KEY environment variable
  pip install requests tqdm
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from base64 import b64encode
from pathlib import Path

import requests
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parents[1]
DATA_DIR    = REPO_ROOT / "data"
IN_MISSING  = DATA_DIR / "reference" / "fr_missing_annual_reports.csv"
OUT_DIR     = DATA_DIR / "ch_pdfs"
CHECKPOINT  = OUT_DIR / ".download_checkpoint.json"

CH_BASE     = "https://api.company-information.service.gov.uk"
CH_API_KEY  = os.environ.get("CH_API_KEY") or os.environ.get("COMPANIES_HOUSE_API_KEY", "")
RATE_CH     = 0.6   # ~1.6 req/s — polite for free tier

ANNUAL_TYPES = {"AA", "AAMD", "AA01", "ACCOUNTS-WITH-ACCOUNTS-EXEMPTION-DATE"}


# ── CH API helpers ────────────────────────────────────────────────────────────

def make_session() -> requests.Session:
    s = requests.Session()
    cred = b64encode(f"{CH_API_KEY}:".encode()).decode()
    s.headers.update({
        "Authorization": f"Basic {cred}",
        "Accept": "application/json",
    })
    return s


SESSION = make_session()


def ch_filing_history(company_number: str) -> list[dict]:
    """Fetch all accounts filings (paginated) for a company."""
    items: list[dict] = []
    start = 0
    page_size = 100
    while True:
        r = SESSION.get(
            f"{CH_BASE}/company/{company_number}/filing-history",
            params={"category": "accounts", "start_index": start, "items_per_page": page_size},
        )
        if r.status_code == 404:
            return []
        r.raise_for_status()
        data = r.json()
        page_items = data.get("items", [])
        items.extend(page_items)
        if len(items) >= data.get("total_count", 0) or not page_items:
            break
        start += page_size
        time.sleep(RATE_CH)
    return items


def find_filing_for_date(filings: list[dict], made_up_date: str) -> dict | None:
    """Find the annual accounts filing matching a specific made_up_date (YYYY-MM-DD)."""
    for f in filings:
        desc_values = f.get("description_values", {})
        f_mud = desc_values.get("made_up_date", "")
        f_type = f.get("type", "")
        # Match on made_up_date and annual filing type
        if f_mud == made_up_date and f_type in ANNUAL_TYPES:
            return f
    # Fallback: match on date field (some filings use 'date' instead)
    for f in filings:
        if f.get("date") == made_up_date and f.get("type", "") in ANNUAL_TYPES:
            return f
    return None


def download_pdf(filing: dict, output_path: Path) -> bool:
    """Download PDF for a filing via document metadata → S3 redirect."""
    links = filing.get("links", {})
    doc_meta_link = links.get("document_metadata")
    if not doc_meta_link:
        return False

    if not doc_meta_link.startswith("http"):
        doc_meta_link = f"{CH_BASE}{doc_meta_link}"

    try:
        # Step 1: get document metadata
        meta_r = SESSION.get(doc_meta_link)
        meta_r.raise_for_status()
        metadata = meta_r.json()
        time.sleep(RATE_CH)

        content_link = metadata.get("links", {}).get("document")
        if not content_link:
            return False

        if not content_link.startswith("http"):
            content_url = f"{CH_BASE}{content_link}"
        else:
            content_url = content_link
        if not content_url.endswith("/content"):
            content_url += "/content"

        # Step 2: request PDF with redirect handling
        r = SESSION.get(
            content_url,
            headers={"Accept": "application/pdf"},
            allow_redirects=False,
            stream=True,
        )

        if r.status_code in (301, 302):
            s3_url = r.headers.get("Location")
            # Download from S3 without auth
            file_r = requests.get(s3_url, stream=True)
            file_r.raise_for_status()
        elif r.status_code == 200:
            file_r = r
        else:
            r.raise_for_status()
            return False

        # Step 3: save to disk
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as out:
            for chunk in file_r.iter_content(chunk_size=8192):
                out.write(chunk)
        return True

    except Exception as exc:
        print(f"    download error: {exc}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not CH_API_KEY:
        print("ERROR: set CH_API_KEY or COMPANIES_HOUSE_API_KEY env var")
        sys.exit(1)

    test_mode = "--test" in sys.argv
    no_resume = "--no-resume" in sys.argv

    # Load missing reports
    rows: list[dict] = []
    with open(IN_MISSING) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    print(f"Loaded {len(rows)} missing report-year pairs")

    # Group by company (one API call per company)
    by_company: dict[str, list[dict]] = {}
    for row in rows:
        key = row["ch_company_number"]
        by_company.setdefault(key, []).append(row)
    print(f"Across {len(by_company)} unique companies")

    if test_mode:
        by_company = dict(list(by_company.items())[:10])
        print(f"  TEST MODE: limited to {len(by_company)} companies")

    # Load checkpoint
    done: set[str] = set()  # "company_number|made_up_date"
    if not no_resume and CHECKPOINT.exists():
        done = set(json.loads(CHECKPOINT.read_text()))
        print(f"  Resuming: {len(done)} filings already downloaded")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"downloaded": 0, "skipped_exists": 0, "not_found": 0, "failed": 0}
    companies = list(by_company.items())

    for i, (comp_num, comp_rows) in enumerate(tqdm(companies, desc="Companies"), 1):
        name = comp_rows[0]["company_name"]

        # Filter to rows we haven't done yet
        pending = [
            r for r in comp_rows
            if f"{comp_num}|{r['ch_made_up_date']}" not in done
        ]
        if not pending:
            continue

        # Fetch filing history once per company
        try:
            filings = ch_filing_history(comp_num)
        except Exception as exc:
            tqdm.write(f"  [{i}] {name} ({comp_num}): API error — {exc}")
            stats["failed"] += len(pending)
            continue
        time.sleep(RATE_CH)

        for row in pending:
            mud = row["ch_made_up_date"]
            ck = f"{comp_num}|{mud}"
            out_path = OUT_DIR / comp_num / f"{comp_num}_{mud}.pdf"

            if out_path.exists():
                stats["skipped_exists"] += 1
                done.add(ck)
                continue

            filing = find_filing_for_date(filings, mud)
            if not filing:
                tqdm.write(f"  [{i}] {name}: no CH filing for made_up_date={mud}")
                stats["not_found"] += 1
                done.add(ck)
                continue

            ok = download_pdf(filing, out_path)
            time.sleep(RATE_CH)

            if ok:
                stats["downloaded"] += 1
                tqdm.write(f"  [{i}] {name}: downloaded {mud} ({out_path.stat().st_size // 1024} KB)")
            else:
                stats["failed"] += 1
                tqdm.write(f"  [{i}] {name}: failed to download {mud}")

            done.add(ck)

        # Checkpoint after each company
        CHECKPOINT.write_text(json.dumps(sorted(done)))

    # Final summary
    print(f"\n{'='*60}")
    print(f"Downloaded:     {stats['downloaded']}")
    print(f"Already exists: {stats['skipped_exists']}")
    print(f"Not found in CH:{stats['not_found']}")
    print(f"Failed:         {stats['failed']}")
    print(f"Output dir:     {OUT_DIR}")


if __name__ == "__main__":
    main()
