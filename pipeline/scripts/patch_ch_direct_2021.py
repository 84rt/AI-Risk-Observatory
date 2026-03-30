#!/usr/bin/env python3
"""Fetch 2021 annual reports for target companies not covered by FR.

For companies in target_manifest with FY2020/2021 entries that have no
annual report in our 2021 corpus, go directly to Companies House (no FR
probe needed — FR doesn't have these).  CH company numbers come straight
from target_manifest so no GLEIF lookup is required.

Three phases (run in order, or --phase all):

  download_ch     Download annual-accounts PDFs from Companies House.
  submit_batch    Upload PDFs to Mistral Files API and submit batch OCR jobs.
  collect_batch   Poll jobs, download results, save markdown files.

Outputs (all under --output-dir, default data/ch_gap_fill_2021_direct/):
  ch_processed/pdfs/{id}.pdf
  ch_processed/markdown/{id}.md
  batch_jobs/{batch_n}.jsonl
  gap_manifest.csv
  batch_state.json

Usage:
  python scripts/patch_ch_direct_2021.py --phase all --dry-run
  python scripts/patch_ch_direct_2021.py --phase download_ch
  python scripts/patch_ch_direct_2021.py --phase submit_batch
  python scripts/patch_ch_direct_2021.py --phase collect_batch
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from src.companies_house import CompaniesHouseClient

DATA_ROOT = REPO_ROOT / "data"
TARGET_MANIFEST = DATA_ROOT / "reference" / "target_manifest.csv"
FR_CONSOLIDATED_META = DATA_ROOT / "FR_consolidated" / "metadata.csv"
GAP_2021_MANIFEST = DATA_ROOT / "ch_gap_fill_2021" / "gap_manifest.csv"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "ch_gap_fill_2021_direct"

MISTRAL_FILES_URL = "https://api.mistral.ai/v1/files"
MISTRAL_BATCH_URL = "https://api.mistral.ai/v1/batch/jobs"
MISTRAL_OCR_MODEL = "mistral-ocr-latest"

MANIFEST_FIELDS = [
    "id", "lei", "company_name", "fiscal_year", "ch_company_number",
    "ch_filing_date", "pdf_path", "markdown_path", "status",
    "batch_job_id", "error",
]
TERMINAL_STATUSES = {"ch_processed", "error"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--phase",
        choices=["download_ch", "submit_batch", "collect_batch", "all"],
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-batch-mb", type=float, default=350.0)
    parser.add_argument("--max-batch-items", type=int, default=None)
    parser.add_argument("--batch-submit-delay", type=float, default=2.0)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--sleep-ch", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Gap discovery
# ---------------------------------------------------------------------------

def find_missing_companies() -> list[dict[str, str]]:
    """Return target companies with no 2021 annual report in our corpus."""
    # Build set of LEIs already covered for release_year 2021
    covered_leis: set[str] = set()
    for row in csv.DictReader(FR_CONSOLIDATED_META.open(encoding="utf-8")):
        if row.get("release_year") == "2021":
            covered_leis.add(row["lei"])
    if GAP_2021_MANIFEST.exists():
        for row in csv.DictReader(GAP_2021_MANIFEST.open(encoding="utf-8")):
            if row.get("status") in ("fr_recovered", "ch_processed"):
                covered_leis.add(row["lei"])

    # Collect target companies for FY2020/2021 not yet covered
    # Prefer FY2021 over FY2020 when a company has both
    candidates: dict[str, dict[str, str]] = {}
    for row in csv.DictReader(TARGET_MANIFEST.open(encoding="utf-8")):
        lei = row["lei"]
        fy = row.get("fiscal_year", "")
        if fy not in ("2020", "2021"):
            continue
        if lei in covered_leis:
            continue
        ch_num = row.get("ch_company_number", "").strip()
        if not ch_num:
            continue
        # Prefer FY2021 entry over FY2020
        existing = candidates.get(lei)
        if existing is None or (fy == "2021" and existing["fiscal_year"] == "2020"):
            candidates[lei] = {
                "id": f"{lei}_{fy}",
                "lei": lei,
                "company_name": row.get("company_name", ""),
                "fiscal_year": fy,
                "ch_company_number": ch_num,
            }

    return list(candidates.values())


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row["id"]: row for row in csv.DictReader(path.open(encoding="utf-8"))}


def save_manifest(path: Path, manifest: dict[str, dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in manifest.values():
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def load_batch_state(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_batch_state(path: Path, jobs: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8")


def _blank_row(item: dict[str, str]) -> dict[str, str]:
    return {k: item.get(k, "") for k in MANIFEST_FIELDS} | {"status": "pending"}


# ---------------------------------------------------------------------------
# CH helpers (reused from patch_ch_gaps.py)
# ---------------------------------------------------------------------------

def _find_ch_filing(ch_client: CompaniesHouseClient, company_number: str, fiscal_year: int) -> dict | None:
    try:
        filings = ch_client.get_filing_history(company_number, category="accounts")
    except Exception:
        return None
    annual = [f for f in filings
              if f.get("type") == "AA" or "annual" in f.get("description", "").lower()]
    if not annual:
        return None
    year_str = str(fiscal_year)
    next_year = str(fiscal_year + 1)
    for filing in annual:
        desc = filing.get("description", "")
        date = filing.get("date", "")
        if year_str in desc or (date and (date.startswith(year_str) or date.startswith(next_year))):
            return filing
    return annual[0]


def _download_ch_pdf(ch_client: CompaniesHouseClient, filing: dict, output_path: Path) -> Path | None:
    links = filing.get("links", {})
    doc_metadata_link = links.get("document_metadata")
    if not doc_metadata_link:
        return None
    try:
        downloaded = ch_client.download_document(doc_metadata_link, output_path, prefer_xhtml=False)
        return downloaded if downloaded.suffix.lower() == ".pdf" else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase 1: download CH PDFs
# ---------------------------------------------------------------------------

def phase_download_ch(
    manifest: dict[str, dict[str, str]],
    *,
    ch_client: CompaniesHouseClient,
    output_dir: Path,
    sleep_ch: float,
    overwrite: bool,
    limit: int | None,
    dry_run: bool,
) -> None:
    candidates = [
        row for row in manifest.values()
        if row["status"] == "pending" and (overwrite or not row.get("pdf_path"))
    ]
    if limit:
        candidates = candidates[:limit]

    print(f"\n[download_ch] {len(candidates)} companies to download")
    if dry_run:
        return

    ok = errors = 0
    for i, row in enumerate(candidates, 1):
        row_id = row["id"]
        company_name = row["company_name"]
        ch_number = row["ch_company_number"]
        fiscal_year = int(row["fiscal_year"])

        pdf_path = output_dir / "ch_processed" / "pdfs" / f"{row_id}.pdf"
        if pdf_path.exists() and not overwrite:
            row["pdf_path"] = str(pdf_path)
            row["status"] = "ch_downloaded"
            ok += 1
            continue

        filing = _find_ch_filing(ch_client, ch_number, fiscal_year)
        if not filing:
            row["status"] = "error"
            row["error"] = f"no CH filing found for {ch_number} FY{fiscal_year}"
            print(f"  [{i}/{len(candidates)}] {company_name}: no CH filing")
            errors += 1
            time.sleep(sleep_ch)
            continue
        row["ch_filing_date"] = filing.get("date", "")

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = _download_ch_pdf(ch_client, filing, pdf_path)
        if downloaded is None:
            row["status"] = "error"
            row["error"] = f"CH PDF download failed for {ch_number}"
            print(f"  [{i}/{len(candidates)}] {company_name}: download failed")
            errors += 1
            time.sleep(sleep_ch)
            continue

        row["pdf_path"] = str(downloaded)
        row["status"] = "ch_downloaded"
        ok += 1
        print(f"  [{i}/{len(candidates)}] {company_name} FY{fiscal_year}: downloaded → {downloaded.name}")
        time.sleep(sleep_ch)

    print(f"[download_ch] done. downloaded={ok}, errors={errors}")


# ---------------------------------------------------------------------------
# Phase 2: submit Mistral batch
# ---------------------------------------------------------------------------

class QuotaExceededError(Exception):
    pass


def _pdf_to_data_url(pdf_path: Path) -> str:
    encoded = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


def _build_ocr_request_body(row_id: str, data_url: str) -> bytes:
    record = {
        "custom_id": row_id,
        "body": {
            "model": MISTRAL_OCR_MODEL,
            "document": {"type": "document_url", "document_url": data_url},
            "include_image_base64": False,
            "extract_header": True,
            "extract_footer": True,
        },
    }
    return (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")


def _upload_jsonl_to_mistral(jsonl_path: Path, *, api_key: str, timeout: int = 120) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    with jsonl_path.open("rb") as f:
        resp = requests.post(
            MISTRAL_FILES_URL,
            headers=headers,
            files={"file": (jsonl_path.name, f, "application/json")},
            data={"purpose": "batch"},
            timeout=timeout,
        )
    if resp.status_code in (402, 429):
        raise QuotaExceededError(f"Mistral {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    return resp.json()["id"]


def _submit_batch_job(file_id: str, *, api_key: str, timeout: int = 60) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"input_files": [file_id], "model": MISTRAL_OCR_MODEL, "endpoint": "/v1/ocr"}
    resp = requests.post(MISTRAL_BATCH_URL, headers=headers, json=payload, timeout=timeout)
    if resp.status_code in (402, 429):
        raise QuotaExceededError(f"Mistral {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    return resp.json()["id"]


def phase_submit_batch(
    manifest: dict[str, dict[str, str]],
    batch_jobs: list[dict[str, Any]],
    *,
    mistral_api_key: str,
    output_dir: Path,
    max_batch_mb: float,
    max_batch_items: int | None,
    batch_submit_delay: float,
    overwrite: bool,
    limit: int | None,
    dry_run: bool,
) -> None:
    active_ids = {
        row_id for job in batch_jobs
        if job.get("status") not in ("SUCCESS", "FAILED")
        for row_id in job.get("id_list", [])
    }
    candidates = [
        row for row in manifest.values()
        if row["status"] == "ch_downloaded"
        and row.get("pdf_path")
        and (overwrite or not row.get("batch_job_id"))
        and row["id"] not in active_ids
    ]
    if limit:
        candidates = candidates[:limit]

    print(f"\n[submit_batch] {len(candidates)} PDFs to batch")
    if not candidates:
        return
    if dry_run:
        total_mb = sum(
            Path(r["pdf_path"]).stat().st_size / (1024 * 1024)
            for r in candidates if Path(r["pdf_path"]).exists()
        )
        print(f"  ~{total_mb:.1f} MB PDFs → ~{total_mb * 1.34:.0f} MB base64 "
              f"→ ~{max(1, int(total_mb * 1.34 / max_batch_mb))} batch(es)")
        return

    batch_dir = output_dir / "batch_jobs"
    batch_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_batch_mb * 1024 * 1024)

    current_lines: list[tuple[str, bytes]] = []
    current_size = 0
    batch_index = len(batch_jobs)

    def _flush(lines: list[tuple[str, bytes]], idx: int) -> None:
        if not lines:
            return
        jsonl_path = batch_dir / f"batch_{idx:04d}.jsonl"
        jsonl_path.write_bytes(b"".join(b for _, b in lines))
        id_list = [row_id for row_id, _ in lines]
        print(f"  Uploading batch {idx} ({len(id_list)} PDFs, "
              f"{jsonl_path.stat().st_size / 1024 / 1024:.1f} MB) …")
        file_id = _upload_jsonl_to_mistral(jsonl_path, api_key=mistral_api_key)
        job_id = _submit_batch_job(file_id, api_key=mistral_api_key)
        print(f"  Batch {idx}: job_id={job_id}")
        batch_jobs.append({
            "batch_index": idx,
            "jsonl_path": str(jsonl_path),
            "mistral_file_id": file_id,
            "job_id": job_id,
            "id_list": id_list,
            "status": "QUEUED",
        })
        for row_id in id_list:
            manifest[row_id]["batch_job_id"] = job_id

    try:
        for row in candidates:
            row_id = row["id"]
            pdf_path = Path(row["pdf_path"])
            if not pdf_path.exists():
                row["status"] = "error"
                row["error"] = f"PDF not found: {pdf_path}"
                continue
            data_url = _pdf_to_data_url(pdf_path)
            line_bytes = _build_ocr_request_body(row_id, data_url)

            size_full = bool(current_lines) and (current_size + len(line_bytes) > max_bytes)
            items_full = bool(max_batch_items) and len(current_lines) >= max_batch_items
            if size_full or items_full:
                _flush(current_lines, batch_index)
                batch_index += 1
                current_lines = []
                current_size = 0
                if batch_submit_delay > 0:
                    time.sleep(batch_submit_delay)

            current_lines.append((row_id, line_bytes))
            current_size += len(line_bytes)

        if current_lines:
            _flush(current_lines, batch_index)

    except QuotaExceededError as exc:
        print(f"\n[submit_batch] Quota exceeded: {exc}")
        print("  Top up your Mistral account and re-run --phase submit_batch.")

    submitted = sum(1 for r in manifest.values() if r.get("batch_job_id"))
    print(f"[submit_batch] done. {len(batch_jobs)} job(s) submitted, {submitted} items queued.")


# ---------------------------------------------------------------------------
# Phase 3: collect batch results
# ---------------------------------------------------------------------------

def _get_job_status(job_id: str, *, api_key: str) -> dict[str, Any]:
    resp = requests.get(f"{MISTRAL_BATCH_URL}/{job_id}",
                        headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    if resp.status_code == 429:
        raise QuotaExceededError(f"429 polling {job_id}")
    resp.raise_for_status()
    return resp.json()


def _download_file(file_id: str, *, api_key: str) -> str:
    resp = requests.get(f"{MISTRAL_FILES_URL}/{file_id}/content",
                        headers={"Authorization": f"Bearer {api_key}"}, timeout=300)
    resp.raise_for_status()
    return resp.text


def _stitch_markdown(pages: list[dict[str, Any]]) -> str:
    ordered = sorted(pages, key=lambda p: int(p.get("index", 0)))
    chunks = [str(p.get("markdown") or "").strip() for p in ordered]
    return "\n\n".join(c for c in chunks if c).strip() + "\n"


def phase_collect_batch(
    manifest: dict[str, dict[str, str]],
    batch_jobs: list[dict[str, Any]],
    *,
    mistral_api_key: str,
    output_dir: Path,
    poll_interval: int,
    dry_run: bool,
) -> None:
    active = [j for j in batch_jobs if j.get("status") not in ("SUCCESS", "FAILED", "COLLECTED")]
    print(f"\n[collect_batch] {len(active)} active job(s)")
    if not active or dry_run:
        return

    md_dir = output_dir / "ch_processed" / "markdown"
    md_dir.mkdir(parents=True, exist_ok=True)

    for job in active:
        job_id = job["job_id"]
        print(f"\n  Monitoring job {job_id} ({len(job.get('id_list', []))} items) …")
        while True:
            try:
                status_data = _get_job_status(job_id, api_key=mistral_api_key)
            except QuotaExceededError:
                print("  429 — re-run collect_batch later.")
                return
            except requests.RequestException as exc:
                print(f"  Poll error: {exc}. Retrying…")
                time.sleep(poll_interval)
                continue

            status = status_data.get("status", "UNKNOWN")
            completed = status_data.get("completed_requests", 0)
            total = status_data.get("total_requests", len(job.get("id_list", [])))
            print(f"    status={status}  {completed}/{total}")
            job["status"] = status
            if status in ("SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"):
                break
            time.sleep(poll_interval)

        output_file_id = status_data.get("output_file")
        if not output_file_id:
            print(f"  No output_file for job {job_id}")
            job["status"] = "FAILED"
            continue

        results_text = _download_file(output_file_id, api_key=mistral_api_key)
        saved = failed = 0
        for line in results_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = record.get("custom_id", "")
            if row_id not in manifest:
                continue
            body = record.get("response", {}).get("body", {})
            error = record.get("error")
            if error or not body:
                manifest[row_id]["status"] = "error"
                manifest[row_id]["error"] = str(error or "empty response")
                failed += 1
                continue
            markdown = _stitch_markdown(body.get("pages") or [])
            if not markdown.strip():
                manifest[row_id]["status"] = "error"
                manifest[row_id]["error"] = "OCR returned empty markdown"
                failed += 1
                continue
            md_path = md_dir / f"{row_id}.md"
            md_path.write_text(markdown, encoding="utf-8")
            manifest[row_id]["status"] = "ch_processed"
            manifest[row_id]["markdown_path"] = str(md_path)
            saved += 1

        job["status"] = "COLLECTED"
        print(f"  Job {job_id}: saved={saved}, failed={failed}")

    total = sum(1 for r in manifest.values() if r["status"] == "ch_processed")
    print(f"\n[collect_batch] done. Total ch_processed: {total}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    load_dotenv(REPO_ROOT / ".env.local", override=True)
    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(REPO_ROOT / "OCR_test" / ".env.local", override=False)
    load_dotenv(REPO_ROOT / "OCR_test" / ".env", override=False)

    ch_api_key = os.environ.get("COMPANIES_HOUSE_API_KEY", "")
    mistral_api_key = os.environ.get("MISTRAL_API_KEY", "")

    phases = ["download_ch", "submit_batch", "collect_batch"] \
        if args.phase == "all" else [args.phase]

    if "download_ch" in phases and not ch_api_key and not args.dry_run:
        print("COMPANIES_HOUSE_API_KEY not set.", file=sys.stderr)
        return 1
    if any(p in phases for p in ("submit_batch", "collect_batch")) \
            and not mistral_api_key and not args.dry_run:
        print("MISTRAL_API_KEY not set.", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "gap_manifest.csv"
    batch_state_path = output_dir / "batch_state.json"

    # Discover missing companies and seed manifest
    missing = find_missing_companies()
    print(f"Missing companies (no 2021 AR in corpus): {len(missing)}")

    manifest = load_manifest(manifest_path)
    new_count = 0
    for item in missing:
        if item["id"] not in manifest:
            manifest[item["id"]] = _blank_row(item)
            new_count += 1
    if new_count:
        print(f"  Added {new_count} new items to manifest")

    from collections import Counter
    status_counts = Counter(r["status"] for r in manifest.values())
    print("  Current status:", dict(sorted(status_counts.items())))

    batch_jobs = load_batch_state(batch_state_path)
    ch_client = CompaniesHouseClient(api_key=ch_api_key) if "download_ch" in phases and not args.dry_run else None

    try:
        if "download_ch" in phases:
            phase_download_ch(
                manifest, ch_client=ch_client, output_dir=output_dir,
                sleep_ch=args.sleep_ch, overwrite=args.overwrite,
                limit=args.limit, dry_run=args.dry_run,
            )

        if "submit_batch" in phases:
            phase_submit_batch(
                manifest, batch_jobs, mistral_api_key=mistral_api_key,
                output_dir=output_dir, max_batch_mb=args.max_batch_mb,
                max_batch_items=args.max_batch_items,
                batch_submit_delay=args.batch_submit_delay,
                overwrite=args.overwrite, limit=args.limit, dry_run=args.dry_run,
            )

        if "collect_batch" in phases:
            phase_collect_batch(
                manifest, batch_jobs, mistral_api_key=mistral_api_key,
                output_dir=output_dir, poll_interval=args.poll_interval,
                dry_run=args.dry_run,
            )

    finally:
        save_manifest(manifest_path, manifest)
        save_batch_state(batch_state_path, batch_jobs)
        print(f"\nState saved → {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
