#!/usr/bin/env python3
"""Patch gaps in the FR consolidated manifest with CH PDFs processed via Mistral OCR (Batch API).

Four phases (run in order, or pass ``--phase all`` to run sequentially):

  probe_fr      Probe FR API for all gap items; those that return 200 are saved
                immediately as fr_recovered.  Items still missing are marked
                needs_ch (or fr_pending if still processing on FR's side).

  download_ch   For each needs_ch item: GLEIF LEI→company-number lookup, then
                Companies House annual-accounts PDF download.

  submit_batch  Group downloaded PDFs into batches (split by JSONL file size),
                upload each JSONL to Mistral Files API, submit batch jobs.
                Batch job IDs are saved to batch_state.json.

  collect_batch Poll all active batch jobs; when a job succeeds, download the
                result JSONL and save one markdown file per item.

Resume safety
  Every phase reads gap_manifest.csv before starting and skips items whose
  status is already terminal (fr_recovered, ch_processed, fr_pending, error).
  Batch job IDs are stored in batch_state.json; re-running collect_batch will
  resume polling any jobs that are still running.

Payment / quota errors
  Both the FR probe (429) and Mistral batch submission (429) are caught.
  The script saves all in-progress state and exits with a clear message so
  you can top up the account and re-run with the same command.

Outputs  (all under --output-dir, default data/ch_gap_fill/)
  fr_recovered/markdown/{pk}.md
  ch_processed/pdfs/{pk}.pdf
  ch_processed/markdown/{pk}.md
  batch_jobs/{batch_n}.jsonl          (uploaded batch request files)
  gap_manifest.csv
  batch_state.json

Usage:
  python scripts/patch_ch_gaps.py --phase all --dry-run
  python scripts/patch_ch_gaps.py --phase probe_fr --limit 200
  python scripts/patch_ch_gaps.py --phase download_ch
  python scripts/patch_ch_gaps.py --phase submit_batch --max-batch-mb 300
  python scripts/patch_ch_gaps.py --phase collect_batch

Credentials required in .env.local:
  FR_API_KEY
  COMPANIES_HOUSE_API_KEY
  MISTRAL_API_KEY
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from src.companies_house import CompaniesHouseClient

FR_API_BASE = "https://api.financialreports.eu"
GLEIF_API_BASE = "https://api.gleif.org/api/v1"
MISTRAL_FILES_URL = "https://api.mistral.ai/v1/files"
MISTRAL_BATCH_URL = "https://api.mistral.ai/v1/batch/jobs"
MISTRAL_OCR_MODEL = "mistral-ocr-latest"

DEFAULT_MANIFEST_RAW = REPO_ROOT / "data" / "FR_dataset" / "manifest_raw.csv"
DEFAULT_CONSOLIDATED_METADATA = REPO_ROOT / "data" / "FR_consolidated" / "metadata.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "ch_gap_fill"
DEFAULT_GAP_2021_MANIFEST = REPO_ROOT / "data" / "ch_gap_fill_2021" / "gap_manifest.csv"
DEFAULT_GAP_2021_DIRECT_MANIFEST = REPO_ROOT / "data" / "ch_gap_fill_2021_direct" / "gap_manifest.csv"
DEFAULT_GAP_FY2020_MANIFEST = REPO_ROOT / "data" / "ch_gap_fill_fy2020" / "gap_manifest.csv"

ANNUAL_REPORT_TYPES = {"Annual Report", "Annual Report (ESEF)"}
FR_ANNUAL_REPORT_CODE = "10-K"
FR_MATCH_WINDOW_DAYS = 270
HAVE_MARKDOWN_STATUSES = {"fr_recovered", "ch_processed"}

TERMINAL_STATUSES = {"fr_recovered", "ch_processed", "fr_pending", "error", "stub", "duplicate"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--phase",
        choices=["probe_fr", "download_ch", "submit_batch", "collect_batch", "all"],
        default="all",
        help="Which phase(s) to run (default: all)",
    )
    parser.add_argument("--manifest-raw", type=Path, default=DEFAULT_MANIFEST_RAW)
    parser.add_argument("--consolidated-metadata", type=Path, default=DEFAULT_CONSOLIDATED_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-year", type=int, default=2022)
    parser.add_argument("--max-year", type=int, default=2025)
    parser.add_argument(
        "--release-year", type=int, default=None,
        help="Filter by release_year instead of fiscal_year (e.g. --release-year 2021)",
    )
    parser.add_argument(
        "--input-csv", type=Path, default=None,
        help="Alternative universe input: ch_period_of_accounts.csv (or similar). "
             "Must have columns: lei, fiscal_year, made_up_date, ch_company_number, company_name. "
             "Probes FR by LEI search instead of by PK. Use with --phase probe_fr.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max gap items to process per phase")
    parser.add_argument("--max-batch-mb", type=float, default=350.0,
                        help="Max JSONL batch file size in MB before splitting (default: 350)")
    parser.add_argument("--max-batch-items", type=int, default=None,
                        help="Max PDFs per batch job (default: unlimited). Use small values e.g. 5 for free-tier accounts.")
    parser.add_argument("--batch-submit-delay", type=float, default=2.0,
                        help="Seconds to wait between batch job submissions (default: 2)")
    parser.add_argument(
        "--sleep-fr",
        type=float,
        default=0.02,
        help="Pause between FR API calls (s). Default 0.02 ~= 50 requests/sec.",
    )
    parser.add_argument("--sleep-gleif", type=float, default=0.5, help="Pause between GLEIF calls (s)")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between batch job status polls (default: 60)")
    parser.add_argument("--include-pending", action="store_true",
                        help="Also process fr_pending items via CH instead of waiting for FR")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process items even if output already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without making API calls")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Gap manifest (CSV state file)
# ---------------------------------------------------------------------------

MANIFEST_FIELDS = [
    "pk", "lei", "company_name", "fiscal_year", "filing_type",
    "status",             # pending | fr_recovered | fr_pending | needs_ch |
                          # downloading | ch_downloaded | ch_processed | error
    "fr_processing_status",
    "ch_company_number", "ch_filing_date",
    "pdf_path", "markdown_path",
    "batch_job_id",       # Mistral batch job this item is in
    "error",
]


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Load gap_manifest.csv into a dict keyed by pk."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["pk"]: row for row in csv.DictReader(f)}


def save_manifest(path: Path, rows: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows.values():
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def _blank_row(pk: str, lei: str, company_name: str, fiscal_year: int, filing_type: str) -> dict[str, str]:
    return {field: "" for field in MANIFEST_FIELDS} | {
        "pk": pk,
        "lei": lei,
        "company_name": company_name,
        "fiscal_year": str(fiscal_year),
        "filing_type": filing_type,
        "status": "pending",
    }


def _pair_key(lei: str, fiscal_year: int | str) -> tuple[str, str]:
    return (str(lei).strip(), str(fiscal_year).strip())


def _status_rank(status: str) -> int:
    ranking = {
        "fr_recovered": 7,
        "ch_processed": 6,
        "fr_pending": 5,
        "needs_ch": 4,
        "ch_downloaded": 3,
        "downloading": 2,
        "pending": 1,
        "error": 0,
        "stub": -1,
        "duplicate": -2,
    }
    return ranking.get(status or "", 0)


def _row_rank(row: dict[str, str]) -> tuple[int, int]:
    pk = str(row.get("pk", "")).strip()
    return (_status_rank(row.get("status", "")), int(pk.isdigit()))


def _build_manifest_pair_index(manifest: dict[str, dict[str, str]]) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for pk, row in manifest.items():
        pair = _pair_key(row.get("lei", ""), row.get("fiscal_year", ""))
        if not pair[0] or not pair[1]:
            continue
        existing_pk = index.get(pair)
        if existing_pk is None or _row_rank(row) > _row_rank(manifest[existing_pk]):
            index[pair] = pk
    return index


def _ensure_manifest_row(
    manifest: dict[str, dict[str, str]],
    pair_index: dict[tuple[str, str], str],
    item: dict[str, str],
) -> tuple[str, dict[str, str]]:
    pair = _pair_key(item["lei"], item["fiscal_year"])
    resolved_pk = item["pk"] if item["pk"] in manifest else None
    if resolved_pk is None and item.get("probe_method") == "lei":
        resolved_pk = pair_index.get(pair)
    if resolved_pk is None:
        resolved_pk = item["pk"]
        manifest[resolved_pk] = _blank_row(
            resolved_pk,
            item["lei"],
            item["company_name"],
            int(item["fiscal_year"]),
            item["filing_type"],
        )
    row = manifest[resolved_pk]
    pair_index[pair] = resolved_pk

    if item.get("company_name") and not row.get("company_name"):
        row["company_name"] = item["company_name"]
    if item.get("filing_type") and not row.get("filing_type"):
        row["filing_type"] = item["filing_type"]
    if item.get("ch_company_number") and not row.get("ch_company_number"):
        row["ch_company_number"] = item["ch_company_number"]
    return resolved_pk, row


def _rekey_manifest_row(
    manifest: dict[str, dict[str, str]],
    pair_index: dict[tuple[str, str], str],
    old_pk: str,
    new_pk: str,
) -> dict[str, str]:
    if old_pk == new_pk:
        row = manifest[new_pk]
        pair_index[_pair_key(row.get("lei", ""), row.get("fiscal_year", ""))] = new_pk
        return row

    row = manifest.pop(old_pk)
    row["pk"] = new_pk
    existing = manifest.get(new_pk)
    if existing is None:
        manifest[new_pk] = row
        target = row
    else:
        for field in MANIFEST_FIELDS:
            if row.get(field) and not existing.get(field):
                existing[field] = row[field]
        existing["pk"] = new_pk
        target = existing

    pair_index[_pair_key(target.get("lei", ""), target.get("fiscal_year", ""))] = new_pk
    return target


# ---------------------------------------------------------------------------
# Batch state file
# ---------------------------------------------------------------------------

def load_batch_state(path: Path) -> list[dict[str, Any]]:
    """Load batch_state.json — list of Mistral batch job records."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_batch_state(path: Path, jobs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Source manifest loading
# ---------------------------------------------------------------------------

def load_gap(
    manifest_raw_path: Path,
    consolidated_path: Path,
    *,
    min_year: int,
    max_year: int,
    release_year: int | None = None,
) -> list[dict[str, str]]:
    """Return manifest_raw rows whose pk is absent from consolidated metadata.

    If ``release_year`` is set, filter by the row's ``release_year`` field
    instead of ``fiscal_year``.  ``min_year``/``max_year`` are ignored in
    that case.
    """
    with consolidated_path.open(newline="", encoding="utf-8") as f:
        consolidated_pks = {row["pk"] for row in csv.DictReader(f)}

    items: list[dict[str, str]] = []
    with manifest_raw_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pk = row["pk"].strip()
            if pk in consolidated_pks:
                continue
            filing_type = row.get("filing_type__name", "").strip()
            if filing_type not in ANNUAL_REPORT_TYPES:
                continue
            try:
                fiscal_year = int(row.get("fiscal_year", "0"))
            except ValueError:
                continue
            if release_year is not None:
                try:
                    row_release_year = int(row.get("release_year", "0"))
                except ValueError:
                    continue
                if row_release_year != release_year:
                    continue
            else:
                if not (min_year <= fiscal_year <= max_year):
                    continue
            items.append({
                "pk": pk,
                "lei": row.get("company__lei", "").strip(),
                "company_name": row.get("company__name", "").strip(),
                "fiscal_year": str(fiscal_year),
                "filing_type": filing_type,
                "probe_method": "pk",
            })
    return items


def load_gap_from_csv(
    input_csv_path: Path,
    *,
    min_year: int,
    max_year: int,
    release_year: int | None = None,
) -> list[dict[str, str]]:
    """Build a gap universe from CH-style company/year rows.

    Expected columns:
      lei, fiscal_year, made_up_date
    Optional columns:
      ch_company_number/company_number, company_name/name, submission_date, filing_type, release_year
    """
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}

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
            row_release_year = str(raw.get("release_year") or "").strip()
            if release_year is not None:
                release_year_str = row_release_year or submission_date[:4]
                if release_year_str != str(release_year):
                    continue
            elif not (min_year <= fiscal_year_int <= max_year):
                continue

            ch_company_number = str(
                raw.get("ch_company_number")
                or raw.get("company_number")
                or ""
            ).strip()
            if ch_company_number.isdigit():
                ch_company_number = ch_company_number.zfill(8)

            filing_type = str(raw.get("filing_type") or "").strip()
            company_name = str(raw.get("company_name") or raw.get("name") or "").strip()
            normalized = {
                "pk": f"{lei}_{fiscal_year}",
                "lei": lei,
                "company_name": company_name,
                "fiscal_year": fiscal_year,
                "filing_type": "Annual Report",
                "probe_method": "lei",
                "made_up_date": str(raw.get("made_up_date") or "").strip(),
                "submission_date": submission_date,
                "ch_company_number": ch_company_number,
                "_source_filing_type": filing_type,
            }
            groups.setdefault(_pair_key(lei, fiscal_year), []).append(normalized)

    items: list[dict[str, str]] = []
    for group in groups.values():
        aa = [row for row in group if row.get("_source_filing_type") == "AA"]
        pool = aa if aa else group
        best = max(pool, key=lambda row: row.get("submission_date", ""))
        best = dict(best)
        best.pop("_source_filing_type", None)
        items.append(best)

    items.sort(key=lambda row: (row["fiscal_year"], row["lei"]))
    return items


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


def load_existing_pairs() -> set[tuple[str, str]]:
    """Return (lei, fiscal_year) pairs that already have usable markdown on disk."""
    pairs: set[tuple[str, str]] = set()

    if DEFAULT_CONSOLIDATED_METADATA.exists():
        with DEFAULT_CONSOLIDATED_METADATA.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lei = str(row.get("lei") or "").strip()
                fiscal_year = str(row.get("fiscal_year") or "").strip()
                src_path = _resolve_repo_relative_path(row.get("src_path", ""))
                if lei and fiscal_year and src_path and src_path.exists():
                    pairs.add(_pair_key(lei, fiscal_year))

    for manifest_path in (
        DEFAULT_OUTPUT_DIR / "gap_manifest.csv",
        DEFAULT_GAP_2021_MANIFEST,
        DEFAULT_GAP_FY2020_MANIFEST,
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
                if lei and fiscal_year and markdown_path and markdown_path.exists():
                    pairs.add(_pair_key(lei, fiscal_year))

    if DEFAULT_GAP_2021_DIRECT_MANIFEST.exists():
        with DEFAULT_GAP_2021_DIRECT_MANIFEST.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ch_processed":
                    continue
                lei = str(row.get("lei") or "").strip()
                fiscal_year = str(row.get("fiscal_year") or "").strip()
                markdown_path = _resolve_repo_relative_path(row.get("markdown_path", ""))
                if lei and fiscal_year and markdown_path and markdown_path.exists():
                    pairs.add(_pair_key(lei, fiscal_year))

    return pairs


# ---------------------------------------------------------------------------
# FR API response helpers
# ---------------------------------------------------------------------------

def _extract_fr_markdown(resp: requests.Response) -> str:
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


def _extract_processing_status(resp: requests.Response) -> str | None:
    try:
        body = resp.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    raw = body.get("processing_status")
    return str(raw).strip() if raw is not None else None


def _processing_status_rank(status: str) -> int:
    status = status.lower()
    if not status:
        return 0
    if any(token in status for token in ("pending", "queued", "running", "processing")):
        return 1
    if any(token in status for token in ("failed", "error")):
        return 2
    return 0


def _fr_pause(delay: float) -> None:
    if delay > 0:
        time.sleep(delay)


def _fr_search_annual_by_lei(
    lei: str,
    *,
    fr_api_key: str,
    made_up_date: str,
    submission_date: str = "",
    sleep_fr: float = 0.0,
    timeout: int = 60,
) -> list[dict[str, str]]:
    """Return FR annual-report candidates for a LEI matched to the CH filing window."""
    made_up = _parse_iso_date(made_up_date)
    if made_up is None:
        return []
    submission = _parse_iso_date(submission_date)

    headers = {"x-api-key": fr_api_key, "Accept": "application/json"}
    params = {
        "company__lei": lei,
        "filing_type__code": FR_ANNUAL_REPORT_CODE,
        "page_size": 100,
    }
    url = f"{FR_API_BASE}/filings/"

    matches: list[dict[str, Any]] = []
    while url:
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        if resp.status_code == 429:
            raise QuotaExceededError(f"FR 429 while searching LEI {lei}")
        _fr_pause(sleep_fr)
        resp.raise_for_status()

        body = resp.json()
        results = body.get("results", []) if isinstance(body, dict) else []
        for filing in results:
            filing_id = str(filing.get("id") or "").strip()
            filing_date = _parse_iso_date(str(filing.get("filing_date") or ""))
            if not filing_id or filing_date is None:
                continue

            delta_days = (filing_date - made_up).days
            if not (0 <= delta_days <= FR_MATCH_WINDOW_DAYS):
                continue

            processing_status = str(filing.get("processing_status") or "").strip()
            submission_delta = abs((filing_date - submission).days) if submission else delta_days
            matches.append({
                "pk": filing_id,
                "processing_status": processing_status,
                "filing_date": filing_date.isoformat(),
                "submission_delta_days": submission_delta,
                "delta_days": delta_days,
                "title": str(filing.get("title") or "").strip(),
            })

        next_url = body.get("next") if isinstance(body, dict) else None
        url = str(next_url).strip() if next_url else ""
        if url.startswith("/"):
            url = f"{FR_API_BASE}{url}"
        params = None

    matches.sort(
        key=lambda filing: (
            _processing_status_rank(filing.get("processing_status", "")),
            int(filing.get("submission_delta_days", 10 ** 9)),
            int(filing.get("delta_days", 10 ** 9)),
            filing.get("pk", ""),
        )
    )
    return matches


# ---------------------------------------------------------------------------
# Phase 1: FR API probe
# ---------------------------------------------------------------------------

def phase_probe_fr(
    manifest: dict[str, dict[str, str]],
    gap_items: list[dict[str, str]],
    *,
    fr_api_key: str,
    output_dir: Path,
    sleep_fr: float,
    overwrite: bool,
    limit: int | None,
    dry_run: bool,
) -> None:
    """Probe FR API for each pending gap item; update manifest in place."""
    pair_index = _build_manifest_pair_index(manifest)
    candidates = []
    for item in gap_items:
        resolved_pk = item["pk"] if item["pk"] in manifest else None
        if resolved_pk is None and item.get("probe_method") == "lei":
            resolved_pk = pair_index.get(_pair_key(item["lei"], item["fiscal_year"]))
        status = manifest.get(resolved_pk, {}).get("status", "pending") if resolved_pk else "pending"
        if overwrite or status not in TERMINAL_STATUSES:
            candidates.append(item)
    if limit:
        candidates = candidates[:limit]

    print(f"\n[probe_fr] {len(candidates)} items to probe")
    if dry_run:
        return

    recovered = 0
    with tqdm(candidates, desc="probe_fr", unit="item") as pbar:
        for item in pbar:
            current_pk, row = _ensure_manifest_row(manifest, pair_index, item)
            row["error"] = ""

            probe_candidates = [{"pk": current_pk, "processing_status": row.get("fr_processing_status", "")}]
            if item.get("probe_method") == "lei":
                try:
                    probe_candidates = _fr_search_annual_by_lei(
                        item["lei"],
                        fr_api_key=fr_api_key,
                        made_up_date=item.get("made_up_date", ""),
                        submission_date=item.get("submission_date", ""),
                        sleep_fr=sleep_fr,
                    )
                except QuotaExceededError:
                    tqdm.write(
                        f"[probe_fr] 429 rate-limited "
                        f"(lei={item['lei']}, fy={item['fiscal_year']}). Save state and re-run."
                    )
                    break
                except requests.RequestException as exc:
                    row["error"] = f"FR search error: {exc}"
                    continue

                if not probe_candidates:
                    row["status"] = "needs_ch"
                    row["fr_processing_status"] = "not_found"
                    pbar.set_postfix(
                        recovered=recovered,
                        pending=sum(1 for r in manifest.values() if r["status"] == "fr_pending"),
                        needs_ch=sum(1 for r in manifest.values() if r["status"] == "needs_ch"),
                    )
                    continue

                canonical_pk = probe_candidates[0]["pk"]
                if current_pk != canonical_pk:
                    row = _rekey_manifest_row(manifest, pair_index, current_pk, canonical_pk)
                    current_pk = canonical_pk
                row["fr_processing_status"] = probe_candidates[0].get("processing_status", "")

            headers = {"x-api-key": fr_api_key, "Accept": "application/json"}
            saw_pending = False
            final_processing_status = row.get("fr_processing_status", "") or "not_found"
            rate_limited = False
            request_failed = False

            for candidate in probe_candidates:
                candidate_pk = candidate["pk"]
                fr_md_path = output_dir / "fr_recovered" / "markdown" / f"{candidate_pk}.md"

                # Re-use already-saved file
                if fr_md_path.exists() and not overwrite:
                    if current_pk != candidate_pk:
                        row = _rekey_manifest_row(manifest, pair_index, current_pk, candidate_pk)
                        current_pk = candidate_pk
                    row["status"] = "fr_recovered"
                    row["markdown_path"] = str(fr_md_path)
                    row["fr_processing_status"] = candidate.get("processing_status", "")
                    row["error"] = ""
                    recovered += 1
                    break

                url = f"{FR_API_BASE}/filings/{candidate_pk}/markdown/"
                try:
                    resp = requests.get(url, headers=headers, timeout=60)
                except requests.RequestException as exc:
                    row["error"] = f"FR request error: {exc}"
                    request_failed = True
                    break

                if resp.status_code == 429:
                    tqdm.write(
                        f"[probe_fr] 429 rate-limited (pk={candidate_pk}). Save state and re-run."
                    )
                    rate_limited = True
                    break
                _fr_pause(sleep_fr)

                if resp.status_code == 200:
                    text = _extract_fr_markdown(resp)
                    if text.strip():
                        if current_pk != candidate_pk:
                            row = _rekey_manifest_row(manifest, pair_index, current_pk, candidate_pk)
                            current_pk = candidate_pk
                        fr_md_path.parent.mkdir(parents=True, exist_ok=True)
                        fr_md_path.write_text(text, encoding="utf-8")
                        row["status"] = "fr_recovered"
                        row["markdown_path"] = str(fr_md_path)
                        row["fr_processing_status"] = candidate.get("processing_status", "")
                        row["error"] = ""
                        recovered += 1
                        break

                    final_processing_status = "empty_body"
                    continue

                if resp.status_code == 404:
                    ps = _extract_processing_status(resp) or candidate.get("processing_status") or "not_found"
                    final_processing_status = ps
                    if "pending" in ps.lower():
                        saw_pending = True
                    continue

                if resp.status_code == 403:
                    final_processing_status = "forbidden"
                    continue

                final_processing_status = f"http_{resp.status_code}"

            if rate_limited:
                break
            if request_failed:
                continue

            if row.get("status") != "fr_recovered":
                row["fr_processing_status"] = final_processing_status
                row["status"] = "fr_pending" if saw_pending else "needs_ch"

            pbar.set_postfix(
                recovered=recovered,
                pending=sum(1 for r in manifest.values() if r["status"] == "fr_pending"),
                needs_ch=sum(1 for r in manifest.values() if r["status"] == "needs_ch"),
            )

    print(f"[probe_fr] done. fr_recovered={recovered}, "
          f"needs_ch={sum(1 for r in manifest.values() if r['status']=='needs_ch')}, "
          f"fr_pending={sum(1 for r in manifest.values() if r['status']=='fr_pending')}")


# ---------------------------------------------------------------------------
# Phase 2: CH download
# ---------------------------------------------------------------------------

def _gleif_company_number(lei: str, *, timeout: int = 30) -> str | None:
    if not lei:
        return None
    url = f"{GLEIF_API_BASE}/lei-records/{lei}"
    try:
        resp = requests.get(url, timeout=timeout, headers={"Accept": "application/vnd.api+json"})
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    reg = (data.get("data", {}).get("attributes", {}).get("entity", {}).get("registeredAs") or "")
    reg = str(reg).strip()
    if not reg:
        return None
    if reg.isdigit():
        reg = reg.zfill(8)
    return reg


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


def phase_download_ch(
    manifest: dict[str, dict[str, str]],
    *,
    ch_client: CompaniesHouseClient,
    output_dir: Path,
    sleep_gleif: float,
    overwrite: bool,
    include_pending: bool,
    limit: int | None,
    dry_run: bool,
) -> None:
    """Download CH PDFs for needs_ch items; update manifest in place."""
    target_statuses = {"needs_ch", "fr_pending"} if include_pending else {"needs_ch"}
    candidates = [
        row for row in manifest.values()
        if row["status"] in target_statuses and (overwrite or not row.get("pdf_path"))
    ]
    if limit:
        candidates = candidates[:limit]

    print(f"\n[download_ch] {len(candidates)} items to download")
    if dry_run:
        return

    ok = 0
    with tqdm(candidates, desc="download_ch", unit="item") as pbar:
        for row in pbar:
            pk, lei = row["pk"], row["lei"]
            company_name = row["company_name"]
            fiscal_year = int(row["fiscal_year"])

            pdf_path = output_dir / "ch_processed" / "pdfs" / f"{pk}.pdf"
            if pdf_path.exists() and not overwrite:
                row["pdf_path"] = str(pdf_path)
                row["status"] = "ch_downloaded"
                ok += 1
                pbar.set_postfix(downloaded=ok, errors=sum(1 for r in candidates if r["status"] == "error"))
                continue

            # Re-use CH number from the source CSV when available; fall back to GLEIF.
            ch_number = str(row.get("ch_company_number") or "").strip()
            if not ch_number:
                ch_number = _gleif_company_number(lei)
                time.sleep(sleep_gleif)
            if not ch_number:
                row["status"] = "error"
                row["error"] = f"GLEIF lookup failed for LEI {lei}"
                tqdm.write(f"[download_ch] {company_name}: GLEIF failed")
                pbar.set_postfix(downloaded=ok, errors=sum(1 for r in candidates if r["status"] == "error"))
                continue
            row["ch_company_number"] = ch_number

            # CH filing
            filing = _find_ch_filing(ch_client, ch_number, fiscal_year)
            if not filing:
                row["status"] = "error"
                row["error"] = f"no CH filing found for {ch_number} FY{fiscal_year}"
                tqdm.write(f"[download_ch] {company_name}: no CH filing")
                pbar.set_postfix(downloaded=ok, errors=sum(1 for r in candidates if r["status"] == "error"))
                continue
            row["ch_filing_date"] = filing.get("date", "")

            # Download PDF
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = _download_ch_pdf(ch_client, filing, pdf_path)
            if downloaded is None:
                row["status"] = "error"
                row["error"] = f"CH PDF download failed for {ch_number}"
                tqdm.write(f"[download_ch] {company_name}: download failed")
                pbar.set_postfix(downloaded=ok, errors=sum(1 for r in candidates if r["status"] == "error"))
                continue

            row["pdf_path"] = str(downloaded)
            row["status"] = "ch_downloaded"
            ok += 1
            pbar.set_postfix(downloaded=ok, errors=sum(1 for r in candidates if r["status"] == "error"))

    print(f"[download_ch] done. downloaded={ok}, "
          f"errors={sum(1 for r in candidates if r['status']=='error')}")


# ---------------------------------------------------------------------------
# Phase 3: Mistral batch submit
# ---------------------------------------------------------------------------

def _pdf_to_data_url(pdf_path: Path) -> str:
    encoded = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


def _build_ocr_request_body(pk: str, data_url: str) -> bytes:
    record = {
        "custom_id": pk,
        "body": {
            "model": MISTRAL_OCR_MODEL,
            "document": {
                "type": "document_url",
                "document_url": data_url,
            },
            "include_image_base64": False,
            "extract_header": True,
            "extract_footer": True,
        },
    }
    return (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")


def _upload_jsonl_to_mistral(jsonl_path: Path, *, api_key: str, timeout: int = 120) -> str:
    """Upload a JSONL file to Mistral Files API and return the file ID."""
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
        raise QuotaExceededError(f"Mistral {resp.status_code} on file upload: {resp.text}")
    resp.raise_for_status()
    return resp.json()["id"]


def _submit_batch_job(file_id: str, *, api_key: str, timeout: int = 60) -> str:
    """Submit a Mistral batch job and return the job ID."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "input_files": [file_id],
        "model": MISTRAL_OCR_MODEL,
        "endpoint": "/v1/ocr",
    }
    resp = requests.post(MISTRAL_BATCH_URL, headers=headers, json=payload, timeout=timeout)
    if resp.status_code in (402, 429):
        raise QuotaExceededError(f"Mistral {resp.status_code} on batch submit: {resp.text}")
    resp.raise_for_status()
    return resp.json()["id"]


class QuotaExceededError(Exception):
    """Raised when Mistral returns 429 (quota / payment limit reached)."""


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
    """Build JSONL batches from downloaded PDFs and submit to Mistral."""
    # Items with a PDF that haven't been OCR'd yet and aren't in an active batch
    active_pks = {pk for job in batch_jobs if job.get("status") not in ("SUCCESS", "FAILED")
                  for pk in job.get("pk_list", [])}
    candidates = [
        row for row in manifest.values()
        if row["status"] == "ch_downloaded"
        and row.get("pdf_path")
        and (overwrite or not row.get("batch_job_id"))
        and row["pk"] not in active_pks
    ]
    if limit:
        candidates = candidates[:limit]

    print(f"\n[submit_batch] {len(candidates)} PDFs to batch")
    if not candidates:
        return
    if dry_run:
        total_mb = sum(
            Path(row["pdf_path"]).stat().st_size / (1024 * 1024)
            for row in candidates
            if Path(row["pdf_path"]).exists()
        )
        print(f"  Estimated total PDF size: {total_mb:.1f} MB → "
              f"~{total_mb * 1.34:.0f} MB base64 → "
              f"~{max(1, int(total_mb * 1.34 / max_batch_mb))} batch(es)")
        return

    batch_dir = output_dir / "batch_jobs"
    batch_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_batch_mb * 1024 * 1024)

    current_lines: list[tuple[str, bytes]] = []   # (pk, jsonl_line_bytes)
    current_size = 0
    batch_index = len(batch_jobs)

    def _flush_batch(lines: list[tuple[str, bytes]], idx: int) -> None:
        if not lines:
            return
        jsonl_path = batch_dir / f"batch_{idx:04d}.jsonl"
        jsonl_path.write_bytes(b"".join(line for _, line in lines))
        pk_list = [pk for pk, _ in lines]
        print(f"  Uploading batch {idx} ({len(pk_list)} PDFs, "
              f"{jsonl_path.stat().st_size / 1024 / 1024:.1f} MB) …")
        try:
            file_id = _upload_jsonl_to_mistral(jsonl_path, api_key=mistral_api_key)
            job_id = _submit_batch_job(file_id, api_key=mistral_api_key)
        except QuotaExceededError as exc:
            print(f"\n[submit_batch] Quota exceeded: {exc}")
            print("  State saved. Top up your Mistral account and re-run --phase submit_batch.")
            raise
        print(f"  Batch {idx}: job_id={job_id}")
        batch_jobs.append({
            "batch_index": idx,
            "jsonl_path": str(jsonl_path),
            "mistral_file_id": file_id,
            "job_id": job_id,
            "pk_list": pk_list,
            "status": "QUEUED",
        })
        for pk in pk_list:
            manifest[pk]["batch_job_id"] = job_id
            manifest[pk]["status"] = "ch_downloaded"  # will be updated by collect_batch

    try:
        for row in candidates:
            pk = row["pk"]
            pdf_path = Path(row["pdf_path"])
            if not pdf_path.exists():
                row["status"] = "error"
                row["error"] = f"PDF not found: {pdf_path}"
                continue
            data_url = _pdf_to_data_url(pdf_path)
            line_bytes = _build_ocr_request_body(pk, data_url)

            size_full = current_lines and current_size + len(line_bytes) > max_bytes
            items_full = max_batch_items and len(current_lines) >= max_batch_items
            if size_full or items_full:
                _flush_batch(current_lines, batch_index)
                batch_index += 1
                current_lines = []
                current_size = 0
                if batch_submit_delay > 0:
                    time.sleep(batch_submit_delay)

            current_lines.append((pk, line_bytes))
            current_size += len(line_bytes)

        if current_lines:
            _flush_batch(current_lines, batch_index)

    except QuotaExceededError:
        # Partial state has been updated — fall through to save
        pass

    print(f"[submit_batch] submitted {len(batch_jobs) - (batch_index - len(batch_jobs))} batch job(s)")


# ---------------------------------------------------------------------------
# Phase 4: collect batch results
# ---------------------------------------------------------------------------

def _get_batch_job_status(job_id: str, *, api_key: str, timeout: int = 30) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(f"{MISTRAL_BATCH_URL}/{job_id}", headers=headers, timeout=timeout)
    if resp.status_code == 429:
        raise QuotaExceededError(f"Mistral 429 polling job {job_id}")
    resp.raise_for_status()
    return resp.json()


def _download_file_content(file_id: str, *, api_key: str, timeout: int = 300) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(f"{MISTRAL_FILES_URL}/{file_id}/content", headers=headers, timeout=timeout)
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
    """Poll active batch jobs, download results and save markdowns."""
    active = [j for j in batch_jobs if j.get("status") not in ("SUCCESS", "FAILED", "COLLECTED")]
    print(f"\n[collect_batch] {len(active)} active job(s) to monitor")
    if not active or dry_run:
        return

    md_dir = output_dir / "ch_processed" / "markdown"
    md_dir.mkdir(parents=True, exist_ok=True)

    for job in active:
        job_id = job["job_id"]
        print(f"\n  Monitoring job {job_id} ({len(job.get('pk_list', []))} items) …")

        # Poll until terminal
        while True:
            try:
                status_data = _get_batch_job_status(job_id, api_key=mistral_api_key)
            except QuotaExceededError as exc:
                print(f"  429 while polling: {exc}. Re-run collect_batch later.")
                return
            except requests.RequestException as exc:
                print(f"  Poll error: {exc}. Will retry.")
                time.sleep(poll_interval)
                continue

            status = status_data.get("status", "UNKNOWN")
            completed = status_data.get("completed_requests", 0)
            total = status_data.get("total_requests", len(job.get("pk_list", [])))
            print(f"    status={status}  {completed}/{total} completed")
            job["status"] = status

            if status in ("SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"):
                break
            time.sleep(poll_interval)

        if status not in ("SUCCESS", "FAILED"):
            print(f"  Job {job_id} ended with status={status} — no results to collect.")
            continue

        # Download output JSONL
        output_file_id = status_data.get("output_file")
        if not output_file_id:
            print(f"  No output_file in job response — cannot collect results.")
            job["status"] = "FAILED"
            continue

        print(f"  Downloading results (file_id={output_file_id}) …")
        try:
            results_text = _download_file_content(output_file_id, api_key=mistral_api_key)
        except requests.RequestException as exc:
            print(f"  Download error: {exc}")
            continue

        # Parse JSONL results
        saved = 0
        failed = 0
        for line in results_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            pk = record.get("custom_id", "")
            body = record.get("response", {}).get("body", {})
            error = record.get("error")

            if pk not in manifest:
                continue

            if error or not body:
                manifest[pk]["status"] = "error"
                manifest[pk]["error"] = str(error or "empty response body")
                failed += 1
                continue

            pages = body.get("pages") or []
            markdown = _stitch_markdown(pages)
            if not markdown.strip():
                manifest[pk]["status"] = "error"
                manifest[pk]["error"] = "OCR returned empty markdown"
                failed += 1
                continue

            md_path = md_dir / f"{pk}.md"
            md_path.write_text(markdown, encoding="utf-8")
            manifest[pk]["status"] = "ch_processed"
            manifest[pk]["markdown_path"] = str(md_path)
            saved += 1

        job["status"] = "COLLECTED"
        print(f"  Job {job_id}: saved={saved}, failed={failed}")

    total_processed = sum(1 for r in manifest.values() if r["status"] == "ch_processed")
    print(f"\n[collect_batch] done. Total ch_processed so far: {total_processed}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    load_dotenv(REPO_ROOT / ".env.local", override=True)
    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(REPO_ROOT / "OCR_test" / ".env.local", override=False)
    load_dotenv(REPO_ROOT / "OCR_test" / ".env", override=False)

    fr_api_key = os.environ.get("FR_API_KEY", "")
    mistral_api_key = os.environ.get("MISTRAL_API_KEY", "")
    ch_api_key = os.environ.get("COMPANIES_HOUSE_API_KEY", "")

    phases = ["probe_fr", "download_ch", "submit_batch", "collect_batch"] \
        if args.phase == "all" else [args.phase]

    # Validate credentials up front
    if "probe_fr" in phases and not fr_api_key and not args.dry_run:
        print("FR_API_KEY not set. Add it to .env.local.", file=sys.stderr)
        return 1
    if "download_ch" in phases and not ch_api_key and not args.dry_run:
        print("COMPANIES_HOUSE_API_KEY not set. Add it to .env.local.", file=sys.stderr)
        return 1
    if ("submit_batch" in phases or "collect_batch" in phases) and not mistral_api_key and not args.dry_run:
        print("MISTRAL_API_KEY not set. Add it to .env.local.", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "gap_manifest.csv"
    batch_state_path = output_dir / "batch_state.json"

    # Load source gap
    skipped_existing = 0
    if args.input_csv is not None:
        gap_items = load_gap_from_csv(
            args.input_csv,
            min_year=args.min_year,
            max_year=args.max_year,
            release_year=args.release_year,
        )
        if not args.overwrite:
            existing_pairs = load_existing_pairs()
            original_count = len(gap_items)
            gap_items = [
                item for item in gap_items
                if _pair_key(item["lei"], item["fiscal_year"]) not in existing_pairs
            ]
            skipped_existing = original_count - len(gap_items)
    else:
        gap_items = load_gap(
            args.manifest_raw, args.consolidated_metadata,
            min_year=args.min_year, max_year=args.max_year,
            release_year=args.release_year,
        )
    year_counts = Counter(item["fiscal_year"] for item in gap_items)
    source_label = f"{args.input_csv}" if args.input_csv is not None else f"{args.manifest_raw}"
    if args.release_year:
        print(f"Gap: {len(gap_items)} Annual Report items (release_year={args.release_year})")
    else:
        print(f"Gap: {len(gap_items)} Annual Report items (FY{args.min_year}–{args.max_year})")
    print(f"  Source: {source_label}")
    if skipped_existing:
        print(f"  Skipped already-have pairs: {skipped_existing}")
    print("  By fiscal year:", dict(sorted(year_counts.items())))

    # Merge into manifest (new items start as pending)
    manifest = load_manifest(manifest_path)
    pair_index = _build_manifest_pair_index(manifest)
    for item in gap_items:
        _ensure_manifest_row(manifest, pair_index, item)

    batch_jobs: list[dict[str, Any]] = load_batch_state(batch_state_path)

    # Status summary
    status_counts = Counter(r["status"] for r in manifest.values())
    print("  Current status:", dict(status_counts))

    if args.dry_run:
        print("\nDry run — no API calls.")
        for phase in phases:
            if phase == "probe_fr":
                phase_probe_fr(manifest, gap_items, fr_api_key=fr_api_key,
                               output_dir=output_dir, sleep_fr=args.sleep_fr,
                               overwrite=args.overwrite, limit=args.limit, dry_run=True)
            elif phase == "download_ch":
                phase_download_ch(manifest, ch_client=None, output_dir=output_dir,
                                  sleep_gleif=args.sleep_gleif, overwrite=args.overwrite,
                                  include_pending=args.include_pending,
                                  limit=args.limit, dry_run=True)
            elif phase == "submit_batch":
                phase_submit_batch(manifest, batch_jobs, mistral_api_key=mistral_api_key,
                                   output_dir=output_dir, max_batch_mb=args.max_batch_mb,
                                   max_batch_items=args.max_batch_items,
                                   batch_submit_delay=args.batch_submit_delay,
                                   overwrite=args.overwrite, limit=args.limit, dry_run=True)
            elif phase == "collect_batch":
                phase_collect_batch(manifest, batch_jobs, mistral_api_key=mistral_api_key,
                                    output_dir=output_dir, poll_interval=args.poll_interval,
                                    dry_run=True)
        return 0

    ch_client = CompaniesHouseClient(api_key=ch_api_key) if "download_ch" in phases else None

    try:
        for phase in phases:
            if phase == "probe_fr":
                phase_probe_fr(manifest, gap_items, fr_api_key=fr_api_key,
                               output_dir=output_dir, sleep_fr=args.sleep_fr,
                               overwrite=args.overwrite, limit=args.limit, dry_run=False)

            elif phase == "download_ch":
                phase_download_ch(manifest, ch_client=ch_client, output_dir=output_dir,
                                  sleep_gleif=args.sleep_gleif, overwrite=args.overwrite,
                                  include_pending=args.include_pending,
                                  limit=args.limit, dry_run=False)

            elif phase == "submit_batch":
                phase_submit_batch(manifest, batch_jobs, mistral_api_key=mistral_api_key,
                                   output_dir=output_dir, max_batch_mb=args.max_batch_mb,
                                   max_batch_items=args.max_batch_items,
                                   batch_submit_delay=args.batch_submit_delay,
                                   overwrite=args.overwrite, limit=args.limit, dry_run=False)

            elif phase == "collect_batch":
                phase_collect_batch(manifest, batch_jobs, mistral_api_key=mistral_api_key,
                                    output_dir=output_dir, poll_interval=args.poll_interval,
                                    dry_run=False)

    finally:
        # Always save state — even if interrupted by quota error or Ctrl-C
        save_manifest(manifest_path, manifest)
        save_batch_state(batch_state_path, batch_jobs)
        print(f"\nState saved → {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
