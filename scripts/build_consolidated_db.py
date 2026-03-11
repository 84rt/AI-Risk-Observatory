#!/usr/bin/env python3
"""Build a single consolidated directory of all in-scope FR markdown files.

Consolidates markdown from multiple download directories into one canonical location:
  data/FR_consolidated/
    metadata.csv       — one row per PK kept
    {pk}.md            — the markdown file (hardlinked from source, not copied)

Selection logic:
  - Only COMPLETED filings (cached/fetched in processing_status.csv, or COMPLETED in batch metadata)
  - Only companies in the universe (ch_coverage.csv)
  - Only annual report types (AR, 10-K, 10-K-ESEF, 10-K-AFS, Annual Report, Annual Report (ESEF))
  - Only publication years 2021–2025
  - One file per (lei, pub_year) slot — picks best PK when multiple exist
    Priority: 10-K-ESEF > 10-K > AR > 10-K-AFS > others
    Tiebreak: source directory priority order

Usage:
  python scripts/build_consolidated_db.py
  python scripts/build_consolidated_db.py --copy    # copy instead of hardlink
  python scripts/build_consolidated_db.py --dry-run # preview without writing
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / "data"
OUT_DIR   = DATA_DIR / "FR_consolidated"

TARGET_YEARS  = {"2021", "2022", "2023", "2024", "2025"}
ANNUAL_TYPES  = {"AR", "10-K", "10-K-ESEF", "10-K-AFS",
                 "Annual Report", "Annual Report (ESEF)"}

# Normalise batch filing type names to manifest codes
TYPE_NORM = {
    "Annual Report": "AR",
    "Annual Report (ESEF)": "10-K-ESEF",
}

# Higher index = lower preference
TYPE_RANK = {"10-K-ESEF": 0, "10-K": 1, "AR": 2, "10-K-AFS": 3}

# Source directory priority — first = preferred when content is identical
SOURCE_PRIORITY = [
    "FR_clean",
    "FR_2026-02-05",
    "FinancialReports_downloaded",
    "FR-2021-to-2023",
    "full annual 2024-2026 batch test 7",
    "full annual 2021-2023 batch test 6",
    "FR_clean_from_frd",
    "FR-UK-2021-2023-test-2",
    "FR_batch_test_2021",
    "FR_frasers-group-plc-filings",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_proc_status() -> dict[str, str]:
    """Map pk -> normalised status ('COMPLETED' | 'PENDING' | ...)."""
    status: dict[str, str] = {}

    ps_path = DATA_DIR / "FR_dataset" / "processing_status.csv"
    with open(ps_path) as f:
        for r in csv.DictReader(f):
            s = r.get("md_status", "")
            status[r["pk"]] = "COMPLETED" if s in ("cached", "fetched") else s

    for batch in ["full annual 2021-2023 batch test 6",
                  "full annual 2024-2026 batch test 7"]:
        meta = DATA_DIR / batch / "metadata.csv"
        if not meta.exists():
            continue
        with open(meta) as f:
            for r in csv.DictReader(f):
                pk = r.get("pk", "")
                if pk and pk not in status:
                    s = r.get("processing_status", "")
                    status[pk] = "COMPLETED" if s == "COMPLETED" else s

    return status


def load_universe() -> tuple[set[str], dict[str, str], dict[str, str]]:
    """Return (universe_leis, lei→segment, lei→name)."""
    leis, seg, name = set(), {}, {}
    with open(DATA_DIR / "FR_dataset" / "ch_coverage.csv") as f:
        for r in csv.DictReader(f):
            lei = r.get("lei", "")
            if lei:
                leis.add(lei)
                seg[lei]  = r.get("market_segment", "")
                name[lei] = r.get("name", "")
    return leis, seg, name


def load_metadata() -> dict[str, dict]:
    """Build pk → metadata dict from manifest + batch files."""
    meta: dict[str, dict] = {}

    with open(DATA_DIR / "FR_dataset" / "manifest.csv") as f:
        for r in csv.DictReader(f):
            meta[r["pk"]] = {
                "pk":               r["pk"],
                "lei":              r.get("company__lei", ""),
                "company":          r.get("company__name", ""),
                "market_segment":   r.get("market_segment", ""),
                "fiscal_year":      r.get("fiscal_year", ""),
                "release_year":     r.get("release_year", ""),
                "release_datetime": r.get("release_datetime", ""),
                "title":            r.get("title", ""),
                "filing_type":      r.get("filing_type__code", ""),
                "meta_source":      "manifest",
            }

    for batch in ["full annual 2021-2023 batch test 6",
                  "full annual 2024-2026 batch test 7"]:
        path = DATA_DIR / batch / "metadata.csv"
        if not path.exists():
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                pk = r.get("pk", "")
                if pk and pk not in meta:
                    lei = r.get("company__lei", "")
                    meta[pk] = {
                        "pk":               pk,
                        "lei":              lei,
                        "company":          r.get("company__name", ""),
                        "market_segment":   "",  # filled later from universe
                        "fiscal_year":      "",
                        "release_year":     r.get("release_datetime", "")[:4],
                        "release_datetime": r.get("release_datetime", ""),
                        "title":            r.get("title", ""),
                        "filing_type":      TYPE_NORM.get(
                                                r.get("filing_type__name", ""),
                                                r.get("filing_type__name", "")),
                        "meta_source":      batch,
                    }
    return meta


def scan_local_files() -> dict[str, Path]:
    """Return pk → canonical local path (first match by SOURCE_PRIORITY)."""
    pk_path: dict[str, Path] = {}
    for src in SOURCE_PRIORITY:
        p = DATA_DIR / src
        if not p.exists():
            continue
        for md in p.glob("**/*.md"):
            pk = md.stem
            if pk not in pk_path:
                pk_path[pk] = md
    return pk_path


def type_rank(filing_type: str) -> int:
    return TYPE_RANK.get(filing_type, 99)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dry_run  = "--dry-run" in sys.argv
    use_copy = "--copy"    in sys.argv

    print("Loading data...")
    proc_status               = load_proc_status()
    universe_leis, seg, uname = load_universe()
    all_meta                  = load_metadata()
    pk_path                   = scan_local_files()

    # ── Filter to eligible PKs ────────────────────────────────────────────────
    eligible: list[dict] = []
    for pk, src_path in pk_path.items():
        if proc_status.get(pk, "") != "COMPLETED":
            continue
        m = all_meta.get(pk)
        if not m:
            continue
        lei = m["lei"]
        yr  = m["release_year"]
        ft  = m["filing_type"]
        if lei not in universe_leis:
            continue
        if yr not in TARGET_YEARS:
            continue
        if ft not in ANNUAL_TYPES:
            continue
        # Fill segment from universe if missing
        if not m["market_segment"]:
            m["market_segment"] = seg.get(lei, "")
        eligible.append({**m, "src_path": src_path})

    print(f"Eligible PKs: {len(eligible)}")

    # ── One file per (lei, pub_year) slot ─────────────────────────────────────
    slot_candidates: dict[tuple, list] = defaultdict(list)
    for rec in eligible:
        slot = (rec["lei"], rec["release_year"])
        slot_candidates[slot].append(rec)

    # Pick best candidate per slot
    def rank_candidate(rec: dict) -> tuple:
        return (type_rank(rec["filing_type"]),
                SOURCE_PRIORITY.index(rec["src_path"].parts[
                    len(DATA_DIR.parts)]) if any(
                    src in str(rec["src_path"]) for src in SOURCE_PRIORITY) else 99)

    selected: list[dict] = []
    for slot, candidates in slot_candidates.items():
        best = min(candidates, key=rank_candidate)
        selected.append(best)

    selected.sort(key=lambda r: (r["release_year"], r["market_segment"], r["company"]))

    print(f"Unique (lei, pub_year) slots: {len(selected)}")
    print(f"  FTSE 350: {sum(1 for r in selected if r['market_segment']=='FTSE 350')}")
    print(f"  AIM:      {sum(1 for r in selected if r['market_segment']=='AIM')}")
    print(f"  Other:    {sum(1 for r in selected if r['market_segment']=='Other')}")

    if dry_run:
        print("\nDry run — no files written.")
        return

    # ── Write output ──────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written = skipped = failed = 0
    for rec in selected:
        src  = rec["src_path"]
        dest = OUT_DIR / f"{rec['pk']}.md"

        if dest.exists():
            skipped += 1
            continue

        try:
            if use_copy:
                import shutil
                shutil.copy2(src, dest)
            else:
                os.link(src, dest)
            written += 1
        except OSError:
            # Cross-device hardlink fails — fall back to copy
            try:
                import shutil
                shutil.copy2(src, dest)
                written += 1
            except Exception as e:
                print(f"  FAILED {rec['pk']}: {e}")
                failed += 1

    # ── Write metadata.csv ────────────────────────────────────────────────────
    meta_path = OUT_DIR / "metadata.csv"
    fields = ["pk", "lei", "company", "market_segment", "release_year",
              "release_datetime", "fiscal_year", "filing_type", "title",
              "meta_source", "src_path"]

    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in selected:
            writer.writerow({**rec, "src_path": str(rec["src_path"])})

    print(f"\nDone.")
    print(f"  Written:  {written}")
    print(f"  Skipped (already exist): {skipped}")
    print(f"  Failed:   {failed}")
    print(f"  metadata.csv: {meta_path}")
    print(f"  Output dir:   {OUT_DIR}")


if __name__ == "__main__":
    main()
