"""
Rebuild FR_clean from two source datasets.

Sources:
  - FR-UK-2021-2023-test-2  (release years 2021-2023)
  - FR_2026-02-05            (release years 2023-2026)

Rules:
  1. Union by PK — 2023 overlap resolved by unique PKs only.
  2. Group by (company_lei, release_year).
  3. Per group: keep the largest ESEF file if any ESEF exists,
     else keep the largest regular file.
  4. Write metadata.csv, hardlink markdown files, write summary.json.
"""

import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path("/Users/84rt/Projects/AI Risk Observatory/data")
SOURCES = [
    BASE / "FR-UK-2021-2023-test-2",
    BASE / "FR_2026-02-05",
]
OUT     = BASE / "FR_clean"
OUT_MD  = OUT / "markdown"

# ── Load & union all rows by PK ───────────────────────────────────────────────
all_rows: dict[str, dict] = {}          # pk -> row
pk_to_md_path: dict[str, Path] = {}    # pk -> markdown file path

for src in SOURCES:
    md_dir = src / "markdown"
    with open(src / "metadata.csv") as f:
        for row in csv.DictReader(f):
            pk = row["pk"]
            if pk not in all_rows:
                all_rows[pk] = row
                row["_source"] = str(src)
            # Track markdown location (prefer first source that has it)
            md_file = md_dir / f"{pk}.md"
            if pk not in pk_to_md_path and md_file.exists():
                pk_to_md_path[pk] = md_file

print(f"Unique filings after union : {len(all_rows):,}")
print(f"Markdown files found       : {len(pk_to_md_path):,}")

# ── Attach markdown size to each row ─────────────────────────────────────────
for pk, row in all_rows.items():
    md = pk_to_md_path.get(pk)
    row["_md_size"] = md.stat().st_size if md else 0
    row["_has_md"]  = md is not None
    row["_is_esef"] = "ESEF" in row["filing_type__name"]
    row["_rel_yr"]  = row.get("release_datetime", "")[:4]
    row["_lei"]     = row.get("company__lei") or row.get("company__name", "")

# ── Group by (lei, release_year) and select one filing per group ──────────────
groups: dict[tuple, list] = defaultdict(list)
for row in all_rows.values():
    groups[(row["_lei"], row["_rel_yr"])].append(row)

kept_rows = []
dropped_rows = []

for (lei, yr), candidates in groups.items():
    # Separate ESEF and regular
    esef    = [r for r in candidates if r["_is_esef"]]
    regular = [r for r in candidates if not r["_is_esef"]]

    if esef:
        # Keep the largest ESEF file (by markdown size)
        winner = max(esef, key=lambda r: r["_md_size"])
        losers = [r for r in candidates if r["pk"] != winner["pk"]]
    else:
        # Keep the largest regular file
        winner = max(regular, key=lambda r: r["_md_size"])
        losers = [r for r in candidates if r["pk"] != winner["pk"]]

    kept_rows.append(winner)
    dropped_rows.extend(losers)

print(f"Kept after dedup           : {len(kept_rows):,}")
print(f"Dropped                    : {len(dropped_rows):,}")

# ── Clean and recreate output directory ──────────────────────────────────────
if OUT_MD.exists():
    shutil.rmtree(OUT_MD)
OUT_MD.mkdir(parents=True, exist_ok=True)

# ── Write metadata.csv ────────────────────────────────────────────────────────
# Collect all original columns across both sources (drop private _* fields),
# preserving a stable order (first-seen wins for ordering).
seen_fields: dict[str, None] = {}
for row in all_rows.values():
    for k in row:
        if not k.startswith("_"):
            seen_fields[k] = None
orig_fields = list(seen_fields)

with open(OUT / "metadata.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=orig_fields, extrasaction="ignore")
    writer.writeheader()
    for row in sorted(kept_rows, key=lambda r: r["pk"]):
        writer.writerow({k: row.get(k, "") for k in orig_fields})

# ── Hardlink (or copy) markdown files ────────────────────────────────────────
linked = 0
missing = 0
for row in kept_rows:
    src_md = pk_to_md_path.get(row["pk"])
    if src_md:
        dst = OUT_MD / src_md.name
        try:
            os.link(src_md, dst)
        except OSError:
            shutil.copy2(src_md, dst)
        linked += 1
    else:
        missing += 1

print(f"Markdown hardlinked/copied : {linked:,}")
print(f"Markdown missing           : {missing:,}")

# ── Write dropped_candidates.csv ─────────────────────────────────────────────
drop_fields = orig_fields + ["_rel_yr", "_is_esef", "_md_size", "_source", "_md_size"]
drop_fields = list(dict.fromkeys(drop_fields))  # deduplicate preserving order
drop_fields = orig_fields + [f for f in ["_rel_yr", "_is_esef", "_md_size", "_source"] if f not in orig_fields]
with open(OUT / "dropped_candidates.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=drop_fields)
    writer.writeheader()
    for row in sorted(dropped_rows, key=lambda r: r["pk"]):
        writer.writerow({k: row.get(k, "") for k in drop_fields})

# ── Write summary.json ────────────────────────────────────────────────────────
from collections import Counter

yr_counts    = Counter(r["_rel_yr"] for r in kept_rows)
type_counts  = Counter(r["filing_type__name"] for r in kept_rows)
companies    = len(set(r["_lei"] for r in kept_rows))

summary = {
    "sources": [str(s) for s in SOURCES],
    "output": str(OUT),
    "total_source_filings": len(all_rows),
    "kept_filings": len(kept_rows),
    "dropped_filings": len(dropped_rows),
    "markdown_linked": linked,
    "markdown_missing": missing,
    "unique_companies": companies,
    "filings_per_release_year": dict(sorted(yr_counts.items())),
    "filing_types": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
    "dedup_rules": {
        "overlap": "unique PK",
        "esef_vs_regular": "prefer ESEF",
        "multi_same_type": "keep largest markdown file",
    },
}

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print()
print("=== Summary ===")
print(json.dumps(summary, indent=2))
