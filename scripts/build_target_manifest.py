"""Build the AIRO target manifest.

Every row is exactly one (company, fiscal_year) pair — no nulls, no stubs.
Companies whose fiscal year cannot be reliably determined are written to a
separate investigation file instead of polluting the manifest.

Fiscal year sourcing — strict hierarchy, no guessing:
  1. fr_manifest       — FR manifest fiscal_year (extracted from filing title).
  2. ch_made_up_date   — period-end date from CH filing history (requires
                         ch_period_of_accounts.csv from fetch_ch_period_of_accounts.py).
  3. interpolated_fr_gap — years between a company's first and last FR fiscal year
                         with no FR entry. Safe: bounded by confirmed FR data.

  Companies with none of the above go to companies_no_fiscal_year.csv.
  Fiscal year is NEVER inferred from CH submission dates.

fr_status values:
  md_available    FR has it, markdown cached/fetched
  fr_pending      FR has it, not yet processed
  fr_failed       FR processing failed
  fr_skipped      FR skipped this filing
  fr_no_status    In FR manifest but no processing record
  not_in_fr       No FR manifest entry for this (company, fiscal_year)

Outputs:
  data/reference/target_manifest.csv         — uniform (company, fiscal_year) rows
  data/reference/companies_no_fiscal_year.csv — companies needing manual investigation
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

TARGET_YEARS = {2021, 2022, 2023, 2024, 2025, 2026}
CH_POA_CSV   = DATA / "FR_dataset/ch_period_of_accounts.csv"


def to_fr_status(md_status: str) -> str:
    return {
        "cached":     "md_available",
        "fetched":    "md_available",
        "PENDING":    "fr_pending",
        "PROCESSING": "fr_pending",
        "QUEUED":     "fr_pending",
        "FAILED":     "fr_failed",
        "SKIPPED":    "fr_skipped",
    }.get(md_status, f"fr_{md_status.lower()}" if md_status else "fr_no_status")


# ── Load CH period-of-accounts (made_up_date) if available ───────────────────
ch_poa_years:   dict[str, list[int]] = defaultdict(list)
ch_poa_details: dict[tuple, dict]    = {}  # (lei, fiscal_year) → row

if CH_POA_CSV.exists():
    print("Loading ch_period_of_accounts (made_up_date)...")
    with open(CH_POA_CSV) as f:
        for row in csv.DictReader(f):
            try:
                fy = int(row["fiscal_year"])
            except (ValueError, TypeError):
                continue
            if fy not in TARGET_YEARS:
                continue
            lei = row["lei"]
            ch_poa_years[lei].append(fy)
            ch_poa_details[(lei, fy)] = row
    for lei in ch_poa_years:
        ch_poa_years[lei] = sorted(set(ch_poa_years[lei]))
    total_poa_pairs = sum(len(v) for v in ch_poa_years.values())
    print(f"  {len(ch_poa_years)} companies, {total_poa_pairs} (company, fiscal_year) pairs")
else:
    print("ch_period_of_accounts.csv not found.")
    print("  Run fetch_ch_period_of_accounts.py to fill fiscal years for non-FR companies.")

# ── Load CH coverage (all companies: UK-incorporated + foreign LSE-listed) ────
print("Loading ch_coverage...")
ch: dict[str, dict] = {}
with open(DATA / "FR_dataset/ch_coverage.csv") as f:
    for row in csv.DictReader(f):
        ch[row["lei"]] = row
print(f"  {len(ch)} companies (UK-incorporated + foreign LSE-listed)")

# ── Load processing status keyed by pk ────────────────────────────────────────
print("Loading processing_status...")
ps_by_pk: dict[str, dict] = {}
with open(DATA / "FR_dataset/processing_status.csv") as f:
    for row in csv.DictReader(f):
        ps_by_pk[row["pk"]] = row
print(f"  {len(ps_by_pk)} processing status rows")

# ── Load FR manifest, indexed by LEI ─────────────────────────────────────────
print("Loading manifest...")
manifest_by_lei: dict[str, list] = defaultdict(list)
total_manifest = 0
with open(DATA / "FR_dataset/manifest.csv") as f:
    for row in csv.DictReader(f):
        try:
            fy = int(row.get("fiscal_year", ""))
        except (ValueError, TypeError):
            continue
        if fy not in TARGET_YEARS:
            continue
        manifest_by_lei[row["company__lei"]].append(row)
        total_manifest += 1
print(f"  {total_manifest} rows in 2021-2025 (across {len(manifest_by_lei)} companies)")

# ── Build manifest rows ───────────────────────────────────────────────────────
print("Building target manifest...")

MANIFEST_FIELDS = [
    "lei",
    "company_name",
    "ch_company_number",
    "market_segment",
    "ch_jurisdiction",
    "fiscal_year",           # always a valid integer string, never blank
    "fiscal_year_source",    # fr_manifest | ch_made_up_date | interpolated_fr_gap
    "fr_pk",
    "fr_filing_type",
    "fr_is_esef",
    "fr_status",
    "md_size",
    "ch_made_up_date",       # from CH, blank if not fetched or not matched
    "ch_submission_date",    # from CH, blank if not fetched or not matched
]

NO_FY_FIELDS = [
    "lei",
    "company_name",
    "ch_company_number",
    "market_segment",
    "ch_jurisdiction",
    "ch_submission_years",   # pipe-separated CH submission years, for manual investigation
]

manifest_rows: list[dict] = []
no_fy_rows:    list[dict] = []


def make_row(lei, name, ch_num, segment, jurisdiction, fiscal_year, fy_source,
             fr_pk="", fr_type="", fr_esef="", fr_status="not_in_fr",
             md_size="", ch_mud="", ch_sub="") -> dict:
    return {
        "lei":               lei,
        "company_name":      name,
        "ch_company_number": ch_num,
        "market_segment":    segment,
        "ch_jurisdiction":   jurisdiction,
        "fiscal_year":       str(fiscal_year),
        "fiscal_year_source": fy_source,
        "fr_pk":             fr_pk,
        "fr_filing_type":    fr_type,
        "fr_is_esef":        fr_esef,
        "fr_status":         fr_status,
        "md_size":           md_size,
        "ch_made_up_date":   ch_mud,
        "ch_submission_date": ch_sub,
    }


for lei, ch_row in ch.items():
    name       = ch_row["name"]
    ch_num     = ch_row["company_number"]
    segment    = ch_row["market_segment"]
    juris      = ch_row.get("jurisdiction", "")
    sub_years  = ch_row.get("filing_years", "")

    fr_entries = manifest_by_lei.get(lei, [])

    # ── Case 1: company has FR manifest entries ───────────────────────────────
    if fr_entries:
        fr_fiscal_years = set()
        for entry in fr_entries:
            try:
                fr_fiscal_years.add(int(entry["fiscal_year"]))
            except (ValueError, TypeError):
                pass

        for entry in fr_entries:
            pk        = entry["pk"]
            ps_row    = ps_by_pk.get(pk, {})
            fr_status = to_fr_status(ps_row.get("md_status", ""))
            try:
                fy_int = int(entry["fiscal_year"])
            except (ValueError, TypeError):
                fy_int = None
            poa = ch_poa_details.get((lei, fy_int), {}) if fy_int else {}

            manifest_rows.append(make_row(
                lei, entry["company__name"], ch_num,
                entry.get("market_segment") or segment, juris,
                entry["fiscal_year"], "fr_manifest",
                fr_pk=pk,
                fr_type=entry.get("filing_type__code", ""),
                fr_esef=entry.get("is_esef", ""),
                fr_status=fr_status,
                md_size=ps_row.get("md_size", ""),
                ch_mud=poa.get("made_up_date", ""),
                ch_sub=poa.get("submission_date", ""),
            ))

        # Gap years: between first and last FR fiscal year, safe to interpolate
        if fr_fiscal_years:
            for gap_year in range(min(fr_fiscal_years), max(fr_fiscal_years) + 1):
                if gap_year in fr_fiscal_years or gap_year not in TARGET_YEARS:
                    continue
                poa = ch_poa_details.get((lei, gap_year), {})
                manifest_rows.append(make_row(
                    lei, name, ch_num, segment, juris,
                    gap_year, "interpolated_fr_gap",
                    ch_mud=poa.get("made_up_date", ""),
                    ch_sub=poa.get("submission_date", ""),
                ))

    # ── Case 2: no FR entries, but CH made_up_date available ─────────────────
    elif ch_poa_years.get(lei):
        for fy in ch_poa_years[lei]:
            poa = ch_poa_details.get((lei, fy), {})
            manifest_rows.append(make_row(
                lei, name, ch_num, segment, juris,
                fy, "ch_made_up_date",
                ch_mud=poa.get("made_up_date", ""),
                ch_sub=poa.get("submission_date", ""),
            ))

    # ── Case 3: no fiscal year data at all → investigation queue ─────────────
    else:
        no_fy_rows.append({
            "lei":               lei,
            "company_name":      name,
            "ch_company_number": ch_num,
            "market_segment":    segment,
            "ch_jurisdiction":   juris,
            "ch_submission_years": sub_years,
        })


# ── Sort and write manifest ───────────────────────────────────────────────────
manifest_rows.sort(key=lambda r: (r["market_segment"], r["company_name"].lower(), r["fiscal_year"]))

out_manifest = DATA / "reference/target_manifest.csv"
out_no_fy    = DATA / "reference/companies_no_fiscal_year.csv"
out_manifest.parent.mkdir(parents=True, exist_ok=True)

with open(out_manifest, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
    w.writeheader()
    w.writerows(manifest_rows)

with open(out_no_fy, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=NO_FY_FIELDS)
    w.writeheader()
    w.writerows(no_fy_rows)

# ── Summary ───────────────────────────────────────────────────────────────────
status_counts  = Counter(r["fr_status"]         for r in manifest_rows)
year_counts    = Counter(r["fiscal_year"]        for r in manifest_rows)
segment_counts = Counter(r["market_segment"]     for r in manifest_rows)
source_counts  = Counter(r["fiscal_year_source"] for r in manifest_rows)

print(f"\nWrote {len(manifest_rows):,} rows  → {out_manifest.relative_to(BASE)}")
print(f"Wrote {len(no_fy_rows):,} companies → {out_no_fy.relative_to(BASE)}")

print("\n── fr_status ─────────────────────────────────────────────────")
for s, n in sorted(status_counts.items(), key=lambda x: -x[1]):
    print(f"  {s:<25s}  {n:5,}  ({100*n/len(manifest_rows):.1f}%)")

print("\n── fiscal_year ───────────────────────────────────────────────")
for yr, n in sorted(year_counts.items()):
    print(f"  {yr}  {n:5,}")

print("\n── fiscal_year_source ────────────────────────────────────────")
for src, n in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"  {src:<30s}  {n:5,}")

print("\n── market_segment ────────────────────────────────────────────")
for seg, n in sorted(segment_counts.items(), key=lambda x: -x[1]):
    print(f"  {seg:<20s}  {n:5,}")

print(f"\n── Company coverage ──────────────────────────────────────────")
print(f"  In scope (UK, LSE):       {len(ch):,}")
print(f"  In manifest (known FY):   {len({r['lei'] for r in manifest_rows}):,}")
print(f"  No FY data (queue):       {len(no_fy_rows):,}")
