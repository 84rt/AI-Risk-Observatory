#!/usr/bin/env python3
"""Detect release-year collision gaps in the FR manifest.

A "collision" occurs when two filings for the same company land in the same
release_year bucket (one on-time, one late-filed from the prior fiscal year).
Dedup picks one winner, silently discarding the other.  The symptom is a gap
in the company's release_year sequence.

Algorithm
---------
For each company:
  1. Compute gaps  (release years in [min, max] that have no manifest entry).
  2. For each gap year G, inspect the adjacent manifest entries (G±1):
       a. candidates_count > 1  →  dedup DID discard something that year.
       b. lag anomaly  →  title_year deviates from the company's typical lag
          (lag = release_year − fiscal_year_inferred_from_title).
          lag ≥ 2 is a near-certain late-filer.
       c. fiscal-span mismatch  →  the adjacent winner already covers a
          fiscal year that "should" belong to the gap year.

Severity levels
---------------
  CONFIRMED  — candidates_count > 1  AND  lag ≥ 2  (old filing beat newer one)
  LIKELY     — candidates_count > 1  (something was discarded; unknown FY)
  POSSIBLE   — lag ≥ 2  (late filer kept, no competing candidate visible)
  PATTERN    — company's typical lag is 1 but adjacent entry has lag 0
               (may mean an early-released current-year report displaced a gap)

Outputs
-------
  data/FR_dataset/collision_flags.csv   — one row per flagged gap
  Prints a summary to stdout.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CSV = REPO_ROOT / "data" / "FR_dataset" / "manifest.csv"
OUTPUT_CSV   = REPO_ROOT / "data" / "FR_dataset" / "collision_flags.csv"


# ── Fiscal-year extraction ────────────────────────────────────────────────────

def extract_fiscal_year(title: str) -> int | None:
    """Return the fiscal year a filing covers, inferred from its title.

    Handles:
      - "Annual Report 2022"            → 2022
      - "Annual Report 2021/22"         → 2022  (slash-short)
      - "Annual Report 2021/2022"       → 2022  (slash-full)
      - "Annual Report 2021-22"         → 2022  (hyphen-short, UK common)
      - "Annual Report 2021-2022"       → 2022  (hyphen-full)
      - "Annual Report and Accounts"    → None  (no year in title)
    Returns None when no year can be inferred.
    """
    # YYYY/YY or YYYY-YY  (e.g. 2021/22 or 2021-22) → take the later year
    m = re.search(r'\b(20\d{2})[/-](2\d)\b', title)
    if m:
        return int(m.group(1)) + 1

    # YYYY/YYYY or YYYY-YYYY  (e.g. 2021/2022 or 2021-2022)
    m = re.search(r'\b(20\d{2})[/-](20\d{2})\b', title)
    if m:
        return max(int(m.group(1)), int(m.group(2)))

    # Bare 4-digit years in plausible range — take the maximum
    years = [int(y) for y in re.findall(r'\b(20[12]\d)\b', title)
             if 2015 <= int(y) <= 2030]
    return max(years) if years else None


# ── Main analysis ─────────────────────────────────────────────────────────────

def load_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def company_typical_lag(entries: list[dict]) -> int | None:
    """Mode lag for entries where the title contains a year."""
    lags = []
    for e in entries:
        fy = extract_fiscal_year(e["title"])
        if fy is not None:
            lags.append(int(e["release_year"]) - fy)
    if not lags:
        return None
    return Counter(lags).most_common(1)[0][0]


def analyse(manifest: list[dict]) -> list[dict]:
    # Index manifest by (lei, release_year)
    by_lei: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in manifest:
        lei = row["company__lei"]
        yr  = int(row["release_year"])
        by_lei[lei][yr] = row

    flags: list[dict] = []

    for lei, year_map in by_lei.items():
        years    = sorted(year_map)
        min_y    = years[0]
        max_y    = years[-1]
        gap_years = sorted(set(range(min_y, max_y + 1)) - set(years))

        if not gap_years:
            continue

        entries       = list(year_map.values())
        typical_lag   = company_typical_lag(entries)

        for gap in gap_years:
            # --- Check G+1 entry ---
            for direction, adj_yr in [("next", gap + 1), ("prev", gap - 1)]:
                entry = year_map.get(adj_yr)
                if entry is None:
                    continue

                cands      = int(entry.get("candidates_count", 1))
                fy         = extract_fiscal_year(entry["title"])
                rel_yr     = int(entry["release_year"])
                lag        = (rel_yr - fy) if fy is not None else None
                lag_str    = f"{lag:+d}" if lag is not None else "n/a"

                # ── Severity classification ──────────────────────────────────
                if lag is not None and lag >= 2 and cands > 1:
                    severity = "CONFIRMED"
                    reason   = (f"candidates_count={cands} + lag={lag_str}: "
                                f"old late-filer competed with newer filing; "
                                f"one FY was lost to dedup")
                elif cands > 1:
                    severity = "LIKELY"
                    reason   = (f"candidates_count={cands}: something was deduped "
                                f"from release_year={adj_yr}; may be FY{gap}")
                elif lag is not None and lag >= 2:
                    severity = "POSSIBLE"
                    reason   = (f"lag={lag_str}: winner is a late-filer covering "
                                f"FY{fy}, not FY{adj_yr - 1}")
                elif (typical_lag == 1 and lag is not None and lag == 0
                      and direction == "prev"):
                    # Company normally has lag=1 but this adjacent entry has lag=0,
                    # meaning a same-year report displaced an older one
                    severity = "PATTERN"
                    reason   = (f"company typical_lag=1 but this entry has lag=0 "
                                f"(FY=release_year); may have pushed FY{gap} out")
                else:
                    continue   # nothing actionable

                flags.append({
                    "company":             entry["company__name"],
                    "lei":                 lei,
                    "segment":             entry["market_segment"],
                    "gap_year":            gap,
                    "adjacent_direction":  direction,
                    "adjacent_rel_year":   adj_yr,
                    "adjacent_pk":         entry["pk"],
                    "adjacent_title":      entry["title"],
                    "title_fiscal_year":   fy if fy is not None else "",
                    "lag":                 lag_str,
                    "typical_lag":         typical_lag if typical_lag is not None else "",
                    "candidates_count":    cands,
                    "is_esef":             entry["is_esef"],
                    "severity":            severity,
                    "reason":              reason,
                })

    return flags


def write_flags(flags: list[dict], path: Path) -> None:
    if not flags:
        print("No collision flags found.")
        return
    fields = [
        "severity", "company", "lei", "segment",
        "gap_year", "adjacent_direction", "adjacent_rel_year",
        "adjacent_pk", "adjacent_title",
        "title_fiscal_year", "lag", "typical_lag",
        "candidates_count", "is_esef", "reason",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sorted(flags, key=lambda r: (r["severity"], r["company"], r["gap_year"])):
            w.writerow(row)
    print(f"Wrote {len(flags)} flags → {path.relative_to(REPO_ROOT)}")


def print_summary(flags: list[dict]) -> None:
    sev_counts = Counter(f["severity"] for f in flags)
    # Unique companies per severity
    sev_cos    = defaultdict(set)
    for f in flags:
        sev_cos[f["severity"]].add(f["lei"])

    print("\n=== Collision flag summary ===")
    for sev in ("CONFIRMED", "LIKELY", "POSSIBLE", "PATTERN"):
        n_flags = sev_counts.get(sev, 0)
        n_cos   = len(sev_cos.get(sev, set()))
        print(f"  {sev:<10}: {n_flags:3d} flags across {n_cos:3d} companies")

    total_cos = len({f["lei"] for f in flags})
    print(f"\n  Total flagged: {len(flags)} flags across {total_cos} companies\n")

    print("=== CONFIRMED cases (most actionable) ===")
    confirmed = [f for f in flags if f["severity"] == "CONFIRMED"]
    for f in sorted(confirmed, key=lambda x: (x["company"], x["gap_year"])):
        print(f"  [{f['gap_year']}] {f['company'][:45]:<45} "
              f"| adj={f['adjacent_rel_year']} lag={f['lag']} "
              f"cands={f['candidates_count']} | {f['adjacent_title'][:50]}")

    print("\n=== Segment breakdown of LIKELY+CONFIRMED ===")
    actionable = [f for f in flags if f["severity"] in ("CONFIRMED", "LIKELY")]
    seg_counts = Counter(f["segment"] for f in actionable)
    for seg, cnt in sorted(seg_counts.items(), key=lambda x: -x[1]):
        print(f"  {seg}: {cnt}")


def main() -> None:
    manifest = load_manifest(MANIFEST_CSV)
    print(f"Loaded {len(manifest)} manifest entries.")

    flags = analyse(manifest)
    print_summary(flags)
    write_flags(flags, OUTPUT_CSV)


if __name__ == "__main__":
    main()
