#!/usr/bin/env python3
"""
Patch the 24k base annotations with updated vendor (v2) and harm (phase1 v2),
then write to dashboard/data/annotations.jsonl ready for build-dashboard-data.mjs.

Patches applied:
  1. vendor_tags, vendor_other, vendor_confidence  <- p2-vendor-...-uk-24k-v2.jsonl
  2. mention_types                                 <- phase1 v2 annotations (harm fix)
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE        = REPO_ROOT / "data/results/uk_annual_reports_24k/annotations.jsonl"
VENDOR_V2   = REPO_ROOT / "data/testbed_runs/p2-vendor-gemini-3-flash-preview-uk-24k-v2.jsonl"
PHASE1_V2   = REPO_ROOT / "data/results/uk_annual_reports_24k_v2/p1-mention_type-gemini-3-flash-preview-uk-24k-v2.phase1_annotations.jsonl"
DEST        = REPO_ROOT / "dashboard/data/annotations.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_by_chunk_id(rows: list[dict]) -> dict[str, dict]:
    return {str(r["chunk_id"]): r for r in rows if r.get("chunk_id")}


def main() -> None:
    print(f"Loading base annotations ({BASE.name})...")
    base = load_jsonl(BASE)
    print(f"  {len(base):,} chunks")

    print(f"Loading vendor v2 ({VENDOR_V2.name})...")
    vendor_idx = index_by_chunk_id(load_jsonl(VENDOR_V2))
    print(f"  {len(vendor_idx):,} vendor results")

    print(f"Loading phase1 v2 ({PHASE1_V2.name})...")
    p1v2_idx = index_by_chunk_id(load_jsonl(PHASE1_V2))
    print(f"  {len(p1v2_idx):,} phase1 v2 results")

    vendor_patched = 0
    harm_changed = 0
    output = []

    for rec in base:
        chunk_id = str(rec.get("chunk_id", ""))
        rec = dict(rec)

        # --- Patch 1: vendor ---
        v = vendor_idx.get(chunk_id)
        if v is not None:
            rec["vendor_tags"]       = v.get("llm_labels", [])
            rec["vendor_other"]      = v.get("llm_other")
            rec["vendor_confidence"] = v.get("vendor_signals", {})
            vendor_patched += 1

        # --- Patch 2: mention_types — harm label only ---
        # We only propagate the harm label from v2. All other labels (adoption,
        # risk, vendor, general_ambiguous, none) stay as they were in v1 so we
        # don't inadvertently import unrelated prompt-drift differences.
        p1 = p1v2_idx.get(chunk_id)
        if p1 is not None:
            old_types = set(rec.get("mention_types") or [])
            v2_has_harm = "harm" in set(p1.get("mention_types") or [])
            v1_has_harm = "harm" in old_types

            if v2_has_harm != v1_has_harm:
                new_types = (old_types | {"harm"}) if v2_has_harm else (old_types - {"harm"})
                rec["mention_types"] = sorted(new_types)
                harm_changed += 1

        output.append(rec)

    print(f"\nPatched:")
    print(f"  vendor:       {vendor_patched:,} chunks updated")
    print(f"  mention_types changed: {harm_changed} chunks")

    DEST.parent.mkdir(parents=True, exist_ok=True)
    with DEST.open("w") as f:
        for rec in output:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(output):,} chunks -> {DEST}")


if __name__ == "__main__":
    main()
