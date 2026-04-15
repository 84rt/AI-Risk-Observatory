#!/usr/bin/env python3
"""
Merge substantiveness scores from a scores JSONL into the dashboard annotations.jsonl.

Matches on chunk_id. Only writes the specific score field(s) present in the scores
file — all other annotation fields are left untouched.

Usage:
    python scripts/merge_substantiveness_scores.py \
        --scores data/results/substantiveness/<run>/...substantiveness_scores.jsonl \
        [--fields vendor_substantiveness adoption_substantiveness risk_substantiveness] \
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
ANNOTATIONS = REPO_ROOT / "dashboard/data/annotations.jsonl"

VALID_FIELDS = {
    "adoption_substantiveness",
    "risk_substantiveness",
    "vendor_substantiveness",
    "risk_sub_substantiveness",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True,
                   help="Path to *.substantiveness_scores.jsonl")
    p.add_argument("--fields", nargs="+", default=None,
                   choices=sorted(VALID_FIELDS),
                   help="Which fields to merge (default: all non-null fields found in scores file)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change without writing anything")
    p.add_argument("--annotations", type=Path, default=ANNOTATIONS,
                   help=f"Path to annotations JSONL (default: {ANNOTATIONS})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scores_path = args.scores if args.scores.is_absolute() else REPO_ROOT / args.scores
    ann_path    = args.annotations

    if not scores_path.exists():
        raise SystemExit(f"Scores file not found: {scores_path}")
    if not ann_path.exists():
        raise SystemExit(f"Annotations file not found: {ann_path}")

    # Load scores — build lookup: chunk_id → {field: value, ...}
    scores_lookup: dict[str, dict] = {}
    with scores_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec.get("chunk_id")
            if not cid:
                continue
            scores_lookup[cid] = rec

    # Determine which fields to merge
    if args.fields:
        fields_to_merge = list(args.fields)
    else:
        # Auto-detect: all valid score fields that have at least one non-null value
        fields_to_merge = [
            f for f in VALID_FIELDS
            if any(rec.get(f) is not None for rec in scores_lookup.values())
        ]

    print(f"Scores file    : {scores_path.name}")
    print(f"Annotations    : {ann_path}")
    print(f"Score records  : {len(scores_lookup):,}")
    print(f"Fields to merge: {fields_to_merge}")
    print()

    # Load annotations
    annotations: list[dict] = []
    with ann_path.open() as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))

    print(f"Annotation records: {len(annotations):,}")

    # Pre-merge counts
    for field in fields_to_merge:
        before = sum(1 for r in annotations if r.get(field) is not None)
        print(f"  {field}: {before:,} filled before merge")
    print()

    # Merge
    updated = 0
    unmatched_scores = set(scores_lookup.keys())
    field_counts: dict[str, int] = {f: 0 for f in fields_to_merge}

    for rec in annotations:
        cid = rec.get("chunk_id")
        score_rec = scores_lookup.get(cid) if cid else None
        if not score_rec:
            continue
        unmatched_scores.discard(cid)
        changed = False
        for field in fields_to_merge:
            value = score_rec.get(field)
            if value is not None and rec.get(field) != value:
                rec[field] = value
                field_counts[field] += 1
                changed = True
        if changed:
            updated += 1

    print(f"Records updated  : {updated:,}")
    for field, count in field_counts.items():
        print(f"  {field}: {count:,} values written")
    if unmatched_scores:
        print(f"  Score chunk_ids with no match in annotations: {len(unmatched_scores):,}")

    if args.dry_run:
        print("\nDry run — no files written.")
        return

    # Backup
    backup_path = ann_path.with_suffix(".jsonl.bak")
    shutil.copy2(ann_path, backup_path)
    print(f"\nBackup : {backup_path}")

    # Write updated annotations
    with ann_path.open("w") as f:
        for rec in annotations:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Written: {ann_path}")

    # Verify
    post: list[dict] = []
    with ann_path.open() as f:
        for line in f:
            if line.strip():
                post.append(json.loads(line))
    print(f"\nVerification ({len(post):,} records):")
    for field in fields_to_merge:
        after = sum(1 for r in post if r.get(field) is not None)
        print(f"  {field}: {after:,} filled")


if __name__ == "__main__":
    main()
