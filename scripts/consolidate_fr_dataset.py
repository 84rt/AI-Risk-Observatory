#!/usr/bin/env python3
"""Consolidate all FR_dataset markdown files into data/FR_dataset/markdown/.

Reads the manifest and phase3 checkpoint, then hardlinks (or copies) every
available markdown file from its source directory into FR_dataset/markdown/,
so the dataset has one single folder with all files.

Already-present files in FR_dataset/markdown/ are left untouched.
"""

import json
import os
import shutil
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA      = REPO_ROOT / "data"

OUT_MD    = DATA / "FR_dataset" / "markdown"

CACHE_DIRS = [
    DATA / "FR_dataset"              / "markdown",   # already-fetched new files
    DATA / "FR_clean"                / "markdown",
    DATA / "FR-UK-2021-2023-test-2"  / "markdown",
    DATA / "FR_2026-02-05"           / "markdown",
]

MANIFEST_PATH    = DATA / "FR_dataset" / "manifest.json"
CHECKPOINT_PATH  = DATA / "FR_dataset" / "phase3_checkpoint.json"

AVAILABLE = {"cached", "fetched"}


def find_source(pk: str) -> Path | None:
    for d in CACHE_DIRS:
        p = d / f"{pk}.md"
        if p.exists():
            return p
    return None


def main() -> None:
    OUT_MD.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    with open(CHECKPOINT_PATH) as f:
        checkpoint = json.load(f)

    status_counter = Counter()
    linked = skipped = missing = 0

    for row in manifest:
        pk = row["pk"]
        md_status = checkpoint.get(pk, {}).get("status", "unknown")

        if md_status not in AVAILABLE:
            status_counter[md_status] += 1
            continue

        dst = OUT_MD / f"{pk}.md"
        if dst.exists():
            skipped += 1
            continue

        src = find_source(pk)
        if src is None:
            print(f"  WARNING: {pk} marked as {md_status} but file not found in any cache dir")
            missing += 1
            continue

        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

        linked += 1

    total = linked + skipped
    print(f"Linked (new):   {linked:,}")
    print(f"Already present:{skipped:,}")
    print(f"Total in folder:{total:,}")
    print(f"File not found: {missing}")
    print(f"\nNot available (skipped from manifest):")
    for status, count in sorted(status_counter.items(), key=lambda x: -x[1]):
        print(f"  {status}: {count}")

    # Final disk usage
    total_size = sum(f.stat().st_size for f in OUT_MD.iterdir() if f.suffix == ".md")
    print(f"\ndata/FR_dataset/markdown/: {total:,} files, {total_size/1e9:.2f} GB (logical size)")


if __name__ == "__main__":
    main()
