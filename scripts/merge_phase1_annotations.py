#!/usr/bin/env python3
"""Merge disjoint phase1 annotation shard files into one JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge phase1 annotation shard JSONL files into a single JSONL."
    )
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Phase1 annotation JSONL shard files to merge.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Merged phase1 annotations JSONL path.",
    )
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()

    merged_by_chunk: dict[str, dict[str, Any]] = {}
    ordered_rows: list[dict[str, Any]] = []

    for path in args.inputs:
        rows = load_jsonl(path)
        print(f"Loaded {len(rows)} rows from {path}")
        for row in rows:
            chunk_id = str(row.get("chunk_id") or "")
            if not chunk_id:
                raise SystemExit(f"Missing chunk_id in {path}")
            if chunk_id in merged_by_chunk:
                raise SystemExit(f"Duplicate chunk_id across inputs: {chunk_id}")
            merged_by_chunk[chunk_id] = row
            ordered_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in ordered_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Merged rows: {len(ordered_rows)}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
