#!/usr/bin/env python3
"""Merge reconciled annotations with the original human golden set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge reconciled annotations into the human golden set."
    )
    parser.add_argument(
        "--human",
        type=Path,
        default=Path("data/golden_set/human/annotations.jsonl"),
        help="Path to original human annotations.jsonl",
    )
    parser.add_argument(
        "--reconciled",
        type=Path,
        default=Path("data/golden_set/reconciled/annotations.jsonl"),
        help="Path to reconciled annotations.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/golden_set/human_reconciled/annotations.jsonl"),
        help="Output path for merged annotations.jsonl",
    )
    parser.add_argument(
        "--include-extra",
        action="store_true",
        help="Include reconciled chunks not found in the human file.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def index_by_chunk_id(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for record in records:
        cid = record.get("chunk_id")
        if cid:
            out[cid] = record
    return out


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()

    human_records = load_jsonl(args.human)
    reconciled_records = load_jsonl(args.reconciled)

    reconciled_by_id = index_by_chunk_id(reconciled_records)
    reconciled_ids = set(reconciled_by_id)

    merged: List[Dict[str, Any]] = []
    used = set()
    for record in human_records:
        cid = record.get("chunk_id")
        if cid and cid in reconciled_by_id:
            merged.append(reconciled_by_id[cid])
            used.add(cid)
        else:
            merged.append(record)

    extra = [rec for rec in reconciled_records if rec.get("chunk_id") not in used]
    if extra and args.include_extra:
        merged.extend(extra)

    write_jsonl(args.output, merged)

    print(f"Human records: {len(human_records)}")
    print(f"Reconciled records: {len(reconciled_records)}")
    print(f"Reconciled used: {len(used)}")
    if extra:
        print(f"Reconciled extras: {len(extra)} (included={args.include_extra})")
    print(f"Output records: {len(merged)}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
