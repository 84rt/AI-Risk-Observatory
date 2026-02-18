#!/usr/bin/env python3
"""
Merge phase1 retry annotations into a base phase1 annotations file.

Only rows that improve from error -> no error are replaced by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge retry phase1 annotations into a base phase1 annotations JSONL."
    )
    p.add_argument("--base", type=Path, required=True, help="Base phase1 annotations JSONL.")
    p.add_argument("--retry", type=Path, required=True, help="Retry phase1 annotations JSONL.")
    p.add_argument("--output", type=Path, required=True, help="Merged output JSONL.")
    p.add_argument(
        "--replace-any",
        action="store_true",
        help="Replace base rows whenever chunk_id exists in retry (default: only error->success).",
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


def has_error(row: dict[str, Any]) -> bool:
    return bool(((row.get("llm_details") or {}).get("phase1_error")))


def main() -> None:
    args = parse_args()

    base_rows = load_jsonl(args.base)
    retry_rows = load_jsonl(args.retry)
    retry_by_chunk = {str(r.get("chunk_id")): r for r in retry_rows if r.get("chunk_id")}

    replaced = 0
    kept = 0

    out_rows: list[dict[str, Any]] = []
    for base in base_rows:
        chunk_id = str(base.get("chunk_id"))
        retry = retry_by_chunk.get(chunk_id)
        if retry is None:
            out_rows.append(base)
            kept += 1
            continue

        if args.replace_any:
            out_rows.append(retry)
            replaced += 1
            continue

        if has_error(base) and not has_error(retry):
            out_rows.append(retry)
            replaced += 1
        else:
            out_rows.append(base)
            kept += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Base rows:    {len(base_rows)}")
    print(f"Retry rows:   {len(retry_rows)}")
    print(f"Replaced:     {replaced}")
    print(f"Unchanged:    {kept}")
    print(f"Wrote merged: {args.output}")


if __name__ == "__main__":
    main()

