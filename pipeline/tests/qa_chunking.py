#!/usr/bin/env python3
"""Quality checks for chunked JSONL outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from src.config import get_settings  # noqa: E402


REQUIRED_FIELDS = [
    "chunk_id",
    "document_id",
    "chunk_text",
    "paragraph_start",
    "paragraph_end",
    "context_before",
    "context_after",
    "matched_keywords",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA checks for chunk JSONL outputs")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID for data/processed/<run_id>/chunks/chunks.jsonl",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=None,
        help="Optional explicit path to chunks.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for QA reports",
    )
    return parser.parse_args()


def _validate_chunk(chunk: Dict) -> List[str]:
    issues = []
    for field in REQUIRED_FIELDS:
        if field not in chunk:
            issues.append(f"missing_{field}")
    if chunk.get("chunk_text") is not None and not str(chunk.get("chunk_text")).strip():
        issues.append("empty_chunk_text")
    if chunk.get("matched_keywords") is not None and not chunk.get("matched_keywords"):
        issues.append("empty_matched_keywords")
    if chunk.get("paragraph_start") is not None and chunk.get("paragraph_end") is not None:
        if chunk["paragraph_start"] > chunk["paragraph_end"]:
            issues.append("invalid_paragraph_range")
    return issues


def run_qa(
    run_id: str,
    chunks_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> List[Dict]:
    settings = get_settings()
    default_path = settings.processed_dir / run_id / "chunks" / "chunks.jsonl"
    chunks_path = chunks_path or default_path

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks.jsonl at {chunks_path}")

    results: List[Dict] = []
    seen_chunk_ids = set()

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                results.append(
                    {
                        "line": line_num,
                        "chunk_id": None,
                        "issues": ["invalid_json"],
                    }
                )
                continue

            issues = _validate_chunk(chunk)
            chunk_id = chunk.get("chunk_id")
            if chunk_id in seen_chunk_ids:
                issues.append("duplicate_chunk_id")
            seen_chunk_ids.add(chunk_id)

            results.append(
                {
                    "line": line_num,
                    "chunk_id": chunk_id,
                    "document_id": chunk.get("document_id"),
                    "issues": issues,
                }
            )

    output_dir = output_dir or chunks_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "qa_chunking_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = output_dir / "qa_chunking_report.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["line", "chunk_id", "document_id", "issues"])
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "line": result.get("line"),
                    "chunk_id": result.get("chunk_id"),
                    "document_id": result.get("document_id"),
                    "issues": ";".join(result.get("issues") or []),
                }
            )

    flagged = sum(1 for r in results if r.get("issues"))
    print(f"✅ QA complete: {len(results)} chunks checked, {flagged} flagged")
    print(f"Report: {json_path}")
    print(f"Report: {csv_path}")
    return results


def main() -> None:
    args = parse_args()
    run_qa(run_id=args.run_id, chunks_path=args.chunks_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
