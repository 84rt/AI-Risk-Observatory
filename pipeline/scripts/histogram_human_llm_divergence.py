#!/usr/bin/env python3
"""Summarize where human vs LLM labels diverge, with histograms."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


FIELDS = ["mention_types", "adoption_types", "risk_taxonomy", "vendor_tags"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build histograms of human vs LLM label divergence."
    )
    parser.add_argument(
        "--human",
        type=Path,
        default=Path("data/golden_set/human/annotations.jsonl"),
        help="Path to human annotations.jsonl",
    )
    parser.add_argument(
        "--llm",
        type=Path,
        default=Path("data/golden_set/llm/annotations.jsonl"),
        help="Path to LLM annotations.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/golden_set/divergence"),
        help="Output directory for histogram artifacts.",
    )
    parser.add_argument(
        "--llm-confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum LLM confidence for a label to be counted.",
    )
    parser.add_argument(
        "--bucket-max",
        type=int,
        default=6,
        help="Max bucket for histogram; values >= bucket-max are grouped.",
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
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        chunk_id = record.get("chunk_id")
        if chunk_id:
            indexed[chunk_id] = record
    return indexed


def normalize_labels(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        return [value]
    return []


def filter_llm_labels(
    record: Dict[str, Any],
    field: str,
    threshold: float,
) -> List[str]:
    labels = normalize_labels(record.get(field))
    if threshold <= 0:
        return labels
    details = record.get("llm_details") or {}
    if field == "mention_types":
        confs = details.get("mention_confidences") or {}
    elif field == "adoption_types":
        confs = details.get("adoption_signals") or details.get("adoption_confidences") or {}
        if isinstance(confs, list):
            confs = {
                str(e.get("type")): float(e.get("signal"))
                for e in confs
                if isinstance(e, dict) and isinstance(e.get("signal"), (int, float))
            }
    elif field == "risk_taxonomy":
        confs = details.get("risk_confidences") or {}
    elif field == "vendor_tags":
        confs = details.get("vendor_confidences") or {}
    else:
        confs = {}
    return [label for label in labels if float(confs.get(label, 0.0)) >= threshold]


def render_bar(count: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return ""
    filled = int(round(width * (count / total)))
    return "#" * filled + "." * (width - filled)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    human_records = load_jsonl(args.human)
    llm_records = load_jsonl(args.llm)
    human_by_id = index_by_chunk_id(human_records)
    llm_by_id = index_by_chunk_id(llm_records)

    shared_ids = sorted(set(human_by_id) & set(llm_by_id))
    total_chunks = len(shared_ids)

    per_chunk_rows: List[Dict[str, Any]] = []
    total_hist = Counter()
    field_hist = {field: Counter() for field in FIELDS}
    disagreements = 0

    for cid in shared_ids:
        human = human_by_id[cid]
        llm = llm_by_id[cid]

        total_diff = 0
        field_diffs: Dict[str, int] = {}

        for field in FIELDS:
            human_set = set(normalize_labels(human.get(field)))
            llm_set = set(filter_llm_labels(llm, field, args.llm_confidence_threshold))
            diff_count = len(human_set.symmetric_difference(llm_set))
            field_diffs[field] = diff_count
            total_diff += diff_count

        if total_diff > 0:
            disagreements += 1

        bucket = (
            f"{args.bucket_max}+"
            if total_diff >= args.bucket_max
            else str(total_diff)
        )
        total_hist[bucket] += 1

        for field in FIELDS:
            diff = field_diffs[field]
            field_bucket = (
                f"{args.bucket_max}+"
                if diff >= args.bucket_max
                else str(diff)
            )
            field_hist[field][field_bucket] += 1

        per_chunk_rows.append(
            {
                "chunk_id": cid,
                "company_name": human.get("company_name"),
                "report_year": human.get("report_year"),
                "total_diff": total_diff,
                "mention_diff": field_diffs["mention_types"],
                "adoption_diff": field_diffs["adoption_types"],
                "risk_diff": field_diffs["risk_taxonomy"],
                "vendor_diff": field_diffs["vendor_tags"],
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "divergence_by_chunk.csv", per_chunk_rows)

    hist_rows: List[Dict[str, Any]] = []
    for bucket, count in sorted(total_hist.items(), key=lambda x: int(x[0].rstrip("+"))):
        hist_rows.append(
            {
                "bucket": bucket,
                "count": count,
                "percent": round(100 * count / total_chunks, 2) if total_chunks else 0.0,
            }
        )
    write_csv(output_dir / "divergence_histogram.csv", hist_rows)

    lines: List[str] = []
    lines.append("Overall divergence histogram (total differing labels per chunk):")
    for row in hist_rows:
        bar = render_bar(row["count"], total_chunks)
        lines.append(f"  {row['bucket']:>4}: {row['count']:>5}  {bar}")
    lines.append("")
    lines.append("Per-field divergence histograms (symmetric diff count per chunk):")
    for field in FIELDS:
        lines.append(f"  {field}:")
        buckets = sorted(field_hist[field].items(), key=lambda x: int(x[0].rstrip('+')))
        for bucket, count in buckets:
            bar = render_bar(count, total_chunks)
            lines.append(f"    {bucket:>4}: {count:>5}  {bar}")
    lines.append("")
    lines.append(
        f"Chunks compared: {total_chunks} | disagreements: {disagreements} "
        f"({round(100 * disagreements / total_chunks, 2) if total_chunks else 0.0}%)"
    )
    lines.append(f"LLM confidence threshold: {args.llm_confidence_threshold}")

    (output_dir / "divergence_histogram.txt").write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\nWrote: {output_dir / 'divergence_by_chunk.csv'}")
    print(f"Wrote: {output_dir / 'divergence_histogram.csv'}")
    print(f"Wrote: {output_dir / 'divergence_histogram.txt'}")


if __name__ == "__main__":
    main()
