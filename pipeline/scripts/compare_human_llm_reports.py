#!/usr/bin/env python3
"""Compare human vs LLM annotations aggregated at report (company-year) level."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


FIELDS = ["mention_types", "adoption_types", "risk_taxonomy", "vendor_tags"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate chunk annotations to report-level and compare human vs LLM."
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
        default=Path("data/golden_set/compare-report-level"),
        help="Output directory for report-level CSVs.",
    )
    parser.add_argument(
        "--llm-confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum LLM confidence for a label to be counted.",
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
        confs = details.get("risk_signals") or details.get("risk_confidences") or {}
        if isinstance(confs, list):
            confs = {
                str(e.get("type")): float(e.get("signal"))
                for e in confs
                if isinstance(e, dict) and isinstance(e.get("signal"), (int, float))
            }
    elif field == "vendor_tags":
        confs = details.get("vendor_confidences") or {}
    else:
        confs = {}
    return [label for label in labels if float(confs.get(label, 0.0)) >= threshold]


def aggregate_records(
    records: Iterable[Dict[str, Any]],
    llm_threshold: float | None = None,
) -> Tuple[Dict[Tuple[str, int], Dict[str, set]], Dict[Tuple[str, int], Dict[str, Any]]]:
    """Aggregate labels to report level by union of chunk labels."""
    by_report: Dict[Tuple[str, int], Dict[str, set]] = defaultdict(
        lambda: {field: set() for field in FIELDS}
    )
    meta: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for rec in records:
        company_id = rec.get("company_id")
        report_year = rec.get("report_year")
        if not company_id or report_year is None:
            continue
        key = (company_id, int(report_year))
        meta[key] = {
            "company_id": company_id,
            "company_name": rec.get("company_name") or company_id,
            "report_year": int(report_year),
        }

        for field in FIELDS:
            if llm_threshold is None:
                labels = normalize_labels(rec.get(field))
            else:
                labels = filter_llm_labels(rec, field, llm_threshold)
            by_report[key][field].update(labels)

    return by_report, meta


def jaccard(a: set, b: set) -> float:
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    human_records = load_jsonl(args.human)
    llm_records = load_jsonl(args.llm)

    human_by_report, meta = aggregate_records(human_records, llm_threshold=None)
    llm_by_report, _ = aggregate_records(llm_records, llm_threshold=args.llm_confidence_threshold)

    report_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    report_keys = sorted(set(human_by_report) | set(llm_by_report))
    for key in report_keys:
        info = meta.get(key, {"company_id": key[0], "company_name": key[0], "report_year": key[1]})
        for field in FIELDS:
            human_set = human_by_report.get(key, {}).get(field, set())
            llm_set = llm_by_report.get(key, {}).get(field, set())
            report_rows.append(
                {
                    "company_id": info["company_id"],
                    "company_name": info["company_name"],
                    "report_year": info["report_year"],
                    "field": field,
                    "human_labels": ",".join(sorted(human_set)),
                    "llm_labels": ",".join(sorted(llm_set)),
                    "jaccard": round(jaccard(human_set, llm_set), 4),
                    "tp": len(human_set & llm_set),
                    "fp": len(llm_set - human_set),
                    "fn": len(human_set - llm_set),
                    "fp_labels": ",".join(sorted(llm_set - human_set)),
                    "fn_labels": ",".join(sorted(human_set - llm_set)),
                }
            )

    for field in FIELDS:
        field_rows = [r for r in report_rows if r["field"] == field]
        if not field_rows:
            continue
        avg_j = mean([r["jaccard"] for r in field_rows])
        avg_tp = mean([r["tp"] for r in field_rows])
        avg_fp = mean([r["fp"] for r in field_rows])
        avg_fn = mean([r["fn"] for r in field_rows])
        summary_rows.append(
            {
                "field": field,
                "reports": len(field_rows),
                "avg_jaccard": round(avg_j, 4),
                "avg_tp": round(avg_tp, 2),
                "avg_fp": round(avg_fp, 2),
                "avg_fn": round(avg_fn, 2),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "report_level_comparison.csv", report_rows)
    write_csv(output_dir / "report_level_summary.csv", summary_rows)
    print(f"Wrote {output_dir / 'report_level_comparison.csv'}")
    print(f"Wrote {output_dir / 'report_level_summary.csv'}")


if __name__ == "__main__":
    main()
