#!/usr/bin/env python3
"""Compare two LLM annotation runs across confidence thresholds."""

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
        description="Compare two LLM runs at chunk and report level across thresholds."
    )
    parser.add_argument(
        "--run-a",
        type=Path,
        required=True,
        help="Path to baseline LLM annotations.jsonl (A).",
    )
    parser.add_argument(
        "--run-b",
        type=Path,
        required=True,
        help="Path to comparison LLM annotations.jsonl (B).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/golden_set/compare-llm-runs"),
        help="Output directory for reports and CSVs.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0,0.1,0.2,0.3",
        help="Comma-separated confidence thresholds (e.g., 0,0.1,0.2).",
    )
    return parser.parse_args()


def parse_thresholds(raw: str) -> List[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


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


def filter_llm_labels(record: Dict[str, Any], field: str, threshold: float) -> List[str]:
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


def index_by_chunk_id(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        cid = rec.get("chunk_id")
        if cid:
            indexed[cid] = rec
    return indexed


def aggregate_by_report(
    records: Iterable[Dict[str, Any]],
    threshold: float,
) -> Tuple[Dict[Tuple[str, int], Dict[str, set]], Dict[Tuple[str, int], Dict[str, Any]]]:
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
            labels = filter_llm_labels(rec, field, threshold)
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


def summarize_pairwise(
    pairs: Iterable[Tuple[str, Dict[str, Any], Dict[str, Any]]],
    threshold: float,
    level: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_rows: List[Dict[str, Any]] = []
    label_rows: List[Dict[str, Any]] = []

    for field in FIELDS:
        j_scores: List[float] = []
        tp = fp = fn = 0
        label_stats: Dict[str, Dict[str, int]] = {}

        for _, a, b in pairs:
            set_a = set(filter_llm_labels(a, field, threshold))
            set_b = set(filter_llm_labels(b, field, threshold))
            j_scores.append(jaccard(set_a, set_b))
            tp += len(set_a & set_b)
            fp += len(set_b - set_a)
            fn += len(set_a - set_b)
            for label in set_a | set_b:
                stats = label_stats.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})
                if label in set_a and label in set_b:
                    stats["tp"] += 1
                elif label in set_b and label not in set_a:
                    stats["fp"] += 1
                elif label in set_a and label not in set_b:
                    stats["fn"] += 1

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        summary_rows.append(
            {
                "level": level,
                "threshold": threshold,
                "field": field,
                "avg_jaccard": round(mean(j_scores), 4) if j_scores else 0.0,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )

        for label, stats in sorted(label_stats.items()):
            ltp = stats["tp"]
            lfp = stats["fp"]
            lfn = stats["fn"]
            lprec = ltp / (ltp + lfp) if ltp + lfp else 0.0
            lrec = ltp / (ltp + lfn) if ltp + lfn else 0.0
            lf1 = 2 * lprec * lrec / (lprec + lrec) if lprec + lrec else 0.0
            label_rows.append(
                {
                    "level": level,
                    "threshold": threshold,
                    "field": field,
                    "label": label,
                    "tp": ltp,
                    "fp": lfp,
                    "fn": lfn,
                    "precision": round(lprec, 4),
                    "recall": round(lrec, 4),
                    "f1": round(lf1, 4),
                }
            )

    return summary_rows, label_rows


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    output_dir = args.output_dir

    run_a_records = load_jsonl(args.run_a)
    run_b_records = load_jsonl(args.run_b)
    run_a_by_id = index_by_chunk_id(run_a_records)
    run_b_by_id = index_by_chunk_id(run_b_records)

    common_ids = sorted(set(run_a_by_id) & set(run_b_by_id))
    missing_a = sorted(set(run_b_by_id) - set(run_a_by_id))
    missing_b = sorted(set(run_a_by_id) - set(run_b_by_id))

    output_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# LLM Run Comparison",
        "",
        f"- Run A: {args.run_a}",
        f"- Run B: {args.run_b}",
        f"- Common chunks: {len(common_ids)}",
        f"- Missing in A: {len(missing_a)}",
        f"- Missing in B: {len(missing_b)}",
        "",
        "## Threshold Summary",
        "",
    ]

    summary_rows_all: List[Dict[str, Any]] = []
    label_rows_all: List[Dict[str, Any]] = []

    for threshold in thresholds:
        pairs = [(cid, run_a_by_id[cid], run_b_by_id[cid]) for cid in common_ids]
        chunk_summary, chunk_labels = summarize_pairwise(pairs, threshold, level="chunk")
        summary_rows_all.extend(chunk_summary)
        label_rows_all.extend(chunk_labels)

        # Report-level aggregation
        a_reports, a_meta = aggregate_by_report(run_a_records, threshold)
        b_reports, b_meta = aggregate_by_report(run_b_records, threshold)
        report_keys = sorted(set(a_reports) | set(b_reports))
        report_pairs = []
        for key in report_keys:
            # Pack into dicts compatible with filter_llm_labels usage
            report_pairs.append(
                (
                    f"{key[0]}:{key[1]}",
                    {field: list(a_reports.get(key, {}).get(field, set())) for field in FIELDS},
                    {field: list(b_reports.get(key, {}).get(field, set())) for field in FIELDS},
                )
            )

        report_summary, report_labels = summarize_pairwise(report_pairs, 0.0, level="report")
        # report_pairs already filtered by threshold, so pass 0.0 for no extra filtering
        for row in report_summary:
            row["threshold"] = threshold
        for row in report_labels:
            row["threshold"] = threshold
        summary_rows_all.extend(report_summary)
        label_rows_all.extend(report_labels)

        report_lines.append(f"### Threshold = {threshold}")
        for row in chunk_summary:
            report_lines.append(
                f"- chunk/{row['field']}: "
                f"avg_jaccard={row['avg_jaccard']}, "
                f"precision={row['precision']}, recall={row['recall']}, f1={row['f1']}"
            )
        for row in report_summary:
            report_lines.append(
                f"- report/{row['field']}: "
                f"avg_jaccard={row['avg_jaccard']}, "
                f"precision={row['precision']}, recall={row['recall']}, f1={row['f1']}"
            )
        report_lines.append("")

    write_csv(output_dir / "summary.csv", summary_rows_all)
    write_csv(output_dir / "label_metrics.csv", label_rows_all)
    (output_dir / "comparison_report.md").write_text("\n".join(report_lines))

    print(f"Wrote {output_dir / 'comparison_report.md'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    print(f"Wrote {output_dir / 'label_metrics.csv'}")


if __name__ == "__main__":
    main()
