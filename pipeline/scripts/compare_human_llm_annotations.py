#!/usr/bin/env python3
"""Compare and summarize human vs LLM chunk annotations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare human vs LLM chunk annotations and write summary artifacts."
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
        default=Path("data/golden_set/compare"),
        help="Output directory for reports and CSVs.",
    )
    parser.add_argument(
        "--max-disagreements",
        type=int,
        default=50,
        help="Max disagreements to write to the disagreements CSV.",
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


def index_by_chunk_id(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        chunk_id = record.get("chunk_id")
        if not chunk_id:
            continue
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


def extract_confidence(record: Dict[str, Any], key: str) -> Dict[str, float]:
    value = record.get(key)
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items() if v is not None}
    return {}


def extract_llm_confidence(record: Dict[str, Any], key: str) -> Dict[str, float]:
    details = record.get("llm_details") or {}
    value = details.get(key)
    if value is None and key.endswith("_signals"):
        value = details.get(key.replace("_signals", "_confidences"))
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        out = {}
        for entry in value:
            if isinstance(entry, dict):
                k = entry.get("type") or entry.get("label")
                v = entry.get("signal")
                if k is not None and isinstance(v, (int, float)):
                    out[str(k)] = float(v)
        return out
    return {}


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
            confs = extract_llm_confidence({"llm_details": {"adoption_signals": confs}}, "adoption_signals")
    elif field == "risk_taxonomy":
        confs = details.get("risk_signals") or details.get("risk_confidences") or {}
        if isinstance(confs, list):
            confs = extract_llm_confidence({"llm_details": {"risk_signals": confs}}, "risk_signals")
    elif field == "vendor_tags":
        confs = details.get("vendor_confidences") or {}
    else:
        confs = {}
    return [label for label in labels if float(confs.get(label, 0.0)) >= threshold]


def compute_label_metrics(
    chunks: Iterable[Tuple[str, Dict[str, Any], Dict[str, Any]]],
    field: str,
    labels: List[str],
    llm_threshold: float,
) -> Tuple[List[Dict[str, Any]], float]:
    metrics: List[Dict[str, Any]] = []
    label_stats = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    jaccard_scores: List[float] = []

    for _, human, llm in chunks:
        human_set = set(normalize_labels(human.get(field)))
        llm_set = set(filter_llm_labels(llm, field, llm_threshold))
        union = human_set | llm_set
        inter = human_set & llm_set
        jaccard_scores.append(1.0 if not union else len(inter) / len(union))
        for label in labels:
            if label in human_set and label in llm_set:
                label_stats[label]["tp"] += 1
            elif label in llm_set and label not in human_set:
                label_stats[label]["fp"] += 1
            elif label in human_set and label not in llm_set:
                label_stats[label]["fn"] += 1

    for label in labels:
        tp = label_stats[label]["tp"]
        fp = label_stats[label]["fp"]
        fn = label_stats[label]["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics.append(
            {
                "label": label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )

    avg_jaccard = mean(jaccard_scores) if jaccard_scores else 0.0
    return metrics, avg_jaccard


def compute_confidence_diffs(
    chunks: Iterable[Tuple[str, Dict[str, Any], Dict[str, Any]]],
    human_key: str,
    llm_key: str,
) -> Tuple[float, int]:
    diffs: List[float] = []
    for _, human, llm in chunks:
        human_conf = extract_confidence(human, human_key)
        llm_conf = extract_llm_confidence(llm, llm_key)
        for label in set(human_conf) & set(llm_conf):
            diffs.append(abs(human_conf[label] - llm_conf[label]))
    return (mean(diffs) if diffs else 0.0), len(diffs)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_bar(count: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return ""
    filled = int(round(width * (count / total)))
    return "#" * filled + "." * (width - filled)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    human_records = load_jsonl(args.human)
    llm_records = load_jsonl(args.llm)
    human_by_id = index_by_chunk_id(human_records)
    llm_by_id = index_by_chunk_id(llm_records)

    common_ids = sorted(set(human_by_id) & set(llm_by_id))
    missing_human = sorted(set(llm_by_id) - set(human_by_id))
    missing_llm = sorted(set(human_by_id) - set(llm_by_id))

    paired = [(cid, human_by_id[cid], llm_by_id[cid]) for cid in common_ids]

    label_sets = {
        "mention_types": sorted(
            {
                *{label for _, h, _ in paired for label in normalize_labels(h.get("mention_types"))},
                *{label for _, _, l in paired for label in normalize_labels(l.get("mention_types"))},
            }
        ),
        "adoption_types": sorted(
            {
                *{label for _, h, _ in paired for label in normalize_labels(h.get("adoption_types"))},
                *{label for _, _, l in paired for label in normalize_labels(l.get("adoption_types"))},
            }
        ),
        "risk_taxonomy": sorted(
            {
                *{label for _, h, _ in paired for label in normalize_labels(h.get("risk_taxonomy"))},
                *{label for _, _, l in paired for label in normalize_labels(l.get("risk_taxonomy"))},
            }
        ),
        "vendor_tags": sorted(
            {
                *{label for _, h, _ in paired for label in normalize_labels(h.get("vendor_tags"))},
                *{label for _, _, l in paired for label in normalize_labels(l.get("vendor_tags"))},
            }
        ),
    }

    summaries: Dict[str, Dict[str, Any]] = {}
    for field, labels in label_sets.items():
        metrics, avg_jaccard = compute_label_metrics(
            paired,
            field,
            labels,
            args.llm_confidence_threshold,
        )
        summaries[field] = {
            "labels": labels,
            "metrics": metrics,
            "avg_jaccard": round(avg_jaccard, 4),
        }
        write_csv(output_dir / f"{field}_metrics.csv", metrics)

    adoption_conf_diff, adoption_conf_n = compute_confidence_diffs(
        paired, "adoption_confidence", "adoption_signals"
    )
    risk_conf_diff, risk_conf_n = compute_confidence_diffs(
        paired, "risk_confidence", "risk_signals"
    )

    disagreements: List[Dict[str, Any]] = []
    for cid, human, llm in paired:
        for field in ["mention_types", "adoption_types", "risk_taxonomy", "vendor_tags"]:
            human_set = set(normalize_labels(human.get(field)))
            llm_set = set(filter_llm_labels(llm, field, args.llm_confidence_threshold))
            if human_set == llm_set:
                continue
            union = human_set | llm_set
            inter = human_set & llm_set
            jaccard = 1.0 if not union else len(inter) / len(union)
            disagreements.append(
                {
                    "chunk_id": cid,
                    "document_id": human.get("document_id"),
                    "field": field,
                    "human_labels": ",".join(sorted(human_set)),
                    "llm_labels": ",".join(sorted(llm_set)),
                    "jaccard": round(jaccard, 4),
                    "chunk_preview": (human.get("chunk_text") or "")[:160].replace("\n", " "),
                }
            )
            if len(disagreements) >= args.max_disagreements:
                break
        if len(disagreements) >= args.max_disagreements:
            break

    write_csv(output_dir / "disagreements.csv", disagreements)

    false_positives: List[Dict[str, Any]] = []
    for cid, human, llm in paired:
        for field in ["mention_types", "adoption_types", "risk_taxonomy", "vendor_tags"]:
            human_set = set(normalize_labels(human.get(field)))
            llm_set = set(filter_llm_labels(llm, field, args.llm_confidence_threshold))
            extras = llm_set - human_set
            if not extras:
                continue
            details = llm.get("llm_details") or {}
            if field == "mention_types":
                confs = details.get("mention_confidences") or {}
            elif field == "adoption_types":
                confs = details.get("adoption_signals") or details.get("adoption_confidences") or {}
                if isinstance(confs, list):
                    confs = extract_llm_confidence({"llm_details": {"adoption_signals": confs}}, "adoption_signals")
            elif field == "risk_taxonomy":
                confs = details.get("risk_signals") or details.get("risk_confidences") or {}
                if isinstance(confs, list):
                    confs = extract_llm_confidence({"llm_details": {"risk_signals": confs}}, "risk_signals")
            elif field == "vendor_tags":
                confs = details.get("vendor_confidences") or {}
            else:
                confs = {}
            for label in sorted(extras):
                false_positives.append(
                    {
                        "chunk_id": cid,
                        "document_id": human.get("document_id"),
                        "field": field,
                        "label": label,
                        "llm_confidence": confs.get(label, None),
                        "chunk_preview": (human.get("chunk_text") or "")[:160].replace("\n", " "),
                    }
                )

    false_positives.sort(key=lambda row: (row["field"], -(row["llm_confidence"] or 0.0)))
    write_csv(output_dir / "false_positives.csv", false_positives)

    report_lines: List[str] = []
    report_lines.append("# Human vs LLM Annotation Comparison\n")
    report_lines.append("## Coverage\n")
    report_lines.append(f"- Human chunks: {len(human_by_id)}")
    report_lines.append(f"- LLM chunks: {len(llm_by_id)}")
    report_lines.append(f"- Common chunks: {len(common_ids)}")
    report_lines.append(f"- Missing in human: {len(missing_human)}")
    report_lines.append(f"- Missing in LLM: {len(missing_llm)}\n")

    report_lines.append("## Agreement\n")
    for field, summary in summaries.items():
        report_lines.append(f"### {field}")
        report_lines.append(f"- Average Jaccard: {summary['avg_jaccard']}")
        report_lines.append("")
        report_lines.append("| Label | TP | FP | FN | Precision | Recall | F1 |")
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in summary["metrics"]:
            report_lines.append(
                f"| {row['label']} | {row['tp']} | {row['fp']} | {row['fn']} | "
                f"{row['precision']} | {row['recall']} | {row['f1']} |"
            )
        report_lines.append("")

    report_lines.append("## Confidence Differences\n")
    report_lines.append(
        f"- Adoption confidence mean abs diff: {round(adoption_conf_diff, 4)} "
        f"(n={adoption_conf_n})"
    )
    report_lines.append(
        f"- Risk confidence mean abs diff: {round(risk_conf_diff, 4)} "
        f"(n={risk_conf_n})"
    )
    report_lines.append("")
    report_lines.append("## LLM Confidence Threshold\n")
    report_lines.append(f"- Threshold applied: {args.llm_confidence_threshold}")
    report_lines.append("")

    report_lines.append("## Label Distributions (Human vs LLM)\n")
    for field, summary in summaries.items():
        labels = summary["labels"]
        report_lines.append(f"### {field}")
        report_lines.append("| Label | Human | LLM | Human Bar | LLM Bar |")
        report_lines.append("| --- | --- | --- | --- | --- |")
        for label in labels:
            human_count = sum(
                label in normalize_labels(human_by_id[cid].get(field)) for cid in common_ids
            )
            llm_count = sum(
                label
                in filter_llm_labels(
                    llm_by_id[cid],
                    field,
                    args.llm_confidence_threshold,
                )
                for cid in common_ids
            )
            report_lines.append(
                f"| {label} | {human_count} | {llm_count} | "
                f"`{render_bar(human_count, len(common_ids))}` | "
                f"`{render_bar(llm_count, len(common_ids))}` |"
            )
        report_lines.append("")

    if missing_human:
        write_csv(
            output_dir / "missing_in_human.csv",
            [{"chunk_id": cid} for cid in missing_human],
        )
    if missing_llm:
        write_csv(
            output_dir / "missing_in_llm.csv",
            [{"chunk_id": cid} for cid in missing_llm],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison_report.md"
    report_path.write_text("\n".join(report_lines))

    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
