#!/usr/bin/env python3
"""Export classifier_testbed run output into LLM annotations format for reconciliation.

Supports both phase 1 (mention_type) and phase 2 (vendor, risk, adoption_type) testbed runs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert testbed run JSONL to LLM annotations JSONL for reconciliation."
    )
    parser.add_argument(
        "--testbed-run",
        type=Path,
        required=True,
        help="Path to testbed run JSONL (data/testbed_runs/<run_id>.jsonl).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override run_id for output records (defaults to testbed file stem).",
    )
    parser.add_argument(
        "--human",
        type=Path,
        default=Path("data/golden_set/human/annotations.jsonl"),
        help="Path to human annotations JSONL for metadata join.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/golden_set/llm"),
        help="Output directory for LLM annotations JSONL.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Limit exported chunks (0 = no limit).",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include records even if chunk_id is missing from human file.",
    )
    parser.add_argument(
        "--confidence-mode",
        choices=["none", "uniform"],
        default="none",
        help=(
            "How to build mention_confidences. "
            "'none' leaves empty; 'uniform' assigns the testbed confidence to each label."
        ),
    )
    parser.add_argument(
        "--classifier-type",
        choices=["mention_type", "vendor", "risk", "adoption_type"],
        default="mention_type",
        help=(
            "Which classifier produced the testbed run. "
            "mention_type (default) reads llm_mention_types; "
            "phase 2 classifiers (vendor, risk, adoption_type) read llm_labels."
        ),
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


def index_by_chunk_id(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("chunk_id")): r for r in records if r.get("chunk_id")}


def load_meta(testbed_run_path: Path) -> Dict[str, Any]:
    meta_path = testbed_run_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return {}


def build_llm_record(
    *,
    run_id: str,
    testbed: Dict[str, Any],
    base: Optional[Dict[str, Any]],
    model_name: Optional[str],
    confidence_mode: str,
    classifier_type: str = "mention_type",
) -> Dict[str, Any]:
    chunk_id = testbed.get("chunk_id")
    confidence = testbed.get("confidence", 0.0)

    # Phase 1 testbed saves llm_mention_types; phase 2 saves llm_labels
    if classifier_type == "mention_type":
        llm_labels = testbed.get("llm_mention_types") or []
    else:
        llm_labels = testbed.get("llm_labels") or []

    record: Dict[str, Any] = dict(base) if base else {}
    record.update(
        {
            "annotation_id": f"llm-{run_id}-{chunk_id}",
            "run_id": run_id,
            "chunk_id": chunk_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "llm",
            "classifier_id": f"llm_testbed_{classifier_type}",
            "classifier_version": "v2",
            "model_name": model_name,
        }
    )

    # Place LLM labels into the correct field based on classifier type
    if classifier_type == "mention_type":
        record["mention_types"] = llm_labels
    elif classifier_type == "vendor":
        record["vendor_tags"] = llm_labels
        record["vendor_other"] = testbed.get("llm_other")
    elif classifier_type == "risk":
        record["risk_taxonomy"] = llm_labels
    elif classifier_type == "adoption_type":
        record["adoption_types"] = llm_labels

    llm_details: Dict[str, Any] = {
        "reasoning": testbed.get("reasoning", ""),
    }
    # Preserve per-label signal/confidence maps when present in testbed output
    if classifier_type == "risk":
        risk_conf = testbed.get("risk_confidences")
        if isinstance(risk_conf, dict) and risk_conf:
            llm_details["risk_confidences"] = {
                str(k): float(v)
                for k, v in risk_conf.items()
                if v is not None and isinstance(v, (int, float))
            }
    if classifier_type == "adoption_type":
        adopt_conf = testbed.get("adoption_signals") or testbed.get("adoption_confidences")
        if isinstance(adopt_conf, dict) and adopt_conf:
            llm_details["adoption_signals"] = {
                str(k): int(v)
                for k, v in adopt_conf.items()
                if v is not None and isinstance(v, (int, float))
            }
        elif isinstance(adopt_conf, list) and adopt_conf:
            llm_details["adoption_signals"] = adopt_conf
    if confidence_mode == "uniform" and llm_labels:
        conf_key = {
            "mention_type": "mention_confidences",
            "vendor": "vendor_signals",
            "risk": "risk_confidences",
            "adoption_type": "adoption_signals",
        }.get(classifier_type, "confidences")
        llm_details[conf_key] = {
            str(label): float(confidence) for label in llm_labels
        }
    record["llm_details"] = llm_details

    return record


def main() -> None:
    args = parse_args()

    testbed_records = load_jsonl(args.testbed_run)
    if not testbed_records:
        raise SystemExit(f"No records found: {args.testbed_run}")

    human_records = load_jsonl(args.human)
    human_by_id = index_by_chunk_id(human_records)

    meta = load_meta(args.testbed_run)
    run_id = args.run_id or meta.get("run_id") or args.testbed_run.stem
    model_name = (meta.get("config") or {}).get("model_name")

    output_dir = args.output_dir
    output_path = output_dir / "annotations.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    with output_path.open("w") as f:
        for record in testbed_records:
            if args.max_chunks and exported >= args.max_chunks:
                break
            chunk_id = record.get("chunk_id")
            if not chunk_id:
                continue
            base = human_by_id.get(chunk_id)
            if base is None and not args.include_missing:
                continue
            llm_record = build_llm_record(
                run_id=run_id,
                testbed=record,
                base=base,
                model_name=model_name,
                confidence_mode=args.confidence_mode,
                classifier_type=args.classifier_type,
            )
            f.write(json.dumps(llm_record) + "\n")
            exported += 1

    print(f"Exported {exported} records to {output_path}")


if __name__ == "__main__":
    main()
