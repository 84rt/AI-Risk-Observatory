#!/usr/bin/env python3
"""
Merge phase 2 classifier results back into the full golden set.

Reads the golden set + the three testbed run JSONL files produced by
run_phase2_classifiers.py, and writes a new complete JSONL with all
phase 2 fields filled in for every chunk.

Usage:
    python3 scripts/merge_phase2_into_golden_set.py \
        --golden-set data/golden_set/human_reconciled/annotations.jsonl \
        --run-suffix full-v1 \
        --model gemini-3-flash-preview \
        --output data/golden_set/phase2_annotated/p2-gemini-3-flash-preview-full-v1.jsonl
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
RUNS_DIR = REPO_ROOT / "data" / "testbed_runs"

CLASSIFIERS = ["risk", "adoption_type", "vendor"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge phase 2 classifier outputs into the full golden set."
    )
    p.add_argument(
        "--golden-set",
        type=Path,
        default=REPO_ROOT / "data" / "golden_set" / "human_reconciled" / "annotations.jsonl",
        help="Input golden set JSONL.",
    )
    p.add_argument(
        "--run-suffix",
        required=True,
        help="Run suffix used in run_phase2_classifiers.py (e.g. full-v1).",
    )
    p.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Model name used in run IDs.",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory containing testbed run JSONL files.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the merged JSONL.",
    )
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def index_by_chunk_id(records: list[dict]) -> dict[str, dict]:
    return {str(r["chunk_id"]): r for r in records if r.get("chunk_id")}


def main() -> None:
    args = parse_args()

    # Load golden set
    golden = load_jsonl(args.golden_set)
    print(f"Loaded {len(golden)} chunks from {args.golden_set.name}")

    # Load each classifier's testbed results
    classifier_data: dict[str, dict[str, dict]] = {}
    for name in CLASSIFIERS:
        run_id = f"p2-{name}-{args.model}-{args.run_suffix}"
        run_path = args.runs_dir / f"{run_id}.jsonl"
        if not run_path.exists():
            print(f"  {name}: {run_path.name} not found, skipping")
            continue
        records = load_jsonl(run_path)
        classifier_data[name] = index_by_chunk_id(records)
        print(f"  {name}: {len(records)} results from {run_path.name}")

    if not classifier_data:
        raise SystemExit("No classifier results found. Run run_phase2_classifiers.py first.")

    # Merge
    merged_count = {name: 0 for name in CLASSIFIERS}
    output_records = []

    for chunk in golden:
        chunk_id = str(chunk.get("chunk_id", ""))
        record = dict(chunk)  # shallow copy

        # Risk
        if "risk" in classifier_data:
            risk_result = classifier_data["risk"].get(chunk_id)
            if risk_result:
                record["risk_taxonomy"] = risk_result.get("llm_labels", [])
                record["risk_signals"] = risk_result.get("risk_signals", [])
                record["risk_confidence"] = risk_result.get("risk_confidences", {})
                record["risk_substantiveness"] = risk_result.get("risk_substantiveness")
                merged_count["risk"] += 1

        # Adoption type
        if "adoption_type" in classifier_data:
            adopt_result = classifier_data["adoption_type"].get(chunk_id)
            if adopt_result:
                record["adoption_types"] = adopt_result.get("llm_labels", [])
                record["adoption_confidence"] = adopt_result.get("adoption_signals", {})
                merged_count["adoption_type"] += 1

        # Vendor
        if "vendor" in classifier_data:
            vendor_result = classifier_data["vendor"].get(chunk_id)
            if vendor_result:
                record["vendor_tags"] = vendor_result.get("llm_labels", [])
                record["vendor_other"] = vendor_result.get("llm_other")
                record["vendor_confidence"] = vendor_result.get("vendor_signals", {})
                merged_count["vendor"] += 1

        output_records.append(record)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in output_records:
            f.write(json.dumps(r) + "\n")

    # Write meta
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "created_at": datetime.now().isoformat(),
        "golden_set": str(args.golden_set),
        "model": args.model,
        "run_suffix": args.run_suffix,
        "total_chunks": len(output_records),
        "merged": merged_count,
    }, indent=2))

    print(f"\nWrote {len(output_records)} chunks -> {args.output}")
    print(f"Merged: risk={merged_count['risk']}, adoption_type={merged_count['adoption_type']}, vendor={merged_count['vendor']}")


if __name__ == "__main__":
    main()
