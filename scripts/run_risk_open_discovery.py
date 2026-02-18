#!/usr/bin/env python3
"""Run conservative open-taxonomy AI risk discovery on golden-set chunks.

This script runs the OpenRiskDiscoveryClassifier on either:
- only chunks with human mention_type "risk" (default), or
- all chunks (--all-chunks)

Outputs:
- <run_id>_results.jsonl: per-chunk classifier output
- <run_id>_summary.json: aggregate run metrics + label distribution
- <run_id>_label_inventory.json: discovered labels with example definitions

Examples:
    python3 scripts/run_risk_open_discovery.py --limit 100
    python3 scripts/run_risk_open_discovery.py --all --all-chunks
    python3 scripts/run_risk_open_discovery.py --limit 20 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - runtime fallback for minimal envs
    def tqdm(iterable, desc: str | None = None):
        if desc:
            print(desc)
        return iterable

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))


# Load .env.local before importing pipeline modules (settings read env at import time)
_env_path = REPO_ROOT / ".env.local"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
        if _k and _k not in os.environ:
            os.environ[_k] = _v

GOLDEN_SET_DEFAULT = REPO_ROOT / "data" / "golden_set" / "human_reconciled" / "annotations.jsonl"
RUNS_DIR_DEFAULT = REPO_ROOT / "data" / "testbed_runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run conservative open-taxonomy risk discovery on golden-set chunks."
    )
    p.add_argument(
        "--golden-set",
        type=Path,
        default=GOLDEN_SET_DEFAULT,
        help="Path to reconciled golden set JSONL.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=RUNS_DIR_DEFAULT,
        help="Directory for run outputs.",
    )
    p.add_argument(
        "--run-id",
        default="",
        help="Optional run id. Default: auto timestamped id.",
    )
    p.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Model name (default: gemini-3-flash-preview).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model temperature (default: 0.0).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max number of selected chunks to run. Ignored if --all is set.",
    )
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N selected chunks.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run all selected chunks (ignore --limit).",
    )
    p.add_argument(
        "--all-chunks",
        action="store_true",
        help="Do not filter to mention_type=risk; run across all chunks.",
    )
    p.add_argument(
        "--use-openrouter",
        action="store_true",
        help="Force OpenRouter mode (for provider/model strings with '/').",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show selected chunk counts and sample IDs only; do not call model.",
    )
    return p.parse_args()


def load_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_chunks(
    chunks: list[dict[str, Any]],
    *,
    risk_only: bool,
    offset: int,
    limit: int,
    use_all: bool,
) -> list[dict[str, Any]]:
    if risk_only:
        selected = [c for c in chunks if "risk" in c.get("mention_types", [])]
    else:
        selected = list(chunks)

    if offset > 0:
        selected = selected[offset:]
    if not use_all and limit > 0:
        selected = selected[:limit]
    return selected


def build_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "firm_id": chunk.get("company_id", "Unknown"),
        "firm_name": chunk.get("company_name", "Unknown"),
        "report_year": chunk.get("report_year", 0),
        "sector": chunk.get("sector", "Unknown"),
        "report_section": (
            chunk.get("report_sections", ["Unknown"])[0]
            if chunk.get("report_sections")
            else "Unknown"
        ),
        "mention_types": chunk.get("mention_types", []),
    }


def normalize_label(token: object) -> str:
    return str(token).strip().lower()


def extract_risk_types(parsed: dict[str, Any]) -> list[str]:
    labels = parsed.get("risk_types", [])
    if not isinstance(labels, list):
        return []
    out: list[str] = []
    for label in labels:
        val = normalize_label(label)
        if val:
            out.append(val)
    return out


def extract_risk_signals(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    raw = parsed.get("risk_signals", [])
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        label = normalize_label(entry.get("type", ""))
        signal = entry.get("signal")
        if label and isinstance(signal, (int, float)):
            out.append({"type": label, "signal": int(signal)})
    return out


def extract_label_definitions(parsed: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    raw = parsed.get("label_definitions", [])
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        label = normalize_label(entry.get("type", ""))
        definition = str(entry.get("definition", "")).strip()
        if label and definition:
            out.append({"type": label, "definition": definition})
    return out


def summarize(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    success_rows = [r for r in rows if r.get("success")]
    error_rows = [r for r in rows if not r.get("success")]

    none_count = 0
    label_chunk_counts: Counter[str] = Counter()
    label_signal_sums: Counter[str] = Counter()
    label_signal_counts: Counter[str] = Counter()
    definitions_by_label: dict[str, Counter[str]] = defaultdict(Counter)

    for row in success_rows:
        labels = set(row.get("llm_risk_types", []))
        if labels == {"none"} or not labels:
            none_count += 1
            continue

        for label in sorted(labels):
            if label == "none":
                continue
            label_chunk_counts[label] += 1

        for sig in row.get("llm_risk_signals", []):
            label = normalize_label(sig.get("type", ""))
            score = sig.get("signal")
            if not label or label == "none" or not isinstance(score, (int, float)):
                continue
            label_signal_sums[label] += int(score)
            label_signal_counts[label] += 1

        for d in row.get("label_definitions", []):
            label = normalize_label(d.get("type", ""))
            definition = str(d.get("definition", "")).strip()
            if label and label != "none" and definition:
                definitions_by_label[label][definition] += 1

    top_labels = []
    for label, chunks in sorted(label_chunk_counts.items(), key=lambda x: (-x[1], x[0])):
        mean_signal = (
            round(label_signal_sums[label] / label_signal_counts[label], 3)
            if label_signal_counts[label] > 0
            else None
        )
        top_labels.append(
            {
                "label": label,
                "chunk_count": chunks,
                "mean_signal": mean_signal,
            }
        )

    label_inventory = {}
    for label, defs in sorted(definitions_by_label.items()):
        label_inventory[label] = {
            "count": int(label_chunk_counts.get(label, 0)),
            "top_definitions": [
                {"definition": d, "count": c}
                for d, c in defs.most_common(5)
            ],
        }

    summary = {
        "num_chunks": len(rows),
        "success_count": len(success_rows),
        "error_count": len(error_rows),
        "none_count": none_count,
        "num_discovered_labels": len(label_chunk_counts),
        "top_labels": top_labels,
    }

    return summary, label_inventory


def main() -> int:
    args = parse_args()

    chunks = load_chunks(args.golden_set)
    selected = select_chunks(
        chunks,
        risk_only=not args.all_chunks,
        offset=args.offset,
        limit=args.limit,
        use_all=args.all,
    )

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Selected chunks: {len(selected)}")
    print(f"Selection mode: {'all_chunks' if args.all_chunks else 'mention_type=risk'}")

    if not selected:
        print("No chunks selected. Exiting.")
        return 1

    sample_ids = [str(c.get("chunk_id", "unknown")) for c in selected[:5]]
    print(f"Sample chunk IDs: {sample_ids}")

    if args.dry_run:
        print("Dry run complete. No model calls made.")
        return 0

    try:
        from src.classifiers.open_risk_classifier import OpenRiskDiscoveryClassifier
    except ModuleNotFoundError as e:
        print("Missing Python dependencies for classifier runtime.")
        print(f"Import error: {e}")
        print("Use your project environment (or install pipeline requirements) and re-run.")
        return 1

    run_id = args.run_id.strip() or (
        f"risk-open-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier = OpenRiskDiscoveryClassifier(
        run_id=run_id,
        model_name=args.model,
        temperature=args.temperature,
        use_openrouter=args.use_openrouter,
    )

    results: list[dict[str, Any]] = []
    for chunk in tqdm(selected, desc="Classifying open risk"):
        chunk_id = chunk.get("chunk_id", chunk.get("annotation_id", "unknown"))
        metadata = build_metadata(chunk)
        result = classifier.classify(chunk.get("chunk_text", ""), metadata)
        parsed = result.classification if isinstance(result.classification, dict) else {}

        row = {
            "run_id": run_id,
            "chunk_id": chunk_id,
            "company_name": chunk.get("company_name", "Unknown"),
            "report_year": chunk.get("report_year", 0),
            "mention_types": chunk.get("mention_types", []),
            "human_risk_taxonomy": chunk.get("risk_taxonomy", []),
            "llm_risk_types": extract_risk_types(parsed),
            "llm_risk_signals": extract_risk_signals(parsed),
            "label_definitions": extract_label_definitions(parsed),
            "substantiveness": parsed.get("substantiveness"),
            "reasoning": parsed.get("reasoning", ""),
            "success": bool(result.success),
            "error": result.error_message,
            "confidence": float(result.confidence_score),
            "model": args.model,
            "classifier_type": result.classifier_type,
        }
        results.append(row)

    results_path = output_dir / f"{run_id}_results.jsonl"
    with results_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_payload, label_inventory = summarize(results)
    summary_payload.update(
        {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "temperature": args.temperature,
            "golden_set": str(args.golden_set),
            "selection": {
                "all_chunks": bool(args.all_chunks),
                "offset": int(args.offset),
                "limit": 0 if args.all else int(args.limit),
                "selected_count": len(selected),
            },
            "results_path": str(results_path),
        }
    )

    summary_path = output_dir / f"{run_id}_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    inventory_path = output_dir / f"{run_id}_label_inventory.json"
    inventory_path.write_text(json.dumps(label_inventory, indent=2))

    print("\nRun complete")
    print(f"run_id: {run_id}")
    print(f"results: {results_path}")
    print(f"summary: {summary_path}")
    print(f"label inventory: {inventory_path}")
    print(f"discovered labels: {summary_payload['num_discovered_labels']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
