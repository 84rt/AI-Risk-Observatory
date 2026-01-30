#!/usr/bin/env python3
"""Two-phase LLM classifier for AI-mention chunks.

Phase 1: Run MentionTypeClassifier on all chunks (checkpoint to disk).
Phase 2: Run downstream classifiers (risk, adoption) based on phase 1 results.

Uses tqdm for progress tracking instead of manual counters.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from src.classifiers import (  # noqa: E402
    AdoptionTypeClassifier,
    MentionTypeClassifier,
    RiskClassifier,
)
from src.config import get_settings  # noqa: E402
from src.utils.validation import validate_classification_response  # noqa: E402

settings = get_settings()


# ---------------------------------------------------------------------------
# Downstream phase registry
# ---------------------------------------------------------------------------

@dataclass
class DownstreamPhase:
    """Declarative definition of a downstream classification phase."""
    name: str
    mention_tag: str
    classifier_class: Type
    build_record: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None


def _build_risk_fields(chunk: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    risk_taxonomy = payload.get("risk_types", []) or []
    risk_confidences = payload.get("confidence_scores", {}) or {}
    return {
        "risk_taxonomy": risk_taxonomy,
        "risk_confidences": risk_confidences,
    }


def _build_adoption_fields(chunk: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    adoption_confidences = payload.get("adoption_confidences", {}) or {}
    adoption_types = tags_from_confidences(adoption_confidences, 0.0)  # threshold applied later
    valid_scores = [
        s for s in adoption_confidences.values() if isinstance(s, (int, float))
    ]
    return {
        "adoption_types_raw": adoption_types,
        "adoption_confidences": adoption_confidences,
        "adoption_confidence": max(valid_scores) if valid_scores else 0.0,
    }


DOWNSTREAM_PHASES: List[DownstreamPhase] = [
    DownstreamPhase(
        name="Risk",
        mention_tag="risk",
        classifier_class=RiskClassifier,
        build_record=_build_risk_fields,
    ),
    DownstreamPhase(
        name="Adoption",
        mention_tag="adoption",
        classifier_class=AdoptionTypeClassifier,
        build_record=_build_adoption_fields,
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-phase LLM annotation for AI-mention chunks.")
    parser.add_argument("--chunks", type=Path, required=True, help="Path to chunks JSONL file.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run_id override.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/golden_set/llm"), help="Output directory.")
    parser.add_argument("--append", action="store_true", help="Resume from existing checkpoints.")
    parser.add_argument("--max-chunks", type=int, default=0, help="Limit chunks to process (0 = no limit).")
    parser.add_argument("--mention-threshold", type=float, default=0.2, help="Confidence threshold for mention tags.")
    parser.add_argument("--downstream-threshold", type=float, default=None, help="Confidence threshold for downstream routing.")
    parser.add_argument("--max-attempts", type=int, default=2, help="Retry attempts per classifier call.")
    parser.add_argument("--max-concurrent", type=int, default=30, help="Max concurrent requests per phase.")
    parser.add_argument("--model", type=str, default=None, help="Override model name.")
    parser.add_argument("--openrouter", action="store_true", help="Route calls through OpenRouter.")
    return parser.parse_args()


def infer_run_id(chunks_path: Path) -> str:
    parts = chunks_path.as_posix().split("/")
    if "processed" in parts:
        idx = parts.index("processed")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown-run"




def tags_from_confidences(confidences: Dict[str, Any], threshold: float) -> List[str]:
    return [
        tag for tag, score in confidences.items()
        if isinstance(score, (int, float)) and score >= threshold
    ]


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    chunks = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return chunks


def load_checkpoint(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load a JSONL checkpoint file, returning {chunk_id: record}."""
    records: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("chunk_id")
            if cid:
                records[cid] = obj
    return records


def classify_with_validation(
    classifier,
    text: str,
    metadata: Dict[str, Any],
    source_file: str,
    max_attempts: int,
) -> Tuple[Dict[str, Any], List[str]]:
    last_messages: List[str] = []
    payload: Dict[str, Any] = {}
    for attempt in range(max_attempts):
        result = classifier.classify(text, metadata, source_file)
        payload = result.classification or {}
        ok, messages = validate_classification_response(payload, classifier.CLASSIFIER_TYPE)
        if ok:
            return payload, []
        last_messages = messages
    return payload, last_messages


def build_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "firm_id": chunk.get("company_id") or "unknown",
        "firm_name": chunk.get("company_name") or "unknown",
        "report_year": chunk.get("report_year") or 0,
        "sector": chunk.get("sector") or "Unknown",
        "report_section": ", ".join(chunk.get("report_sections") or []),
    }


# ---------------------------------------------------------------------------
# Phase 1: Mention Type Classification
# ---------------------------------------------------------------------------

def run_phase1(
    chunks: List[Dict[str, Any]],
    checkpoint_path: Path,
    args: argparse.Namespace,
    run_id: str,
    model_name: str,
) -> Dict[str, Dict[str, Any]]:
    """Classify mention types for all chunks. Returns {chunk_id: phase1_record}."""

    existing = load_checkpoint(checkpoint_path) if args.append else {}
    to_process = [c for c in chunks if c.get("chunk_id") not in existing]

    total = len(to_process)
    if total == 0:
        print(f"[Phase 1 – Mention Type] All {len(chunks)} chunks already in checkpoint, skipping.")
        return existing

    print(f"[Phase 1 – Mention Type] Starting: {total} chunks")

    classifier = MentionTypeClassifier(
        run_id=run_id,
        model_name=model_name,
        use_openrouter=args.openrouter,
    )

    def classify_one(chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        chunk_id = chunk["chunk_id"]
        text = (chunk.get("chunk_text") or "")
        metadata = build_metadata(chunk)

        payload, errors = classify_with_validation(
            classifier, text, metadata, str(args.chunks), args.max_attempts
        )

        mention_confidences = payload.get("confidence_scores", {}) or {}
        mention_types = tags_from_confidences(mention_confidences, args.mention_threshold)

        return {
            "chunk_id": chunk_id,
            "mention_types": mention_types,
            "mention_confidences": mention_confidences,
            "chunk": chunk,
            "errors": errors,
        }

    # Use thread_map for concurrent processing with tqdm progress bar
    results = thread_map(
        classify_one,
        to_process,
        max_workers=args.max_concurrent,
        desc="Phase 1: Mention Types",
        unit="chunk",
    )

    # Filter out None results and count stats
    valid_results = [r for r in results if r is not None]
    success_count = sum(1 for r in valid_results if not r.get("errors"))
    error_count = sum(1 for r in valid_results if r.get("errors"))

    # Write results to checkpoint (append mode for resumability)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as f:
        for record in valid_results:
            # Remove errors field before saving
            save_record = {k: v for k, v in record.items() if k != "errors"}
            f.write(json.dumps(save_record) + "\n")

    # Merge with existing
    for r in valid_results:
        existing[r["chunk_id"]] = r

    # Summary
    tag_counts: Dict[str, int] = {}
    for rec in existing.values():
        for tag in rec.get("mention_types", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    summary = ", ".join(f"{k}={v}" for k, v in sorted(tag_counts.items()))
    print(f"[Phase 1 – Mention Type] Complete: {len(valid_results)}/{total} "
          f"| success: {success_count} | errors: {error_count}")
    print(f"[Phase 1 – Mention Type] Results: {summary}")

    return existing


# ---------------------------------------------------------------------------
# Phase 2: Downstream Classification
# ---------------------------------------------------------------------------

def run_phase2(
    phase1_records: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    run_id: str,
    model_name: str,
    downstream_threshold: float,
) -> List[Dict[str, Any]]:
    """Run downstream classifiers and produce final annotation records."""

    # Collect all (chunk_id, phase_def) pairs that need downstream calls
    @dataclass
    class DownstreamTask:
        chunk_id: str
        phase: DownstreamPhase
        chunk: Dict[str, Any]
        text: str
        metadata: Dict[str, Any]

    tasks_by_phase: Dict[str, List[DownstreamTask]] = {}
    for cid, rec in phase1_records.items():
        chunk = rec["chunk"]
        mention_confidences = rec.get("mention_confidences", {})
        text = (chunk.get("chunk_text") or "")
        metadata = build_metadata(chunk)

        for phase_def in DOWNSTREAM_PHASES:
            conf = mention_confidences.get(phase_def.mention_tag, 0.0)
            if isinstance(conf, (int, float)) and conf >= downstream_threshold:
                # Add mention_types to metadata for downstream classifiers
                enriched_metadata = {
                    **metadata,
                    "mention_types": rec.get("mention_types", []),
                }
                task = DownstreamTask(
                    chunk_id=cid,
                    phase=phase_def,
                    chunk=chunk,
                    text=text,
                    metadata=enriched_metadata,
                )
                tasks_by_phase.setdefault(phase_def.name, []).append(task)

    # Instantiate classifiers
    classifiers: Dict[str, Any] = {}
    for phase_def in DOWNSTREAM_PHASES:
        classifiers[phase_def.name] = phase_def.classifier_class(
            run_id=run_id,
            model_name=model_name,
            use_openrouter=args.openrouter,
        )

    # Store downstream results: {chunk_id: {phase_name: fields_dict}}
    downstream_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Process each phase with tqdm
    for phase_def in DOWNSTREAM_PHASES:
        phase_tasks = tasks_by_phase.get(phase_def.name, [])
        if not phase_tasks:
            continue

        print(f"[Phase 2 – {phase_def.name}] Queued: {len(phase_tasks)} chunks")
        classifier = classifiers[phase_def.name]

        def run_task(task: DownstreamTask) -> Tuple[str, str, Dict[str, Any]]:
            payload, errors = classify_with_validation(
                classifier, task.text, task.metadata, str(args.chunks), args.max_attempts
            )

            if task.phase.build_record:
                fields = task.phase.build_record(task.chunk, payload)
            else:
                fields = {"payload": payload}

            return task.chunk_id, task.phase.name, fields

        # Use thread_map for concurrent processing with tqdm
        results = thread_map(
            run_task,
            phase_tasks,
            max_workers=args.max_concurrent,
            desc=f"Phase 2: {phase_def.name}",
            unit="chunk",
        )

        # Collect results
        for chunk_id, phase_name, fields in results:
            if chunk_id not in downstream_results:
                downstream_results[chunk_id] = {}
            downstream_results[chunk_id][phase_name] = fields

    # Build final annotation records
    annotations: List[Dict[str, Any]] = []
    for cid, rec in tqdm(phase1_records.items(), desc="Building annotations", unit="record"):
        chunk = rec["chunk"]
        mention_types = rec.get("mention_types", [])
        mention_confidences = rec.get("mention_confidences", {})

        if not mention_types:
            mention_types = ["none"]

        ds = downstream_results.get(cid, {})

        # Adoption fields
        adoption_fields = ds.get("Adoption", {})
        adoption_confidences = adoption_fields.get("adoption_confidences", {})
        adoption_types = tags_from_confidences(adoption_confidences, downstream_threshold)
        if adoption_confidences and not adoption_types:
            adoption_types = ["none"]
        adoption_confidence = adoption_fields.get("adoption_confidence")

        # Risk fields
        risk_fields = ds.get("Risk", {})
        risk_taxonomy = risk_fields.get("risk_taxonomy", [])
        risk_confidences = risk_fields.get("risk_confidences", {})

        record = {
            "annotation_id": f"llm-{run_id}-{cid}",
            "run_id": run_id,
            "chunk_id": cid,
            "document_id": chunk.get("document_id"),
            "company_id": chunk.get("company_id"),
            "company_name": chunk.get("company_name"),
            "report_year": chunk.get("report_year"),
            "report_sections": chunk.get("report_sections"),
            "chunk_text": chunk.get("chunk_text"),
            "matched_keywords": chunk.get("matched_keywords"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mention_types": mention_types,
            "adoption_types": adoption_types,
            "adoption_confidence": adoption_confidence,
            "risk_taxonomy": risk_taxonomy,
            "risk_substantiveness": None,
            "vendor_tags": ["vendor"] if "vendor" in mention_types else [],
            "vendor_other": None,
            "llm_details": {
                "model": model_name,
                "mention_confidences": mention_confidences,
                "adoption_confidences": adoption_confidences,
                "risk_confidences": risk_confidences,
                "vendor_confidences": {},
            },
        }
        annotations.append(record)

    return annotations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    chunks_path = args.chunks
    run_id = args.run_id or infer_run_id(chunks_path)
    output_dir: Path = args.output_dir
    phase1_path = output_dir / "phase1_mentions.jsonl"
    annotations_path = output_dir / "annotations.jsonl"

    downstream_threshold = (
        args.downstream_threshold
        if args.downstream_threshold is not None
        else settings.downstream_confidence_threshold
    )

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file at {chunks_path}")

    # Load all chunks
    all_chunks = load_chunks(chunks_path)
    all_chunks = [c for c in all_chunks if c.get("chunk_id") and (c.get("chunk_text") or "").strip()]
    if args.max_chunks:
        all_chunks = all_chunks[:args.max_chunks]

    if not all_chunks:
        print("No chunks to process.")
        return

    print(f"Loaded {len(all_chunks)} chunks from {chunks_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1
    phase1_records = run_phase1(
        chunks=all_chunks,
        checkpoint_path=phase1_path,
        args=args,
        run_id=run_id,
        model_name=args.model or settings.gemini_model,
    )

    # Phase 2
    model_name = args.model or settings.gemini_model
    annotations = run_phase2(
        phase1_records=phase1_records,
        args=args,
        run_id=run_id,
        model_name=model_name,
        downstream_threshold=downstream_threshold,
    )

    # Write final annotations
    with annotations_path.open("w", encoding="utf-8") as f:
        for record in annotations:
            f.write(json.dumps(record) + "\n")

    print(f"\nDone. Wrote {len(annotations)} annotations to {annotations_path}")


if __name__ == "__main__":
    main()
