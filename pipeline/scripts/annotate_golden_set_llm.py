#!/usr/bin/env python3
"""LLM classifier for AI-mention chunks (multi-step, JSONL output)."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Ensure src is importable
import sys

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from src.classifiers import (  # noqa: E402
    AdoptionTypeClassifier,
    MentionTypeClassifier,
    RiskClassifier,
    VendorClassifier,
)
from src.config import get_settings  # noqa: E402
from src.classifiers.risk_classifier import RISK_CATEGORIES  # noqa: E402
from src.utils.prompt_loader import get_prompt_template  # noqa: E402
from src.utils.validation import validate_classification_response  # noqa: E402

settings = get_settings()


PROMPT_PREAMBLE = (
    "You are labeling AI-related mentions in UK annual report excerpts. "
    "Follow the label schema exactly, use only allowed keys, and return JSON only. "
    "Do not invent facts. If unsure, return low confidence scores. "
    "If there is no AI-related signal, return empty labels."
)


class GoldenMentionTypeClassifier(MentionTypeClassifier):
    def get_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        return f"{PROMPT_PREAMBLE}\n\n{super().get_prompt(text, metadata)}"


class GoldenAdoptionTypeClassifier(AdoptionTypeClassifier):
    def get_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        return f"{PROMPT_PREAMBLE}\n\n{super().get_prompt(text, metadata)}"


class GoldenRiskClassifier(RiskClassifier):
    def get_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        return f"{PROMPT_PREAMBLE}\n\n{super().get_prompt(text, metadata)}"


class GoldenVendorClassifier(VendorClassifier):
    def get_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        return f"{PROMPT_PREAMBLE}\n\n{super().get_prompt(text, metadata)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM annotation for AI-mention chunks.")
    parser.add_argument(
        "--chunks",
        type=Path,
        required=True,
        help="Path to chunks.jsonl or chunks_gemma.jsonl",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id override (defaults to inferred from chunks path).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/golden_set/llm"),
        help="Output directory for annotations and progress files.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output and skip already processed chunk_ids.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Stop after processing this many chunks (0 = no limit).",
    )
    parser.add_argument(
        "--mention-threshold",
        type=float,
        default=0.1,
        help="Confidence threshold for mention type inclusion.",
    )
    parser.add_argument(
        "--downstream-threshold",
        type=float,
        default=None,
        help="Confidence threshold for downstream classifiers.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Retry attempts if LLM returns invalid JSON for a classifier.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=150,
        help="Batch size for OpenRouter requests (default: 150).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Max concurrent batch requests per stage.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Max characters per chunk to send to the LLM.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override Gemini model name for classification.",
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        help="Route classifier calls through OpenRouter.",
    )
    return parser.parse_args()


def infer_run_id(chunks_path: Path) -> str:
    parts = chunks_path.as_posix().split("/")
    if "processed" in parts:
        idx = parts.index("processed")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown-run"


def load_processed_chunk_ids(path: Path) -> set:
    if not path.exists():
        return set()
    chunk_ids = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = obj.get("chunk_id")
            if chunk_id:
                chunk_ids.add(chunk_id)
    return chunk_ids


def load_chunks(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def classify_with_validation(
    classifier,
    text: str,
    metadata: Dict[str, Any],
    source_file: str,
    max_attempts: int,
) -> Tuple[Dict[str, Any], List[str]]:
    last_messages: List[str] = []
    for attempt in range(max_attempts):
        result = classifier.classify(text, metadata, source_file)
        payload = result.classification or {}
        ok, messages = validate_classification_response(payload, classifier.CLASSIFIER_TYPE)
        if ok:
            return payload, []
        last_messages = messages
        if attempt < max_attempts - 1:
            continue
    return payload, last_messages


def trim_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n\n[...truncated...]\n\n" + text[-tail:]


def call_openrouter(prompt: str, model: str, timeout: int = 120) -> str:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    url = f"{settings.openrouter_base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": settings.max_tokens,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text[:500]}")
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def build_batch_items(chunks: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    items = []
    for chunk in chunks:
        text = trim_text(chunk.get("chunk_text") or "", max_chars)
        items.append({
            "chunk_id": chunk.get("chunk_id"),
            "text": text,
        })
    return items


def build_batch_prompt(key: str, items: List[Dict[str, Any]]) -> str:
    template = get_prompt_template(key)
    payload = json.dumps(items, ensure_ascii=False)
    if key == "risk_batch":
        category_descriptions = "\n".join([
            f"- **{key}**: {val['name']} - {val['description']}"
            for key, val in RISK_CATEGORIES.items()
        ])
        return template.format(
            items=payload,
            risk_categories=category_descriptions,
            risk_keys=list(RISK_CATEGORIES.keys()),
        )
    return template.format(items=payload)


def parse_batch_response(
    response_text: str,
    classifier_type: str,
) -> Dict[str, Dict[str, Any]]:
    parsed = json.loads(response_text)
    if not isinstance(parsed, list):
        raise ValueError("Batch response is not a JSON array.")
    results: Dict[str, Dict[str, Any]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id")
        if not chunk_id:
            continue
        payload = {k: v for k, v in item.items() if k != "chunk_id"}
        results[str(chunk_id)] = payload
    return results


def run_batch_classifier(
    key: str,
    classifier_type: str,
    chunks: List[Dict[str, Any]],
    model: str,
    max_attempts: int,
    max_chars: int,
) -> Dict[str, Dict[str, Any]]:
    items = build_batch_items(chunks, max_chars)
    prompt = build_batch_prompt(key, items)
    last_error = None
    for attempt in range(max_attempts):
        try:
            response_text = call_openrouter(prompt, model)
            results = parse_batch_response(response_text, classifier_type)
            return results
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                continue
    if len(chunks) > 1:
        mid = len(chunks) // 2
        left = run_batch_classifier(key, classifier_type, chunks[:mid], model, max_attempts, max_chars)
        right = run_batch_classifier(key, classifier_type, chunks[mid:], model, max_attempts, max_chars)
        merged = {**left, **right}
        return merged
    raise RuntimeError(f"Batch {classifier_type} failed: {last_error}")


def run_batches_parallel(
    key: str,
    classifier_type: str,
    chunks: List[Dict[str, Any]],
    model: str,
    max_attempts: int,
    batch_size: int,
    max_concurrent: int,
    max_chars: int,
) -> Dict[str, Dict[str, Any]]:
    batches = []
    for i in range(0, len(chunks), batch_size):
        batches.append(chunks[i:i + batch_size])

    results: Dict[str, Dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_map = {
            executor.submit(
                run_batch_classifier,
                key,
                classifier_type,
                batch,
                model,
                max_attempts,
                max_chars,
            ): batch
            for batch in batches
        }
        for future in concurrent.futures.as_completed(future_map):
            batch_results = future.result()
            results.update(batch_results)
    return results


def build_record(
    chunk: Dict[str, Any],
    run_id: str,
    mention_payload: Dict[str, Any],
    adoption_payload: Optional[Dict[str, Any]],
    risk_payload: Optional[Dict[str, Any]],
    vendor_payload: Optional[Dict[str, Any]],
    mention_threshold: float,
    downstream_threshold: float,
) -> Dict[str, Any]:
    mention_types = mention_payload.get("mention_types", [])
    mention_confidences = mention_payload.get("confidence_scores", {}) or {}
    if not isinstance(mention_types, list):
        mention_types = []

    if not mention_types:
        mention_types = [
            tag for tag, score in mention_confidences.items()
            if isinstance(score, (int, float)) and score >= mention_threshold
        ]

    if not mention_types:
        mention_types = ["none"]

    adoption_types: List[str] = []
    adoption_confidence = None
    adoption_confidences = {}
    if adoption_payload:
        adoption_confidences = adoption_payload.get("adoption_confidences", {}) or {}
        adoption_types = [
            key for key, score in adoption_confidences.items()
            if isinstance(score, (int, float)) and score >= downstream_threshold
        ]
        if not adoption_types:
            adoption_types = ["none"]
        valid_scores = [
            score for score in adoption_confidences.values()
            if isinstance(score, (int, float))
        ]
        adoption_confidence = max(valid_scores) if valid_scores else 0.0

    risk_taxonomy: List[str] = []
    risk_substantiveness = None
    risk_confidences = {}
    if risk_payload:
        risk_taxonomy = risk_payload.get("risk_types", []) or []
        risk_confidences = risk_payload.get("confidence_scores", {}) or {}
        if risk_taxonomy:
            risk_substantiveness = risk_payload.get("substantiveness_score", None)

    vendor_tags: List[str] = []
    vendor_other = None
    vendor_confidences = {}
    if vendor_payload:
        vendor_confidences = vendor_payload.get("vendor_confidences", {}) or {}
        vendor_tags = [
            key for key, score in vendor_confidences.items()
            if isinstance(score, (int, float)) and score >= downstream_threshold
        ]
        other_vendor = vendor_payload.get("other_vendor") or ""
        if other_vendor:
            vendor_other = other_vendor
            if "other" not in vendor_tags:
                vendor_tags.append("other")

    record = {
        "annotation_id": f"llm-{run_id}-{chunk.get('chunk_id')}",
        "run_id": run_id,
        "chunk_id": chunk.get("chunk_id"),
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
        "risk_substantiveness": risk_substantiveness,
        "vendor_tags": vendor_tags,
        "vendor_other": vendor_other,
        "llm_details": {
            "model": settings.gemini_model,
            "mention_confidences": mention_confidences,
            "adoption_confidences": adoption_confidences,
            "risk_confidences": risk_confidences,
            "vendor_confidences": vendor_confidences,
        },
    }
    return record


def main() -> None:
    args = parse_args()
    chunks_path = args.chunks
    run_id = args.run_id or infer_run_id(chunks_path)
    output_dir = args.output_dir
    annotations_path = output_dir / "annotations.jsonl"
    progress_path = output_dir / "progress.json"

    downstream_threshold = (
        args.downstream_threshold
        if args.downstream_threshold is not None
        else settings.downstream_confidence_threshold
    )

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file at {chunks_path}")

    processed_ids = load_processed_chunk_ids(annotations_path) if args.append else set()

    if not args.openrouter:
        raise RuntimeError("Batch mode requires --openrouter.")

    model_name = args.model or settings.gemini_model

    mention_classifier = GoldenMentionTypeClassifier(
        run_id=run_id,
        model_name=model_name,
        use_openrouter=args.openrouter,
    )
    adoption_classifier = GoldenAdoptionTypeClassifier(
        run_id=run_id,
        model_name=model_name,
        use_openrouter=args.openrouter,
    )
    risk_classifier = GoldenRiskClassifier(
        run_id=run_id,
        model_name=model_name,
        use_openrouter=args.openrouter,
    )
    vendor_classifier = GoldenVendorClassifier(
        run_id=run_id,
        model_name=model_name,
        use_openrouter=args.openrouter,
    )

    chunks = []
    for chunk in load_chunks(chunks_path):
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            continue
        if chunk_id in processed_ids:
            continue
        text = chunk.get("chunk_text") or ""
        if not text.strip():
            continue
        chunks.append(chunk)
        if args.max_chunks and len(chunks) >= args.max_chunks:
            break

    if not chunks:
        print("No chunks to process.")
        return

    def batch_iter(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
        for i in range(0, len(items), size):
            yield items[i:i + size]

    mention_results: Dict[str, Dict[str, Any]] = {}
    adoption_results: Dict[str, Dict[str, Any]] = {}
    risk_results: Dict[str, Dict[str, Any]] = {}
    vendor_results: Dict[str, Dict[str, Any]] = {}

    # Stage 1: mention types (batched + parallel)
    mention_payloads = run_batches_parallel(
        key="mention_type_batch",
        classifier_type="mention_type",
        chunks=chunks,
        model=model_name,
        max_attempts=args.max_attempts,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        max_chars=args.max_chars,
    )
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id"))
        payload = mention_payloads.get(chunk_id)
        if payload:
            ok, _ = validate_classification_response(payload, "mention_type")
            if ok:
                mention_results[chunk_id] = payload
                continue
        metadata = {
            "firm_id": chunk.get("company_id") or "unknown",
            "firm_name": chunk.get("company_name") or "unknown",
            "report_year": chunk.get("report_year") or 0,
            "sector": chunk.get("sector") or "Unknown",
            "report_section": ", ".join(chunk.get("report_sections") or []),
        }
        mention_payload, _ = classify_with_validation(
            mention_classifier,
            trim_text(chunk.get("chunk_text") or "", args.max_chars),
            metadata,
            str(chunks_path),
            args.max_attempts,
        )
        mention_results[chunk_id] = mention_payload

    # Stage 2: route chunks
    adoption_chunks = []
    risk_chunks = []
    vendor_chunks = []
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id"))
        mention_payload = mention_results.get(chunk_id, {})
        mention_confidences = mention_payload.get("confidence_scores", {}) or {}
        mention_types = mention_payload.get("mention_types", [])
        if not isinstance(mention_types, list):
            mention_types = []
        if not mention_types:
            mention_types = [
                tag for tag, score in mention_confidences.items()
                if isinstance(score, (int, float)) and score >= args.mention_threshold
            ]
        if "adoption" in mention_types and mention_confidences.get("adoption", 0.0) >= downstream_threshold:
            adoption_chunks.append(chunk)
        if "risk" in mention_types and mention_confidences.get("risk", 0.0) >= downstream_threshold:
            risk_chunks.append(chunk)
        if "vendor" in mention_types and mention_confidences.get("vendor", 0.0) >= downstream_threshold:
            vendor_chunks.append(chunk)

    # Stage 3: adoption batch (batched + parallel)
    if adoption_chunks:
        adoption_payloads = run_batches_parallel(
            key="adoption_type_batch",
            classifier_type="adoption",
            chunks=adoption_chunks,
            model=model_name,
            max_attempts=args.max_attempts,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            max_chars=args.max_chars,
        )
        for chunk in adoption_chunks:
            chunk_id = str(chunk.get("chunk_id"))
            payload = adoption_payloads.get(chunk_id)
            if payload:
                ok, _ = validate_classification_response(payload, "adoption")
                if ok:
                    adoption_results[chunk_id] = payload
                    continue
            metadata = {
                "firm_id": chunk.get("company_id") or "unknown",
                "firm_name": chunk.get("company_name") or "unknown",
                "report_year": chunk.get("report_year") or 0,
                "sector": chunk.get("sector") or "Unknown",
            }
            adoption_payload, _ = classify_with_validation(
                adoption_classifier,
                trim_text(chunk.get("chunk_text") or "", args.max_chars),
                metadata,
                str(chunks_path),
                args.max_attempts,
            )
            adoption_results[chunk_id] = adoption_payload

    # Stage 4: risk batch (batched + parallel)
    if risk_chunks:
        risk_payloads = run_batches_parallel(
            key="risk_batch",
            classifier_type="risk",
            chunks=risk_chunks,
            model=model_name,
            max_attempts=args.max_attempts,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            max_chars=args.max_chars,
        )
        for chunk in risk_chunks:
            chunk_id = str(chunk.get("chunk_id"))
            payload = risk_payloads.get(chunk_id)
            if payload:
                ok, _ = validate_classification_response(payload, "risk")
                if ok:
                    risk_results[chunk_id] = payload
                    continue
            metadata = {
                "firm_id": chunk.get("company_id") or "unknown",
                "firm_name": chunk.get("company_name") or "unknown",
                "report_year": chunk.get("report_year") or 0,
                "sector": chunk.get("sector") or "Unknown",
            }
            risk_payload, _ = classify_with_validation(
                risk_classifier,
                trim_text(chunk.get("chunk_text") or "", args.max_chars),
                metadata,
                str(chunks_path),
                args.max_attempts,
            )
            risk_results[chunk_id] = risk_payload

    # Stage 5: vendor batch (batched + parallel)
    if vendor_chunks:
        vendor_payloads = run_batches_parallel(
            key="vendor_batch",
            classifier_type="vendor",
            chunks=vendor_chunks,
            model=model_name,
            max_attempts=args.max_attempts,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            max_chars=args.max_chars,
        )
        for chunk in vendor_chunks:
            chunk_id = str(chunk.get("chunk_id"))
            payload = vendor_payloads.get(chunk_id)
            if payload:
                ok, _ = validate_classification_response(payload, "vendor")
                if ok:
                    vendor_results[chunk_id] = payload
                    continue
            metadata = {
                "firm_id": chunk.get("company_id") or "unknown",
                "firm_name": chunk.get("company_name") or "unknown",
                "report_year": chunk.get("report_year") or 0,
                "sector": chunk.get("sector") or "Unknown",
            }
            vendor_payload, _ = classify_with_validation(
                vendor_classifier,
                trim_text(chunk.get("chunk_text") or "", args.max_chars),
                metadata,
                str(chunks_path),
                args.max_attempts,
            )
            vendor_results[chunk_id] = vendor_payload

    output_dir.mkdir(parents=True, exist_ok=True)
    with annotations_path.open("a", encoding="utf-8") as f:
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id"))
            record = build_record(
                chunk=chunk,
                run_id=run_id,
                mention_payload=mention_results.get(chunk_id, {}),
                adoption_payload=adoption_results.get(chunk_id),
                risk_payload=risk_results.get(chunk_id),
                vendor_payload=vendor_results.get(chunk_id),
                mention_threshold=args.mention_threshold,
                downstream_threshold=downstream_threshold,
            )
            f.write(json.dumps(record) + "\n")
            processed_ids.add(chunk_id)

    progress = {
        "run_id": run_id,
        "last_chunk_id": chunks[-1].get("chunk_id"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "total": len(chunks),
            "adoption": len(adoption_chunks),
            "risk": len(risk_chunks),
            "vendor": len(vendor_chunks),
        },
    }
    progress_path.write_text(json.dumps(progress, indent=2))

    print(f"Finished. Annotated {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
