"""Helper utilities for classifier workbench scripts."""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import get_settings


def load_chunks(path, limit: int = None) -> list[dict]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(path) as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                if limit and len(chunks) >= limit:
                    break
    return chunks


def run_mention_classification(
    chunks: list[dict],
    mention_clf,
    adoption_clf=None,
    risk_clf=None,
    vendor_clf=None,
    *,
    run_downstream: bool = True,
    run_vendor_if_mentioned: bool = True,
    confidence_threshold: float = 0.0,
    tqdm_func=None,
) -> list[tuple[dict, dict]]:
    """Run mention classifier (and optional downstream classifiers) over chunks."""
    progress = tqdm_func or (lambda x, **_: x)
    results = []

    for i, chunk in enumerate(progress(chunks, desc="Mention type classification")):
        print(f"\n[{i+1}/{len(chunks)}] Processing: {chunk['company_name']} ({chunk['report_year']})")

        metadata = {
            "firm_id": chunk.get("company_id", "Unknown"),
            "firm_name": chunk.get("company_name", "Unknown"),
            "report_year": chunk.get("report_year", 0),
            "sector": "Unknown",
            "report_section": chunk.get("report_sections", ["Unknown"])[0]
            if chunk.get("report_sections")
            else "Unknown",
        }

        result = mention_clf.classify(chunk["chunk_text"], metadata)

        classification = result.classification
        raw_classification = classification
        schema_echo = _looks_like_schema(classification)
        if schema_echo:
            print(
                "  WARNING: Model returned a schema-like response; "
                f"treating as empty classification. Keys={list(classification.keys())}"
            )
            classification = None
        mention_types = []
        if isinstance(classification, dict) and "mention_types" in classification:
            mention_types = [
                str(mt.value) if hasattr(mt, "value") else str(mt)
                for mt in classification["mention_types"]
            ]

        llm_result = {
            "chunk_id": chunk["chunk_id"],
            "mention_types": mention_types,
            "confidence": result.confidence_score,
            "reasoning": result.reasoning,
            "schema_echo": schema_echo,
            "raw_classification": raw_classification,
            "adoption_types": [],
            "risk_taxonomy": [],
            "vendor_tags": [],
        }

        if run_downstream:
            downstream_metadata = dict(metadata)
            downstream_metadata["mention_types"] = mention_types

            if "adoption" in mention_types and adoption_clf:
                adoption_result = adoption_clf.classify(chunk["chunk_text"], downstream_metadata)
                adoption_classification = adoption_result.classification
                adoption_types = []
                if isinstance(adoption_classification, dict):
                    adoption_confidences = adoption_classification.get("adoption_confidences", {}) or {}
                    if isinstance(adoption_confidences, dict):
                        adoption_types = [
                            k for k, v in adoption_confidences.items()
                            if isinstance(v, (int, float)) and v > confidence_threshold
                        ]
                llm_result["adoption_types"] = adoption_types
                llm_result["raw_adoption"] = adoption_classification

            if "risk" in mention_types and risk_clf:
                risk_result = risk_clf.classify(chunk["chunk_text"], downstream_metadata)
                risk_classification = risk_result.classification
                risk_types = []
                if isinstance(risk_classification, dict):
                    rt = risk_classification.get("risk_types", []) or []
                    if isinstance(rt, list):
                        risk_types = [
                            str(r.value) if hasattr(r, "value") else str(r)
                            for r in rt
                            if str(r) != "none"
                        ]
                llm_result["risk_taxonomy"] = risk_types
                llm_result["raw_risk"] = risk_classification

            if run_vendor_if_mentioned and "vendor" in mention_types and vendor_clf:
                vendor_result = vendor_clf.classify(chunk["chunk_text"], metadata)
                vendor_classification = vendor_result.classification
                vendor_tags = []
                if isinstance(vendor_classification, dict):
                    vendor_confidences = vendor_classification.get("vendor_confidences", {}) or {}
                    if isinstance(vendor_confidences, dict):
                        vendor_tags = [
                            k for k, v in vendor_confidences.items()
                            if isinstance(v, (int, float)) and v > confidence_threshold
                        ]
                    other_vendor = vendor_classification.get("other_vendor")
                    if other_vendor:
                        vendor_tags.append(f"other:{other_vendor}")
                llm_result["vendor_tags"] = vendor_tags
                llm_result["raw_vendor"] = vendor_classification

        results.append((chunk, llm_result))

        print(f"  Result: {mention_types} (conf={result.confidence_score:.2f})")
        print(f"  Reasoning: {result.reasoning[:100] if result.reasoning else 'N/A'}...")

    return results


def save_phase1_results(
    results: list[tuple[dict, dict]],
    run_id: str,
    output_path: Path | None = None,
) -> Path:
    """Save Phase 1 results to JSONL under data/processed/<run_id>/classifications/."""
    settings = get_settings()
    if output_path is None:
        output_dir = settings.processed_dir / run_id / "classifications"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "phase1_results.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk, llm_result in results:
            record = {
                "run_id": run_id,
                "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "chunk": chunk,
                "llm_result": llm_result,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def _looks_like_schema(classification: object) -> bool:
    """Heuristic check for schema-echo responses."""
    if not isinstance(classification, dict):
        return False

    keys = set(classification.keys())
    if {"title", "description", "type", "properties"}.issubset(keys):
        return True
    if "properties" in keys and "type" in keys and isinstance(classification.get("properties"), dict):
        return True
    for v in classification.values():
        if isinstance(v, str) and "schema" in v.lower():
            return True
    return False


def compare_one(chunk: dict, llm_result: dict) -> dict:
    """Compare human annotation with LLM result."""
    def to_set(value):
        if value is None:
            return set()
        if isinstance(value, list):
            return set(value)
        return {value}

    human = {
        "mention_types": to_set(chunk.get("mention_types")),
        "adoption_types": to_set(chunk.get("adoption_types")),
        "risk_taxonomy": to_set(chunk.get("risk_taxonomy")),
        "vendor_tags": to_set(chunk.get("vendor_tags")),
        "risk_substantiveness": to_set(chunk.get("risk_substantiveness")),
    }
    llm = {
        "mention_types": to_set(llm_result.get("mention_types")),
        "adoption_types": to_set(llm_result.get("adoption_types")),
        "risk_taxonomy": to_set(llm_result.get("risk_taxonomy")),
        "vendor_tags": to_set(llm_result.get("vendor_tags")),
        "risk_substantiveness": to_set(llm_result.get("risk_substantiveness")),
    }

    def match_status(h: set, l: set) -> str:
        if h == l:
            return "EXACT"
        elif h & l:
            return "PARTIAL"
        else:
            return "DIFF"

    return {
        "chunk_id": chunk["chunk_id"],
        "company": chunk["company_name"],
        "mention_match": match_status(human["mention_types"], llm["mention_types"]),
        "human_mention": list(human["mention_types"]),
        "llm_mention": list(llm["mention_types"]),
        "human_adoption": list(human["adoption_types"]),
        "llm_adoption": list(llm["adoption_types"]),
    }


def print_comparison(comp: dict, show_text: bool = False, text: str = ""):
    """Pretty print a comparison result."""
    match_symbol = {"EXACT": "✓", "PARTIAL": "~", "DIFF": "✗"}

    print(f"\n{'='*80}")
    print(f"Chunk: {comp['chunk_id'][:40]}...")
    print(f"Company: {comp['company']}")
    print(f"{'='*80}")

    if show_text and text:
        print(f"\nText:\n{text[:500]}...")

    print(f"\nMention Types: {match_symbol.get(comp['mention_match'], '?')} {comp['mention_match']}")
    print(f"  Human: {comp['human_mention']}")
    print(f"  LLM:   {comp['llm_mention']}")

    if comp.get("human_adoption") or comp.get("llm_adoption"):
        print(f"\nAdoption Types:")
        print(f"  Human: {comp['human_adoption']}")
        print(f"  LLM:   {comp['llm_adoption']}")


def inspect_chunk(results: list[tuple[dict, dict]], index: int):
    """Inspect a single chunk in detail with human vs LLM comparison."""
    if index >= len(results):
        print(f"Index {index} out of range (0-{len(results)-1})")
        return

    chunk, llm_result = results[index]

    print("="*80)
    print(f"CHUNK {index}")
    print("="*80)
    print(f"Company: {chunk['company_name']}")
    print(f"Year: {chunk['report_year']}")
    print(f"Section: {chunk.get('report_sections', ['?'])[0]}")
    print(f"Keywords: {chunk.get('matched_keywords', [])}")

    print(f"\n--- TEXT ---")
    print(chunk["chunk_text"])

    print(f"\n--- HUMAN ANNOTATION ---")
    print(f"Mention types: {chunk.get('mention_types', [])}")
    print(f"Adoption types: {chunk.get('adoption_types', [])}")
    print(f"Risk taxonomy: {chunk.get('risk_taxonomy', [])}")

    print(f"\n--- LLM ANNOTATION ---")
    print(f"Mention types: {llm_result['mention_types']}")
    print(f"Confidence: {llm_result['confidence']:.2f}" if llm_result['confidence'] else "Confidence: N/A")
    confidence_scores = (
        llm_result.get("raw_classification", {}) or {}
    ).get("confidence_scores", {})
    if confidence_scores:
        print("Confidence scores:")
        for k, v in confidence_scores.items():
            if isinstance(v, (int, float)):
                print(f"  - {k}: {v:.2f}")
            else:
                print(f"  - {k}: {v}")
    else:
        print("Confidence scores: N/A")
    print(f"Reasoning: {llm_result.get('reasoning', 'N/A')}")
