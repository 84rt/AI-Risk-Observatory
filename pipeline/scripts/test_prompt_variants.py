#!/usr/bin/env python3
"""Test prompt variants against classifier_dev test chunks.

Runs mention_type, adoption_type, and risk prompts in 3 reasoning variants
(full, no_reasoning, limited_reasoning) against the test chunks and saves
results in the standard annotation schema as llm_{variant}.jsonl.

Usage:
    python scripts/test_prompt_variants.py --model gemini-2.0-flash
    python scripts/test_prompt_variants.py --model google/gemini-2.0-flash-001 --openrouter
    python scripts/test_prompt_variants.py --variant-label my_v1,my_v2,my_v3
"""

from __future__ import annotations

import argparse
import json
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from src.config import get_settings
from src.utils.prompt_loader import get_prompt_template
from src.classifiers.risk_classifier import RISK_CATEGORIES
from src.utils.validation import parse_json_response

settings = get_settings()

# Prompt variant suffixes to test
VARIANTS = ["", "_no_reasoning", "_limited_reasoning"]
VARIANT_LABELS = ["full", "no_reasoning", "limited_reasoning"]

# Downstream confidence threshold
DOWNSTREAM_THRESHOLD = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test prompt variants on dev chunks.")
    parser.add_argument("--model", type=str, default=None, help="Model name override.")
    parser.add_argument("--openrouter", action="store_true", help="Use OpenRouter API.")
    parser.add_argument("--chunks", type=Path,
                        default=PIPELINE_ROOT.parent / "data" / "classifier_dev" / "test_chunks.jsonl",
                        help="Path to test chunks JSONL.")
    parser.add_argument("--output-dir", type=Path,
                        default=PIPELINE_ROOT.parent / "data" / "classifier_dev",
                        help="Output directory.")
    parser.add_argument("--mention-threshold", type=float, default=0.1,
                        help="Confidence threshold for mention tags.")
    parser.add_argument("--variant-label", type=str, default=None,
                        help="Comma-separated labels for output files (default: auto from suffix).")
    return parser.parse_args()


def call_llm(prompt: str, model_name: str, use_openrouter: bool) -> str:
    """Call LLM and return raw response text."""
    if use_openrouter:
        import requests
        url = f"{settings.openrouter_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": settings.max_tokens,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text[:500]}")
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    else:
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        generation_config = {
            "temperature": 0.0,
            "max_output_tokens": settings.max_tokens,
            "response_mime_type": "application/json",
        }
        model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
        response = model.generate_content(prompt)
        return response.text


def build_mention_prompt(template_key: str, chunk: Dict[str, Any]) -> str:
    """Build a mention_type prompt from a chunk."""
    template = get_prompt_template(template_key)
    return template.format(
        firm_name=chunk.get("company_name") or "Unknown",
        sector=chunk.get("sector") or "Unknown",
        report_year=chunk.get("report_year") or "Unknown",
        report_section=", ".join(chunk.get("report_sections") or []),
        text=chunk.get("chunk_text") or "",
    )


def build_adoption_prompt(template_key: str, chunk: Dict[str, Any]) -> str:
    """Build an adoption_type prompt from a chunk."""
    template = get_prompt_template(template_key)
    return template.format(
        firm_name=chunk.get("company_name") or "Unknown",
        sector=chunk.get("sector") or "Unknown",
        report_year=chunk.get("report_year") or "Unknown",
        text=chunk.get("chunk_text") or "",
    )


def build_risk_prompt(template_key: str, chunk: Dict[str, Any]) -> str:
    """Build a risk prompt from a chunk."""
    template = get_prompt_template(template_key)
    category_descriptions = "\n".join([
        f"- **{key}**: {val['name']} - {val['description']}"
        for key, val in RISK_CATEGORIES.items()
    ])
    return template.format(
        firm_name=chunk.get("company_name") or "Unknown",
        sector=chunk.get("sector") or "Unknown",
        report_year=chunk.get("report_year") or "Unknown",
        text=chunk.get("chunk_text") or "",
        risk_categories=category_descriptions,
        risk_keys=list(RISK_CATEGORIES.keys()),
    )


def classify_chunk(prompt: str, model_name: str, use_openrouter: bool) -> Dict[str, Any]:
    """Call LLM and parse JSON response."""
    try:
        raw = call_llm(prompt, model_name, use_openrouter)
        parsed, error = parse_json_response(raw)
        if error or parsed is None:
            return {"_error": error or "Empty response", "_raw": raw}
        return parsed
    except Exception as e:
        return {"_error": str(e)}


def main() -> None:
    args = parse_args()
    model_name = args.model or settings.gemini_model

    # Load chunks
    chunks = []
    with args.chunks.open() as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} test chunks")
    print(f"Model: {model_name} | OpenRouter: {args.openrouter}")
    print(f"Variants: {VARIANT_LABELS}")
    print()

    # Results: {variant_label: {chunk_id: {phase: payload}}}
    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for variant_suffix, variant_label in zip(VARIANTS, VARIANT_LABELS):
        print(f"{'='*60}")
        print(f"VARIANT: {variant_label}")
        print(f"{'='*60}")

        variant_results: Dict[str, Dict[str, Any]] = {}

        # Phase 1: Mention type
        mention_key = f"mention_type{variant_suffix}"
        print(f"\n  [Phase 1] Prompt key: {mention_key}")

        for chunk in chunks:
            cid = chunk["chunk_id"]
            short_id = cid.split("chunk-")[1][:12]
            prompt = build_mention_prompt(mention_key, chunk)
            result = classify_chunk(prompt, model_name, args.openrouter)

            mention_confidences = result.get("confidence_scores", {})
            mention_types = [
                tag for tag, score in mention_confidences.items()
                if isinstance(score, (int, float)) and score >= args.mention_threshold
            ]

            variant_results[cid] = {
                "mention": result,
                "mention_types": mention_types,
                "mention_confidences": mention_confidences,
            }

            print(f"    {short_id} | {chunk.get('company_name', '')[:20]:20s} | "
                  f"types={mention_types} | conf={mention_confidences}")
            time.sleep(0.5)  # rate limiting

        # Phase 2: Downstream classifiers
        adoption_key = f"adoption_type{variant_suffix}"
        risk_key = f"risk{variant_suffix}"

        # Adoption
        print(f"\n  [Phase 2a] Prompt key: {adoption_key}")
        for chunk in chunks:
            cid = chunk["chunk_id"]
            short_id = cid.split("chunk-")[1][:12]
            mc = variant_results[cid]["mention_confidences"]
            adoption_conf = mc.get("adoption", 0.0)
            if isinstance(adoption_conf, (int, float)) and adoption_conf >= DOWNSTREAM_THRESHOLD:
                prompt = build_adoption_prompt(adoption_key, chunk)
                result = classify_chunk(prompt, model_name, args.openrouter)
                variant_results[cid]["adoption"] = result
                ac = result.get("adoption_confidences", {})
                print(f"    {short_id} | adoption_conf={ac}")
                time.sleep(0.5)
            else:
                print(f"    {short_id} | SKIPPED (adoption conf={adoption_conf:.2f})")

        # Risk
        print(f"\n  [Phase 2b] Prompt key: {risk_key}")
        for chunk in chunks:
            cid = chunk["chunk_id"]
            short_id = cid.split("chunk-")[1][:12]
            mc = variant_results[cid]["mention_confidences"]
            risk_conf = mc.get("risk", 0.0)
            if isinstance(risk_conf, (int, float)) and risk_conf >= DOWNSTREAM_THRESHOLD:
                prompt = build_risk_prompt(risk_key, chunk)
                result = classify_chunk(prompt, model_name, args.openrouter)
                variant_results[cid]["risk"] = result
                rt = result.get("risk_types", [])
                print(f"    {short_id} | risk_types={rt}")
                time.sleep(0.5)
            else:
                print(f"    {short_id} | SKIPPED (risk conf={risk_conf:.2f})")

        all_results[variant_label] = variant_results
        print()

    # Save results in standard annotation schema per variant
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output labels
    if args.variant_label:
        output_labels = [l.strip() for l in args.variant_label.split(",")]
        if len(output_labels) != len(VARIANT_LABELS):
            print(f"WARNING: --variant-label count ({len(output_labels)}) != variants ({len(VARIANT_LABELS)}), using defaults")
            output_labels = VARIANT_LABELS
    else:
        output_labels = VARIANT_LABELS

    for variant_label, output_label in zip(VARIANT_LABELS, output_labels):
        variant_results = all_results[variant_label]
        out_path = args.output_dir / f"llm_{output_label}.jsonl"
        with out_path.open("w") as f:
            for cid, data in variant_results.items():
                mention_confidences = data.get("mention_confidences", {})
                mention_types = data.get("mention_types", [])

                # Adoption
                adoption_raw = data.get("adoption", {})
                adoption_confidences = adoption_raw.get("adoption_confidences", {})
                adoption_types = [
                    k for k, v in adoption_confidences.items()
                    if isinstance(v, (int, float)) and v >= args.mention_threshold
                ]

                # Risk
                risk_raw = data.get("risk", {})
                risk_confidences = risk_raw.get("confidence_scores", {})
                risk_taxonomy = [
                    k for k, v in risk_confidences.items()
                    if isinstance(v, (int, float)) and v >= args.mention_threshold
                ]
                risk_substantiveness = risk_raw.get("substantiveness_score")

                # Vendor (not run in variant tests)
                vendor_tags: List[str] = []

                record = {
                    "chunk_id": cid,
                    "mention_types": mention_types,
                    "mention_confidences": mention_confidences,
                    "adoption_types": adoption_types,
                    "adoption_confidences": adoption_confidences,
                    "risk_taxonomy": risk_taxonomy,
                    "risk_confidences": risk_confidences,
                    "risk_substantiveness": risk_substantiveness,
                    "vendor_tags": vendor_tags,
                    "vendor_other": None,
                }
                f.write(json.dumps(record) + "\n")
        print(f"Saved: {out_path}")

    # Build comparison CSV
    csv_path = args.output_dir / "variant_comparison.csv"
    rows = []
    for chunk in chunks:
        cid = chunk["chunk_id"]
        row = {
            "chunk_id": cid,
            "company": chunk.get("company_name", ""),
            "year": chunk.get("report_year", ""),
        }
        for variant_label in VARIANT_LABELS:
            vr = all_results[variant_label].get(cid, {})
            mc = vr.get("mention_confidences", {})
            mt = vr.get("mention_types", [])
            ac = vr.get("adoption", {}).get("adoption_confidences", {})
            rt = vr.get("risk", {}).get("risk_types", [])
            rc = vr.get("risk", {}).get("confidence_scores", {})

            prefix = f"llm_{variant_label}"
            row[f"{prefix}_mention_types"] = "|".join(sorted(mt))
            row[f"{prefix}_mention_conf"] = json.dumps(mc)
            row[f"{prefix}_adoption_conf"] = json.dumps(ac)
            row[f"{prefix}_risk_types"] = "|".join(sorted(rt)) if rt else ""
            row[f"{prefix}_risk_conf"] = json.dumps(rc)

        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved comparison: {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
