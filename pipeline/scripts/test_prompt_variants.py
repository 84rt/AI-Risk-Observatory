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
from src.utils.prompt_loader import get_prompt_messages
from src.classifiers.risk_classifier import RISK_CATEGORIES
from src.utils.validation import parse_json_response

settings = get_settings()

# Prompt variant suffixes to test
REASONING_POLICIES = ["short", "none", "limited"]
VARIANT_LABELS = ["full", "no_reasoning", "limited_reasoning"]
REASONING_LABELS = {
    "short": "full",
    "none": "no_reasoning",
    "limited": "limited_reasoning",
}

# Downstream confidence threshold
DOWNSTREAM_THRESHOLD = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test prompt variants on dev chunks.")
    parser.add_argument("--model", type=str, default=None, help="Model name override (single model).")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model slugs to test (overrides --model).")
    parser.add_argument("--openrouter", action="store_true", help="Use OpenRouter API.")
    parser.add_argument("--chunks", type=Path,
                        default=PIPELINE_ROOT.parent / "data" / "classifier_dev" / "test_chunks.jsonl",
                        help="Path to test chunks JSONL.")
    parser.add_argument("--output-dir", type=Path,
                        default=PIPELINE_ROOT.parent / "data" / "classifier_dev",
                        help="Output directory.")
    parser.add_argument("--mention-threshold", type=float, default=0.2,
                        help="Confidence threshold for mention tags.")
    parser.add_argument("--best-of-n", type=int, default=1,
                        help="Run each prompt N times per chunk to measure consistency.")
    parser.add_argument("--reasoning-policies", type=str, default=None,
                        help="Comma-separated reasoning policies: short,none,limited.")
    parser.add_argument("--variant-label", type=str, default=None,
                        help="Comma-separated labels for output files (default: auto from suffix).")
    return parser.parse_args()


def call_llm(system_prompt: str, user_prompt: str, model_name: str, use_openrouter: bool) -> str:
    """Call LLM and return raw response text."""
    if use_openrouter:
        import requests
        url = f"{settings.openrouter_base_url}/chat/completions"
        system_content = "Return ONLY valid JSON."
        if system_prompt:
            system_content = f"{system_content}\n\n{system_prompt}"
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt},
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
        if system_prompt:
            prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
        else:
            prompt = user_prompt
        response = model.generate_content(prompt)
        return response.text


def build_mention_prompt(template_key: str, chunk: Dict[str, Any], reasoning_policy: str) -> tuple[str, str]:
    """Build a mention_type prompt from a chunk."""
    return get_prompt_messages(
        template_key,
        reasoning_policy=reasoning_policy,
        firm_name=chunk.get("company_name") or "Unknown",
        sector=chunk.get("sector") or "Unknown",
        report_year=chunk.get("report_year") or "Unknown",
        report_section=", ".join(chunk.get("report_sections") or []),
        text=chunk.get("chunk_text") or "",
    )


def build_adoption_prompt(template_key: str, chunk: Dict[str, Any], reasoning_policy: str) -> tuple[str, str]:
    """Build an adoption_type prompt from a chunk."""
    return get_prompt_messages(
        template_key,
        reasoning_policy=reasoning_policy,
        firm_name=chunk.get("company_name") or "Unknown",
        sector=chunk.get("sector") or "Unknown",
        report_year=chunk.get("report_year") or "Unknown",
        text=chunk.get("chunk_text") or "",
    )


def build_risk_prompt(template_key: str, chunk: Dict[str, Any], reasoning_policy: str) -> tuple[str, str]:
    """Build a risk prompt from a chunk."""
    category_descriptions = "\n".join([
        f"- **{key}**: {val['name']} - {val['description']}"
        for key, val in RISK_CATEGORIES.items()
    ])
    return get_prompt_messages(
        template_key,
        reasoning_policy=reasoning_policy,
        firm_name=chunk.get("company_name") or "Unknown",
        sector=chunk.get("sector") or "Unknown",
        report_year=chunk.get("report_year") or "Unknown",
        text=chunk.get("chunk_text") or "",
        risk_categories=category_descriptions,
        risk_keys=list(RISK_CATEGORIES.keys()),
    )


def classify_chunk(system_prompt: str, user_prompt: str, model_name: str, use_openrouter: bool) -> Dict[str, Any]:
    """Call LLM and parse JSON response."""
    try:
        raw = call_llm(system_prompt, user_prompt, model_name, use_openrouter)
        parsed, error = parse_json_response(raw)
        if error or parsed is None:
            return {"_error": error or "Empty response", "_raw": raw}
        return parsed
    except Exception as e:
        return {"_error": str(e)}


def _model_label(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def main() -> None:
    args = parse_args()
    if args.models:
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_list = [args.model or settings.gemini_model]
    if not model_list:
        raise ValueError("No models provided.")

    # Load chunks
    chunks = []
    with args.chunks.open() as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} test chunks")
    print(f"Models: {model_list} | OpenRouter: {args.openrouter}")
    if args.reasoning_policies:
        reasoning_policies = [p.strip() for p in args.reasoning_policies.split(",") if p.strip()]
    else:
        reasoning_policies = REASONING_POLICIES
    if not reasoning_policies:
        raise ValueError("No reasoning policies provided.")
    unknown = [p for p in reasoning_policies if p not in REASONING_LABELS]
    if unknown:
        raise ValueError(f"Unknown reasoning policies: {unknown}")
    variant_labels = [REASONING_LABELS[p] for p in reasoning_policies]

    print(f"Variants: {variant_labels}")
    print(f"Best-of-N: {args.best_of_n}")
    print()

    # Results: {model_label: {variant_label: {chunk_id: {phase: payload}}}}
    all_results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}

    for model_name in model_list:
        model_label = _model_label(model_name)
        model_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for reasoning_policy, variant_label in zip(reasoning_policies, variant_labels):
            print(f"{'='*60}")
            print(f"MODEL: {model_name} | VARIANT: {variant_label}")
            print(f"{'='*60}")

            variant_results: Dict[str, Dict[str, Any]] = {}

            # Phase 1: Mention type
            mention_key = "mention_type"
            print(f"\n  [Phase 1] Prompt key: {mention_key}")

            for chunk in chunks:
                cid = chunk["chunk_id"]
                short_id = cid.split("chunk-")[1][:12]
                system_prompt, user_prompt = build_mention_prompt(mention_key, chunk, reasoning_policy)

                runs: List[Dict[str, Any]] = []
                for _ in range(max(1, args.best_of_n)):
                    result = classify_chunk(system_prompt, user_prompt, model_name, args.openrouter)
                    runs.append(result)
                    time.sleep(0.5)  # rate limiting

                result = runs[0]
                mention_confidences = result.get("confidence_scores", {})
                mention_types = [
                    tag for tag, score in mention_confidences.items()
                    if isinstance(score, (int, float)) and score >= args.mention_threshold
                ]

                variant_results[cid] = {
                    "mention": result,
                    "mention_runs": runs if args.best_of_n > 1 else None,
                    "mention_types": mention_types,
                    "mention_confidences": mention_confidences,
                }

                print(f"    {short_id} | {chunk.get('company_name', '')[:20]:20s} | "
                      f"types={mention_types} | conf={mention_confidences}")

            # Phase 2: Downstream classifiers
            adoption_key = "adoption_type"
            risk_key = "risk"

            # Adoption
            print(f"\n  [Phase 2a] Prompt key: {adoption_key}")
            for chunk in chunks:
                cid = chunk["chunk_id"]
                short_id = cid.split("chunk-")[1][:12]
                mc = variant_results[cid]["mention_confidences"]
                adoption_conf = mc.get("adoption", 0.0)
                if isinstance(adoption_conf, (int, float)) and adoption_conf >= DOWNSTREAM_THRESHOLD:
                    system_prompt, user_prompt = build_adoption_prompt(adoption_key, chunk, reasoning_policy)
                    runs = []
                    for _ in range(max(1, args.best_of_n)):
                        result = classify_chunk(system_prompt, user_prompt, model_name, args.openrouter)
                        runs.append(result)
                        time.sleep(0.5)
                    result = runs[0]
                    variant_results[cid]["adoption"] = result
                    variant_results[cid]["adoption_runs"] = runs if args.best_of_n > 1 else None
                    ac = result.get("adoption_signals") or result.get("adoption_confidences", {})
                    print(f"    {short_id} | adoption_signals={ac}")
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
                    system_prompt, user_prompt = build_risk_prompt(risk_key, chunk, reasoning_policy)
                    runs = []
                    for _ in range(max(1, args.best_of_n)):
                        result = classify_chunk(system_prompt, user_prompt, model_name, args.openrouter)
                        runs.append(result)
                        time.sleep(0.5)
                    result = runs[0]
                    variant_results[cid]["risk"] = result
                    variant_results[cid]["risk_runs"] = runs if args.best_of_n > 1 else None
                    rt = result.get("risk_types", [])
                    print(f"    {short_id} | risk_types={rt}")
                else:
                    print(f"    {short_id} | SKIPPED (risk conf={risk_conf:.2f})")

            model_results[variant_label] = variant_results
            print()

        all_results[model_label] = model_results

    # Save results in standard annotation schema per variant
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output labels
    if args.variant_label:
        output_labels = [l.strip() for l in args.variant_label.split(",")]
        if len(output_labels) != len(variant_labels):
            print(f"WARNING: --variant-label count ({len(output_labels)}) != variants ({len(variant_labels)}), using defaults")
            output_labels = variant_labels
    else:
        output_labels = variant_labels

    for model_label, model_results in all_results.items():
        for variant_label, output_label in zip(variant_labels, output_labels):
            variant_results = model_results[variant_label]
            out_path = args.output_dir / f"llm_{model_label}_{output_label}.jsonl"
            with out_path.open("w") as f:
                for cid, data in variant_results.items():
                    mention_confidences = data.get("mention_confidences", {})
                    mention_types = data.get("mention_types", [])

                    # Adoption
                    adoption_raw = data.get("adoption", {})
                    adoption_signals = adoption_raw.get("adoption_signals") or adoption_raw.get("adoption_confidences", {})
                    signal_map = {}
                    if isinstance(adoption_signals, list):
                        for entry in adoption_signals:
                            if isinstance(entry, dict):
                                k = entry.get("type")
                                v = entry.get("signal")
                                if k is not None and isinstance(v, (int, float)):
                                    signal_map[str(k)] = float(v)
                    elif isinstance(adoption_signals, dict):
                        signal_map = {
                            str(k): float(v)
                            for k, v in adoption_signals.items()
                            if isinstance(v, (int, float))
                        }
                    adoption_types = [
                        k for k, v in signal_map.items()
                        if isinstance(v, (int, float)) and v >= args.mention_threshold
                    ]

                    # Risk
                    risk_raw = data.get("risk", {})
                    risk_signals = risk_raw.get("risk_signals") or risk_raw.get("confidence_scores", {})
                    risk_map = {}
                    if isinstance(risk_signals, list):
                        for entry in risk_signals:
                            if isinstance(entry, dict):
                                k = entry.get("type")
                                v = entry.get("signal")
                                if k is not None and isinstance(v, (int, float)):
                                    risk_map[str(k)] = float(v)
                    elif isinstance(risk_signals, dict):
                        risk_map = {
                            str(k): float(v)
                            for k, v in risk_signals.items()
                            if isinstance(v, (int, float))
                        }
                    risk_taxonomy = [
                        k for k, v in risk_map.items()
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
                        "adoption_signals": adoption_signals,
                        "risk_taxonomy": risk_taxonomy,
                        "risk_signals": risk_signals,
                        "risk_substantiveness": risk_substantiveness,
                        "vendor_tags": vendor_tags,
                        "vendor_other": None,
                    }
                    f.write(json.dumps(record) + "\n")
            print(f"Saved: {out_path}")

            if args.best_of_n > 1:
                runs_path = args.output_dir / f"llm_{model_label}_{output_label}_runs.jsonl"
                with runs_path.open("w") as f:
                    for cid, data in variant_results.items():
                        f.write(json.dumps({
                            "chunk_id": cid,
                            "mention_runs": data.get("mention_runs"),
                            "adoption_runs": data.get("adoption_runs"),
                            "risk_runs": data.get("risk_runs"),
                        }) + "\n")
                print(f"Saved: {runs_path}")

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
        for model_label, model_results in all_results.items():
            for variant_label in variant_labels:
                vr = model_results[variant_label].get(cid, {})
                mc = vr.get("mention_confidences", {})
                mt = vr.get("mention_types", [])
                ac = vr.get("adoption", {}).get("adoption_signals") or vr.get("adoption", {}).get("adoption_confidences", {})
                rt = vr.get("risk", {}).get("risk_types", [])
                rc = vr.get("risk", {}).get("risk_signals") or vr.get("risk", {}).get("confidence_scores", {})

                prefix = f"llm_{model_label}_{variant_label}"
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
