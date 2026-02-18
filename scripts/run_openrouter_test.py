#!/usr/bin/env python3
"""Run classifiers against the golden set sequentially (Gemini or OpenRouter).

Benchmarks any model against human-annotated golden set using the same
classifiers, prompts, and output format as run_phase2_classifiers.py.

Gemini models (no '/' in name) use native structured output via the Gemini API.
OpenRouter models (with '/' in name) use prompt-injected schema.

Examples:
    # Gemini 2.5 Flash (native structured output)
    python3 scripts/run_openrouter_test.py --model gemini-2.5-flash --all
    python3 scripts/run_openrouter_test.py --model gemini-2.5-flash --limit 5 --dry-run

    # OpenRouter models
    python3 scripts/run_openrouter_test.py --model z-ai/glm-5 --limit 20
    python3 scripts/run_openrouter_test.py --model z-ai/glm-5 --classifiers risk --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, desc: str | None = None, **kwargs):
        if desc:
            print(desc)
        return iterable

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

# Load .env.local before importing pipeline modules (Settings reads env at import time)
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
RUNS_DIR = REPO_ROOT / "data" / "testbed_runs"

# ---------------------------------------------------------------------------
# Classifier config — mirrors run_phase2_classifiers.py + adds mention_type
# ---------------------------------------------------------------------------

CLASSIFIER_CONFIG: dict[str, dict[str, Any]] = {}  # populated after imports


def _init_classifier_config() -> None:
    """Lazy init so imports happen after env loading."""
    from src.classifiers.mention_type_classifier import MentionTypeClassifier
    from src.classifiers.risk_classifier import RiskClassifier
    from src.classifiers.adoption_type_classifier import AdoptionTypeClassifier
    from src.classifiers.vendor_classifier import VendorClassifier

    CLASSIFIER_CONFIG.update({
        "mention_type": {
            "cls": MentionTypeClassifier,
            "human_field": "mention_types",
            "filter_mention": None,  # all chunks
        },
        "risk": {
            "cls": RiskClassifier,
            "human_field": "risk_taxonomy",
            "filter_mention": "risk",
        },
        "adoption_type": {
            "cls": AdoptionTypeClassifier,
            "human_field": "adoption_types",
            "filter_mention": "adoption",
        },
        "vendor": {
            "cls": VendorClassifier,
            "human_field": "vendor_tags",
            "filter_mention": "vendor",
        },
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run classifiers on golden set sequentially (Gemini or OpenRouter)."
    )
    p.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Model name. Gemini models use native API; names with '/' use OpenRouter.",
    )
    p.add_argument(
        "--use-openrouter",
        action="store_true",
        help="Force OpenRouter mode even for models without '/' in the name.",
    )
    p.add_argument(
        "--classifiers",
        default="mention_type,risk,adoption_type,vendor",
        help="Comma-separated classifiers to run in order.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max chunks per classifier (default: 20).",
    )
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N chunks.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Ignore --limit, run all matching chunks.",
    )
    p.add_argument(
        "--run-suffix",
        default="test",
        help="Suffix for run ID (default: test).",
    )
    p.add_argument(
        "--golden-set",
        type=Path,
        default=GOLDEN_SET_DEFAULT,
        help="Path to golden set JSONL.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model temperature (default: 0.0).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show counts, no API calls.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading & filtering
# ---------------------------------------------------------------------------


def load_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def filter_chunks(
    chunks: list[dict[str, Any]],
    filter_mention: str | None,
    offset: int,
    limit: int,
    use_all: bool,
) -> list[dict[str, Any]]:
    if filter_mention:
        selected = [c for c in chunks if filter_mention in c.get("mention_types", [])]
    else:
        selected = list(chunks)
    if offset > 0:
        selected = selected[offset:]
    if not use_all and limit > 0:
        selected = selected[:limit]
    return selected


# ---------------------------------------------------------------------------
# Normalization helpers (inlined from run_phase2_classifiers.py)
# ---------------------------------------------------------------------------


def normalize_label_token(value: object) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    token = str(value)
    for prefix in ("RiskType.", "AdoptionType."):
        if token.startswith(prefix):
            return token.split(".", 1)[1]
    return token


def _try_parse_json(text: str) -> dict | None:
    """Best-effort JSON parser for occasionally malformed model output."""
    if not text:
        return None

    attempts: list[str] = [text]

    stripped = text.strip()
    if stripped.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        attempts.append(cleaned.strip())

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        attempts.append(stripped[start : end + 1])

    for candidate in list(attempts):
        attempts.append(re.sub(r",\s*([}\]])", r"\1", candidate))

    for candidate in attempts:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Per-classifier label extraction
# ---------------------------------------------------------------------------


def extract_llm_labels(classifier_name: str, parsed: dict) -> list[str]:
    if classifier_name == "mention_type":
        types = parsed.get("mention_types", [])
        if isinstance(types, str):
            types = [types]
        return [normalize_label_token(t) for t in types if t]

    if classifier_name == "risk":
        raw = []
        for rt in parsed.get("risk_types", []):
            token = normalize_label_token(rt)
            if token and token != "none":
                raw.append(token)
        return sorted(set(raw))

    if classifier_name == "adoption_type":
        signals = parsed.get("adoption_signals", [])
        if isinstance(signals, list):
            return [
                normalize_label_token(e.get("type"))
                for e in signals
                if isinstance(e, dict)
                and isinstance(e.get("signal"), (int, float))
                and e["signal"] > 0
            ]
        # Legacy dict format
        conf = parsed.get("adoption_confidences", {})
        if isinstance(conf, dict):
            return [
                normalize_label_token(k)
                for k, v in conf.items()
                if isinstance(v, (int, float)) and v > 0
            ]
        return []

    if classifier_name == "vendor":
        vendors = parsed.get("vendors", [])
        out = []
        for v in vendors:
            if not isinstance(v, dict):
                continue
            tag = v.get("vendor", "")
            if hasattr(tag, "value"):
                tag = tag.value
            tag = str(tag).strip().lower()
            if tag:
                out.append(tag)
        return out

    return []


def extract_confidence(classifier_name: str, parsed: dict) -> float:
    """Return a confidence score (0.0-1.0) from the parsed response."""
    if classifier_name == "mention_type":
        scores = parsed.get("confidence_scores", {})
        if isinstance(scores, dict):
            valid = [v for v in scores.values() if isinstance(v, (int, float))]
            return max(valid) if valid else 0.0
        return 0.0

    if classifier_name == "risk":
        signals = parsed.get("risk_signals", [])
        if isinstance(signals, list):
            vals = [
                e.get("signal")
                for e in signals
                if isinstance(e, dict) and isinstance(e.get("signal"), (int, float))
            ]
            return max(vals) / 3.0 if vals else 0.0
        scores = parsed.get("confidence_scores", {})
        if isinstance(scores, dict):
            valid = [v for v in scores.values() if isinstance(v, (int, float))]
            return max(valid) / 3.0 if valid else 0.0
        return 0.0

    if classifier_name == "adoption_type":
        signals = parsed.get("adoption_signals", [])
        if isinstance(signals, list):
            vals = [
                e.get("signal")
                for e in signals
                if isinstance(e, dict) and isinstance(e.get("signal"), (int, float))
            ]
            return max(vals) / 3.0 if vals else 0.0
        conf = parsed.get("adoption_confidences", {})
        if isinstance(conf, dict):
            valid = [v for v in conf.values() if isinstance(v, (int, float))]
            return max(valid) / 3.0 if valid else 0.0
        return 0.0

    if classifier_name == "vendor":
        vendors = parsed.get("vendors", [])
        if isinstance(vendors, list):
            vals = [
                v.get("signal", 0)
                for v in vendors
                if isinstance(v, dict) and isinstance(v.get("signal"), (int, float))
            ]
            return max(vals) / 3.0 if vals else 0.0
        return 0.0

    return 0.0


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------


def build_metadata(chunk: dict) -> dict:
    return {
        "firm_id": chunk.get("company_id", "Unknown"),
        "firm_name": chunk.get("company_name", "Unknown"),
        "report_year": chunk.get("report_year", 0),
        "sector": "Unknown",
        "report_section": (
            chunk.get("report_sections", ["Unknown"])[0]
            if chunk.get("report_sections")
            else "Unknown"
        ),
        "mention_types": chunk.get("mention_types", []),
    }


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------


def make_run_id(classifier_name: str, model: str, suffix: str, use_openrouter: bool = False) -> str:
    model_slug = model.replace("/", "-")
    prefix = "or" if use_openrouter else "seq"
    return f"{prefix}-{classifier_name}-{model_slug}-{suffix}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    classifier_names = [c.strip() for c in args.classifiers.split(",") if c.strip()]

    # Import classifiers
    try:
        _init_classifier_config()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you are in the pipeline venv and dependencies are installed.")
        return 1

    for name in classifier_names:
        if name not in CLASSIFIER_CONFIG:
            print(f"Unknown classifier: {name}. Choose from: {list(CLASSIFIER_CONFIG)}")
            return 1

    # Load chunks
    chunks = load_chunks(args.golden_set)
    backend = "OpenRouter" if (args.use_openrouter or "/" in args.model) else "Gemini API"
    print(f"Loaded {len(chunks)} chunks from {args.golden_set.name}")
    print(f"Model: {args.model} ({backend})")
    print(f"Classifiers: {classifier_names}")
    print()

    all_run_paths: list[Path] = []

    for classifier_name in classifier_names:
        config = CLASSIFIER_CONFIG[classifier_name]
        use_or = args.use_openrouter or ("/" in args.model)
        run_id = make_run_id(classifier_name, args.model, args.run_suffix, use_or)

        # Filter
        selected = filter_chunks(
            chunks,
            filter_mention=config["filter_mention"],
            offset=args.offset,
            limit=args.limit,
            use_all=args.all,
        )

        filter_desc = f"mention_type={config['filter_mention']}" if config["filter_mention"] else "all"
        print(f"--- {classifier_name} ({run_id}) ---")
        print(f"  Filter: {filter_desc} -> {len(selected)} chunks")

        if not selected:
            print("  No chunks matched. Skipping.")
            print()
            continue

        sample_ids = [str(c.get("chunk_id", "?")) for c in selected[:3]]
        print(f"  Sample IDs: {sample_ids}")

        if args.dry_run:
            print("  [dry-run] Skipping API calls.")
            print()
            continue

        # Instantiate classifier (auto-detects Gemini vs OpenRouter based on model name)
        cls = config["cls"]
        use_or = args.use_openrouter or ("/" in args.model)
        classifier = cls(
            run_id=run_id,
            model_name=args.model,
            temperature=args.temperature,
            use_openrouter=use_or,
        )

        human_field = config["human_field"]
        results: list[dict[str, Any]] = []
        success_count = 0
        error_count = 0

        for chunk in tqdm(selected, desc=f"  {classifier_name}"):
            chunk_id = chunk.get("chunk_id", chunk.get("annotation_id", "unknown"))
            metadata = build_metadata(chunk)

            result = classifier.classify(chunk.get("chunk_text", ""), metadata)
            parsed = result.classification if isinstance(result.classification, dict) else {}

            if result.success:
                llm_labels = extract_llm_labels(classifier_name, parsed)
                confidence = extract_confidence(classifier_name, parsed)
                reasoning = parsed.get("reasoning", result.reasoning or "")
                error = None
                success_count += 1
            else:
                llm_labels = []
                confidence = 0.0
                reasoning = result.error_message or ""
                error = result.error_message
                error_count += 1

            row: dict[str, Any] = {
                "chunk_id": chunk_id,
                "company_name": chunk.get("company_name", "Unknown"),
                "report_year": chunk.get("report_year", 0),
                "human_labels": chunk.get(human_field, []),
                "llm_labels": llm_labels,
                "confidence": confidence,
                "reasoning": reasoning,
                "chunk_text": chunk.get("chunk_text", ""),
                "error": error,
            }

            # Classifier-specific extras (match phase2 output schema)
            if classifier_name == "risk":
                row["risk_signals"] = parsed.get("risk_signals", [])
                row["risk_substantiveness"] = parsed.get("substantiveness")
            if classifier_name == "adoption_type":
                row["adoption_signals"] = parsed.get("adoption_signals", [])
            if classifier_name == "vendor":
                row["vendor_signals"] = {}
                for v in parsed.get("vendors", []):
                    if isinstance(v, dict):
                        tag = v.get("vendor", "")
                        if hasattr(tag, "value"):
                            tag = tag.value
                        tag = str(tag).strip().lower()
                        sig = v.get("signal", 0)
                        if isinstance(sig, (int, float)) and sig > 0 and tag:
                            row["vendor_signals"][tag] = max(
                                row["vendor_signals"].get(tag, 0), int(sig)
                            )

            results.append(row)

        # Write outputs
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        run_path = RUNS_DIR / f"{run_id}.jsonl"
        with run_path.open("w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        meta = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "classifier": classifier_name,
            "temperature": args.temperature,
            "num_chunks": len(results),
            "success_count": success_count,
            "error_count": error_count,
            "golden_set": str(args.golden_set),
            "offset": args.offset,
            "limit": 0 if args.all else args.limit,
        }
        meta_path = run_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))

        # Summary
        label_counts: Counter[str] = Counter()
        for r in results:
            for lbl in r.get("llm_labels", []):
                label_counts[lbl] += 1

        print(f"  Success: {success_count} | Errors: {error_count}")
        print(f"  Label distribution: {dict(label_counts.most_common(10))}")
        print(f"  Output: {run_path.name}")
        print()
        all_run_paths.append(run_path)

    if args.dry_run:
        print("Dry run complete. No API calls made.")
    else:
        print("All runs complete.")
        for p in all_run_paths:
            print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
