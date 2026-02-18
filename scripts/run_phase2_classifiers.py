#!/usr/bin/env python3
"""
Phase 2 Golden Set Production Pipeline.

Runs all 3 phase 2 classifiers (risk, adoption_type, vendor) on the full
golden set via Gemini Batch API, then exports results for reconciliation.

Usage:
    python3 scripts/run_phase2_classifiers.py \
        --classifiers risk,adoption_type,vendor \
        --model gemini-3-flash-preview \
        --run-suffix full-v1

    # Dry run (prepare JSONL only, no submission):
    python3 scripts/run_phase2_classifiers.py --dry-run

    # Resume polling for previously submitted jobs:
    python3 scripts/run_phase2_classifiers.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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

from src.classifiers.adoption_type_classifier import AdoptionTypeClassifier
from src.classifiers.base_classifier import _clean_schema_for_gemini
from src.classifiers.mention_type_classifier import MentionTypeClassifier
from src.classifiers.risk_classifier import RiskClassifier
from src.classifiers.vendor_classifier import VendorClassifier
from src.utils.batch_api import BatchClient
from src.utils.normalization import (
    normalize_risk_labels as normalize_risk_labels_shared,
    normalize_risk_substantiveness as normalize_risk_substantiveness_shared,
    normalize_signal_to_unit_interval,
    risk_signals_from_payload,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUNS_DIR = REPO_ROOT / "data" / "testbed_runs"
GOLDEN_SET_DEFAULT = REPO_ROOT / "data" / "golden_set" / "human_reconciled" / "annotations.jsonl"
EXPORT_SCRIPT = PIPELINE_DIR / "scripts" / "export_testbed_run_for_reconcile.py"

CLASSIFIER_CONFIG: dict[str, dict[str, Any]] = {
    "mention_type": {
        "cls": MentionTypeClassifier,
        "human_field": "mention_types",
        "filter_mention": None,  # all chunks
    },
    "adoption_type": {
        "cls": AdoptionTypeClassifier,
        "human_field": "adoption_types",
        "filter_mention": "adoption",
    },
    "risk": {
        "cls": RiskClassifier,
        "human_field": "risk_taxonomy",
        "filter_mention": "risk",
    },
    "vendor": {
        "cls": VendorClassifier,
        "human_field": "vendor_tags",
        "filter_mention": "vendor",
    },
}

PIPELINE_SCHEMA_VERSION = "phase2_testbed_v2_2"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run phase 2 classifiers on the golden set via Gemini Batch API."
    )
    p.add_argument(
        "--classifiers",
        default="risk,adoption_type,vendor",
        help="Comma-separated list of classifiers to run (default: all three).",
    )
    p.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Gemini model name (default: gemini-3-flash-preview).",
    )
    p.add_argument(
        "--run-suffix",
        default="full",
        help="Suffix for run IDs: p2-{classifier}-{model}-{suffix}.",
    )
    p.add_argument(
        "--golden-set",
        type=Path,
        default=GOLDEN_SET_DEFAULT,
        help="Path to human-reconciled golden set JSONL.",
    )
    p.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between batch status checks (default: 30).",
    )
    p.add_argument(
        "--max-poll-time",
        type=int,
        default=86400,
        help="Maximum seconds to wait for batch completion before timing out (default: 86400 = 24h).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0).",
    )
    p.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help=(
            "Thinking budget for Gemini. Set 0 to explicitly disable thinking "
            "and preserve output tokens for JSON."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare JSONL files without submitting to the API.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume: skip prepare+submit, poll existing jobs from batch_meta.json.",
    )
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip automatic export step after parsing results.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading & filtering
# ---------------------------------------------------------------------------


def load_env() -> None:
    """Load .env.local into os.environ (same as testbed)."""
    env_path = REPO_ROOT / ".env.local"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_chunks(golden_set_path: Path) -> list[dict]:
    chunks = []
    with golden_set_path.open() as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks from {golden_set_path.name}")
    return chunks


def filter_chunks(chunks: list[dict], classifier_name: str) -> list[dict]:
    mention = CLASSIFIER_CONFIG[classifier_name]["filter_mention"]
    if mention is None:
        filtered = list(chunks)
        print(f"  {classifier_name}: {len(filtered)} chunks (all)")
    else:
        filtered = [c for c in chunks if mention in c.get("mention_types", [])]
        print(f"  {classifier_name}: {len(filtered)} chunks (mention_type={mention})")
    return filtered


# ---------------------------------------------------------------------------
# Normalization helpers (mirrored from phase2_testbed.py)
# ---------------------------------------------------------------------------


def normalize_label_token(value: object) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    token = str(value)
    for prefix in ("RiskType.", "AdoptionType."):
        if token.startswith(prefix):
            return token.split(".", 1)[1]
    return token


def normalize_risk_labels(labels: list[str]) -> list[str]:
    return normalize_risk_labels_shared(labels)


def normalize_risk_substantiveness(value: object) -> str | None:
    return normalize_risk_substantiveness_shared(value)


def risk_signal_map(parsed: dict) -> dict[str, int]:
    raw = risk_signals_from_payload(parsed)
    return {
        normalize_label_token(k): int(v)
        for k, v in raw.items()
        if isinstance(v, (int, float))
    }


def risk_signal_entries(parsed: dict) -> list[dict[str, int | str]]:
    signals_list = parsed.get("risk_signals")
    if not isinstance(signals_list, list):
        return []
    out: list[dict[str, int | str]] = []
    seen: set[str] = set()
    for entry in signals_list:
        if not isinstance(entry, dict):
            continue
        key = normalize_label_token(entry.get("type"))
        val = entry.get("signal")
        if not key or not isinstance(val, (int, float)):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append({"type": key, "signal": int(val)})
    return out


def adoption_signal_map(parsed: dict) -> dict[str, int]:
    signals_list = parsed.get("adoption_signals")
    if isinstance(signals_list, list):
        out: dict[str, int] = {}
        for entry in signals_list:
            if isinstance(entry, dict):
                key = entry.get("type")
                val = entry.get("signal")
                if key is not None and isinstance(val, (int, float)):
                    out[normalize_label_token(key)] = int(val)
        return out
    legacy = parsed.get("adoption_confidences")
    if isinstance(legacy, dict):
        return {
            normalize_label_token(k): int(v)
            for k, v in legacy.items()
            if isinstance(v, (int, float))
        }
    return {}


def extract_risk_labels(parsed: dict) -> list[str]:
    """Extract applied risk labels (signal threshold = 0 in production)."""
    raw_types = []
    for rt in parsed.get("risk_types", []):
        raw = normalize_label_token(rt)
        if raw and raw != "none":
            raw_types.append(raw)
    return sorted(set(raw_types))


# ---------------------------------------------------------------------------
# Build metadata (same as testbed)
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


def make_run_id(classifier_name: str, model: str, suffix: str) -> str:
    return f"p2-{classifier_name}-{model}-{suffix}"


# ---------------------------------------------------------------------------
# PREPARE: Write batch JSONL
# ---------------------------------------------------------------------------


def prepare_batch_jsonl(
    run_id: str,
    classifier_name: str,
    filtered_chunks: list[dict],
    temperature: float = 0.0,
    thinking_budget: int | None = 0,
) -> Path:
    """Write batch request JSONL for one classifier."""
    config = CLASSIFIER_CONFIG[classifier_name]
    cls = config["cls"]
    temp_classifier = cls(run_id="batch-prep", model_name="unused")
    response_schema = _clean_schema_for_gemini(cls.RESPONSE_MODEL.model_json_schema())

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = RUNS_DIR / f"{run_id}.batch_input.jsonl"

    with jsonl_path.open("w") as f:
        for i, chunk in enumerate(filtered_chunks):
            metadata = build_metadata(chunk)
            system_prompt, user_prompt = temp_classifier.get_prompt_messages(
                chunk["chunk_text"], metadata
            )
            generation_config: dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": 2048,
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            }
            if thinking_budget is not None:
                generation_config["thinking_config"] = {
                    "thinking_budget": int(thinking_budget)
                }
            line = {
                "key": str(i),
                "request": {
                    "contents": [{"parts": [{"text": user_prompt}], "role": "user"}],
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "generation_config": generation_config,
                },
            }
            f.write(json.dumps(line) + "\n")

    print(f"  Wrote {len(filtered_chunks)} requests -> {jsonl_path.name}")
    return jsonl_path


# ---------------------------------------------------------------------------
# PARSE: Download + parse batch results
# ---------------------------------------------------------------------------


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


def _salvage_risk_payload(text: str) -> dict | None:
    """Regex salvage for truncated/malformed risk JSON."""
    if not text:
        return None

    payload: dict[str, Any] = {}

    m_types = re.search(r'"risk_types"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
    risk_types: list[str] = []
    if m_types:
        risk_types = re.findall(r'"([a-z_]+)"', m_types.group(1))

    risk_signals: list[dict[str, Any]] = []
    for t, s in re.findall(
        r'"type"\s*:\s*"([a-z_]+)"\s*,\s*"signal"\s*:\s*([123])',
        text,
        flags=re.DOTALL,
    ):
        risk_signals.append({"type": t, "signal": int(s)})

    m_sub = re.search(r'"substantiveness"\s*:\s*"([a-z_]+)"', text, flags=re.DOTALL)
    substantiveness = normalize_risk_substantiveness(m_sub.group(1) if m_sub else None)

    m_reason = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text, flags=re.DOTALL)
    reasoning = m_reason.group(1) if m_reason else ""

    if not risk_types and risk_signals:
        risk_types = [str(e["type"]) for e in risk_signals]
    if not risk_types:
        return None

    payload["risk_types"] = risk_types
    payload["risk_signals"] = risk_signals
    payload["substantiveness"] = substantiveness or "boilerplate"
    if reasoning:
        payload["reasoning"] = reasoning
    return payload


def _extract_confidence(parsed: dict) -> float:
    """Return a normalized confidence score on a 0.0-1.0 scale."""
    if "adoption_signals" in parsed or "adoption_confidences" in parsed:
        scores = adoption_signal_map(parsed)
        valid = [v for v in scores.values() if isinstance(v, (int, float))]
        return normalize_signal_to_unit_interval(max(valid), 3.0) if valid else 0.0
    if "risk_signals" in parsed or "confidence_scores" in parsed:
        scores = risk_signal_map(parsed)
        valid = [v for v in scores.values() if isinstance(v, (int, float))]
        return normalize_signal_to_unit_interval(max(valid), 3.0) if valid else 0.0
    if "vendors" in parsed and isinstance(parsed["vendors"], list):
        signals = [v.get("signal", 0) for v in parsed["vendors"] if isinstance(v, dict)]
        return normalize_signal_to_unit_interval(max(signals), 3.0) if signals else 0.0
    return 0.0


def download_and_parse(
    classifier_name: str,
    job_name: str,
    filtered_chunks: list[dict],
    batch: BatchClient,
) -> list[dict]:
    """Download batch results and parse into testbed-format records."""
    config = CLASSIFIER_CONFIG[classifier_name]
    human_field = config["human_field"]

    # --- Extract labels per classifier ---
    def _extract_llm_labels(parsed: dict) -> list[str]:
        if classifier_name == "mention_type":
            types = parsed.get("mention_types", [])
            if isinstance(types, str):
                types = [types]
            return [normalize_label_token(t) for t in types if t]
        elif classifier_name == "risk":
            return extract_risk_labels(parsed)
        elif classifier_name == "adoption_type":
            return [
                k
                for k, v in adoption_signal_map(parsed).items()
                if isinstance(v, (int, float)) and v > 0
            ]
        elif classifier_name == "vendor":
            return [
                (
                    v.get("vendor").value
                    if hasattr(v.get("vendor"), "value")
                    else str(v.get("vendor", ""))
                )
                for v in parsed.get("vendors", [])
                if isinstance(v, dict)
            ]
        return []

    def _human_labels(chunk: dict) -> list[str]:
        labels = chunk.get(human_field, [])
        return normalize_risk_labels(labels) if classifier_name == "risk" else labels

    def _make_error(chunk: dict, chunk_id: str, error: str) -> dict:
        return {
            "chunk_id": chunk_id,
            "company_name": chunk.get("company_name", "Unknown"),
            "report_year": chunk.get("report_year", 0),
            "human_labels": _human_labels(chunk),
            "llm_labels": [],
            "confidence": 0.0,
            "reasoning": error,
            "chunk_text": chunk.get("chunk_text", ""),
            "error": error,
            "risk_confidences": {},
            "risk_signals": [],
            "risk_substantiveness": None,
            "adoption_signals": {},
        }

    # --- Download raw responses ---
    job = batch.client.batches.get(name=job_name)
    if job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"  WARNING: job state is {job.state.name}, cannot parse results")
        return [
            _make_error(c, c.get("chunk_id", f"unknown_{i}"), f"batch {job.state.name}")
            for i, c in enumerate(filtered_chunks)
        ]

    output_lines: list[dict] = []
    dest = getattr(job, "dest", None)
    result_file_name = getattr(dest, "file_name", None) if dest else None

    if result_file_name:
        raw_bytes = batch.client.files.download(file=result_file_name)
        output_text = raw_bytes.decode("utf-8")
        output_lines = [json.loads(line) for line in output_text.splitlines() if line.strip()]
        print(f"  Downloaded {len(output_lines)} responses from {result_file_name}")

    if not output_lines and dest and getattr(dest, "inlined_responses", None):
        for ir in dest.inlined_responses:
            entry: dict = {}
            key = getattr(ir, "key", None)
            if key is not None:
                entry["key"] = str(key)
            resp = getattr(ir, "response", None)
            if resp:
                entry["response"] = resp.model_dump() if hasattr(resp, "model_dump") else {}
            output_lines.append(entry)
        print(f"  Extracted {len(output_lines)} inlined responses")

    if not output_lines:
        print(f"  ERROR: No responses found for {classifier_name}")
        return [
            _make_error(c, c.get("chunk_id", f"unknown_{i}"), "no responses")
            for i, c in enumerate(filtered_chunks)
        ]

    if len(output_lines) != len(filtered_chunks):
        print(f"  WARNING: {len(output_lines)} responses != {len(filtered_chunks)} chunks")

    # Build lookup by key
    response_by_key: dict[str, dict] = {}
    missing_key_count = 0
    for entry in output_lines:
        key = entry.get("key")
        if key is not None:
            response_by_key[str(key)] = entry
        else:
            missing_key_count += 1

    if not response_by_key and output_lines:
        print("  WARNING: No response keys found; falling back to positional matching.")
        response_by_key = {str(i): e for i, e in enumerate(output_lines)}
    elif missing_key_count:
        print(f"  WARNING: {missing_key_count} responses missing keys.")

    # --- Parse each response ---
    results = []
    matched = 0
    errors: list[str] = []

    for i, chunk in enumerate(filtered_chunks):
        chunk_id = chunk.get("chunk_id", chunk.get("annotation_id", f"unknown_{i}"))
        entry = response_by_key.get(str(i))

        if entry is None:
            error = f"Key '{i}': no response returned"
            errors.append(error)
            results.append(_make_error(chunk, chunk_id, error))
            continue

        if "error" in entry and entry["error"]:
            error = f"Key '{i}': API error: {entry['error']}"
            errors.append(error)
            results.append(_make_error(chunk, chunk_id, error))
            continue

        try:
            response_obj = entry.get("response", {})
            response_text = None
            candidates = response_obj.get("candidates", [])
            for cand in candidates:
                parts = cand.get("content", {}).get("parts", [])
                text_parts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
                if text_parts:
                    response_text = "\n".join(text_parts).strip()
                    break

            if not response_text:
                error = f"Key '{i}': empty response text"
                errors.append(error)
                results.append(_make_error(chunk, chunk_id, error))
                continue

            parsed = _try_parse_json(response_text)
            if parsed is None and classifier_name == "risk":
                parsed = _salvage_risk_payload(response_text)
            if parsed is None:
                raise ValueError("unable to parse model JSON response")

            llm_labels = _extract_llm_labels(parsed)
            confidence = _extract_confidence(parsed)

            risk_conf = (
                {str(k): v for k, v in risk_signal_map(parsed).items() if isinstance(v, (int, float))}
                if classifier_name == "risk"
                else {}
            )
            r_signals = risk_signal_entries(parsed) if classifier_name == "risk" else []
            r_substantiveness = (
                normalize_risk_substantiveness(parsed.get("substantiveness"))
                if classifier_name == "risk"
                else None
            )
            adopt_conf = (
                {str(k): v for k, v in adoption_signal_map(parsed).items() if isinstance(v, (int, float))}
                if classifier_name == "adoption_type"
                else {}
            )
            vendor_sigs = {}
            if classifier_name == "vendor" and isinstance(parsed.get("vendors"), list):
                for v in parsed["vendors"]:
                    if isinstance(v, dict):
                        tag = v.get("vendor", "")
                        if hasattr(tag, "value"):
                            tag = tag.value
                        tag = str(tag).strip().lower()
                        sig = v.get("signal", 0)
                        if isinstance(sig, (int, float)) and sig > 0:
                            vendor_sigs[tag] = max(vendor_sigs.get(tag, 0), int(sig))

            results.append(
                {
                    "chunk_id": chunk_id,
                    "company_name": chunk.get("company_name", "Unknown"),
                    "report_year": chunk.get("report_year", 0),
                    "human_labels": _human_labels(chunk),
                    "llm_labels": llm_labels,
                    "confidence": confidence,
                    "reasoning": parsed.get("reasoning", ""),
                    "chunk_text": chunk.get("chunk_text", ""),
                    "human_other": chunk.get("vendor_other"),
                    "llm_other": parsed.get("other_vendor"),
                    "risk_confidences": risk_conf,
                    "risk_signals": r_signals,
                    "risk_substantiveness": r_substantiveness,
                    "adoption_signals": adopt_conf,
                    "vendor_signals": vendor_sigs,
                }
            )
            matched += 1

        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            error = f"Key '{i}': parse error: {e}"
            errors.append(error)
            results.append(_make_error(chunk, chunk_id, error))

    print(f"  Matched: {matched}/{len(filtered_chunks)} | Errors: {len(errors)}")
    for err in errors[:10]:
        print(f"    - {err}")
    if len(errors) > 10:
        print(f"    ... and {len(errors) - 10} more errors")

    return results


# ---------------------------------------------------------------------------
# SAVE: Write testbed-format output
# ---------------------------------------------------------------------------


def save_run(run_id: str, results: list[dict], config: dict) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RUNS_DIR / f"{run_id}.jsonl"
    with run_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    meta = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "config": config,
        "num_chunks": len(results),
    }
    run_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  Saved {len(results)} results -> {run_path.name}")
    return run_path


# ---------------------------------------------------------------------------
# EXPORT: Run export script programmatically
# ---------------------------------------------------------------------------


def run_export(
    run_id: str,
    classifier_name: str,
    testbed_run_path: Path,
    golden_set_path: Path,
) -> Path | None:
    """Run export_testbed_run_for_reconcile.py as a subprocess."""
    output_dir = REPO_ROOT / "data" / "golden_set" / "llm" / run_id
    cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--testbed-run",
        str(testbed_run_path),
        "--run-id",
        run_id,
        "--human",
        str(golden_set_path),
        "--output-dir",
        str(output_dir),
        "--classifier-type",
        classifier_name,
        "--include-missing",
    ]
    print(f"  Exporting: {' '.join(cmd[-6:])}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Export FAILED: {result.stderr.strip()}")
        return None
    print(f"  {result.stdout.strip()}")
    return output_dir / "annotations.jsonl"


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    load_env()

    classifier_names = [c.strip() for c in args.classifiers.split(",") if c.strip()]
    for name in classifier_names:
        if name not in CLASSIFIER_CONFIG:
            raise SystemExit(f"Unknown classifier: {name}. Choose from: {list(CLASSIFIER_CONFIG)}")

    print("=" * 60)
    print("Phase 2 Production Pipeline")
    print("=" * 60)
    print(f"  Classifiers: {classifier_names}")
    print(f"  Model:       {args.model}")
    print(f"  Suffix:      {args.run_suffix}")
    print(f"  Golden set:  {args.golden_set}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"  Resume:      {args.resume}")
    print(f"  Poll every:  {args.poll_interval}s")
    print(f"  Poll max:    {args.max_poll_time}s")
    print()

    # Load chunks once
    chunks = load_chunks(args.golden_set)

    # Build run IDs and filter chunks per classifier
    run_ids: dict[str, str] = {}
    filtered: dict[str, list[dict]] = {}
    for name in classifier_names:
        run_ids[name] = make_run_id(name, args.model, args.run_suffix)
        filtered[name] = filter_chunks(chunks, name)
    print()

    batch = BatchClient(runs_dir=RUNS_DIR)

    # -----------------------------------------------------------------------
    # PREPARE + SUBMIT (or RESUME)
    # -----------------------------------------------------------------------

    job_names: dict[str, str] = {}
    jsonl_paths: dict[str, Path] = {}

    if args.resume:
        # Load job_names from batch_meta.json files
        print("RESUME: Loading job names from batch_meta.json files...")
        for name in classifier_names:
            meta_path = RUNS_DIR / f"{run_ids[name]}.batch_meta.json"
            if not meta_path.exists():
                print(f"  ERROR: {meta_path.name} not found, cannot resume {name}")
                continue
            meta = json.loads(meta_path.read_text())
            job_names[name] = meta["job_name"]
            print(f"  {name}: {meta['job_name']}")
        print()
    else:
        # PREPARE
        print("PREPARE: Building batch JSONL files...")
        for name in classifier_names:
            jsonl_paths[name] = prepare_batch_jsonl(
                run_ids[name],
                name,
                filtered[name],
                temperature=args.temperature,
                thinking_budget=args.thinking_budget,
            )
        print()

        if args.dry_run:
            print("DRY RUN: JSONL files created. Skipping submit/poll/parse.")
            print("\nGenerated files:")
            for name in classifier_names:
                print(f"  {jsonl_paths[name]}")
            return

        # SUBMIT
        print("SUBMIT: Uploading and submitting batch jobs...")
        for name in classifier_names:
            try:
                job_name = batch.submit(
                    run_id=run_ids[name],
                    jsonl_path=jsonl_paths[name],
                    model_name=args.model,
                )
                job_names[name] = job_name
            except Exception as e:
                print(f"  ERROR submitting {name}: {e}")
        print()

    if not job_names:
        raise SystemExit("No jobs to poll. Exiting.")

    # -----------------------------------------------------------------------
    # POLL
    # -----------------------------------------------------------------------

    print("POLL: Waiting for batch jobs to complete...")
    try:
        final_statuses = batch.poll_until_complete(
            jobs=job_names,
            interval=args.poll_interval,
            max_time=args.max_poll_time,
        )
    except KeyboardInterrupt:
        print("\nInterrupted! Attempting to parse any completed jobs...")
        final_statuses = {}
        for name, job_name in job_names.items():
            try:
                status = batch.check_status(job_name)
                final_statuses[name] = status
            except Exception:
                final_statuses[name] = {"state": "INTERRUPTED", "job_name": job_name}
    print()

    # -----------------------------------------------------------------------
    # PARSE + SAVE + EXPORT
    # -----------------------------------------------------------------------

    exported_paths: dict[str, Path] = {}

    for name in classifier_names:
        status = final_statuses.get(name, {})
        state = status.get("state", "UNKNOWN")

        print(f"--- {name} ({run_ids[name]}) ---")
        print(f"  Final state: {state}")

        if state != "JOB_STATE_SUCCEEDED":
            print(f"  Skipping parse (state={state})")
            continue

        # PARSE
        job_name = job_names[name]
        results = download_and_parse(name, job_name, filtered[name], batch)

        # SAVE
        run_path = save_run(
            run_ids[name],
            results,
            {
                "classifier_name": name,
                "model_name": args.model,
                "temperature": args.temperature,
                "batch_mode": True,
                "num_filtered": len(filtered[name]),
                "job_name": job_name,
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "prompt_key": getattr(CLASSIFIER_CONFIG[name]["cls"], "PROMPT_KEY", None),
                "response_schema": getattr(
                    getattr(CLASSIFIER_CONFIG[name]["cls"], "RESPONSE_MODEL", None),
                    "__name__",
                    None,
                ),
                "classifier_contract_version": getattr(
                    CLASSIFIER_CONFIG[name]["cls"],
                    "SCHEMA_VERSION",
                    None,
                ),
            },
        )

        # EXPORT
        if not args.skip_export:
            export_path = run_export(run_ids[name], name, run_path, args.golden_set)
            if export_path:
                exported_paths[name] = export_path

        print()

    # -----------------------------------------------------------------------
    # Print reconcile + merge commands
    # -----------------------------------------------------------------------

    if exported_paths:
        print("=" * 60)
        print("NEXT STEPS: Reconcile and merge")
        print("=" * 60)
        for name, export_path in exported_paths.items():
            run_id = run_ids[name]
            # Map classifier_name to focus-field
            focus = {"risk": "risk", "adoption_type": "adoption", "vendor": "vendor"}[name]
            reconciled_dir = REPO_ROOT / "data" / "golden_set" / "reconciled" / run_id
            print(f"\n# {name}:")
            print(
                f"python3 pipeline/scripts/reconcile_annotations.py \\\n"
                f"  --human {args.golden_set} \\\n"
                f"  --llm {export_path} \\\n"
                f"  --output-dir {reconciled_dir} \\\n"
                f"  --only-disagreements --focus-field {focus} --resume"
            )
            print(
                f"\npython3 pipeline/scripts/merge_reconciled_golden_set.py \\\n"
                f"  --human {args.golden_set} \\\n"
                f"  --reconciled {reconciled_dir / 'annotations.jsonl'} \\\n"
                f"  --output {args.golden_set}"
            )
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name in classifier_names:
        status = final_statuses.get(name, {})
        state = status.get("state", "NOT_SUBMITTED")
        exported = "yes" if name in exported_paths else "no"
        print(f"  {name:20s}  state={state:25s}  exported={exported}")


if __name__ == "__main__":
    main()
