#!/usr/bin/env python3
"""
Standalone Substantiveness Classifier Testbed.

Runs the dedicated substantiveness_v2 prompt on golden set chunks via Gemini
Batch API, then compares results against:
  1. Human-reconciled golden set annotations (risk_substantiveness)
  2. Risk classifier testbed run (risk_substantiveness as LLM side-output)

Usage:
    # Dry run — generate batch JSONL without submitting:
    python3 scripts/run_substantiveness_testbed.py --dry-run

    # Full run:
    python3 scripts/run_substantiveness_testbed.py --model gemini-3-flash-preview

    # Resume polling for a previously submitted job:
    python3 scripts/run_substantiveness_testbed.py --resume

    # Compare-only (skip batch, just compare existing results):
    python3 scripts/run_substantiveness_testbed.py --compare-only \
        --results-file data/testbed_runs/substantiveness-v2-gemini-3-flash-preview-v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

# Load .env.local before importing pipeline modules
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

from src.classifiers.base_classifier import _clean_schema_for_gemini
from src.classifiers.schemas import SubstantivenessResponseV2
from src.utils.batch_api import BatchClient
from src.utils.prompt_loader import get_prompt_messages as render_prompt_messages

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUNS_DIR = REPO_ROOT / "data" / "testbed_runs"
GOLDEN_SET_DEFAULT = (
    REPO_ROOT / "data" / "golden_set" / "human_reconciled" / "annotations.jsonl"
)
RISK_TESTBED_DEFAULT = (
    RUNS_DIR / "batch-p2-risk-gemini-3-flash-preview-schema-v2.1-full.jsonl"
)
PROMPT_KEY = "substantiveness_v2"
VALID_LEVELS = {"boilerplate", "moderate", "substantive"}

_USER_TEMPLATE = (
    "## EXCERPT\n"
    '"""\n'
    "{text}\n"
    '"""\n'
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run standalone substantiveness_v2 classifier testbed."
    )
    p.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Gemini model name (default: gemini-3-flash-preview).",
    )
    p.add_argument(
        "--run-suffix",
        default="v1",
        help="Suffix for run ID: substantiveness-v2-{model}-{suffix}.",
    )
    p.add_argument(
        "--golden-set",
        type=Path,
        default=GOLDEN_SET_DEFAULT,
        help="Path to human-reconciled golden set JSONL.",
    )
    p.add_argument(
        "--risk-testbed",
        type=Path,
        default=RISK_TESTBED_DEFAULT,
        help="Path to risk classifier testbed JSONL (for comparison).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0).",
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
        help="Max seconds to wait for batch (default: 86400 = 24h).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare JSONL only, do not submit.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume polling for previously submitted job.",
    )
    p.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip batch, just run comparison on existing results.",
    )
    p.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Path to existing results JSONL (for --compare-only).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open() as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks from {path.name}")
    return chunks


def load_risk_testbed(path: Path) -> dict[str, str]:
    """Load risk_substantiveness from risk testbed run, keyed by chunk_id."""
    mapping: dict[str, str] = {}
    if not path.exists():
        print(f"  WARNING: Risk testbed not found: {path.name}")
        return mapping
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec.get("chunk_id")
            val = rec.get("risk_substantiveness")
            if cid and val and val in VALID_LEVELS:
                mapping[cid] = val
    print(f"  Loaded {len(mapping)} risk_substantiveness values from {path.name}")
    return mapping


# ---------------------------------------------------------------------------
# Prompt + schema helpers
# ---------------------------------------------------------------------------


def build_prompt(chunk: dict) -> tuple[str, str]:
    """Build (system, user) prompts for one chunk."""
    text = chunk.get("chunk_text", "")
    max_chars = 30000
    if len(text) > max_chars:
        text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

    firm_name = chunk.get("company_name", "Unknown Company")
    report_year = chunk.get("report_year", "Unknown")
    sector = "Unknown"
    report_section = (
        chunk.get("report_sections", ["Unknown"])[0]
        if chunk.get("report_sections")
        else "Unknown"
    )

    return render_prompt_messages(
        PROMPT_KEY,
        reasoning_policy="short",
        user_template=_USER_TEMPLATE,
        firm_name=firm_name,
        sector=sector,
        report_year=report_year,
        report_section=report_section,
        text=text,
    )


def get_response_schema() -> dict:
    raw = SubstantivenessResponseV2.model_json_schema()
    return _clean_schema_for_gemini(raw)


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------


def make_run_id(model: str, suffix: str) -> str:
    return f"substantiveness-v2-{model}-{suffix}"


# ---------------------------------------------------------------------------
# PREPARE: Write batch JSONL
# ---------------------------------------------------------------------------


def prepare_batch_jsonl(
    run_id: str,
    chunks: list[dict],
    temperature: float = 0.0,
) -> Path:
    response_schema = get_response_schema()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = RUNS_DIR / f"{run_id}.batch_input.jsonl"

    with jsonl_path.open("w") as f:
        for i, chunk in enumerate(chunks):
            system_prompt, user_prompt = build_prompt(chunk)
            generation_config: dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": 1024,
                "response_mime_type": "application/json",
                "response_schema": response_schema,
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

    print(f"  Wrote {len(chunks)} requests -> {jsonl_path.name}")
    return jsonl_path


# ---------------------------------------------------------------------------
# PARSE: Download + parse batch results
# ---------------------------------------------------------------------------


def _try_parse_json(text: str) -> dict | None:
    if not text:
        return None
    attempts = [text]
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


def download_and_parse(
    job_name: str,
    chunks: list[dict],
    batch: BatchClient,
) -> list[dict]:
    """Download batch results and parse into per-chunk records."""
    job = batch.client.batches.get(name=job_name)
    if job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"  WARNING: job state is {job.state.name}")
        return []

    # Download responses
    output_lines: list[dict] = []
    dest = getattr(job, "dest", None)
    result_file_name = getattr(dest, "file_name", None) if dest else None

    if result_file_name:
        raw_bytes = batch.client.files.download(file=result_file_name)
        output_text = raw_bytes.decode("utf-8")
        output_lines = [
            json.loads(line) for line in output_text.splitlines() if line.strip()
        ]
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
        print("  ERROR: No responses found")
        return []

    if len(output_lines) != len(chunks):
        print(f"  WARNING: {len(output_lines)} responses != {len(chunks)} chunks")

    # Build lookup by key
    response_by_key: dict[str, dict] = {}
    for entry in output_lines:
        key = entry.get("key")
        if key is not None:
            response_by_key[str(key)] = entry

    if not response_by_key and output_lines:
        response_by_key = {str(i): e for i, e in enumerate(output_lines)}

    # Parse each response
    results = []
    matched = 0
    errors = 0

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", chunk.get("annotation_id", f"unknown_{i}"))
        entry = response_by_key.get(str(i))

        if entry is None or ("error" in entry and entry["error"]):
            results.append({
                "chunk_id": chunk_id,
                "substantiveness_v2": None,
                "confidence": 0.0,
                "reasoning": "no response or API error",
                "error": True,
            })
            errors += 1
            continue

        try:
            response_obj = entry.get("response", {})
            response_text = None
            candidates = response_obj.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    response_text = parts[0].get("text")

            if not response_text:
                results.append({
                    "chunk_id": chunk_id,
                    "substantiveness_v2": None,
                    "confidence": 0.0,
                    "reasoning": "empty response",
                    "error": True,
                })
                errors += 1
                continue

            parsed = _try_parse_json(response_text)
            if parsed is None:
                raise ValueError("unable to parse JSON")

            label = parsed.get("substantiveness", "").lower().strip()
            if label not in VALID_LEVELS:
                label = None

            results.append({
                "chunk_id": chunk_id,
                "substantiveness_v2": label,
                "confidence": float(parsed.get("confidence", 0.0)),
                "reasoning": parsed.get("reasoning", ""),
                "error": False,
            })
            matched += 1

        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            results.append({
                "chunk_id": chunk_id,
                "substantiveness_v2": None,
                "confidence": 0.0,
                "reasoning": f"parse error: {e}",
                "error": True,
            })
            errors += 1

    print(f"  Parsed: {matched}/{len(chunks)} | Errors: {errors}")
    return results


# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------


def save_results(run_id: str, results: list[dict], chunks: list[dict], config: dict) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RUNS_DIR / f"{run_id}.jsonl"

    # Merge chunk context into results
    chunk_by_id = {
        c.get("chunk_id", c.get("annotation_id")): c for c in chunks
    }

    with run_path.open("w") as f:
        for r in results:
            chunk = chunk_by_id.get(r["chunk_id"], {})
            record = {
                "chunk_id": r["chunk_id"],
                "company_name": chunk.get("company_name", "Unknown"),
                "report_year": chunk.get("report_year", 0),
                "chunk_text": chunk.get("chunk_text", ""),
                "substantiveness_v2": r["substantiveness_v2"],
                "confidence": r["confidence"],
                "reasoning": r["reasoning"],
            }
            f.write(json.dumps(record) + "\n")

    meta = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "config": config,
        "num_chunks": len(results),
        "num_errors": sum(1 for r in results if r.get("error")),
    }
    run_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  Saved {len(results)} results -> {run_path.name}")
    return run_path


# ---------------------------------------------------------------------------
# COMPARE: Agreement analysis
# ---------------------------------------------------------------------------

LEVEL_ORDER = ["boilerplate", "moderate", "substantive"]


def _print_confusion_matrix(
    title: str,
    pairs: list[tuple[str, str]],
    row_label: str = "standalone_v2",
    col_label: str = "reference",
) -> None:
    """Print a 3x3 confusion matrix."""
    matrix: dict[str, dict[str, int]] = {
        r: {c: 0 for c in LEVEL_ORDER} for r in LEVEL_ORDER
    }
    for v2, ref in pairs:
        if v2 in VALID_LEVELS and ref in VALID_LEVELS:
            matrix[v2][ref] += 1

    total = len(pairs)
    agree = sum(1 for v2, ref in pairs if v2 == ref)
    pct = agree / total * 100 if total else 0

    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"  Agreement: {agree}/{total} ({pct:.1f}%)")
    print(f"{'=' * 60}")

    # Header
    col_w = 14
    print(f"  {row_label:>14s} \\ {col_label:<s}")
    print(f"  {'':>14s}   {'boilerplate':>{col_w}} {'moderate':>{col_w}} {'substantive':>{col_w}}  {'total':>{col_w}}")
    print(f"  {'':>14s}   {'-' * col_w} {'-' * col_w} {'-' * col_w}  {'-' * col_w}")

    for row in LEVEL_ORDER:
        row_total = sum(matrix[row].values())
        vals = "   ".join(f"{matrix[row][c]:>{col_w - 3}}" for c in LEVEL_ORDER)
        print(f"  {row:>14s}   {vals}  {row_total:>{col_w - 3}}")

    col_totals = "   ".join(
        f"{sum(matrix[r][c] for r in LEVEL_ORDER):>{col_w - 3}}" for c in LEVEL_ORDER
    )
    print(f"  {'total':>14s}   {col_totals}  {total:>{col_w - 3}}")


def _print_disagreements(
    title: str,
    disagreements: list[dict],
    max_show: int = 10,
) -> None:
    """Print sample disagreements."""
    if not disagreements:
        print(f"\n  No disagreements for: {title}")
        return

    print(f"\n--- Disagreements: {title} ({len(disagreements)} total, showing up to {max_show}) ---")
    for d in disagreements[:max_show]:
        text_snip = d["chunk_text"][:120].replace("\n", " ")
        print(
            f"  [{d['chunk_id'][:20]}] "
            f"v2={d['v2']:<13s} ref={d['ref']:<13s} "
            f"conf={d['confidence']:.2f}  "
            f"\"{text_snip}...\""
        )


def run_comparison(
    results: list[dict],
    chunks: list[dict],
    risk_testbed_path: Path,
) -> None:
    """Run comparison against golden set and risk testbed."""
    # Build lookups
    result_by_id = {r["chunk_id"]: r for r in results}
    chunk_by_id = {
        c.get("chunk_id", c.get("annotation_id")): c for c in chunks
    }

    # --- 1. Compare vs golden set (risk_substantiveness) ---
    gs_pairs: list[tuple[str, str]] = []
    gs_disagreements: list[dict] = []

    for chunk in chunks:
        cid = chunk.get("chunk_id", chunk.get("annotation_id"))
        gs_val = chunk.get("risk_substantiveness")
        if not gs_val or gs_val not in VALID_LEVELS:
            continue
        result = result_by_id.get(cid)
        if not result or not result.get("substantiveness_v2"):
            continue
        v2_val = result["substantiveness_v2"]
        gs_pairs.append((v2_val, gs_val))
        if v2_val != gs_val:
            gs_disagreements.append({
                "chunk_id": cid,
                "v2": v2_val,
                "ref": gs_val,
                "confidence": result.get("confidence", 0.0),
                "chunk_text": chunk.get("chunk_text", ""),
            })

    if gs_pairs:
        _print_confusion_matrix(
            "vs Golden Set (human-reconciled risk_substantiveness)",
            gs_pairs,
            row_label="standalone_v2",
            col_label="golden_set",
        )
        _print_disagreements("vs Golden Set", gs_disagreements)
    else:
        print("\n  No overlapping golden set annotations with risk_substantiveness.")

    # --- 2. Compare vs risk testbed run ---
    risk_map = load_risk_testbed(risk_testbed_path)

    rt_pairs: list[tuple[str, str]] = []
    rt_disagreements: list[dict] = []

    for result in results:
        cid = result["chunk_id"]
        v2_val = result.get("substantiveness_v2")
        rt_val = risk_map.get(cid)
        if not v2_val or not rt_val:
            continue
        rt_pairs.append((v2_val, rt_val))
        if v2_val != rt_val:
            chunk = chunk_by_id.get(cid, {})
            rt_disagreements.append({
                "chunk_id": cid,
                "v2": v2_val,
                "ref": rt_val,
                "confidence": result.get("confidence", 0.0),
                "chunk_text": chunk.get("chunk_text", ""),
            })

    if rt_pairs:
        _print_confusion_matrix(
            "vs Risk Testbed (LLM risk_substantiveness side-output)",
            rt_pairs,
            row_label="standalone_v2",
            col_label="risk_testbed",
        )
        _print_disagreements("vs Risk Testbed", rt_disagreements)
    else:
        print("\n  No overlapping risk testbed results for comparison.")

    # --- 3. Distribution summary ---
    v2_dist = Counter(r.get("substantiveness_v2") for r in results if r.get("substantiveness_v2"))
    print(f"\n--- Standalone v2 distribution ---")
    for level in LEVEL_ORDER:
        cnt = v2_dist.get(level, 0)
        pct = cnt / sum(v2_dist.values()) * 100 if v2_dist else 0
        print(f"  {level:>14s}: {cnt:4d} ({pct:5.1f}%)")

    # Confidence stats
    confs = [r["confidence"] for r in results if r.get("substantiveness_v2")]
    if confs:
        avg = sum(confs) / len(confs)
        lo = min(confs)
        hi = max(confs)
        print(f"\n  Confidence: mean={avg:.2f}, min={lo:.2f}, max={hi:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    run_id = make_run_id(args.model, args.run_suffix)

    print("=" * 60)
    print("Substantiveness v2 Testbed")
    print("=" * 60)
    print(f"  Run ID:      {run_id}")
    print(f"  Model:       {args.model}")
    print(f"  Golden set:  {args.golden_set}")
    print(f"  Risk testbed:{args.risk_testbed}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"  Resume:      {args.resume}")
    print(f"  Compare-only:{args.compare_only}")
    print()

    # Load chunks
    chunks = load_chunks(args.golden_set)

    # --compare-only: just load existing results and run comparison
    if args.compare_only:
        results_path = args.results_file or (RUNS_DIR / f"{run_id}.jsonl")
        if not results_path.exists():
            raise SystemExit(f"Results file not found: {results_path}")
        results = []
        with results_path.open() as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        print(f"Loaded {len(results)} results from {results_path.name}")
        run_comparison(results, chunks, args.risk_testbed)
        return

    batch = BatchClient(runs_dir=RUNS_DIR)

    # PREPARE + SUBMIT (or RESUME)
    if args.resume:
        meta_path = RUNS_DIR / f"{run_id}.batch_meta.json"
        if not meta_path.exists():
            raise SystemExit(f"Metadata not found: {meta_path}")
        meta = json.loads(meta_path.read_text())
        job_name = meta["job_name"]
        print(f"RESUME: {job_name}")
    else:
        # PREPARE
        print("PREPARE: Building batch JSONL...")
        jsonl_path = prepare_batch_jsonl(run_id, chunks, temperature=args.temperature)

        if args.dry_run:
            print(f"\nDRY RUN complete. Generated: {jsonl_path}")
            return

        # SUBMIT
        print("\nSUBMIT: Uploading and submitting batch job...")
        job_name = batch.submit(
            run_id=run_id,
            jsonl_path=jsonl_path,
            model_name=args.model,
        )

    # POLL
    print("\nPOLL: Waiting for batch job...")
    try:
        final_statuses = batch.poll_until_complete(
            jobs={"substantiveness_v2": job_name},
            interval=args.poll_interval,
            max_time=args.max_poll_time,
        )
    except KeyboardInterrupt:
        print("\nInterrupted!")
        return

    status = final_statuses.get("substantiveness_v2", {})
    state = status.get("state", "UNKNOWN")
    print(f"\nFinal state: {state}")

    if state != "JOB_STATE_SUCCEEDED":
        print(f"Batch did not succeed (state={state}). Exiting.")
        return

    # PARSE
    print("\nPARSE: Downloading and parsing results...")
    results = download_and_parse(job_name, chunks, batch)

    if not results:
        print("No results parsed. Exiting.")
        return

    # SAVE
    print("\nSAVE:")
    save_results(
        run_id,
        results,
        chunks,
        config={
            "model": args.model,
            "temperature": args.temperature,
            "prompt_key": PROMPT_KEY,
            "response_schema": "SubstantivenessResponseV2",
            "job_name": job_name,
        },
    )

    # COMPARE
    print("\nCOMPARE:")
    run_comparison(results, chunks, args.risk_testbed)

    print(f"\n{'=' * 60}")
    print(f"Done. Results: {RUNS_DIR / f'{run_id}.jsonl'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
