#!/usr/bin/env python3
"""
Online (concurrent) substantiveness classification.

Runs substantiveness classifiers via direct async API calls instead of the
Batch API. Supports resume via a checkpoint file so interrupted runs can
continue without reprocessing completed chunks.

Usage:
    python scripts/run_substantiveness_online.py [--labels vendor adoption risk] [--concurrency 50]

Options:
    --labels      Which classifiers to run (default: vendor adoption risk)
    --concurrency Max concurrent requests (default: 50)
    --run-id      Override auto-generated run ID (for resuming a run)
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import tqdm.asyncio

SCRIPT_DIR   = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
REPO_ROOT    = PIPELINE_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR))

# Load .env.local
env_path = REPO_ROOT / ".env.local"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

from google import genai
from google.genai import types
from pydantic import BaseModel

from src.classifiers.base_classifier import _clean_schema_for_gemini
from src.classifiers.schemas import (
    AdoptionSubstantivenessResponse,
    RiskSubstantivenessResponse,
    VendorSubstantivenessResponse,
)
from src.utils.prompt_loader import get_prompt_messages

# ── Config ────────────────────────────────────────────────────────────────────

SOURCE_ANNOTATIONS = (
    REPO_ROOT
    / "data/results/uk_annual_reports_24k"
    / "annotations.jsonl"
)

MODEL       = "gemini-3-flash-preview"
TEMPERATURE = 0.0
THINKING    = 0
MAX_OUTPUT_TOKENS = 2048

CLASSIFIER_CONFIG: dict[str, tuple[str, type[BaseModel]]] = {
    "adoption": ("adoption_substantiveness_v1", AdoptionSubstantivenessResponse),
    "risk":     ("risk_substantiveness_v1",     RiskSubstantivenessResponse),
    "vendor":   ("vendor_substantiveness_v1",   VendorSubstantivenessResponse),
}

# ── Async worker ──────────────────────────────────────────────────────────────

async def classify_chunk(
    client: genai.Client,
    chunk: dict,
    prompt_name: str,
    response_schema: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 4,
) -> dict:
    """Run one substantiveness classification, returning a result dict."""
    chunk_id = chunk["chunk_id"]
    report_year    = str(chunk.get("report_year", "Unknown"))
    report_section = (chunk.get("report_sections") or ["Unknown"])[0]

    system_prompt, user_prompt = get_prompt_messages(
        prompt_name,
        reasoning_policy="short",
        firm_name=chunk.get("company_name", "Unknown"),
        sector=chunk.get("sector", "Unknown"),
        report_year=report_year,
        report_section=report_section,
        mention_types=", ".join(chunk.get("mention_types") or []),
        text=chunk.get("chunk_text", "")[:30000],
    )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_schema=response_schema,
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING),
    )

    delay = 2.0
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=MODEL,
                        contents=user_prompt,
                        config=config,
                    ),
                    timeout=60.0,
                )
                text = response.text
                parsed = json.loads(text)
                return {"chunk_id": chunk_id, "parsed": parsed}
            except Exception as e:
                err_str = str(e) or f"{type(e).__name__} (no message)"
                is_rate_limit = "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower()
                is_server_err = "503" in err_str or "500" in err_str
                is_timeout = "TimeoutError" in type(e).__name__ or "timeout" in err_str.lower()
                if (is_rate_limit or is_server_err or is_timeout) and attempt < max_retries - 1:
                    await asyncio.sleep(delay * (2 ** attempt))
                    continue
                return {"chunk_id": chunk_id, "error": err_str}

    return {"chunk_id": chunk_id, "error": "max retries exceeded"}


async def run_classifier(
    label: str,
    chunks: list[dict],
    prompt_name: str,
    schema_cls: type[BaseModel],
    out_dir: Path,
    run_id: str,
    concurrency: int,
) -> list[dict]:
    """Run all chunks for one classifier concurrently, with checkpoint resume."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client  = genai.Client(api_key=api_key)

    response_schema = _clean_schema_for_gemini(schema_cls.model_json_schema())
    results_path    = out_dir / f"{run_id}-{label}.results.jsonl"
    checkpoint_path = out_dir / f"{run_id}-{label}.checkpoint.json"

    # Load existing checkpoint
    done: dict[str, dict] = {}
    if results_path.exists():
        with results_path.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["chunk_id"]] = r

    remaining = [c for c in chunks if c["chunk_id"] not in done]
    print(f"\n── {label} ──")
    print(f"  Total : {len(chunks):,}  |  Done: {len(done):,}  |  Remaining: {len(remaining):,}")

    if not remaining:
        print("  Already complete — skipping.")
        return list(done.values())

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        classify_chunk(client, chunk, prompt_name, response_schema, semaphore)
        for chunk in remaining
    ]

    # Incremental write as tasks complete
    results_file = results_path.open("a")
    new_results: list[dict] = []
    errors = 0

    try:
        for coro in tqdm.asyncio.tqdm.as_completed(tasks, total=len(tasks), desc=label):
            result = await coro
            results_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            results_file.flush()
            new_results.append(result)
            if "error" in result:
                errors += 1
    finally:
        results_file.close()

    all_results = list(done.values()) + new_results
    print(f"  Completed {len(new_results):,} new  |  Errors: {errors}")

    checkpoint_path.write_text(json.dumps({"done": len(all_results), "errors": errors}))
    return all_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels", nargs="+",
        choices=["adoption", "risk", "vendor"],
        default=["vendor", "adoption", "risk"],
    )
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    run_id  = args.run_id or f"substantiveness-online-{datetime.now().strftime('%Y%m%d-%H%M')}"
    out_dir = REPO_ROOT / "data/results/substantiveness" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load source annotations
    all_chunks: list[dict] = []
    with open(SOURCE_ANNOTATIONS) as f:
        for line in f:
            if line.strip():
                all_chunks.append(json.loads(line))

    chunk_sets = {
        "adoption": [c for c in all_chunks if "adoption" in (c.get("mention_types") or [])],
        "risk":     [c for c in all_chunks if "risk"     in (c.get("mention_types") or [])],
        "vendor":   [c for c in all_chunks if "vendor"   in (c.get("mention_types") or [])],
    }

    print(f"Run ID      : {run_id}")
    print(f"Output dir  : {out_dir}")
    print(f"Concurrency : {args.concurrency}")
    print(f"Labels      : {args.labels}")
    print(f"Total chunks: {len(all_chunks):,}")
    for lbl in args.labels:
        print(f"  {lbl:10s}: {len(chunk_sets[lbl]):,}")

    # Run each classifier
    all_results: dict[str, list[dict]] = {}
    for label in args.labels:
        prompt_name, schema_cls = CLASSIFIER_CONFIG[label]
        results = asyncio.run(run_classifier(
            label=label,
            chunks=chunk_sets[label],
            prompt_name=prompt_name,
            schema_cls=schema_cls,
            out_dir=out_dir,
            run_id=run_id,
            concurrency=args.concurrency,
        ))
        all_results[label] = results

    # Build lookup maps
    subs: dict[str, dict[str, str | None]] = {lbl: {} for lbl in args.labels}
    for label, results in all_results.items():
        for r in results:
            if "parsed" in r:
                subs[label][r["chunk_id"]] = r["parsed"].get("substantiveness")

    # Write merged scores
    scored: dict[str, dict] = {}
    for label in args.labels:
        for chunk in chunk_sets[label]:
            cid = chunk["chunk_id"]
            if cid not in scored:
                scored[cid] = {
                    "chunk_id":               cid,
                    "company_name":           chunk.get("company_name"),
                    "report_year":            chunk.get("report_year"),
                    "mention_types":          chunk.get("mention_types", []),
                    "adoption_substantiveness": None,
                    "risk_substantiveness":     None,
                    "vendor_substantiveness":   None,
                }
            if label == "adoption":
                scored[cid]["adoption_substantiveness"] = subs["adoption"].get(cid)
            elif label == "risk":
                scored[cid]["risk_substantiveness"] = subs["risk"].get(cid)
            elif label == "vendor":
                scored[cid]["vendor_substantiveness"] = subs["vendor"].get(cid)

    final_out = out_dir / f"{run_id}.substantiveness_scores.jsonl"
    with open(final_out, "w") as f:
        for record in scored.values():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nOutput: {final_out}")
    print(f"  Total chunks scored         : {len(scored):,}")
    for label in args.labels:
        col = f"{label}_substantiveness"
        n = sum(1 for r in scored.values() if r.get(col))
        print(f"  {col:35s}: {n:,}")


if __name__ == "__main__":
    main()
