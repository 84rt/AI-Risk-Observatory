#!/usr/bin/env python3
"""
Batch classification pipeline for UK annual reports chunks.

Phase 1: Mention type classifier on all chunks (is this a real AI mention?)
Phase 2: Downstream classifiers on confirmed AI mentions
         - adoption type  (if mention_type includes "adoption")
         - risk type      (if mention_type includes "risk")
         - vendor         (if mention_type includes "vendor")

Cell-based workflow (run cells in order):
  1. SETUP         — load chunks, configure run
  2. PHASE 1 PREP  — build batch request file
  3. PHASE 1 SUB   — submit to Gemini Batch API
  4. PHASE 1 POLL  — check status (re-run until done)
  5. PHASE 1 SAVE  — download results, save to disk
  6. PHASE 2 PREP  — filter to confirmed mentions, build batch requests
  7. PHASE 2 SUB   — submit downstream batches
  8. PHASE 2 POLL  — check status
  9. PHASE 2 SAVE  — download and merge all results
"""

# %% ─── SETUP ────────────────────────────────────────────────────────────────

import json
import os
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_DIR.parent
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

from src.utils.batch_api import BatchClient

# ── Paths ─────────────────────────────────────────────────────────────────────
CHUNKS_PATH  = REPO_ROOT / "data" / "uk_annual_reports_export" / "chunks" / "chunks.jsonl"
RUNS_DIR     = REPO_ROOT / "data" / "uk_annual_reports_export" / "chunks" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ── Run config ────────────────────────────────────────────────────────────────
RUN_DATE    = datetime.now().strftime("%Y%m%d-%H%M")
RUN_ID      = f"uk-reports-{RUN_DATE}"
MODEL       = "gemini-2.0-flash"
TEMPERATURE = 0.0
THINKING    = 0        # set >0 to enable thinking tokens

# ── Load chunks ───────────────────────────────────────────────────────────────
chunks: list[dict] = []
with open(CHUNKS_PATH) as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

print(f"Run ID   : {RUN_ID}")
print(f"Model    : {MODEL}")
print(f"Chunks   : {len(chunks):,}")
print(f"Runs dir : {RUNS_DIR}")

batch = BatchClient(runs_dir=RUNS_DIR)


# %% ─── PHASE 1: PREPARE REQUESTS ────────────────────────────────────────────

p1_run_id = f"{RUN_ID}-phase1"
p1_input  = batch.prepare_requests(
    run_id=p1_run_id,
    chunks=chunks,
    temperature=TEMPERATURE,
    thinking_budget=THINKING,
)
print(f"Phase 1 input: {p1_input}")


# %% ─── PHASE 1: SUBMIT ───────────────────────────────────────────────────────

p1_job = batch.submit(p1_run_id, p1_input, model_name=MODEL)
print(f"Phase 1 job: {p1_job}")

# Save job name to disk so you can recover it if the session restarts
(RUNS_DIR / f"{p1_run_id}.job_name.txt").write_text(p1_job)


# %% ─── PHASE 1: CHECK STATUS (re-run this cell until SUCCEEDED) ─────────────

# Uncomment to restore job name after session restart:
# p1_job = (RUNS_DIR / f"{p1_run_id}.job_name.txt").read_text().strip()

batch.check_status(p1_job)


# %% ─── PHASE 1: SAVE RESULTS ────────────────────────────────────────────────

p1_results = batch.get_results(p1_job, chunks)

if p1_results:
    p1_out = RUNS_DIR / f"{p1_run_id}.results.jsonl"
    with open(p1_out, "w") as f:
        for r in p1_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Phase 1 results saved: {p1_out}")
    print(f"  Total    : {len(p1_results):,}")

    errors   = sum(1 for r in p1_results if r.get("error"))
    non_none = sum(1 for r in p1_results if r.get("llm_mention_types") and r["llm_mention_types"] != ["none"])
    print(f"  Errors   : {errors:,}")
    print(f"  AI mentions confirmed : {non_none:,}")
else:
    print("No results — check job status.")


# %% ─── PHASE 2: PREPARE ─────────────────────────────────────────────────────
#
# Filter to chunks confirmed as real AI mentions, then build per-classifier
# batch request files for adoption, risk, and vendor classifiers.

from src.utils.prompt_loader import get_prompt_messages
from src.classifiers.schemas import (
    AdoptionTypeResponse,
    RiskResponse,
    VendorResponse,
    AdoptionSubstantivenessResponse,
    RiskSubstantivenessResponse,
    VendorSubstantivenessResponse,
)
from src.classifiers.base_classifier import _clean_schema_for_gemini
from pydantic import BaseModel
import re

# Load phase 1 results if not already in memory
if "p1_results" not in dir() or not p1_results:
    p1_out = RUNS_DIR / f"{p1_run_id}.results.jsonl"
    with open(p1_out) as f:
        p1_results = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(p1_results):,} phase 1 results from disk")

# Build lookup: chunk_id → mention_types
p1_lookup: dict[str, list[str]] = {
    r["chunk_id"]: r.get("llm_mention_types", [])
    for r in p1_results
    if not r.get("error")
}

# Annotate original chunks and filter to confirmed mentions
confirmed_chunks = []
for chunk in chunks:
    mention_types = p1_lookup.get(chunk["chunk_id"], [])
    if mention_types and mention_types != ["none"]:
        chunk = dict(chunk)
        chunk["mention_types"] = mention_types
        confirmed_chunks.append(chunk)

print(f"Confirmed AI mentions: {len(confirmed_chunks):,} / {len(chunks):,}")

# Subsets by downstream classifier
adoption_chunks = [c for c in confirmed_chunks if "adoption" in c["mention_types"]]
risk_chunks     = [c for c in confirmed_chunks if "risk"     in c["mention_types"]]
vendor_chunks   = [c for c in confirmed_chunks if "vendor"   in c["mention_types"]]

print(f"  adoption : {len(adoption_chunks):,}")
print(f"  risk     : {len(risk_chunks):,}")
print(f"  vendor   : {len(vendor_chunks):,}")


def prepare_phase2_batch(
    run_id: str,
    p2_chunks: list[dict],
    prompt_name: str,
    schema_cls: type[BaseModel],
) -> Path:
    """Write a phase 2 batch JSONL for a given downstream classifier."""
    response_schema = _clean_schema_for_gemini(schema_cls.model_json_schema())
    out_path = RUNS_DIR / f"{run_id}.batch_input.jsonl"

    with out_path.open("w") as f:
        for i, chunk in enumerate(p2_chunks):
            filing_year    = chunk["filing_date"][:4] if chunk.get("filing_date") else "Unknown"
            system_prompt, user_prompt = get_prompt_messages(
                prompt_name,
                reasoning_policy="short",
                firm_name=chunk.get("company_name", "Unknown"),
                sector="Unknown",
                report_year=filing_year,
                report_section="Unknown",
                mention_types=", ".join(chunk.get("mention_types", [])),
                text=chunk["chunk_text"][:30000],
            )
            line = {
                "key": str(i),
                "request": {
                    "contents": [{"parts": [{"text": user_prompt}], "role": "user"}],
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "generation_config": {
                        "temperature": TEMPERATURE,
                        "max_output_tokens": 2048,
                        "response_mime_type": "application/json",
                        "response_schema": response_schema,
                        "thinking_config": {"thinking_budget": THINKING},
                    },
                },
            }
            f.write(json.dumps(line) + "\n")

    print(f"Wrote {len(p2_chunks)} requests → {out_path.name}")
    return out_path


p2_run_id = f"{RUN_ID}-phase2"

adoption_input  = prepare_phase2_batch(f"{p2_run_id}-adoption",       adoption_chunks, "adoption_type",            AdoptionTypeResponse)
risk_input      = prepare_phase2_batch(f"{p2_run_id}-risk",           risk_chunks,     "risk_v5",                  RiskResponse)
vendor_input    = prepare_phase2_batch(f"{p2_run_id}-vendor",         vendor_chunks,   "vendor",                   VendorResponse)
adoption_sub_input = prepare_phase2_batch(f"{p2_run_id}-adoption-sub", adoption_chunks, "adoption_substantiveness_v1", AdoptionSubstantivenessResponse)
risk_sub_input     = prepare_phase2_batch(f"{p2_run_id}-risk-sub",     risk_chunks,     "risk_substantiveness_v1",    RiskSubstantivenessResponse)
vendor_sub_input   = prepare_phase2_batch(f"{p2_run_id}-vendor-sub",   vendor_chunks,   "vendor_substantiveness_v1",  VendorSubstantivenessResponse)


# %% ─── PHASE 2: SUBMIT ───────────────────────────────────────────────────────

p2_jobs: dict[str, str] = {}

if adoption_chunks:
    p2_jobs["adoption"]     = batch.submit(f"{p2_run_id}-adoption",     adoption_input,     model_name=MODEL)
    p2_jobs["adoption-sub"] = batch.submit(f"{p2_run_id}-adoption-sub", adoption_sub_input, model_name=MODEL)
if risk_chunks:
    p2_jobs["risk"]         = batch.submit(f"{p2_run_id}-risk",         risk_input,         model_name=MODEL)
    p2_jobs["risk-sub"]     = batch.submit(f"{p2_run_id}-risk-sub",     risk_sub_input,     model_name=MODEL)
if vendor_chunks:
    p2_jobs["vendor"]       = batch.submit(f"{p2_run_id}-vendor",       vendor_input,       model_name=MODEL)
    p2_jobs["vendor-sub"]   = batch.submit(f"{p2_run_id}-vendor-sub",   vendor_sub_input,   model_name=MODEL)

# Persist job names
(RUNS_DIR / f"{p2_run_id}.jobs.json").write_text(json.dumps(p2_jobs, indent=2))
print("Phase 2 jobs submitted:", p2_jobs)


# %% ─── PHASE 2: CHECK STATUS (re-run until all SUCCEEDED) ───────────────────

# Uncomment to restore after session restart:
# p2_jobs = json.loads((RUNS_DIR / f"{p2_run_id}.jobs.json").read_text())

for label, job_name in p2_jobs.items():
    print(f"\n── {label} ──")
    batch.check_status(job_name)


# %% ─── PHASE 2: SAVE & MERGE RESULTS ────────────────────────────────────────

def get_p2_raw(label: str, job_name: str, p2_chunks: list[dict]) -> list[dict]:
    """Download phase 2 batch results and return parsed JSON per chunk."""
    job = batch.client.batches.get(name=job_name)
    if job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"{label}: not ready ({job.state.name})")
        return []

    dest = getattr(job, "dest", None)
    result_file = getattr(dest, "file_name", None) if dest else None
    if not result_file:
        print(f"{label}: no result file found")
        return []

    raw_bytes = batch.client.files.download(file=result_file)
    output_lines = [json.loads(l) for l in raw_bytes.decode("utf-8").splitlines() if l.strip()]

    results = []
    for i, chunk in enumerate(p2_chunks):
        entry = next((e for e in output_lines if str(e.get("key")) == str(i)), None)
        if not entry or entry.get("error"):
            results.append({"chunk_id": chunk["chunk_id"], "error": str(entry.get("error") if entry else "missing")})
            continue
        try:
            parts = entry["response"]["candidates"][0]["content"]["parts"]
            parsed = json.loads(parts[0]["text"])
            results.append({"chunk_id": chunk["chunk_id"], "parsed": parsed})
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            results.append({"chunk_id": chunk["chunk_id"], "error": str(e)})

    out = RUNS_DIR / f"{p2_run_id}-{label}.results.jsonl"
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    errors = sum(1 for r in results if r.get("error"))
    print(f"{label}: {len(results):,} results ({errors} errors) → {out.name}")
    return results


adoption_results     = get_p2_raw("adoption",     p2_jobs.get("adoption",     ""), adoption_chunks)
risk_results         = get_p2_raw("risk",         p2_jobs.get("risk",         ""), risk_chunks)
vendor_results       = get_p2_raw("vendor",       p2_jobs.get("vendor",       ""), vendor_chunks)
adoption_sub_results = get_p2_raw("adoption-sub", p2_jobs.get("adoption-sub", ""), adoption_chunks)
risk_sub_results     = get_p2_raw("risk-sub",     p2_jobs.get("risk-sub",     ""), risk_chunks)
vendor_sub_results   = get_p2_raw("vendor-sub",   p2_jobs.get("vendor-sub",   ""), vendor_chunks)

# Merge all results keyed by chunk_id
merged: dict[str, dict] = {}

for chunk in confirmed_chunks:
    cid = chunk["chunk_id"]
    merged[cid] = {
        "chunk_id":       cid,
        "filing_id":      chunk.get("filing_id"),
        "company_name":   chunk.get("company_name"),
        "company_slug":   chunk.get("company_slug"),
        "company_ticker": chunk.get("company_ticker"),
        "filing_date":    chunk.get("filing_date"),
        "filing_type":    chunk.get("filing_type"),
        "chunk_text":     chunk.get("chunk_text"),
        "word_count":     chunk.get("word_count"),
        "matched_keywords": chunk.get("matched_keywords"),
        "mention_types":            chunk.get("mention_types", []),
        "adoption_types":           [],
        "risk_types":               [],
        "vendor_tags":              [],
        "adoption_substantiveness": None,
        "risk_substantiveness":     None,
        "vendor_substantiveness":   None,
    }

for r in adoption_results:
    if r["chunk_id"] in merged and "parsed" in r:
        p = r["parsed"]
        signals = p.get("adoption_signals") or p.get("adoption_confidences") or {}
        if isinstance(signals, list):
            merged[r["chunk_id"]]["adoption_types"] = [e["type"] for e in signals if isinstance(e, dict)]
        elif isinstance(signals, dict):
            merged[r["chunk_id"]]["adoption_types"] = list(signals.keys())

for r in risk_results:
    if r["chunk_id"] in merged and "parsed" in r:
        p = r["parsed"]
        risk_types = [str(rt) for rt in (p.get("risk_types") or []) if str(rt) != "none"]
        merged[r["chunk_id"]]["risk_types"] = risk_types

for r in vendor_results:
    if r["chunk_id"] in merged and "parsed" in r:
        p = r["parsed"]
        vendor_tags = list((p.get("vendor_confidences") or {}).keys())
        if p.get("other_vendor"):
            vendor_tags.append(f"other:{p['other_vendor']}")
        merged[r["chunk_id"]]["vendor_tags"] = vendor_tags

for r in adoption_sub_results:
    if r["chunk_id"] in merged and "parsed" in r:
        merged[r["chunk_id"]]["adoption_substantiveness"] = r["parsed"].get("substantiveness")

for r in risk_sub_results:
    if r["chunk_id"] in merged and "parsed" in r:
        merged[r["chunk_id"]]["risk_substantiveness"] = r["parsed"].get("substantiveness")

for r in vendor_sub_results:
    if r["chunk_id"] in merged and "parsed" in r:
        merged[r["chunk_id"]]["vendor_substantiveness"] = r["parsed"].get("substantiveness")

# Save final merged output
final_out = RUNS_DIR / f"{p2_run_id}.classified_chunks.jsonl"
with open(final_out, "w") as f:
    for record in merged.values():
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\nFinal output: {final_out}")
print(f"  Classified chunks: {len(merged):,}")
