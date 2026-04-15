#!/usr/bin/env python3
"""
Standalone substantiveness batch classification.

Loads confirmed AI-mention chunks from an existing annotations JSONL (phase 1 already done),
then runs three new standalone substantiveness classifiers as Gemini batch jobs:
  - adoption_substantiveness_v1  (adoption chunks)
  - risk_substantiveness_v1      (risk chunks)
  - vendor_substantiveness_v1    (vendor chunks)

A chunk with multiple mention types is scored by all relevant classifiers.

Output is written to a versioned subdirectory under data/results/substantiveness/
and never touches the existing annotations or DB.

Cell-based workflow (run cells in order):
  1. SETUP    — configure run, load source annotations
  2. PREPARE  — build batch request files
  3. SUBMIT   — submit to Gemini Batch API
  4. POLL     — check status (re-run until all SUCCEEDED)
  5. SAVE     — download results, write output JSONL
"""

# %% ─── SETUP ────────────────────────────────────────────────────────────────

import json
import os
import sys
from datetime import datetime
from pathlib import Path

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

from src.utils.batch_api import BatchClient
from src.utils.prompt_loader import get_prompt_messages
from src.classifiers.schemas import (
    AdoptionSubstantivenessResponse,
    RiskSubstantivenessResponse,
    VendorSubstantivenessResponse,
)
from src.classifiers.base_classifier import _clean_schema_for_gemini
from pydantic import BaseModel

# ── Source annotations ────────────────────────────────────────────────────────
SOURCE_ANNOTATIONS = (
    REPO_ROOT
    / "data/results/definitive_main_market_1000"
    / "p2-gemini-3-flash-preview-definitive-main-market-1000-v2.annotations.jsonl"
)

# ── Output directory (versioned, never touches existing data) ─────────────────
RUN_DATE = datetime.now().strftime("%Y%m%d-%H%M")
RUN_ID   = f"substantiveness-{RUN_DATE}"
OUT_DIR  = REPO_ROOT / "data/results/substantiveness" / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model config ──────────────────────────────────────────────────────────────
MODEL       = "gemini-3-flash-preview"
TEMPERATURE = 0.0
THINKING    = 0

batch = BatchClient(runs_dir=OUT_DIR)

# ── Load and filter source annotations ───────────────────────────────────────
all_chunks: list[dict] = []
with open(SOURCE_ANNOTATIONS) as f:
    for line in f:
        if line.strip():
            all_chunks.append(json.loads(line))

adoption_chunks = [c for c in all_chunks if "adoption" in (c.get("mention_types") or [])]
risk_chunks     = [c for c in all_chunks if "risk"     in (c.get("mention_types") or [])]
vendor_chunks   = [c for c in all_chunks if "vendor"   in (c.get("mention_types") or [])]

print(f"Run ID       : {RUN_ID}")
print(f"Output dir   : {OUT_DIR}")
print(f"Source       : {SOURCE_ANNOTATIONS.name}")
print(f"Total chunks : {len(all_chunks):,}")
print(f"  adoption   : {len(adoption_chunks):,}")
print(f"  risk       : {len(risk_chunks):,}")
print(f"  vendor     : {len(vendor_chunks):,}")


# %% ─── PREPARE ──────────────────────────────────────────────────────────────

def prepare_batch(
    label: str,
    chunks: list[dict],
    prompt_name: str,
    schema_cls: type[BaseModel],
) -> Path:
    """Write a Gemini batch input JSONL for a substantiveness classifier."""
    response_schema = _clean_schema_for_gemini(schema_cls.model_json_schema())
    out_path = OUT_DIR / f"{RUN_ID}-{label}.batch_input.jsonl"

    with out_path.open("w") as f:
        for i, chunk in enumerate(chunks):
            report_year   = str(chunk.get("report_year", "Unknown"))
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
            line = {
                "key": str(i),
                "request": {
                    "contents": [{"parts": [{"text": user_prompt}], "role": "user"}],
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "generation_config": {
                        "temperature": TEMPERATURE,
                        "max_output_tokens": 1024,
                        "response_mime_type": "application/json",
                        "response_schema": response_schema,
                        "thinking_config": {"thinking_budget": THINKING},
                    },
                },
            }
            f.write(json.dumps(line) + "\n")

    print(f"Prepared {len(chunks):,} requests → {out_path.name}")
    return out_path


adoption_input = prepare_batch("adoption", adoption_chunks, "adoption_substantiveness_v1", AdoptionSubstantivenessResponse)
risk_input     = prepare_batch("risk",     risk_chunks,     "risk_substantiveness_v1",     RiskSubstantivenessResponse)
vendor_input   = prepare_batch("vendor",   vendor_chunks,   "vendor_substantiveness_v1",   VendorSubstantivenessResponse)


# %% ─── SUBMIT ────────────────────────────────────────────────────────────────

jobs: dict[str, str] = {}

if adoption_chunks:
    jobs["adoption"] = batch.submit(f"{RUN_ID}-adoption", adoption_input, model_name=MODEL)
if risk_chunks:
    jobs["risk"]     = batch.submit(f"{RUN_ID}-risk",     risk_input,     model_name=MODEL)
if vendor_chunks:
    jobs["vendor"]   = batch.submit(f"{RUN_ID}-vendor",   vendor_input,   model_name=MODEL)

(OUT_DIR / f"{RUN_ID}.jobs.json").write_text(json.dumps(jobs, indent=2))
print("Jobs submitted:", jobs)


# %% ─── POLL ─────────────────────────────────────────────────────────────────

# Uncomment to restore after session restart:
# jobs = json.loads((OUT_DIR / f"{RUN_ID}.jobs.json").read_text())

for label, job_name in jobs.items():
    print(f"\n── {label} ──")
    batch.check_status(job_name)


# %% ─── SAVE ─────────────────────────────────────────────────────────────────

def get_results(label: str, job_name: str, chunks: list[dict]) -> list[dict]:
    """Download batch results and return parsed records keyed by chunk_id."""
    job = batch.client.batches.get(name=job_name)
    if job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"{label}: not ready ({job.state.name})")
        return []

    dest = getattr(job, "dest", None)
    result_file = getattr(dest, "file_name", None) if dest else None
    if not result_file:
        print(f"{label}: no result file found")
        return []

    raw_bytes    = batch.client.files.download(file=result_file)
    output_lines = [json.loads(l) for l in raw_bytes.decode("utf-8").splitlines() if l.strip()]

    results = []
    for i, chunk in enumerate(chunks):
        entry = next((e for e in output_lines if str(e.get("key")) == str(i)), None)
        if not entry or entry.get("error"):
            results.append({
                "chunk_id": chunk["chunk_id"],
                "error": str(entry.get("error") if entry else "missing"),
            })
            continue
        try:
            parts  = entry["response"]["candidates"][0]["content"]["parts"]
            parsed = json.loads(parts[0]["text"])
            results.append({"chunk_id": chunk["chunk_id"], "parsed": parsed})
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            results.append({"chunk_id": chunk["chunk_id"], "error": str(e)})

    raw_out = OUT_DIR / f"{RUN_ID}-{label}.results.jsonl"
    with open(raw_out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    errors = sum(1 for r in results if r.get("error"))
    print(f"{label}: {len(results):,} results ({errors} errors) → {raw_out.name}")
    return results


adoption_results = get_results("adoption", jobs.get("adoption", ""), adoption_chunks)
risk_results     = get_results("risk",     jobs.get("risk",     ""), risk_chunks)
vendor_results   = get_results("vendor",   jobs.get("vendor",   ""), vendor_chunks)

# Build lookup: chunk_id → substantiveness per classifier
adoption_sub: dict[str, str | None] = {
    r["chunk_id"]: r["parsed"].get("substantiveness")
    for r in adoption_results if "parsed" in r
}
risk_sub: dict[str, str | None] = {
    r["chunk_id"]: r["parsed"].get("substantiveness")
    for r in risk_results if "parsed" in r
}
vendor_sub: dict[str, str | None] = {
    r["chunk_id"]: r["parsed"].get("substantiveness")
    for r in vendor_results if "parsed" in r
}

# Write merged output — one record per scored chunk, all classifiers that ran
scored: dict[str, dict] = {}
for chunk in adoption_chunks + risk_chunks + vendor_chunks:
    cid = chunk["chunk_id"]
    if cid in scored:
        continue
    scored[cid] = {
        "chunk_id":               cid,
        "company_name":           chunk.get("company_name"),
        "report_year":            chunk.get("report_year"),
        "mention_types":          chunk.get("mention_types", []),
        "adoption_substantiveness": adoption_sub.get(cid),
        "risk_substantiveness":     risk_sub.get(cid),
        "vendor_substantiveness":   vendor_sub.get(cid),
    }

final_out = OUT_DIR / f"{RUN_ID}.substantiveness_scores.jsonl"
with open(final_out, "w") as f:
    for record in scored.values():
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

total   = len(scored)
scored_adoption = sum(1 for r in scored.values() if r["adoption_substantiveness"])
scored_risk     = sum(1 for r in scored.values() if r["risk_substantiveness"])
scored_vendor   = sum(1 for r in scored.values() if r["vendor_substantiveness"])

print(f"\nOutput: {final_out}")
print(f"  Total chunks scored  : {total:,}")
print(f"  adoption_substantiveness scored : {scored_adoption:,}")
print(f"  risk_substantiveness scored     : {scored_risk:,}")
print(f"  vendor_substantiveness scored   : {scored_vendor:,}")
