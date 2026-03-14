#!/usr/bin/env python3
"""Build a CNI sector mapping for all companies in the universe.

Uses the ISIC sub-industry codes already provided by the FR API (99.9% coverage)
as the primary signal, with three phases:

  Phase 1 — static
    Apply hard ISIC→CNI rules for codes with a clear, unambiguous CNI relationship.
    Writes data/reference/company_cni_sectors.csv with all statically-classified
    companies, leaving the rest blank.

  Phase 2 — submit
    For every company without a static mapping, build one Gemini prompt per company
    (using name + ISIC code + description + tagline from the FR API) and submit the
    whole batch to the Gemini Batch API. Saves the job name to a checkpoint file.

  Phase 3 — collect
    Check batch status. When complete, download results, merge into
    company_cni_sectors.csv, and run the ISIC-consistency sanity check.

Outputs
───────
  data/reference/company_cni_sectors.csv   — one row per company
  data/reference/cni_batch_checkpoint.json — batch job metadata (phases 2–3)

Data model per company
───────────────────────
  cni_sector_primary  — first (most relevant) sector; use for all visualisation
  cni_sectors         — pipe-separated full list (1–3 sectors)
  cni_sector_count    — number of sectors assigned
  source              — static | llm_gemini | no_isic

Usage
─────
  python3 scripts/build_cni_sector_mapping.py --phase static
  python3 scripts/build_cni_sector_mapping.py --phase submit
  python3 scripts/build_cni_sector_mapping.py --phase collect
  python3 scripts/build_cni_sector_mapping.py --phase status
  python3 scripts/build_cni_sector_mapping.py --dry-run   # phase 1 preview only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / "data"
REF_DIR   = DATA_DIR / "reference"
RUNS_DIR  = REF_DIR / "cni_batch_runs"


def checkpoint_path(tag: str) -> Path:
    return REF_DIR / f"cni_batch_checkpoint_{tag}.json"

# ── CNI sector taxonomy (official UK post-2024) ───────────────────────────────

CNI_SECTORS = [
    "Chemicals",
    "Civil Nuclear",
    "Communications",
    "Data Infrastructure",
    "Defence",
    "Emergency Services",
    "Energy",
    "Finance",
    "Food",
    "Government",
    "Health",
    "Space",
    "Transport",
    "Water",
    "Other",
]

CNI_DESCRIPTIONS = {
    "Chemicals":          "Industrial and chemical manufacturing.",
    "Civil Nuclear":      "Nuclear power generation and nuclear waste management.",
    "Communications":     "Telecoms, internet infrastructure, and broadcast.",
    "Data Infrastructure":"Data centres and related hosting/processing infrastructure.",
    "Defence":            "Armed forces and supporting defence industry.",
    "Emergency Services": "Police, fire, ambulance, and coast guard.",
    "Energy":             "Oil, gas, coal, electricity generation, and renewables — extraction, refining, and distribution.",
    "Finance":            "Banking, investment funds, and insurance.",
    "Food":               "Food production, processing, supply chain, and retail.",
    "Government":         "Public administration, courts, and regional government.",
    "Health":             "Hospitals, medical services, pharmaceuticals, and public health.",
    "Space":              "Satellite, communication, and space-based technology.",
    "Transport":          "Rail, road, air, and maritime transport.",
    "Water":              "Water supply, sewerage, and wastewater treatment.",
    "Other":              "Not a CNI sector.",
}

# Company-specific overrides for cases where the FR API leaves industry metadata
# blank but we have a verified local correction.
MANUAL_COMPANY_OVERRIDES: dict[str, dict[str, object]] = {
    "2138001MKE18HLW9YX42": {
        "isic_code": "0729",
        "isic_name": "Mining of other non-ferrous metal ores",
        "isic_section_code": "B",
        "isic_section_name": "Mining and quarrying",
        # Mining alone does not imply a clear CNI relationship here, so keep the
        # manual assignment conservative.
        "cni_sectors": ["Other"],
        "source": "manual_override",
    },
}

# ── Static ISIC → CNI rules ───────────────────────────────────────────────────
# Only codes with a CLEAR and UNAMBIGUOUS CNI relationship.
# Lookup order: exact 4-digit → 3-digit prefix → 2-digit prefix.
# More specific rules always win.

EXACT_RULES: dict[str, str] = {
    # Specific 4-digit codes that would otherwise be caught by a broader rule
    # incorrectly, or that need pinning to a specific sector.
    "1920": "Energy",             # Petroleum refining (in div 19, not elsewhere)
    "2540": "Defence",            # Weapons and ammunition
    "3040": "Defence",            # Military fighting vehicles
    "5122": "Space",              # Space transport (within div 51 → Transport)
    "6310": "Data Infrastructure",# Data processing and hosting
    "6311": "Data Infrastructure",# Web portals / data hosting (sub of 6310)
    "8424": "Emergency Services", # Public order and safety (within div 84 → Government)
}

PREFIX_3_RULES: dict[str, str] = {
    "091": "Energy",   # Support activities for petroleum and natural gas extraction
}

PREFIX_2_RULES: dict[str, str] = {
    "01": "Food",           # Agriculture
    "03": "Food",           # Fishing and aquaculture
    "05": "Energy",         # Coal mining
    "06": "Energy",         # Oil and gas extraction
    "10": "Food",           # Food product manufacturing
    "11": "Food",           # Beverage manufacturing
    "20": "Chemicals",      # Chemical manufacturing (all sub-codes)
    "21": "Health",         # Pharmaceutical manufacturing
    "35": "Energy",         # Electricity and gas utilities
    "36": "Water",          # Water collection, treatment and supply
    "37": "Water",          # Sewerage
    "38": "Water",          # Waste collection and treatment
    "49": "Transport",      # Land transport and pipelines
    "50": "Transport",      # Water transport (shipping)
    "51": "Transport",      # Air transport (except 5122 → Space, caught above)
    "60": "Communications", # Broadcasting
    "61": "Communications", # Telecommunications
    "64": "Finance",        # Financial service activities
    "65": "Finance",        # Insurance and pension funding
    "66": "Finance",        # Activities auxiliary to finance and insurance
    "84": "Government",     # Public administration (except 8424, caught above)
    "86": "Health",         # Health services
}


def lookup_isic(code: str) -> str | None:
    """Return CNI sector for an ISIC code, or None if no static rule applies."""
    code = str(code).strip().zfill(4)
    if code in EXACT_RULES:
        return EXACT_RULES[code]
    prefix3 = code[:3]
    if prefix3 in PREFIX_3_RULES:
        return PREFIX_3_RULES[prefix3]
    prefix2 = code[:2]
    return PREFIX_2_RULES.get(prefix2)


def apply_company_override(lei: str, fr: dict, name: str) -> tuple[dict, dict[str, object] | None]:
    """Merge any verified company-level override into the FR payload."""
    merged = {
        "company_name": name,
        "isic_code": "",
        "isic_name": "",
        "isic_section_code": "",
        "isic_section_name": "",
        "description": "",
        "tagline": "",
        **fr,
    }
    override = MANUAL_COMPANY_OVERRIDES.get(lei)
    if not override:
        return merged, None
    for key in (
        "isic_code",
        "isic_name",
        "isic_section_code",
        "isic_section_name",
        "description",
        "tagline",
    ):
        if key in override:
            merged[key] = override[key]
    return merged, override


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_env() -> None:
    env_path = REPO_ROOT / ".env.local"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def load_fr_companies() -> dict[str, dict]:
    """Load FR API company data keyed by LEI."""
    with open(DATA_DIR / "FR_dataset" / "companies.json") as f:
        raw = json.load(f)
    result = {}
    for c in raw:
        lei = c.get("lei", "")
        if not lei:
            continue
        si  = c.get("sub_industry") or {}
        sec = c.get("sector") or {}
        result[lei] = {
            "company_name":      c.get("name", ""),
            "isic_code":         si.get("code", ""),
            "isic_name":         si.get("name", ""),
            "isic_section_code": sec.get("code", ""),
            "isic_section_name": sec.get("name", ""),
            "description":       c.get("description") or "",
            "tagline":           c.get("tagline") or "",
        }
    return result


def load_universe() -> dict[str, str]:
    """Load all in-scope companies from ch_coverage. Returns {lei: name}."""
    universe = {}
    with open(DATA_DIR / "FR_dataset" / "ch_coverage.csv") as f:
        for r in csv.DictReader(f):
            if r.get("lei"):
                universe[r["lei"]] = r["name"]
    return universe


def load_existing_output() -> dict[str, dict]:
    """Load company_cni_sectors.csv if it exists. Returns {lei: row}."""
    out_path = REF_DIR / "company_cni_sectors.csv"
    if not out_path.exists():
        return {}
    with open(out_path) as f:
        return {r["lei"]: r for r in csv.DictReader(f) if r.get("lei")}


OUT_FIELDS = [
    "lei", "company_name",
    "cni_sector_primary", "cni_sectors", "cni_sector_count",
    "isic_code", "isic_name", "isic_section_code", "isic_section_name",
    "source",
]


def make_row(lei: str, name: str, sectors: list[str], fr: dict, source: str) -> dict:
    sectors = [s for s in sectors if s in CNI_SECTORS]
    if not sectors:
        sectors = ["Other"]
    return {
        "lei":               lei,
        "company_name":      name,
        "cni_sector_primary": sectors[0],
        "cni_sectors":       "|".join(sectors),
        "cni_sector_count":  len(sectors),
        "isic_code":         fr.get("isic_code", ""),
        "isic_name":         fr.get("isic_name", ""),
        "isic_section_code": fr.get("isic_section_code", ""),
        "isic_section_name": fr.get("isic_section_name", ""),
        "source":            source,
    }


def write_output(rows: list[dict]) -> None:
    out_path = REF_DIR / "company_cni_sectors.csv"
    REF_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {out_path.relative_to(REPO_ROOT)}")


# ── Phase 1: Static mapping ───────────────────────────────────────────────────

def phase_static(dry_run: bool = False) -> None:
    print("=== Phase 1: Static ISIC mapping ===\n")

    universe   = load_universe()
    fr_data    = load_fr_companies()
    existing   = load_existing_output()

    static_rows: list[dict] = []
    needs_llm:  list[dict]  = []   # {lei, company_name, ...fr fields}
    no_isic:    list[dict]  = []

    for lei, name in sorted(universe.items(), key=lambda x: x[1]):
        fr, override = apply_company_override(
            lei,
            fr_data.get(lei, {}),
            name,
        )
        code   = fr.get("isic_code", "")
        sector = lookup_isic(code) if code else None

        if override and override.get("cni_sectors"):
            static_rows.append(make_row(
                lei,
                name,
                list(override["cni_sectors"]),
                fr,
                str(override.get("source", "manual_override")),
            ))
        elif sector:
            static_rows.append(make_row(lei, name, [sector], fr, "static"))
        elif code:
            needs_llm.append({"lei": lei, "company_name": name, **fr})
        else:
            no_isic.append({"lei": lei, "company_name": name, **fr})

    total = len(universe)
    print(f"Universe:       {total} companies")
    print(f"Static mapped:  {len(static_rows)} ({100*len(static_rows)/total:.1f}%)")
    print(f"Needs LLM:      {len(needs_llm)} ({100*len(needs_llm)/total:.1f}%)")
    print(f"No ISIC code:   {len(no_isic)}")

    if dry_run:
        print("\nDry run — nothing written.")
        # Show sample of each bucket
        print("\nSample static:")
        for r in static_rows[:5]:
            print(f"  {r['company_name']:<45s} {r['isic_code']}  → {r['cni_sector_primary']}")
        print("\nSample needs_llm:")
        for r in needs_llm[:5]:
            print(f"  {r['company_name']:<45s} {r['isic_code']}  {r['isic_name']}")
        return

    # Build full output: static rows + empty placeholders for LLM/no_isic
    all_rows: list[dict] = list(static_rows)
    for item in needs_llm + no_isic:
        all_rows.append({
            "lei":               item["lei"],
            "company_name":      item["company_name"],
            "cni_sector_primary":"",
            "cni_sectors":       "",
            "cni_sector_count":  "",
            "isic_code":         item.get("isic_code", ""),
            "isic_name":         item.get("isic_name", ""),
            "isic_section_code": item.get("isic_section_code", ""),
            "isic_section_name": item.get("isic_section_name", ""),
            "source":            "no_isic" if not item.get("isic_code") else "pending_llm",
        })

    all_rows.sort(key=lambda r: r["company_name"])
    write_output(all_rows)
    print(f"\nPhase 1 complete. Run --phase submit next.")


# ── Phase 2: Submit batch ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a sector analyst classifying companies into UK Critical National \
Infrastructure (CNI) sectors.

Given the information below about a company, assign it to the most appropriate \
CNI sector(s). Return between 1 and 3 sectors, ordered from most to least relevant.

CNI sectors:
{sector_list}

Rules:
- Only assign a CNI sector when the company has a genuine relationship with \
that infrastructure sector — as an operator, direct supplier, or essential \
service provider.
- Civil Nuclear covers nuclear power generation and waste management specifically \
(not general electricity generation).
- Defence includes weapons manufacturers, military vehicle/aircraft makers, and \
defence electronics — not general engineering.
- Government covers companies whose primary business is delivering outsourced \
public services (e.g. prisons, benefits processing, government IT).
- Data Infrastructure covers companies operating data centres and cloud hosting \
infrastructure — not general software companies.
- Space covers satellite operators and space-based technology — not general \
aerospace or aviation.
- Use "Other" when no CNI sector clearly applies.
- Output ONLY valid JSON: {{"sectors": ["Sector1"], "reasoning": "One sentence."}}
"""


def build_sector_list() -> str:
    return "\n".join(f"  {s}: {CNI_DESCRIPTIONS[s]}" for s in CNI_SECTORS)


# Response schema — enforces exact sector names via enum, eliminating all
# normalisation guesswork. The model MUST return names from this list.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sectors": {
            "type": "array",
            "items": {"type": "string", "enum": CNI_SECTORS},
            "minItems": 1,
            "maxItems": 3,
        },
        "reasoning": {"type": "string"},
    },
    "required": ["sectors"],
}


def phase_submit(model: str = "gemini-3-flash-preview", tag: str = "") -> None:
    print("=== Phase 2: Submit batch ===\n")

    load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    existing = load_existing_output()
    fr_data  = load_fr_companies()

    # Collect companies still needing LLM
    pending = [
        row for row in existing.values()
        if row.get("source") == "pending_llm"
    ]
    print(f"Companies pending LLM classification: {len(pending)}")
    if not pending:
        print("Nothing to submit. All companies are already classified.")
        return

    # Build JSONL
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    run_id    = f"cni-sectors-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    jsonl_path = RUNS_DIR / f"{run_id}.batch_input.jsonl"

    sector_list  = build_sector_list()
    system_text  = SYSTEM_PROMPT.format(sector_list=sector_list)

    with jsonl_path.open("w") as f:
        for i, row in enumerate(pending):
            lei  = row["lei"]
            name = row["company_name"]
            fr   = fr_data.get(lei, {})
            code = fr.get("isic_code", row.get("isic_code", ""))
            isic_name = fr.get("isic_name", row.get("isic_name", ""))
            desc = (fr.get("description") or "")[:1000]   # cap at 1000 chars
            tagline = fr.get("tagline") or ""

            user_text = (
                f"Company: {name}\n"
                f"ISIC Code: {code} — {isic_name}\n"
                f"Tagline: {tagline}\n"
                f"Description: {desc}\n"
            )

            line = {
                "key": str(i),
                "request": {
                    "contents": [{"parts": [{"text": user_text}], "role": "user"}],
                    "system_instruction": {"parts": [{"text": system_text}]},
                    "generation_config": {
                        "temperature": 0.0,
                        "max_output_tokens": 2048,
                        "response_mime_type": "application/json",
                        "response_json_schema": RESPONSE_SCHEMA,
                        "thinking_config": {"thinking_budget": 512},
                    },
                },
            }
            f.write(json.dumps(line) + "\n")

    # Pre-submission validation: parse first 3 lines back to catch format bugs
    print(f"Wrote {len(pending)} requests → {jsonl_path.name}")
    print("Validating JSONL...")
    with jsonl_path.open() as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            parsed = json.loads(line)
            assert "key" in parsed and "request" in parsed, f"Line {i}: missing key/request"
            req = parsed["request"]
            assert "contents" in req and "system_instruction" in req, f"Line {i}: missing contents/system_instruction"
    print(f"  OK — first 3 requests validated")

    # Submit via Gemini Batch API
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model_name = model if model.startswith("models/") else f"models/{model}"

    uploaded = client.files.upload(
        file=str(jsonl_path),
        config=types.UploadFileConfig(
            display_name=run_id,
            mime_type="jsonl",
        ),
    )
    print(f"Uploaded: {uploaded.name}")

    batch_job = client.batches.create(
        model=model_name,
        src=uploaded.name,
        config={"display_name": run_id},
    )
    job_name = batch_job.name
    print(f"Submitted batch job: {job_name}")
    print(f"  State: {batch_job.state.name}")
    print(f"  Requests: {len(pending)}")

    # Save checkpoint
    checkpoint = {
        "job_name":    job_name,
        "run_id":      run_id,
        "model":       model_name,
        "num_pending": len(pending),
        "jsonl_path":  str(jsonl_path),
        "uploaded_file": uploaded.name,
        "submitted_at": __import__("datetime").datetime.now().isoformat(),
        # Store index → lei mapping for result collection
        "index_to_lei": {str(i): row["lei"] for i, row in enumerate(pending)},
        "tag": tag,
    }
    cp_path = checkpoint_path(tag)
    cp_path.write_text(json.dumps(checkpoint, indent=2))
    print(f"\nCheckpoint saved → {cp_path.relative_to(REPO_ROOT)}")
    print(f"\nRun --phase status --tag {tag} to check progress.")
    print(f"Run --phase collect --tag {tag} once the job succeeds.")


# ── Phase 3 helper: status ────────────────────────────────────────────────────

def phase_status(tag: str = "") -> None:
    print("=== Batch status ===\n")
    load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    cp_path = checkpoint_path(tag)
    if not cp_path.exists():
        print(f"No checkpoint found at {cp_path.name}. Run --phase submit --tag {tag} first.")
        return

    cp = json.loads(cp_path.read_text())
    job_name = cp["job_name"]

    from google import genai
    client = genai.Client(api_key=api_key)
    job = client.batches.get(name=job_name)

    state = job.state.name
    print(f"Job:   {job_name}")
    print(f"State: {state}")
    if hasattr(job, "batch_stats") and job.batch_stats:
        stats = job.batch_stats
        total     = getattr(stats, "total_count", "?")
        succeeded = getattr(stats, "success_count", "?")
        failed    = getattr(stats, "failed_count", "?")
        print(f"Progress: {succeeded}/{total} succeeded, {failed} failed")


# ── Phase 4: Collect results ──────────────────────────────────────────────────

def phase_collect(tag: str = "") -> None:
    print("=== Phase 3: Collect results ===\n")

    load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    cp_path = checkpoint_path(tag)
    if not cp_path.exists():
        print(f"No checkpoint found at {cp_path.name}. Run --phase submit --tag {tag} first.")
        return

    cp          = json.loads(cp_path.read_text())
    job_name    = cp["job_name"]
    index_to_lei = cp["index_to_lei"]

    from google import genai
    client = genai.Client(api_key=api_key)
    job = client.batches.get(name=job_name)

    if job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"Batch not ready yet: {job.state.name}")
        print("Run --phase status to check progress.")
        return

    # Download results
    output_lines: list[dict] = []
    dest = getattr(job, "dest", None)
    result_file = getattr(dest, "file_name", None) if dest else None
    if result_file:
        raw = client.files.download(file=result_file)
        for line in raw.decode("utf-8").splitlines():
            if line.strip():
                output_lines.append(json.loads(line))
        print(f"Downloaded {len(output_lines)} responses")
    elif dest and getattr(dest, "inlined_responses", None):
        for ir in dest.inlined_responses:
            entry = {}
            if getattr(ir, "key", None) is not None:
                entry["key"] = str(ir.key)
            if getattr(ir, "response", None):
                entry["response"] = ir.response.model_dump() if hasattr(ir.response, "model_dump") else {}
            output_lines.append(entry)
        print(f"Extracted {len(output_lines)} inlined responses")
    else:
        print("ERROR: No responses found in batch output.")
        return

    # Parse results
    llm_results: dict[str, list[str]] = {}  # lei → sectors list
    parse_errors: list[dict] = []

    for entry in output_lines:
        key = str(entry.get("key", ""))
        lei = index_to_lei.get(key)
        if not lei:
            parse_errors.append({"key": key, "lei": None, "reason": "key not in index"})
            continue

        # Check for API-level error on this request
        if entry.get("error"):
            parse_errors.append({"key": key, "lei": lei, "reason": f"api_error: {entry['error']}"})
            llm_results[lei] = ["Other"]
            continue

        try:
            candidates = entry.get("response", {}).get("candidates", [])
            if not candidates:
                raise ValueError("no candidates")

            finish_reason = candidates[0].get("finishReason", "")
            if finish_reason == "MAX_TOKENS":
                raise ValueError(f"MAX_TOKENS — increase max_output_tokens")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise ValueError(f"empty parts (finishReason={finish_reason})")

            text = parts[0].get("text", "")
            parsed = json.loads(text)
            sectors = parsed.get("sectors", [])
            # response_schema enforces enum values, but validate anyway
            valid = [s for s in sectors if s in CNI_SECTORS]
            if not valid:
                raise ValueError(f"no valid sectors in: {sectors}")
            llm_results[lei] = valid[:3]

        except Exception as e:
            parse_errors.append({"key": key, "lei": lei, "reason": str(e)})
            llm_results[lei] = ["Other"]

    succeeded = len(llm_results) - len([e for e in parse_errors if e.get("lei")])
    print(f"Parsed: {len(llm_results)} results | {len(parse_errors)} errors")

    if parse_errors:
        errors_path = REF_DIR / "cni_batch_errors.json"
        errors_path.write_text(json.dumps(parse_errors, indent=2))
        print(f"Error details → {errors_path.relative_to(REPO_ROOT)}")
        # Show first 5
        for e in parse_errors[:5]:
            print(f"  key={e['key']} lei={e.get('lei','?')}: {e['reason']}")

    # Merge into existing output
    existing  = load_existing_output()
    fr_data   = load_fr_companies()
    universe  = load_universe()

    merged: list[dict] = []
    for lei, name in sorted(universe.items(), key=lambda x: x[1]):
        fr = fr_data.get(lei, {})
        row = existing.get(lei, {})

        if row.get("source") == "pending_llm" and lei in llm_results:
            merged.append(make_row(lei, name, llm_results[lei], fr, "llm_gemini"))
        else:
            merged.append(row if row else {
                "lei": lei, "company_name": name,
                "cni_sector_primary": "", "cni_sectors": "",
                "cni_sector_count": "", "isic_code": fr.get("isic_code", ""),
                "isic_name": fr.get("isic_name", ""),
                "isic_section_code": fr.get("isic_section_code", ""),
                "isic_section_name": fr.get("isic_section_name", ""),
                "source": "no_isic",
            })

    write_output(merged)

    # ── Sanity check: ISIC code consistency ───────────────────────────────────
    print("\n=== Sanity check: ISIC code consistency ===")
    from collections import defaultdict, Counter

    isic_to_sectors: dict[str, list[list[str]]] = defaultdict(list)
    for row in merged:
        code = row.get("isic_code", "")
        sectors_str = row.get("cni_sectors", "")
        if code and sectors_str:
            isic_to_sectors[code].append(sectors_str.split("|"))

    inconsistent = []
    for code, sector_lists in sorted(isic_to_sectors.items()):
        primary_counts = Counter(sl[0] for sl in sector_lists)
        if len(primary_counts) > 1:
            top1, top2 = primary_counts.most_common(2)
            inconsistent.append((code, len(sector_lists), primary_counts))

    if inconsistent:
        print(f"\n{len(inconsistent)} ISIC codes with inconsistent primary sectors:")
        for code, n, counts in inconsistent[:20]:
            isic_name = fr_data.get(
                next((r["lei"] for r in merged if r.get("isic_code") == code), ""), {}
            ).get("isic_name", "")
            print(f"  {code}  ({isic_name[:40]})")
            for sector, cnt in counts.most_common():
                print(f"    {cnt:3}x  {sector}")
    else:
        print("All ISIC codes map consistently to the same primary sector.")

    # Summary
    from collections import Counter as C
    sc = C(r.get("cni_sector_primary", "") for r in merged if r.get("cni_sector_primary"))
    src = C(r.get("source", "") for r in merged)
    print("\n── CNI sector distribution ───────────────────────────────────")
    for sector, n in sorted(sc.items(), key=lambda x: -x[1]):
        print(f"  {sector:<25s}  {n:4}  ({100*n/len(merged):.1f}%)")
    print("\n── Source ────────────────────────────────────────────────────")
    for s, n in sorted(src.items(), key=lambda x: -x[1]):
        print(f"  {s:<20s}  {n:4}")


# ── Phase: load-db ────────────────────────────────────────────────────────────

DB_PATH = REPO_ROOT / "pipeline" / "pipeline" / "data" / "airo.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS company_cni_sectors (
    lei                 TEXT PRIMARY KEY,
    company_name        TEXT NOT NULL,
    cni_sector_primary  TEXT,
    cni_sectors         TEXT,
    cni_sector_count    INTEGER,
    isic_code           TEXT,
    isic_name           TEXT,
    isic_section_code   TEXT,
    isic_section_name   TEXT,
    source              TEXT,
    loaded_at           TEXT NOT NULL
)
"""

def phase_load_db() -> None:
    import sqlite3
    from datetime import datetime

    print("=== Load CNI sectors into database ===\n")

    rows = list(load_existing_output().values())
    if not rows:
        print("No data found. Run --phase static first.")
        return

    classified = [r for r in rows if r.get("cni_sector_primary")]
    print(f"Rows to load: {len(classified)} classified + {len(rows) - len(classified)} unclassified = {len(rows)} total")
    print(f"Database: {DB_PATH.relative_to(REPO_ROOT)}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(CREATE_TABLE_SQL)

    loaded_at = datetime.now().isoformat()
    upserted = 0
    for r in rows:
        conn.execute("""
            INSERT INTO company_cni_sectors
                (lei, company_name, cni_sector_primary, cni_sectors, cni_sector_count,
                 isic_code, isic_name, isic_section_code, isic_section_name, source, loaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(lei) DO UPDATE SET
                company_name        = excluded.company_name,
                cni_sector_primary  = excluded.cni_sector_primary,
                cni_sectors         = excluded.cni_sectors,
                cni_sector_count    = excluded.cni_sector_count,
                isic_code           = excluded.isic_code,
                isic_name           = excluded.isic_name,
                isic_section_code   = excluded.isic_section_code,
                isic_section_name   = excluded.isic_section_name,
                source              = excluded.source,
                loaded_at           = excluded.loaded_at
        """, (
            r.get("lei"), r.get("company_name"),
            r.get("cni_sector_primary") or None,
            r.get("cni_sectors") or None,
            int(r["cni_sector_count"]) if r.get("cni_sector_count") else None,
            r.get("isic_code") or None,
            r.get("isic_name") or None,
            r.get("isic_section_code") or None,
            r.get("isic_section_name") or None,
            r.get("source"), loaded_at,
        ))
        upserted += 1

    conn.commit()

    # Summary query
    total   = conn.execute("SELECT COUNT(*) FROM company_cni_sectors").fetchone()[0]
    by_src  = conn.execute("""
        SELECT source, COUNT(*) FROM company_cni_sectors GROUP BY source ORDER BY COUNT(*) DESC
    """).fetchall()
    by_sect = conn.execute("""
        SELECT cni_sector_primary, COUNT(*) FROM company_cni_sectors
        WHERE cni_sector_primary IS NOT NULL
        GROUP BY cni_sector_primary ORDER BY COUNT(*) DESC
    """).fetchall()

    conn.close()

    print(f"\nUpserted {upserted} rows → company_cni_sectors ({total} total in table)")
    print("\n── By source ─────────────────────────────────────────────")
    for src, n in by_src:
        print(f"  {(src or 'NULL'):<20s}  {n:4}")
    print("\n── By CNI sector (primary) ───────────────────────────────")
    for sect, n in by_sect:
        print(f"  {(sect or 'NULL'):<25s}  {n:4}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["static", "submit", "status", "collect", "load-db"],
                    default="static", help="Which phase to run (default: static)")
    ap.add_argument("--model", default="gemini-3-flash-preview",
                    help="Gemini model for batch submit (default: gemini-3-flash-preview)")
    ap.add_argument("--tag", default="",
                    help="Label for this batch run — used to name the checkpoint file "
                         "so multiple batches can run in parallel (e.g. --tag flash3)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Phase 1 only: preview counts without writing files")
    args = ap.parse_args()

    # Default tag to model name (sanitised) if not provided
    tag = args.tag or args.model.replace("/", "-").replace(".", "-")

    if args.phase == "static":
        phase_static(dry_run=args.dry_run)
    elif args.phase == "submit":
        phase_submit(model=args.model, tag=tag)
    elif args.phase == "status":
        phase_status(tag=tag)
    elif args.phase == "collect":
        phase_collect(tag=tag)
    elif args.phase == "load-db":
        phase_load_db()


if __name__ == "__main__":
    main()
