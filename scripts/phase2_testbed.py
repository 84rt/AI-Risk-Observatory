#!/usr/bin/env python3
"""
Phase 2 Classifier Testbed - Test downstream classifiers against reconciled human baseline.

Downstream classifiers: adoption_type, risk, vendor.
Uses reconciled human mention_types to filter chunks (isolates phase 2 accuracy from phase 1).

Cell-based workflow:
1. Run SETUP cell
2. Run FUNCTIONS cell (collapse after)
3. Load chunks, verify counts
4. Configure classifier + run (sync or batch)
5. Analyze results
"""

#%% SETUP & IMPORTS
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from src.classifiers.adoption_type_classifier import AdoptionTypeClassifier
from src.classifiers.risk_classifier import RiskClassifier
from src.classifiers.vendor_classifier import VendorClassifier
from src.classifiers.base_classifier import _clean_schema_for_gemini

### CLEAR PROMPT CACHE (important when updating prompts)
from src.utils.prompt_loader import _load_prompt_yaml, get_prompt_template
_load_prompt_yaml.cache_clear()
print("Prompt cache cleared")
print(f"Available prompts: {', '.join(_load_prompt_yaml().keys())}")

#%% VIEW PROMPT (change key as needed, re-run to inspect)
_load_prompt_yaml.cache_clear()
PROMPT_KEY = "vendor"  # adoption_type | risk | vendor | mention_type_v3
print(f"\n{'='*60}\nPROMPT: {PROMPT_KEY}\n{'='*60}")
print(get_prompt_template(PROMPT_KEY))

from src.utils.batch_api import BatchClient

# Paths
GOLDEN_SET = REPO_ROOT / "data" / "golden_set" / "human_reconciled" / "annotations.jsonl"
RUNS_DIR = REPO_ROOT / "data" / "testbed_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

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

print(f"Golden set: {GOLDEN_SET}")
print(f"Runs dir: {RUNS_DIR}")


#%% FUNCTIONS (collapse this after running)

# --- Data loading ---

def load_all_chunks() -> list[dict]:
    """Load all 474 chunks from golden set."""
    chunks = []
    with GOLDEN_SET.open() as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks from golden set.")
    return chunks


def filter_chunks_for_classifier(
    chunks: list[dict],
    classifier_name: str,
    limit: int = 0,
) -> list[dict]:
    """Filter chunks by human mention_types for a given downstream classifier.

    - adoption_type: chunks where "adoption" in mention_types
    - risk:          chunks where "risk" in mention_types

    Args:
        limit: If > 0, return only the first `limit` matching chunks (for quick iteration).
    """
    if classifier_name == "adoption_type":
        filtered = [
            c for c in chunks
            if "adoption" in c.get("mention_types", [])
        ]
    elif classifier_name == "risk":
        filtered = [
            c for c in chunks
            if "risk" in c.get("mention_types", [])
        ]
    elif classifier_name == "vendor":
        filtered = [
            c for c in chunks
            if "vendor" in c.get("mention_types", [])
        ]
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    if limit > 0:
        filtered = filtered[:limit]
    return filtered


# --- Label normalization (backwards compat with old golden set terminology) ---

RISK_LABEL_ALIASES = {
    "strategic_market": "strategic_competitive",
    "regulatory": "regulatory_compliance",
    "workforce": "workforce_impacts",
    "environmental": "environmental_impact",
}


def normalize_risk_labels(labels: list[str]) -> list[str]:
    """Map legacy risk label names to current taxonomy."""
    return [RISK_LABEL_ALIASES.get(l, l) for l in labels]


# --- Classifier config ---

CLASSIFIER_CONFIG = {
    "adoption_type": {
        "cls": AdoptionTypeClassifier,
        "human_field": "adoption_types",
        "extract_llm_labels": lambda classification: [
            k for k, v in classification.get("adoption_confidences", {}).items()
            if isinstance(v, (int, float)) and v > 0
        ],
    },
    "risk": {
        "cls": RiskClassifier,
        "human_field": "risk_taxonomy",
        "extract_llm_labels": lambda classification: [
            str(rt.value) if hasattr(rt, "value") else str(rt)
            for rt in classification.get("risk_types", [])
            if str(rt.value if hasattr(rt, "value") else rt) != "none"
        ],
    },
    "vendor": {
        "cls": VendorClassifier,
        "human_field": "vendor_tags",
        "extract_llm_labels": lambda classification: [
            (v.get("vendor").value if hasattr(v.get("vendor"), "value") else str(v.get("vendor", "")))
            for v in classification.get("vendors", [])
            if isinstance(v, dict)
        ],
    },
}


def build_metadata(chunk: dict) -> dict:
    """Build metadata dict for classifier from a golden set chunk."""
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


# --- Classifier runner ---

def run_phase2(
    run_id: str,
    classifier_name: str,
    chunks: list[dict],
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    thinking_budget: int = 0,
    limit: int = 0,
) -> list[dict]:
    """Run a phase 2 downstream classifier on filtered chunks.

    1. Look up config from CLASSIFIER_CONFIG
    2. Filter chunks using filter_chunks_for_classifier()
    3. Instantiate classifier
    4. For each chunk: call classifier.classify(text, metadata)
    5. Extract llm_labels using config's extract_llm_labels()
    6. Build result dict and save

    Args:
        limit: If > 0, only classify the first `limit` matching chunks.
    """
    config = CLASSIFIER_CONFIG[classifier_name]
    cls = config["cls"]
    human_field = config["human_field"]
    extract_llm_labels = config["extract_llm_labels"]

    filtered = filter_chunks_for_classifier(chunks, classifier_name, limit=limit)
    print(f"Filtered {len(filtered)} chunks for {classifier_name} (from {len(chunks)} total)")

    classifier = cls(
        run_id=run_id,
        model_name=model_name,
        temperature=temperature,
        thinking_budget=thinking_budget,
        use_openrouter=False,
    )

    results = []
    for chunk in tqdm(filtered, desc=f"Classifying ({classifier_name})"):
        metadata = build_metadata(chunk)
        result = classifier.classify(chunk["chunk_text"], metadata)

        llm_labels = []
        if result.classification:
            llm_labels = extract_llm_labels(result.classification)

        human_labels = chunk.get(human_field, [])
        if classifier_name == "risk":
            human_labels = normalize_risk_labels(human_labels)

        # Extract other_vendor from LLM response if available
        llm_other_vendor = None
        if result.classification:
            llm_other_vendor = result.classification.get("other_vendor")

        results.append({
            "chunk_id": chunk.get("chunk_id", chunk.get("annotation_id", "unknown")),
            "company_name": chunk.get("company_name", "Unknown"),
            "report_year": chunk.get("report_year", 0),
            "human_labels": human_labels,
            "llm_labels": llm_labels,
            "confidence": result.confidence_score,
            "reasoning": result.reasoning,
            "chunk_text": chunk["chunk_text"],
            "human_other": chunk.get("vendor_other"),
            "llm_other": llm_other_vendor,
        })

    save_run(run_id, results, {
        "classifier_name": classifier_name,
        "model_name": model_name,
        "temperature": temperature,
        "thinking_budget": thinking_budget,
        "num_filtered": len(filtered),
    })
    return results


# --- Persistence (same pattern as phase 1) ---

def get_run_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.jsonl"


def save_run(run_id: str, results: list[dict], config: dict) -> Path:
    """Save run results and metadata."""
    run_path = get_run_path(run_id)
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
    print(f"Saved {len(results)} results to {run_path.name}")
    return run_path


def load_run(run_id: str) -> list[dict] | None:
    """Load cached run results."""
    run_path = get_run_path(run_id)
    if not run_path.exists():
        return None
    return [json.loads(line) for line in run_path.read_text().splitlines() if line.strip()]


def list_runs() -> list[dict]:
    """List all cached runs."""
    runs = []
    for path in sorted(RUNS_DIR.glob("*.jsonl")):
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        runs.append({"run_id": path.stem, "path": path, "meta": meta})
    return runs


# --- Comparison & display ---

def compare_sets(human: list[str], llm: list[str]) -> str:
    """Compare two label sets: EXACT, PARTIAL, or DIFF."""
    h, l = set(human), set(llm)
    if h == l:
        return "EXACT"
    if h & l:
        return "PARTIAL"
    return "DIFF"


def show_summary(results: list[dict]) -> dict:
    """Print summary stats and return counts."""
    counts = {"EXACT": 0, "PARTIAL": 0, "DIFF": 0}
    for r in results:
        match = compare_sets(r["human_labels"], r["llm_labels"])
        counts[match] += 1

    total = len(results)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total chunks: {total}")
    print(f"  EXACT:   {counts['EXACT']:3d} ({counts['EXACT']/total:6.1%})")
    print(f"  PARTIAL: {counts['PARTIAL']:3d} ({counts['PARTIAL']/total:6.1%})")
    print(f"  DIFF:    {counts['DIFF']:3d} ({counts['DIFF']/total:6.1%})")
    agreement = counts["EXACT"] + counts["PARTIAL"]
    print(f"  Agreement: {agreement} ({agreement/total:.1%})")
    return counts


def show_table(results: list[dict], diff_only: bool = False) -> None:
    """Print comparison table."""
    print("\n" + "=" * 100)
    header = "COMPARISON TABLE"
    if diff_only:
        header += " (differences only)"
    print(header)
    print("=" * 100)
    print(f"{'#':<3} {'Company':<22} {'Year':<5} {'Match':<6} {'Human':<30} {'LLM':<30}")
    print("-" * 100)

    for i, r in enumerate(results):
        match = compare_sets(r["human_labels"], r["llm_labels"])
        if diff_only and match == "EXACT":
            continue

        symbol = {"EXACT": "  ", "PARTIAL": " ~", "DIFF": "XX"}[match]
        print(
            f"{i:<3} "
            f"{r['company_name'][:22]:<22} "
            f"{r['report_year']:<5} "
            f"{symbol:<6} "
            f"{str(sorted(r['human_labels'])):<30} "
            f"{str(sorted(r['llm_labels'])):<30}"
        )


def show_details(results: list[dict], indices: list[int] | None = None) -> None:
    """Show detailed view for specific chunks (or all disagreements if indices=None)."""
    print("\n" + "=" * 80)
    print("DETAILED VIEW")
    print("=" * 80)

    for i, r in enumerate(results):
        if indices is not None and i not in indices:
            continue
        match = compare_sets(r["human_labels"], r["llm_labels"])
        if indices is None and match == "EXACT":
            continue

        print(f"\n[{i}] {r['company_name']} ({r['report_year']})")
        print(f"    Chunk: {r['chunk_id']}")
        print(f"    Match: {match}")
        human_other = f" (other: {r['human_other']})" if r.get("human_other") else ""
        llm_other = f" (other: {r['llm_other']})" if r.get("llm_other") else ""
        print(f"    Human: {sorted(r['human_labels'])}{human_other}")
        print(f"    LLM:   {sorted(r['llm_labels'])}{llm_other} (conf: {r['confidence']:.2f})")
        print(f"    Reasoning: {r['reasoning']}")
        print(f"    Text: {r['chunk_text'][:500]}")
        print("-" * 80)


def show_diff_summary(results: list[dict]) -> dict:
    """Summarize label deltas and single-label transitions."""
    added_counts: dict[str, int] = {}
    removed_counts: dict[str, int] = {}
    transition_counts: dict[str, int] = {}

    for r in results:
        human = set(r["human_labels"])
        llm = set(r["llm_labels"])

        for label in sorted(llm - human):
            added_counts[label] = added_counts.get(label, 0) + 1
        for label in sorted(human - llm):
            removed_counts[label] = removed_counts.get(label, 0) + 1

        if len(human) == 1 and len(llm) == 1 and human != llm:
            src = next(iter(human))
            dst = next(iter(llm))
            key = f"{src} -> {dst}"
            transition_counts[key] = transition_counts.get(key, 0) + 1

    def _print_block(title: str, items: dict[str, int], prefix: str = "") -> None:
        print("\n" + title)
        print("-" * len(title))
        for label, count in sorted(items.items(), key=lambda x: (-x[1], x[0])):
            print(f"{prefix}{label} x{count}")

    print("\n" + "=" * 60)
    print("DIFF SUMMARY")
    print("=" * 60)

    _print_block("Added Labels", added_counts, prefix="Added ")
    _print_block("Removed Labels", removed_counts, prefix="Removed ")
    _print_block("Single-Label Transitions", transition_counts, prefix="")

    return {
        "added": added_counts,
        "removed": removed_counts,
        "transitions": transition_counts,
    }


# --- Batch API helpers ---

def prepare_batch_requests(
    classifier_name: str,
    filtered_chunks: list[dict],
    temperature: float = 0.0,
    thinking_budget: int = 0,
) -> list[dict]:
    """Build batch requests for any downstream classifier.

    Uses each classifier's get_prompt_messages() and RESPONSE_MODEL.model_json_schema()
    to construct Gemini batch API request dicts.
    """
    config = CLASSIFIER_CONFIG[classifier_name]
    cls = config["cls"]
    # Instantiate just to get prompts (won't call API)
    temp_classifier = cls(run_id="batch-prep", model_name="unused")

    response_schema = _clean_schema_for_gemini(
        cls.RESPONSE_MODEL.model_json_schema()
    )

    # Wrap user prompt with a short integer index for order-preserving matching
    index_wrapper = (
        "## CHUNK_INDEX\n{chunk_index}\n\n"
        "## NOTE\nEcho this integer as \"chunk_id\" in the JSON output.\n\n"
        "## EXCERPT\n{user_prompt}"
    )

    requests = []
    for i, chunk in enumerate(filtered_chunks):
        metadata = build_metadata(chunk)
        system_prompt, user_prompt = temp_classifier.get_prompt_messages(
            chunk["chunk_text"], metadata
        )
        user_prompt_with_id = index_wrapper.format(
            chunk_index=i, user_prompt=user_prompt,
        )
        req_config = {
            "system_instruction": system_prompt,
            "temperature": temperature,
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        }
        if thinking_budget > 0:
            req_config["thinking_config"] = {"thinking_budget": thinking_budget}
        requests.append({
            "contents": [{"parts": [{"text": user_prompt_with_id}], "role": "user"}],
            "config": req_config,
        })

    print(f"Prepared {len(requests)} batch requests for {classifier_name}.")
    return requests


def parse_batch_results(
    classifier_name: str,
    job_name: str,
    filtered_chunks: list[dict],
    batch: BatchClient,
) -> list[dict] | None:
    """Retrieve and parse batch results for a downstream classifier.

    Matches responses to chunks by chunk_id echoed in the LLM response
    (Gemini batch API does NOT guarantee response ordering).
    Falls back to index-based matching when chunk_id is missing.
    """
    job = batch.client.batches.get(name=job_name)

    if job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"Batch not ready: {job.state.name}")
        return None

    if not hasattr(job, "dest") or not hasattr(job.dest, "inlined_responses"):
        print("No inlined_responses found.")
        return None

    responses = job.dest.inlined_responses
    print(f"Retrieved {len(responses)} responses.")

    if len(responses) != len(filtered_chunks):
        print(f"WARNING: {len(responses)} responses != {len(filtered_chunks)} chunks")

    config = CLASSIFIER_CONFIG[classifier_name]
    human_field = config["human_field"]
    extract_llm_labels = config["extract_llm_labels"]

    def _human_labels(chunk: dict) -> list[str]:
        labels = chunk.get(human_field, [])
        return normalize_risk_labels(labels) if classifier_name == "risk" else labels

    num_chunks = len(filtered_chunks)

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
        }

    def _extract_response_text(resp_obj) -> str | None:
        if resp_obj is None:
            return None
        response = getattr(resp_obj, "response", None)
        if response is None:
            return None
        # Primary: response.text
        text = getattr(response, "text", None)
        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8", errors="ignore")
        if isinstance(text, str) and text.strip():
            return text
        # Fallback: candidates[0].content.parts[*].text
        candidates = getattr(response, "candidates", None)
        if candidates is None and isinstance(response, dict):
            candidates = response.get("candidates")
        if candidates:
            cand0 = candidates[0]
            content = getattr(cand0, "content", None)
            if content is None and isinstance(cand0, dict):
                content = cand0.get("content")
            parts = getattr(content, "parts", None) if content is not None else None
            if parts is None and isinstance(content, dict):
                parts = content.get("parts")
            if parts:
                for part in parts:
                    part_text = getattr(part, "text", None)
                    if part_text is None and isinstance(part, dict):
                        part_text = part.get("text")
                    if isinstance(part_text, (bytes, bytearray)):
                        part_text = part_text.decode("utf-8", errors="ignore")
                    if isinstance(part_text, str) and part_text.strip():
                        return part_text
        return None

    def _extract_confidence(parsed: dict) -> float:
        """Extract max confidence/signal from a parsed response."""
        # Adoption: adoption_confidences dict
        if "adoption_confidences" in parsed:
            scores = parsed["adoption_confidences"]
            valid = [v for v in scores.values() if isinstance(v, (int, float))]
            return max(valid) if valid else 0.0
        # Risk: confidence_scores dict
        if "confidence_scores" in parsed:
            scores = parsed["confidence_scores"]
            valid = [v for v in scores.values() if isinstance(v, (int, float))]
            return max(valid) if valid else 0.0
        # Vendor: vendors list with signal scores
        if "vendors" in parsed and isinstance(parsed["vendors"], list):
            signals = [v.get("signal", 0) for v in parsed["vendors"] if isinstance(v, dict)]
            return max(signals) / 3.0 if signals else 0.0
        return 0.0

    results = []
    seen_indices: set[int] = set()
    index_hits = 0

    for i, resp in enumerate(responses):
        fallback_chunk = filtered_chunks[i] if i < num_chunks else {}
        fallback_id = fallback_chunk.get(
            "chunk_id", fallback_chunk.get("annotation_id", f"unknown_{i}")
        )

        if resp.error:
            results.append(_make_error(fallback_chunk, fallback_id, str(resp.error)))
            continue

        try:
            response_text = _extract_response_text(resp)
            if not response_text:
                results.append(_make_error(fallback_chunk, fallback_id, "Empty response text"))
                continue
            parsed = json.loads(response_text)

            # Match by echoed integer index (no positional fallback)
            echoed = parsed.get("chunk_id")
            if echoed is None:
                results.append(_make_error({}, f"unmatched_resp_{i}", f"Response {i}: missing chunk_id echo"))
                continue
            try:
                idx = int(echoed)
            except (ValueError, TypeError):
                results.append(_make_error({}, f"unmatched_resp_{i}", f"Response {i}: chunk_id '{echoed}' not an integer"))
                continue
            if idx < 0 or idx >= num_chunks:
                results.append(_make_error({}, f"unmatched_resp_{i}", f"Response {i}: chunk_id {idx} out of range [0, {num_chunks})"))
                continue
            if idx in seen_indices:
                results.append(_make_error({}, f"unmatched_resp_{i}", f"Response {i}: duplicate chunk_id {idx}"))
                continue

            seen_indices.add(idx)
            index_hits += 1
            chunk = filtered_chunks[idx]
            chunk_id = chunk.get(
                "chunk_id", chunk.get("annotation_id", f"unknown_{i}")
            )

            llm_labels = extract_llm_labels(parsed)
            confidence = _extract_confidence(parsed)

            results.append({
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
            })

        except (json.JSONDecodeError, AttributeError, KeyError, ValueError) as e:
            results.append(_make_error({}, f"parse_error_{i}", f"Parse error: {e}"))

    unmatched = len(results) - index_hits
    print(f"Matched: {index_hits}/{len(responses)} | Unmatched/errors: {unmatched}")
    if unmatched:
        for r in results:
            if r.get("error"):
                print(f"  - {r['chunk_id']}: {r['error']}")

    return results


print("Functions loaded.")


#%% LOAD ALL CHUNKS
chunks = load_all_chunks()

# Print distribution of mention_types and available phase 2 labels
mention_type_counts: dict[str, int] = {}
for c in chunks:
    for mt in c.get("mention_types", []):
        mention_type_counts[mt] = mention_type_counts.get(mt, 0) + 1

print(f"\nMention type distribution ({len(chunks)} chunks):")
for mt, count in sorted(mention_type_counts.items(), key=lambda x: -x[1]):
    print(f"  {mt}: {count}")

# Phase 2 label counts
adoption_chunks = filter_chunks_for_classifier(chunks, "adoption_type")
risk_chunks = filter_chunks_for_classifier(chunks, "risk")
vendor_chunks = filter_chunks_for_classifier(chunks, "vendor")
print(f"\nPhase 2 classifier chunks:")
print(f"  adoption_type: {len(adoption_chunks)} chunks")
print(f"  risk:          {len(risk_chunks)} chunks")
print(f"  vendor:        {len(vendor_chunks)} chunks (vendor mention_type)")


#%% CONFIG
CLASSIFIER_NAME = "risk"  # | adoption_type | risk | vendor
RUN_ID = "p2-risk-gemini-3-flash-v2"
MODEL_NAME = "gemini-3-flash-preview"
TEMPERATURE = 0.0
THINKING_BUDGET = 0 # I believe that this should be set to 0 for batch mode.
LIMIT = 0  # 0 = all matching chunks, >0 = first N (for quick iteration)

# Refresh & display active prompt (re-run this cell after editing classifiers.yaml)
_load_prompt_yaml.cache_clear()
# print(f"\n{'='*60}\nACTIVE PROMPT: {CLASSIFIER_NAME}\n{'='*60}")
# print(get_prompt_template(CLASSIFIER_NAME))


#%% SYNC: Run or load from cache
cached = load_run(RUN_ID)
if cached:
    print(f"Loaded {len(cached)} results from cache: {RUN_ID}")
    sync_results = cached
else:
    print(f"Running {CLASSIFIER_NAME} classifier: {RUN_ID} with {MODEL_NAME}")
    sync_results = run_phase2(
        RUN_ID,
        CLASSIFIER_NAME,
        chunks,
        MODEL_NAME,
        TEMPERATURE,
        THINKING_BUDGET,
        limit=LIMIT,
    )
    print(f"Done. Saved to {get_run_path(RUN_ID)}")


#%% BATCH: Initialize & filter ############################ BATCH MODE ############################
batch = BatchClient(runs_dir=RUNS_DIR)

BATCH_RUN_ID = f"batch-p2-{CLASSIFIER_NAME}-gemini-3-flash-v1"
BATCH_MODEL = "gemini-3-flash-preview"

# Filter chunks for this classifier (always re-run this before submit OR parse)
filtered_chunks = filter_chunks_for_classifier(chunks, CLASSIFIER_NAME, limit=LIMIT)
print(f"Filtered {len(filtered_chunks)} chunks for {CLASSIFIER_NAME} to {len(filtered_chunks)} chunks")

#%% BATCH: Prepare & submit (skip this cell when reloading existing results)
batch_requests = prepare_batch_requests(CLASSIFIER_NAME, filtered_chunks, temperature=TEMPERATURE, thinking_budget=THINKING_BUDGET)
batch_job_name = batch.submit(BATCH_RUN_ID, batch_requests, model_name=BATCH_MODEL)

#%% BATCH: List all recent jobs
batch.list_jobs()

#%% BATCH: Check status (run this periodically)
batch_job_name = "batches/b1dvgrjr3umit887c5nntsg9k1vxrh80es1x"  # risk batch job name
# batch_job_name = "batches/kzhm6o1p0qtj2v01fz9en7o7bq3f37pgxzwj"  # adoption_type batch job name
batch.check_status(batch_job_name)

#%% BATCH: Get results (paste batch_job_name if reconnecting to a previous job)

batch_results = parse_batch_results(CLASSIFIER_NAME, batch_job_name, filtered_chunks, batch)
if batch_results:
    save_run(BATCH_RUN_ID, batch_results, {
        "classifier_name": CLASSIFIER_NAME,
        "model_name": BATCH_MODEL,
        "batch_mode": True,
        "num_filtered": len(filtered_chunks),
    })


#%% ############################ ANALYSIS ############################
# Choose which results to analyze: sync_results or batch_results

results = batch_results  # | sync_results | batch_results
show_summary(results)

#%% SHOW FULL TABLE
show_table(results)

#%% SHOW DIFFERENCES ONLY
show_table(results, diff_only=True)

#%% SHOW DIFF SUMMARY (actionable label deltas)
show_diff_summary(results)

#%% SHOW DISAGREEMENT DETAILS
show_details(results)

#%% INSPECT SPECIFIC CHUNK(S) - change indices as needed
show_details(results, indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33])