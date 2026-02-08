#!/usr/bin/env python3
"""
Classifier Testbed - Compare LLM classifier against human baseline.

Cell-based workflow:
1. Run SETUP cell
2. Run FUNCTIONS cell (collapse after)
3. Use action cells as needed: LOAD DATA, RUN CLASSIFIER, COMPARE, etc.
"""

#%% SETUP & IMPORTS
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
PIPELINE_DIR = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR))

from src.classifiers.llm_classifier_v2 import LLMClassifierV2

### CLEAR PROMPT CACHE (important when updating prompts)
from src.utils.prompt_loader import _load_prompt_yaml
_load_prompt_yaml.cache_clear() 
print("Prompt cache cleared")

from src.utils.prompt_loader import get_prompt_template
print(get_prompt_template("mention_type_v3")[:10000])

# Paths
# Baseline annotations: use reconciled gold set after human/LLM review.
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
def load_chunks(limit: int = 50, offset: int = 0) -> list[dict]:
    """Load chunks from golden set, optionally skipping the first `offset` chunks."""
    chunks = []
    skipped = 0
    with GOLDEN_SET.open() as f:
        for line in f:
            if line.strip():
                if skipped < offset:
                    skipped += 1
                    continue
                chunks.append(json.loads(line))
                if len(chunks) >= limit:
                    break
    return chunks


def get_run_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.jsonl"


def list_runs() -> list[dict]:
    """List all cached runs."""
    runs = []
    for path in sorted(RUNS_DIR.glob("*.jsonl")):
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        runs.append({"run_id": path.stem, "path": path, "meta": meta})
    return runs


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
    return run_path


def load_run(run_id: str) -> list[dict] | None:
    """Load cached run results."""
    run_path = get_run_path(run_id)
    if not run_path.exists():
        return None
    return [json.loads(line) for line in run_path.read_text().splitlines() if line.strip()]


def run_classifier(
    run_id: str,
    chunks: list[dict],
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    thinking_budget: int = 0,
) -> list[dict]:
    """Run classifier on chunks."""
    classifier = LLMClassifierV2(
        run_id=run_id,
        model_name=model_name,
        temperature=temperature,
        thinking_budget=thinking_budget,
        use_openrouter=False,
    )

    results = []
    for chunk in tqdm(chunks, desc="Classifying"):
        metadata = {
            "firm_id": chunk.get("company_id", "Unknown"),
            "firm_name": chunk.get("company_name", "Unknown"),
            "report_year": chunk.get("report_year", 0),
            "sector": "Unknown",
            "report_section": (chunk.get("report_sections", ["Unknown"])[0]
                              if chunk.get("report_sections") else "Unknown"),
        }

        result = classifier.classify(chunk["chunk_text"], metadata)

        llm_types = []
        if result.classification and "mention_types" in result.classification:
            llm_types = [
                str(mt.value) if hasattr(mt, "value") else str(mt)
                for mt in result.classification["mention_types"]
            ]

        results.append({
            "chunk_id": chunk.get("chunk_id", chunk.get("annotation_id", "unknown")),
            "company_name": chunk.get("company_name", "Unknown"),
            "report_year": chunk.get("report_year", 0),
            "human_mention_types": chunk.get("mention_types", []),
            "llm_mention_types": llm_types,
            "confidence": result.confidence_score,
            "reasoning": result.reasoning,
            "chunk_text": chunk["chunk_text"],
        })

    save_run(run_id, results, {"model_name": model_name, "temperature": temperature, "thinking_budget": thinking_budget})
    return results


def compare_sets(
    human: list[str],
    llm: list[str],
    exclude_added_general_ambiguous: bool = False,
) -> str:
    """Compare two label sets: EXACT, PARTIAL, or DIFF."""
    h, l = set(human), set(llm)
    if exclude_added_general_ambiguous and "general_ambiguous" in l and "general_ambiguous" not in h:
        l = set(l)
        l.discard("general_ambiguous")
    if h == l:
        return "EXACT"
    if h & l:
        return "PARTIAL"
    return "DIFF"


def show_summary(results: list[dict], exclude_added_general_ambiguous: bool = False) -> dict:
    """Print summary stats and return counts."""
    counts = {"EXACT": 0, "PARTIAL": 0, "DIFF": 0}
    for r in results:
        match = compare_sets(
            r["human_mention_types"],
            r["llm_mention_types"],
            exclude_added_general_ambiguous=exclude_added_general_ambiguous,
        )
        counts[match] += 1

    total = len(results)
    print("\n" + "=" * 60)
    print("SUMMARY")
    if exclude_added_general_ambiguous:
        print("(excluding added general_ambiguous)")
    print("=" * 60)
    print(f"Total chunks: {total}")
    print(f"  EXACT:   {counts['EXACT']:3d} ({counts['EXACT']/total:6.1%})")
    print(f"  PARTIAL: {counts['PARTIAL']:3d} ({counts['PARTIAL']/total:6.1%})")
    print(f"  DIFF:    {counts['DIFF']:3d} ({counts['DIFF']/total:6.1%})")
    agreement = counts["EXACT"] + counts["PARTIAL"]
    print(f"  Agreement: {agreement} ({agreement/total:.1%})")
    return counts


def show_table(
    results: list[dict],
    diff_only: bool = False,
    exclude_added_general_ambiguous: bool = False,
) -> None:
    """Print comparison table."""
    print("\n" + "=" * 100)
    header = "COMPARISON TABLE"
    if diff_only:
        header += " (differences only)"
    if exclude_added_general_ambiguous:
        header += " (excluding added general_ambiguous)"
    print(header)
    print("=" * 100)
    print(f"{'#':<3} {'Company':<22} {'Year':<5} {'Match':<6} {'Human':<30} {'LLM':<30}")
    print("-" * 100)

    for i, r in enumerate(results):
        match = compare_sets(
            r["human_mention_types"],
            r["llm_mention_types"],
            exclude_added_general_ambiguous=exclude_added_general_ambiguous,
        )
        if diff_only and match == "EXACT":
            continue

        symbol = {"EXACT": "  ", "PARTIAL": " ~", "DIFF": "XX"}[match]
        print(
            f"{i:<3} "
            f"{r['company_name'][:22]:<22} "
            f"{r['report_year']:<5} "
            f"{symbol:<6} "
            f"{str(sorted(r['human_mention_types'])):<30} "
            f"{str(sorted(r['llm_mention_types'])):<30}"
        )


def show_details(
    results: list[dict],
    indices: list[int] | None = None,
) -> None:
    """Show detailed view for specific chunks (or all disagreements if indices=None)."""
    print("\n" + "=" * 80)
    print("DETAILED VIEW")
    print("=" * 80)

    for i, r in enumerate(results):
        if indices is not None and i not in indices:
            continue
        match = compare_sets(r["human_mention_types"], r["llm_mention_types"])
        if indices is None and match == "EXACT":
            continue

        print(f"\n[{i}] {r['company_name']} ({r['report_year']})")
        print(f"    Chunk: {r['chunk_id']}")
        print(f"    Match: {match}")
        print(f"    Human: {sorted(r['human_mention_types'])}")
        print(f"    LLM:   {sorted(r['llm_mention_types'])} (conf: {r['confidence']:.2f})")
        print(f"    Reasoning: {r['reasoning']}")
        print(f"    Text: {r['chunk_text']}")
        print("-" * 80)


def show_diff_summary(results: list[dict], exclude_added_general_ambiguous: bool = False) -> dict:
    """Summarize label deltas and single-label transitions."""
    added_counts: dict[str, int] = {}
    removed_counts: dict[str, int] = {}
    transition_counts: dict[str, int] = {}

    for r in results:
        human = set(r["human_mention_types"])
        llm = set(r["llm_mention_types"])

        # Added/removed labels
        for label in sorted(llm - human):
            if exclude_added_general_ambiguous and label == "general_ambiguous":
                continue
            added_counts[label] = added_counts.get(label, 0) + 1
        for label in sorted(human - llm):
            removed_counts[label] = removed_counts.get(label, 0) + 1

        # Single-label transitions (actionable prompt tuning signal)
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
    if exclude_added_general_ambiguous:
        print("(excluding added general_ambiguous)")
    print("=" * 60)

    _print_block("Added Labels", added_counts, prefix="Added ")
    _print_block("Removed Labels", removed_counts, prefix="Removed ")
    _print_block("Single-Label Transitions", transition_counts, prefix="")

    return {
        "added": added_counts,
        "removed": removed_counts,
        "transitions": transition_counts,
    }

def refresh_human_labels(results: list[dict], chunks: list[dict]) -> None:
    human_by_id = {
        c.get("chunk_id", c.get("annotation_id")): c.get("mention_types", [])
        for c in chunks
    }
    for r in results:
        cid = r.get("chunk_id")
        if cid in human_by_id:
            r["human_mention_types"] = human_by_id[cid]

print("Functions loaded.")


#%% LIST EXISTING RUNS
runs = list_runs()
if runs:
    print("Cached runs:")
    for r in runs:
        m = r["meta"]
        print(f"  {r['run_id']}: {m.get('config', {}).get('model_name', '?')}, "
              f"{m.get('num_chunks', '?')} chunks, {m.get('created_at', '?')[:10]}")
else:
    print("No cached runs found.")


#%% LOAD GOLDEN SET CHUNKS
chunks = load_chunks(limit=324, offset=150)
print(f"Loaded {len(chunks)} chunks from golden set (offset=150).")

#%% SYNC CONFIG: Set run parameters
RUN_ID = "gemini-3-flash-v3-324chunks-offset150"
MODEL_NAME = "gemini-3-flash-preview"
TEMPERATURE = 0.0
THINKING_BUDGET = 0


#%% RUN SYNC: Classifier (or load from cache)
cached = load_run(RUN_ID)
if cached:
    print(f"Loaded {len(cached)} results from cache: {RUN_ID}")
    sync_results = cached
else:
    print(f"Running classifier: {RUN_ID} with {MODEL_NAME}")
    sync_results = run_classifier(RUN_ID, chunks, MODEL_NAME, TEMPERATURE, THINKING_BUDGET)
    print(f"Done. Saved to {get_run_path(RUN_ID)}")


#%% ############################ BATCH MODE ############################
# BATCH: Initialize client
from src.utils.batch_api import BatchClient
batch = BatchClient(runs_dir=RUNS_DIR)

BATCH_RUN_ID = "batch-gemini-3-flash-v3-324chunks-offset150-new"
BATCH_MODEL = "gemini-3-flash-preview"

batch_requests = batch.prepare_requests(chunks, temperature=TEMPERATURE, thinking_budget=THINKING_BUDGET)

#%% BATCH: Submit job
batch_job_name = batch.submit(BATCH_RUN_ID, batch_requests, model_name=BATCH_MODEL)

#%% BATCH: Check status (run this periodically)
batch_job_name= "batches/2j0ufhrtjqml6fubzlb6u1ifur1us0qg462c"
batch.check_status(batch_job_name)
"""
python3 scripts/watch_batch_status.py --run-id 
# batch_job_name = ""
"""

#%% BATCH: List all recent jobs
batch.list_jobs()

#%% BATCH: Get results when done
batch_results = batch.get_results(batch_job_name, chunks)
if batch_results:
    save_run(BATCH_RUN_ID, batch_results, {"model_name": BATCH_MODEL, "batch_mode": True})

#%% ############################ ANALYSIS ############################
# Choose which results to analyze: sync_results or batch_results
results = batch_results  # | sync_results | batch_results | 
exclude_gen_ambig = False

#%% REFRESH HUMAN LABELS FROM BASELINE (no LLM rerun)
refresh_human_labels(results, chunks)
show_summary(results)

#%% SHOW FULL TABLE
show_table(results)


#%% SHOW DIFFERENCES ONLY
show_table(results, diff_only=True, exclude_added_general_ambiguous=exclude_gen_ambig)

#%% SHOW DIFF SUMMARY (actionable label deltas)
show_diff_summary(results, exclude_added_general_ambiguous=exclude_gen_ambig)
#%% SHOW DISAGREEMENT DETAILS
show_details(results)

#%% INSPECT SPECIFIC CHUNK(S) - change indices as needed
show_details(results, indices=[17])

#############################################################################
"""

  1) Export

  python3 pipeline/scripts/export_testbed_run_for_reconcile.py \
    --testbed-run data/testbed_runs/batch-gemini-3-flash-v3-324chunks-offset150.jsonl \
    --human data/golden_set/human_reconciled/annotations.jsonl \
    --output-dir data/golden_set/llm/batch-gemini-3-flash-v3-324chunks-offset150 \
    --confidence-mode uniform

  2) Reconcile (mention-type disagreements only)

  python3 pipeline/scripts/reconcile_annotations.py \
    --human data/golden_set/human_reconciled/annotations.jsonl \
    --llm data/golden_set/llm/batch-gemini-3-flash-v3-324chunks-offset150/annotations.jsonl \
    --output-dir data/golden_set/reconciled/batch-gemini-3-flash-v3-324chunks-offset150 \
    --only-disagreements \
    --show-llm-reasoning \
    --max-chunks 324 \
    --resume

  3) Merge into baseline

  python3 pipeline/scripts/merge_reconciled_golden_set.py \
    --human data/golden_set/human_reconciled/annotations.jsonl \
    --reconciled data/golden_set/reconciled/batch-gemini-3-flash-v3-324chunks-offset150/annotations.jsonl \
    --output data/golden_set/human_reconciled/annotations.jsonl

"""
#%%
#%% ############################ SANITY CHECKS ############################
def check_run_alignment(run_id: str, limit: int, offset: int) -> None:
    """Verify run chunk_id order matches the golden-set slice used to submit the batch."""
    run_path = get_run_path(run_id)
    if not run_path.exists():
        print(f"Run file not found: {run_path}")
        return

    run = [json.loads(l) for l in run_path.read_text().splitlines() if l.strip()]
    chunks = load_chunks(limit=limit, offset=offset)

    print(f"run_len={len(run)} chunks_len={len(chunks)}")
    mismatch = None
    for i, (r, c) in enumerate(zip(run, chunks)):
        if r.get("chunk_id") != c.get("chunk_id"):
            mismatch = (i, r.get("chunk_id"), c.get("chunk_id"))
            break

    if mismatch:
        idx, run_id_at, chunk_id_at = mismatch
        print(f"FIRST MISMATCH @ index {idx}")
        print(f"  run chunk_id:   {run_id_at}")
        print(f"  slice chunk_id: {chunk_id_at}")
    else:
        print("Order check: OK")


def check_run_text_alignment(run_id: str) -> None:
    """Verify chunk_text in run file matches the golden set by chunk_id."""
    run_path = get_run_path(run_id)
    if not run_path.exists():
        print(f"Run file not found: {run_path}")
        return

    human_by_id = {
        r["chunk_id"]: r
        for r in map(json.loads, GOLDEN_SET.read_text().splitlines())
        if r.get("chunk_id")
    }
    run = [json.loads(l) for l in run_path.read_text().splitlines() if l.strip()]

    mismatches = []
    for r in run:
        h = human_by_id.get(r.get("chunk_id"))
        if not h:
            continue
        if (r.get("chunk_text") or "").strip() != (h.get("chunk_text") or "").strip():
            mismatches.append(r.get("chunk_id"))

    print(f"text mismatches: {len(mismatches)}")
    if mismatches:
        print(f"sample: {mismatches[:5]}")


def check_run_errors(run_id: str) -> None:
    """Count parse/error entries in a run file."""
    run_path = get_run_path(run_id)
    if not run_path.exists():
        print(f"Run file not found: {run_path}")
        return

    run = [json.loads(l) for l in run_path.read_text().splitlines() if l.strip()]
    err = sum(1 for r in run if r.get("error"))
    print(f"errors: {err}")


#%% SANITY CHECKS
# check_run_alignment("batch-gemini-3-flash-v3-324chunks-offset150", limit=324, offset=150)
# check_run_text_alignment("batch-gemini-3-flash-v3-324chunks-offset150")
check_run_errors("batch-gemini-3-flash-v3-324chunks-offset150")