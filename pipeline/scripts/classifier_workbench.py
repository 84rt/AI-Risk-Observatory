#!/usr/bin/env python3
#%% SETUP & IMPORTS
import json
import sys
from pathlib import Path

# Add pipeline to path
PIPELINE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE_DIR))

from src.classifiers import (
    MentionTypeClassifier,
    AdoptionTypeClassifier,
    RiskClassifier,
    VendorClassifier,
)
from progress_helper import tqdm
from helper import (
    load_chunks,
    run_mention_classification,
    save_phase1_results,
    compare_one,
    print_comparison,
    inspect_chunk,
)
from visualize_helper import visualize_all, visualize_summary_table, visualize_report_summary
from tests_helper import (
    run_best_of_n,
    best_of_n_batch_detailed,
    run_model_family,
    model_family_batch,
    validate_models,
    run_thinking_levels,
    thinking_levels_batch,
)

## %% CONFIG - (Edit these as needed)
RUN_ID = "workbench"
MAX_CHUNKS = 10  # Number of chunks to process (set to 20 for model family test)
SAVE_RESULTS = False

# Model configuration (None = use default from settings)
# Models with native thinking: gemini-2.5-flash, gemini-2.5-pro, gemini-3-flash-preview
MODEL_NAME = "gemini-2.5-flash"  # e.g., "gemini-2.5-flash", "google/gemini-3-flash-preview", "openai/gpt-4o-mini"
TEMPERATURE = 0.0  # we want the classifier to be deterministic

# Paths
GOLDEN_SET_DIR = PIPELINE_DIR.parent / "data" / "golden_set"
HUMAN_ANNOTATIONS = GOLDEN_SET_DIR / "human" / "annotations.jsonl"

# Confidence threshold for including a label
CONFIDENCE_THRESHOLD = 0.0

# Downstream classifier controls (set RUN_DOWNSTREAM to True to run in Phase 1)
RUN_DOWNSTREAM = True
RUN_VENDOR_IF_MENTIONED = True

## %% LOAD GOLDEN SET CHUNKS
# Load human-annotated chunks
chunks = load_chunks(HUMAN_ANNOTATIONS, limit=MAX_CHUNKS)
print(f"Loaded {len(chunks)} chunks from golden set")

# Show first chunk
print("\n--- First Chunk ---")
print(f"Company: {chunks[0]['company_name']}")
print(f"Year: {chunks[0]['report_year']}")
print(f"Text: {chunks[0]['chunk_text'][:200]}...")
print(f"Human labels: mention_types={chunks[0].get('mention_types', [])}")

# %% PHASE 1: RUN MENTION TYPE CLASSIFIER

print("\n" + "="*80)
print("PHASE 1: MENTION TYPE CLASSIFICATION")
print("="*80)

# Initialize classifiers with config
clf_kwargs = {"run_id": RUN_ID, "model_name": MODEL_NAME, "temperature": TEMPERATURE}
mention_clf = MentionTypeClassifier(**clf_kwargs)
adoption_clf = AdoptionTypeClassifier(**clf_kwargs) if RUN_DOWNSTREAM else None
risk_clf = RiskClassifier(**clf_kwargs) if RUN_DOWNSTREAM else None
vendor_clf = VendorClassifier(**clf_kwargs) if RUN_DOWNSTREAM else None

# Store results
results = run_mention_classification(
    chunks,
    mention_clf,
    adoption_clf=adoption_clf,
    risk_clf=risk_clf,
    vendor_clf=vendor_clf,
    run_downstream=RUN_DOWNSTREAM,
    run_vendor_if_mentioned=RUN_VENDOR_IF_MENTIONED,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    tqdm_func=tqdm,
)

if SAVE_RESULTS:
    saved_path = save_phase1_results(results, run_id=RUN_ID)
    print(f"Saved Phase 1 results to {saved_path}")

# %% COMPARE MENTION TYPE RESULTS
print("\n" + "="*80)
print("MENTION TYPE COMPARISON: Human vs LLM")
print("="*80)

comparisons = []
for chunk, llm_result in tqdm(results, desc="Comparing results"):
    comp = compare_one(chunk, llm_result)
    comparisons.append(comp)
    print_comparison(comp, show_text=False)

# Summary stats
exact = sum(1 for c in comparisons if c["mention_match"] == "EXACT")
partial = sum(1 for c in comparisons if c["mention_match"] == "PARTIAL")
diff = sum(1 for c in comparisons if c["mention_match"] == "DIFF")
total = len(comparisons)

print(f"\n{'='*80}\nSUMMARY\n{'='*80}\nTotal chunks: {total}")
print(f"Exact match:  {exact} ({exact/total*100:.0f}%)")
print(f"Partial:      {partial} ({partial/total*100:.0f}%)")
print(f"Different:    {diff} ({diff/total*100:.0f}%)")
print(f"Agreement:    {(exact + partial)/total*100:.0f}%")

# VISUALIZE ALL RESULTS SIDE-BY-SIDE
visualize_all(results, show_text=True, max_text_len=250)
visualize_summary_table(results)
visualize_report_summary(results)


#%% DEBUG: VIEW RAW LLM REQUEST/RESPONSE
# The debug logs are saved to: data/logs/pipeline/debug/
# Each classification creates a JSON file with full request/response

import os
from datetime import datetime

debug_dir = PIPELINE_DIR.parent / "data" / "logs" / "pipeline" / "debug"
if debug_dir.exists():
    # Get most recent debug files
    debug_files = sorted(debug_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
    if debug_files:
        print(f"\nMost recent debug log: {debug_files[0].name}")
        print(f"Total debug logs: {len(debug_files)}")

        # Load and show the most recent one
        with open(debug_files[0]) as f:
            debug_data = json.load(f)

        print(f"\nTimestamp: {debug_data.get('timestamp')}")
        print(f"Classifier: {debug_data.get('classifier_type')}")
        print(f"Model: {debug_data['request'].get('model')}")
        print(f"\nSystem prompt preview:")
        print(debug_data['request'].get('system_prompt', '')[:500] + "...")
else:
    print("No debug logs found yet. Run a classification first.")

#%% INSPECT SINGLE CHUNK IN DETAIL
inspect_chunk(results, 3)

#%% QUICK TEST: SINGLE TEXT CLASSIFICATION

test_text = """
Digital technology is transforming Croda, reshaping markets and driving value for customers, employees, and the broader business ecosystem. Customers demand greater product transparency and more intuitive digital experiences. By embracing digital and data innovation, Croda maintains its competitive edge, meets customer and employee needs, enhances operational efficiency, and fosters sustainable growth.

"""

test_metadata = {
    "firm_id": "TEST",
    "firm_name": "Test Company",
    "report_year": 2024,
    "sector": "Technology",
    "report_section": "Strategic Report",
}

# Classify (uses same model/temperature from config)
clf = MentionTypeClassifier(run_id="quick-test", model_name=MODEL_NAME, temperature=TEMPERATURE)
result = clf.classify(test_text.strip(), test_metadata)

# Extract mention types
classification = result.classification
mention_types = []
if isinstance(classification, dict) and "mention_types" in classification:
    mention_types = [str(mt.value) if hasattr(mt, 'value') else str(mt)
                    for mt in classification["mention_types"]]

print("="*80)
print("QUICK TEST RESULT")
print("="*80)
print(f"Text: {test_text[:100].strip()}...")
print(f"\nMention Types: {mention_types}")
print(f"Primary Label: {result.primary_label}")
print(f"Confidence: {result.confidence_score:.2f}")
confidence_scores = {}
if isinstance(classification, dict):
    confidence_scores = classification.get("confidence_scores", {}) or {}
if confidence_scores:
    print("Confidence scores:")
    for k, v in confidence_scores.items():
        if isinstance(v, (int, float)):
            print(f"  - {k}: {v:.2f}")
        else:
            print(f"  - {k}: {v}")
else:
    print("Confidence scores: N/A")
print(f"Reasoning: {result.reasoning}")

# Optional: run downstream classifiers in quick test
if RUN_DOWNSTREAM:
    downstream_metadata = dict(test_metadata)
    downstream_metadata["mention_types"] = mention_types

    quick_kwargs = {"run_id": "quick-test", "model_name": MODEL_NAME, "temperature": TEMPERATURE}
    if "adoption" in mention_types:
        adoption_result = AdoptionTypeClassifier(**quick_kwargs).classify(
            test_text.strip(), downstream_metadata
        )
        print(f"\nAdoption types: {adoption_result.classification}")

    if "risk" in mention_types:
        risk_result = RiskClassifier(**quick_kwargs).classify(
            test_text.strip(), downstream_metadata
        )
        print(f"\nRisk taxonomy: {risk_result.classification}")

    if RUN_VENDOR_IF_MENTIONED and "vendor" in mention_types:
        vendor_result = VendorClassifier(**quick_kwargs).classify(
            test_text.strip(), test_metadata
        )
        print(f"\nVendor tags: {vendor_result.classification}")

#%% BEST-OF-N CONSISTENCY TEST
# Test if model gives consistent results across multiple runs
# Temperature=0 should be deterministic; try 0.3-0.7 to test variance

consistency_result = run_best_of_n(
    MentionTypeClassifier,
    chunks[3],
    n=5,
    temperature=0.7,  # the only time we want to use non-0.0 temperature
    model_name=MODEL_NAME,  # Use the configured model
    tqdm_func=tqdm,
)

#%% BEST-OF-N BATCH TEST
# Run consistency test across multiple chunks

batch_results = best_of_n_batch_detailed(
    MentionTypeClassifier,
    chunks[:10],
    n=10,
    temperature=0.7,  # the only time we want to use non-0.0 temperature
    model_name=MODEL_NAME,  # Use the configured model
    tqdm_func=tqdm,
)

#%% MODEL FAMILY TEST (single chunk)
# Compare classifications across different models

model_comparison = run_model_family(
    MentionTypeClassifier,
    chunks[3],
    models=[
        "gemini-2.0-flash",              # Gemini API (default)
        "google/gemini-3-flash-preview",        # OpenRouter
        "openai/gpt-4o-mini",                   # OpenRouter
        "openai/gpt-5-nano",
    ],
    tqdm_func=tqdm,
)

#%% MODEL FAMILY BATCH TEST
# Compare model agreement across ALL chunks

MODELS_TO_TEST = [
    "gemini-2.0-flash",                     # Gemini API (direct)
    "google/gemini-3-flash-preview",        # OpenRouter
    "openai/gpt-4o-mini",                   # OpenRouter
    "openai/gpt-5-nano",                    # OpenRouter
    "anthropic/claude-sonnet-4.5",          # OpenRouter
    "anthropic/claude-haiku-4.5",           # OpenRouter
]

batch_model_results = model_family_batch(
    MentionTypeClassifier,
    chunks,  # Run on all loaded chunks
    models=MODELS_TO_TEST,
    temperature=0.0,
    tqdm_func=tqdm,
)

#%% THINKING LEVELS TEST (single chunk)
# Test different thinking budgets on a single chunk
# Models with thinking support: gemini-2.5-flash, gemini-2.5-pro, gemini-3-flash-preview

THINKING_MODEL = "gemini-2.5-flash"  # Must support native thinking
THINKING_BUDGETS = [0, 1024, 4096, 8192]  # 0=disabled, then increasing budgets

thinking_result = run_thinking_levels(
    MentionTypeClassifier,
    chunks[3],
    budgets=THINKING_BUDGETS,
    model_name=THINKING_MODEL,
    tqdm_func=tqdm,
)

#%% THINKING LEVELS BATCH TEST
# Compare thinking budgets across ALL chunks

thinking_batch_results = thinking_levels_batch(
    MentionTypeClassifier,
    chunks,  # Run on all loaded chunks
    budgets=THINKING_BUDGETS,
    model_name=THINKING_MODEL,
    temperature=0.0,
    tqdm_func=tqdm,
)
