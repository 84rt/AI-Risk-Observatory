#!/usr/bin/env python3
#%% SETUP & IMPORTS
import json
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable, **kwargs):
        return iterable

# Add pipeline to path
PIPELINE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE_DIR))

from src.classifiers import (
    MentionTypeClassifier,
    AdoptionTypeClassifier,
    RiskClassifier,
    VendorClassifier,
)

##%% CONFIG - (Edit these as needed)
RUN_ID = "workbench"
MAX_CHUNKS = 5  # Number of chunks to process

# Paths
GOLDEN_SET_DIR = PIPELINE_DIR.parent / "data" / "golden_set"
HUMAN_ANNOTATIONS = GOLDEN_SET_DIR / "human" / "annotations.jsonl"

# Confidence threshold for including a label
CONFIDENCE_THRESHOLD = 0.0

# Downstream classifier controls (set RUN_DOWNSTREAM to True to run in Phase 1)
RUN_DOWNSTREAM = True
RUN_VENDOR_IF_MENTIONED = True

##%% LOAD GOLDEN SET CHUNKS
def load_chunks(path: Path, limit: int = None) -> list[dict]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(path) as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                if limit and len(chunks) >= limit:
                    break
    return chunks

# Load human-annotated chunks
chunks = load_chunks(HUMAN_ANNOTATIONS, limit=MAX_CHUNKS)
print(f"Loaded {len(chunks)} chunks from golden set")

# Show first chunk
print("\n--- First Chunk ---")
print(f"Company: {chunks[0]['company_name']}")
print(f"Year: {chunks[0]['report_year']}")
print(f"Text: {chunks[0]['chunk_text'][:200]}...")
print(f"Human labels: mention_types={chunks[0].get('mention_types', [])}")

## %% HELPER: SIDE-BY-SIDE VISUALIZATION
def visualize_all(results_list: list[tuple[dict, dict]], show_text: bool = True, max_text_len: int = 300):
    """
    Visualize side-by-side comparison of ALL classification fields for each chunk.

    Args:
        results_list: List of (chunk, llm_result) tuples
        show_text: Whether to show chunk text
        max_text_len: Max characters to show for text
    """
    for i, (chunk, llm_result) in enumerate(results_list):
        print("\n" + "█" * 90)
        print(f"  CHUNK {i}: {chunk.get('company_name', 'Unknown')} ({chunk.get('report_year', '?')})")
        print("█" * 90)

        # Chunk metadata
        print(f"\n📋 ID: {chunk.get('chunk_id', 'N/A')[:50]}...")
        print(f"📂 Section: {chunk.get('report_sections', ['N/A'])[0][:60]}")
        print(f"🔑 Keywords: {', '.join(chunk.get('matched_keywords', [])) or 'none'}")

        # Text preview
        if show_text:
            text = chunk.get('chunk_text', '')[:max_text_len]
            if len(chunk.get('chunk_text', '')) > max_text_len:
                text += "..."
            print(f"\n📝 TEXT:\n{'-'*90}")
            print(text)
            print(f"{'-'*90}")

        # Side-by-side comparison table
        print(f"\n{'─'*90}")
        print(f"{'FIELD':<25} │ {'HUMAN':<30} │ {'LLM':<30}")
        print(f"{'─'*90}")

        # Define all fields to compare
        fields = [
            ("mention_types", "mention_types", "mention_types"),
            ("adoption_types", "adoption_types", "adoption_types"),
            ("risk_taxonomy", "risk_taxonomy", "risk_taxonomy"),
            ("vendor_tags", "vendor_tags", "vendor_tags"),
            ("risk_substantiveness", "risk_substantiveness", "risk_substantiveness"),
        ]

        for field_name, human_key, llm_key in fields:
            # Get human value
            h_val = chunk.get(human_key, [])
            if isinstance(h_val, list):
                h_str = ", ".join(str(v) for v in h_val) if h_val else "—"
            elif h_val is None:
                h_str = "—"
            else:
                h_str = str(h_val)

            # Get LLM value
            l_val = llm_result.get(llm_key, [])
            if isinstance(l_val, list):
                # Handle enum values
                l_str = ", ".join(
                    str(v.value) if hasattr(v, 'value') else str(v)
                    for v in l_val
                ) if l_val else "—"
            elif l_val is None:
                l_str = "—"
            else:
                l_str = str(l_val)

            # Compare
            h_set = set(h_val) if isinstance(h_val, list) else {h_val}
            l_set = set(l_val) if isinstance(l_val, list) else {l_val}

            if h_set == l_set:
                match = "✓"
            elif h_set & l_set:
                match = "~"
            elif not h_val and not l_val:
                match = "·"  # Both empty
            else:
                match = "✗"

            # Truncate long values
            h_str = h_str[:28] + ".." if len(h_str) > 30 else h_str
            l_str = l_str[:28] + ".." if len(l_str) > 30 else l_str

            print(f"{field_name:<25} │ {h_str:<30} │ {l_str:<27} {match}")

        print(f"{'─'*90}")

        # Show confidence and reasoning if available
        if llm_result.get('confidence'):
            print(f"\n🎯 LLM Confidence: {llm_result['confidence']:.2f}")
        if llm_result.get('reasoning'):
            reasoning = llm_result['reasoning'][:200]
            if len(llm_result.get('reasoning', '')) > 200:
                reasoning += "..."
            print(f"💭 LLM Reasoning: {reasoning}")

        print()


def visualize_summary_table(results_list: list[tuple[dict, dict]]):
    """
    Print a compact summary table of all chunks with match status for each field.
    """
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: Human vs LLM Comparison")
    print("=" * 100)
    print(f"{'#':<4} {'Company':<25} {'Mention':<10} {'Adoption':<10} {'Risk':<10} {'Vendor':<10}")
    print("-" * 100)

    stats = {"mention": {"exact": 0, "partial": 0, "diff": 0},
             "adoption": {"exact": 0, "partial": 0, "diff": 0},
             "risk": {"exact": 0, "partial": 0, "diff": 0},
             "vendor": {"exact": 0, "partial": 0, "diff": 0}}

    def match_status(human_val, llm_val) -> str:
        h_set = set(human_val) if isinstance(human_val, list) else set()
        l_set = set(llm_val) if isinstance(llm_val, list) else set()
        # Convert enums to strings for comparison
        l_set = {str(v.value) if hasattr(v, 'value') else str(v) for v in l_set}

        if h_set == l_set:
            return "✓ EXACT"
        elif h_set & l_set:
            return "~ PART"
        elif not h_set and not l_set:
            return "· EMPTY"
        else:
            return "✗ DIFF"

    for i, (chunk, llm_result) in enumerate(results_list):
        company = chunk.get('company_name', 'Unknown')[:23]

        m_status = match_status(chunk.get('mention_types', []), llm_result.get('mention_types', []))
        a_status = match_status(chunk.get('adoption_types', []), llm_result.get('adoption_types', []))
        r_status = match_status(chunk.get('risk_taxonomy', []), llm_result.get('risk_taxonomy', []))
        v_status = match_status(chunk.get('vendor_tags', []), llm_result.get('vendor_tags', []))

        # Update stats
        for field, status in [("mention", m_status), ("adoption", a_status),
                               ("risk", r_status), ("vendor", v_status)]:
            if "EXACT" in status or "EMPTY" in status:
                stats[field]["exact"] += 1
            elif "PART" in status:
                stats[field]["partial"] += 1
            else:
                stats[field]["diff"] += 1

        print(f"{i:<4} {company:<25} {m_status:<10} {a_status:<10} {r_status:<10} {v_status:<10}")

    # Print totals
    total = len(results_list)
    print("-" * 100)
    print(f"{'TOTALS':<30}", end="")
    for field in ["mention", "adoption", "risk", "vendor"]:
        exact_pct = stats[field]["exact"] / total * 100 if total > 0 else 0
        print(f"{exact_pct:>5.0f}% ✓   ", end="")
    print()
    print("=" * 100)

    # Legend
    print("\nLegend: ✓=Exact match  ~=Partial overlap  ✗=Different  ·=Both empty")


## HELPER: FORMAT COMPARISON
def compare_one(chunk: dict, llm_result: dict) -> dict:
    """Compare human annotation with LLM result."""
    human = {
        "mention_types": set(chunk.get("mention_types", [])),
        "adoption_types": set(chunk.get("adoption_types", [])),
        "risk_taxonomy": set(chunk.get("risk_taxonomy", [])),
    }
    llm = {
        "mention_types": set(llm_result.get("mention_types", [])),
        "adoption_types": set(llm_result.get("adoption_types", [])),
        "risk_taxonomy": set(llm_result.get("risk_taxonomy", [])),
    }

    def match_status(h: set, l: set) -> str:
        if h == l:
            return "EXACT"
        elif h & l:
            return "PARTIAL"
        else:
            return "DIFF"

    return {
        "chunk_id": chunk["chunk_id"],
        "company": chunk["company_name"],
        "mention_match": match_status(human["mention_types"], llm["mention_types"]),
        "human_mention": list(human["mention_types"]),
        "llm_mention": list(llm["mention_types"]),
        "human_adoption": list(human["adoption_types"]),
        "llm_adoption": list(llm["adoption_types"]),
    }


def print_comparison(comp: dict, show_text: bool = False, text: str = ""):
    """Pretty print a comparison result."""
    match_symbol = {"EXACT": "✓", "PARTIAL": "~", "DIFF": "✗"}

    print(f"\n{'='*80}")
    print(f"Chunk: {comp['chunk_id'][:40]}...")
    print(f"Company: {comp['company']}")
    print(f"{'='*80}")

    if show_text and text:
        print(f"\nText:\n{text[:500]}...")

    print(f"\nMention Types: {match_symbol.get(comp['mention_match'], '?')} {comp['mention_match']}")
    print(f"  Human: {comp['human_mention']}")
    print(f"  LLM:   {comp['llm_mention']}")

    if comp.get("human_adoption") or comp.get("llm_adoption"):
        print(f"\nAdoption Types:")
        print(f"  Human: {comp['human_adoption']}")
        print(f"  LLM:   {comp['llm_adoption']}")

## %% PHASE 1: RUN MENTION TYPE CLASSIFIER

print("\n" + "="*80)
print("PHASE 1: MENTION TYPE CLASSIFICATION")
print("="*80)

# Initialize classifier
mention_clf = MentionTypeClassifier(run_id=RUN_ID)
adoption_clf = AdoptionTypeClassifier(run_id=RUN_ID) if RUN_DOWNSTREAM else None
risk_clf = RiskClassifier(run_id=RUN_ID) if RUN_DOWNSTREAM else None
vendor_clf = VendorClassifier(run_id=RUN_ID) if RUN_DOWNSTREAM else None

# Store results
results = []

for i, chunk in enumerate(tqdm(chunks, desc="Mention type classification")):
    print(f"\n[{i+1}/{len(chunks)}] Processing: {chunk['company_name']} ({chunk['report_year']})")

    # Build metadata for classifier
    metadata = {
        "firm_id": chunk.get("company_id", "Unknown"),
        "firm_name": chunk.get("company_name", "Unknown"),
        "report_year": chunk.get("report_year", 0),
        "sector": "Unknown",
        "report_section": chunk.get("report_sections", ["Unknown"])[0] if chunk.get("report_sections") else "Unknown",
    }

    # Classify
    result = mention_clf.classify(chunk["chunk_text"], metadata)

    # Extract mention types from classification dict
    classification = result.classification
    mention_types = []
    if isinstance(classification, dict) and "mention_types" in classification:
        mention_types = [str(mt.value) if hasattr(mt, 'value') else str(mt)
                        for mt in classification["mention_types"]]

    # Store result with chunk info
    llm_result = {
        "chunk_id": chunk["chunk_id"],
        "mention_types": mention_types,
        "confidence": result.confidence_score,
        "reasoning": result.reasoning,
        "raw_classification": classification,
        "adoption_types": [],
        "risk_taxonomy": [],
        "vendor_tags": [],
    }

    # Run downstream classifiers based on mention types
    if RUN_DOWNSTREAM:
        downstream_metadata = dict(metadata)
        downstream_metadata["mention_types"] = mention_types

        if "adoption" in mention_types and adoption_clf:
            adoption_result = adoption_clf.classify(chunk["chunk_text"], downstream_metadata)
            adoption_classification = adoption_result.classification
            adoption_types = []
            if isinstance(adoption_classification, dict):
                adoption_confidences = adoption_classification.get("adoption_confidences", {}) or {}
                if isinstance(adoption_confidences, dict):
                    adoption_types = [
                        k for k, v in adoption_confidences.items()
                        if isinstance(v, (int, float)) and v > CONFIDENCE_THRESHOLD
                    ]
            llm_result["adoption_types"] = adoption_types
            llm_result["raw_adoption"] = adoption_classification

        if "risk" in mention_types and risk_clf:
            risk_result = risk_clf.classify(chunk["chunk_text"], downstream_metadata)
            risk_classification = risk_result.classification
            risk_types = []
            if isinstance(risk_classification, dict):
                rt = risk_classification.get("risk_types", []) or []
                if isinstance(rt, list):
                    risk_types = [
                        str(r.value) if hasattr(r, "value") else str(r)
                        for r in rt
                        if str(r) != "none"
                    ]
            llm_result["risk_taxonomy"] = risk_types
            llm_result["raw_risk"] = risk_classification

        if RUN_VENDOR_IF_MENTIONED and "vendor" in mention_types and vendor_clf:
            vendor_result = vendor_clf.classify(chunk["chunk_text"], metadata)
            vendor_classification = vendor_result.classification
            vendor_tags = []
            if isinstance(vendor_classification, dict):
                vendor_confidences = vendor_classification.get("vendor_confidences", {}) or {}
                if isinstance(vendor_confidences, dict):
                    vendor_tags = [
                        k for k, v in vendor_confidences.items()
                        if isinstance(v, (int, float)) and v > CONFIDENCE_THRESHOLD
                    ]
                other_vendor = vendor_classification.get("other_vendor")
                if other_vendor:
                    vendor_tags.append(f"other:{other_vendor}")
            llm_result["vendor_tags"] = vendor_tags
            llm_result["raw_vendor"] = vendor_classification
    results.append((chunk, llm_result))

    # Show result
    print(f"  Result: {mention_types} (conf={result.confidence_score:.2f})")
    print(f"  Reasoning: {result.reasoning[:100] if result.reasoning else 'N/A'}...")

##%% COMPARE MENTION TYPE RESULTS
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

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total chunks: {total}")
print(f"Exact match:  {exact} ({exact/total*100:.0f}%)")
print(f"Partial:      {partial} ({partial/total*100:.0f}%)")
print(f"Different:    {diff} ({diff/total*100:.0f}%)")
print(f"Agreement:    {(exact + partial)/total*100:.0f}%")

##%% VISUALIZE ALL RESULTS SIDE-BY-SIDE
# Show detailed side-by-side comparison for all chunks
visualize_all(results, show_text=True, max_text_len=250)

##%% VISUALIZE SUMMARY TABLE
# Show compact summary table with match status per field
visualize_summary_table(results)


##%% DEBUG: VIEW RAW LLM REQUEST/RESPONSE
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
def inspect_chunk(index: int):
    """Inspect a single chunk in detail with human vs LLM comparison."""
    if index >= len(results):
        print(f"Index {index} out of range (0-{len(results)-1})")
        return

    chunk, llm_result = results[index]

    print("="*80)
    print(f"CHUNK {index}")
    print("="*80)
    print(f"Company: {chunk['company_name']}")
    print(f"Year: {chunk['report_year']}")
    print(f"Section: {chunk.get('report_sections', ['?'])[0]}")
    print(f"Keywords: {chunk.get('matched_keywords', [])}")

    print(f"\n--- TEXT ---")
    print(chunk["chunk_text"])

    print(f"\n--- HUMAN ANNOTATION ---")
    print(f"Mention types: {chunk.get('mention_types', [])}")
    print(f"Adoption types: {chunk.get('adoption_types', [])}")
    print(f"Risk taxonomy: {chunk.get('risk_taxonomy', [])}")

    print(f"\n--- LLM ANNOTATION ---")
    print(f"Mention types: {llm_result['mention_types']}")
    print(f"Confidence: {llm_result['confidence']:.2f}" if llm_result['confidence'] else "Confidence: N/A")
    confidence_scores = (
        llm_result.get("raw_classification", {}) or {}
    ).get("confidence_scores", {})
    if confidence_scores:
        print("Confidence scores:")
        for k, v in confidence_scores.items():
            if isinstance(v, (int, float)):
                print(f"  - {k}: {v:.2f}")
            else:
                print(f"  - {k}: {v}")
    else:
        print("Confidence scores: N/A")
    print(f"Reasoning: {llm_result.get('reasoning', 'N/A')}")

inspect_chunk(1)

#%% QUICK TEST: SINGLE TEXT CLASSIFICATION

test_text = """



"""

test_metadata = {
    "firm_id": "TEST",
    "firm_name": "Test Company",
    "report_year": 2024,
    "sector": "Technology",
    "report_section": "Strategic Report",
}

# Classify
clf = MentionTypeClassifier(run_id="quick-test")
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

    if "adoption" in mention_types:
        adoption_result = AdoptionTypeClassifier(run_id="quick-test").classify(
            test_text.strip(), downstream_metadata
        )
        print(f"\nAdoption types: {adoption_result.classification}")

    if "risk" in mention_types:
        risk_result = RiskClassifier(run_id="quick-test").classify(
            test_text.strip(), downstream_metadata
        )
        print(f"\nRisk taxonomy: {risk_result.classification}")

    if RUN_VENDOR_IF_MENTIONED and "vendor" in mention_types:
        vendor_result = VendorClassifier(run_id="quick-test").classify(
            test_text.strip(), test_metadata
        )
        print(f"\nVendor tags: {vendor_result.classification}")
