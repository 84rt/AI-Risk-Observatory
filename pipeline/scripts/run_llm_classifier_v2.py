#!/usr/bin/env python3
"""Run LLMClassifierV2 on the first N golden set chunks and compare to baseline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

# Allow importing pipeline modules
PIPELINE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE_DIR))

from progress_helper import tqdm
from helper import load_chunks

from src.classifiers import MentionTypeClassifier
from src.classifiers.llm_classifier_v2 import LLMClassifierV2


HUMAN_ANNOTATIONS = "data/golden_set/human/annotations.jsonl"
MODEL_NAME = "gemini-3-flash-preview"
MAX_CHUNKS = 10


def _extract_mention_types(classification: object) -> List[str]:
    if isinstance(classification, dict) and "mention_types" in classification:
        return [
            str(mt.value) if hasattr(mt, "value") else str(mt)
            for mt in classification["mention_types"]
        ]
    return []


def _compare_sets(human: List[str], llm: List[str]) -> str:
    h = set(human)
    l = set(llm)
    if h == l:
        return "EXACT"
    if h & l:
        return "PARTIAL"
    return "DIFF"


def main() -> None:
    chunks = load_chunks(HUMAN_ANNOTATIONS, limit=MAX_CHUNKS)
    print(f"Loaded {len(chunks)} chunks from {HUMAN_ANNOTATIONS}")

    baseline = MentionTypeClassifier(
        run_id="baseline-mention-v1",
        model_name=MODEL_NAME,
        temperature=0.0,
        thinking_budget=0,
        use_openrouter=False,
    )
    v2 = LLMClassifierV2(
        run_id="mention-v2",
        model_name=MODEL_NAME,
        temperature=0.0,
        thinking_budget=0,
        use_openrouter=False,
    )

    counts = {
        "baseline": {"EXACT": 0, "PARTIAL": 0, "DIFF": 0},
        "v2": {"EXACT": 0, "PARTIAL": 0, "DIFF": 0},
    }

    print("\nRunning classifiers on first 10 chunks...\n")
    for chunk in tqdm(chunks, desc="Classifying"):
        metadata: Dict[str, object] = {
            "firm_id": chunk.get("company_id", "Unknown"),
            "firm_name": chunk.get("company_name", "Unknown"),
            "report_year": chunk.get("report_year", 0),
            "sector": "Unknown",
            "report_section": (
                chunk.get("report_sections", ["Unknown"])[0]
                if chunk.get("report_sections")
                else "Unknown"
            ),
        }

        human = chunk.get("mention_types", []) or []

        base_result = baseline.classify(chunk["chunk_text"], metadata)
        v2_result = v2.classify(chunk["chunk_text"], metadata)

        base_labels = _extract_mention_types(base_result.classification)
        v2_labels = _extract_mention_types(v2_result.classification)

        base_match = _compare_sets(human, base_labels)
        v2_match = _compare_sets(human, v2_labels)

        counts["baseline"][base_match] += 1
        counts["v2"][v2_match] += 1

        print(
            f"- {chunk['company_name']} ({chunk['report_year']}): "
            f"human={sorted(human)} | v1={sorted(base_labels)} ({base_match}) | "
            f"v2={sorted(v2_labels)} ({v2_match})"
        )

    total = len(chunks)
    print("\nSummary (mention_types vs human):")
    for key in ("baseline", "v2"):
        exact = counts[key]["EXACT"]
        partial = counts[key]["PARTIAL"]
        diff = counts[key]["DIFF"]
        agree = exact + partial
        print(
            f"  {key}: exact={exact} ({exact/total:.0%}), "
            f"partial={partial} ({partial/total:.0%}), "
            f"diff={diff} ({diff/total:.0%}), "
            f"agreement={agree} ({agree/total:.0%})"
        )


if __name__ == "__main__":
    main()
