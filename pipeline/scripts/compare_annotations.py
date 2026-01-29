#!/usr/bin/env python3
"""Compare human vs LLM annotations for debugging and analysis.

Usage:
    python scripts/compare_annotations.py --human data/golden_set/human/annotations.jsonl \
                                          --llm data/golden_set/llm-test/annotations.jsonl \
                                          --chunk-id <chunk_id>

    # Or show all chunks:
    python scripts/compare_annotations.py --human data/golden_set/human/annotations.jsonl \
                                          --llm data/golden_set/llm-test/annotations.jsonl \
                                          --all

    # Or show specific chunk by index:
    python scripts/compare_annotations.py --human data/golden_set/human/annotations.jsonl \
                                          --llm data/golden_set/llm-test/annotations.jsonl \
                                          --index 3
"""

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional


def load_annotations(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load annotations from JSONL file, keyed by chunk_id."""
    annotations = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                annotations[r['chunk_id']] = r
    return annotations


def format_list(items: list, default: str = "-") -> str:
    """Format a list for display."""
    if not items:
        return default
    return ", ".join(str(item) for item in items)


def print_comparison(
    chunk_id: str,
    human: Optional[Dict[str, Any]],
    llm: Optional[Dict[str, Any]],
    show_text: bool = True,
    text_width: int = 100,
) -> None:
    """Print a side-by-side comparison of human vs LLM annotations."""

    print("=" * text_width)
    print(f"CHUNK ID: {chunk_id}")
    print("=" * text_width)

    if not human and not llm:
        print("  No annotations found for this chunk.")
        return

    # Get the annotation that exists for metadata
    ann = llm or human

    print(f"Company: {ann.get('company_name', 'Unknown')}")
    print(f"Year: {ann.get('report_year', 'Unknown')}")
    print(f"Section: {format_list(ann.get('report_sections', []))}")
    print(f"Keywords: {format_list(ann.get('matched_keywords', []))}")
    print()

    # Show chunk text
    if show_text:
        print("-" * text_width)
        print("CHUNK TEXT:")
        print("-" * text_width)
        text = ann.get('chunk_text', '')
        # Wrap text for readability
        wrapped = textwrap.fill(text, width=text_width - 4)
        for line in wrapped.split('\n'):
            print(f"  {line}")
        print()

    # Compare annotations
    print("-" * text_width)
    print("ANNOTATIONS COMPARISON:")
    print("-" * text_width)

    def get_field(d: Optional[Dict], key: str, default=None):
        if d is None:
            return default
        return d.get(key, default)

    # Mention types
    h_mention = get_field(human, 'mention_types', [])
    l_mention = get_field(llm, 'mention_types', [])
    match = "✓" if set(h_mention) == set(l_mention) else ("~" if set(h_mention) & set(l_mention) else "✗")

    print(f"  Mention Types:")
    print(f"    HUMAN: [{format_list(h_mention, 'none')}]")
    print(f"    LLM:   [{format_list(l_mention, 'none')}]  {match}")
    print()

    # Adoption types
    h_adopt = get_field(human, 'adoption_types', [])
    l_adopt = get_field(llm, 'adoption_types', [])
    match = "✓" if set(h_adopt) == set(l_adopt) else ("~" if set(h_adopt) & set(l_adopt) else "✗")

    print(f"  Adoption Types:")
    print(f"    HUMAN: [{format_list(h_adopt)}]")
    print(f"    LLM:   [{format_list(l_adopt)}]  {match}")
    print()

    # Risk taxonomy
    h_risk = get_field(human, 'risk_taxonomy', [])
    l_risk = get_field(llm, 'risk_taxonomy', [])
    match = "✓" if set(h_risk) == set(l_risk) else ("~" if set(h_risk) & set(l_risk) else "✗")

    print(f"  Risk Taxonomy:")
    print(f"    HUMAN: [{format_list(h_risk)}]")
    print(f"    LLM:   [{format_list(l_risk)}]  {match}")
    print()

    # Confidence scores (LLM only)
    if llm and 'llm_details' in llm:
        details = llm['llm_details']
        print(f"  LLM Confidence Scores:")

        mention_conf = details.get('mention_confidences', {})
        if mention_conf:
            conf_str = ", ".join(
                f"{k}:{v:.2f}" for k, v in mention_conf.items()
                if v is not None and v > 0
            )
            print(f"    Mention: {conf_str or '-'}")

        adopt_conf = details.get('adoption_confidences', {})
        if adopt_conf:
            conf_str = ", ".join(
                f"{k}:{v:.2f}" for k, v in adopt_conf.items()
                if v is not None and v > 0
            )
            print(f"    Adoption: {conf_str or '-'}")

        risk_conf = details.get('risk_confidences', {})
        if risk_conf:
            conf_str = ", ".join(
                f"{k}:{v:.2f}" for k, v in risk_conf.items()
                if v is not None and v > 0
            )
            print(f"    Risk: {conf_str or '-'}")

    print("=" * text_width)
    print()


def print_summary(human_annotations: Dict, llm_annotations: Dict) -> None:
    """Print a summary of all comparisons."""

    # Get all chunk IDs
    all_ids = set(human_annotations.keys()) | set(llm_annotations.keys())

    exact = 0
    partial = 0
    mismatch = 0

    print("=" * 80)
    print("SUMMARY: All Chunk Comparisons")
    print("=" * 80)
    print(f"{'#':<4} {'Match':<8} {'Human Mention':<30} {'LLM Mention':<30}")
    print("-" * 80)

    for i, chunk_id in enumerate(sorted(all_ids), 1):
        h = human_annotations.get(chunk_id)
        l = llm_annotations.get(chunk_id)

        h_mention = set(h.get('mention_types', [])) if h else set()
        l_mention = set(l.get('mention_types', [])) if l else set()

        if h_mention == l_mention:
            match = "✓ EXACT"
            exact += 1
        elif h_mention & l_mention:
            match = "~ PART"
            partial += 1
        else:
            match = "✗ DIFF"
            mismatch += 1

        h_str = ", ".join(sorted(h_mention)) if h_mention else "none"
        l_str = ", ".join(sorted(l_mention)) if l_mention else "none"

        print(f"{i:<4} {match:<8} {h_str:<30} {l_str:<30}")

    print("-" * 80)
    total = len(all_ids)
    print(f"Total: {total} chunks")
    print(f"  Exact:   {exact} ({exact/total*100:.0f}%)")
    print(f"  Partial: {partial} ({partial/total*100:.0f}%)")
    print(f"  Differ:  {mismatch} ({mismatch/total*100:.0f}%)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare human vs LLM annotations")
    parser.add_argument("--human", type=Path, required=True, help="Path to human annotations JSONL")
    parser.add_argument("--llm", type=Path, required=True, help="Path to LLM annotations JSONL")
    parser.add_argument("--chunk-id", type=str, help="Specific chunk ID to compare")
    parser.add_argument("--index", type=int, help="Chunk index (1-based) to compare")
    parser.add_argument("--all", action="store_true", help="Show all chunks")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    parser.add_argument("--no-text", action="store_true", help="Hide chunk text")
    parser.add_argument("--width", type=int, default=100, help="Text width for wrapping")

    args = parser.parse_args()

    # Load annotations
    human_annotations = load_annotations(args.human) if args.human.exists() else {}
    llm_annotations = load_annotations(args.llm) if args.llm.exists() else {}

    print(f"Loaded {len(human_annotations)} human annotations")
    print(f"Loaded {len(llm_annotations)} LLM annotations")
    print()

    if args.summary:
        print_summary(human_annotations, llm_annotations)
        return

    if args.chunk_id:
        print_comparison(
            args.chunk_id,
            human_annotations.get(args.chunk_id),
            llm_annotations.get(args.chunk_id),
            show_text=not args.no_text,
            text_width=args.width,
        )
    elif args.index:
        # Get chunk by index
        all_ids = sorted(set(human_annotations.keys()) | set(llm_annotations.keys()))
        if 1 <= args.index <= len(all_ids):
            chunk_id = all_ids[args.index - 1]
            print_comparison(
                chunk_id,
                human_annotations.get(chunk_id),
                llm_annotations.get(chunk_id),
                show_text=not args.no_text,
                text_width=args.width,
            )
        else:
            print(f"Error: Index {args.index} out of range (1-{len(all_ids)})")
    elif args.all:
        all_ids = sorted(set(human_annotations.keys()) | set(llm_annotations.keys()))
        for chunk_id in all_ids:
            print_comparison(
                chunk_id,
                human_annotations.get(chunk_id),
                llm_annotations.get(chunk_id),
                show_text=not args.no_text,
                text_width=args.width,
            )
        print_summary(human_annotations, llm_annotations)
    else:
        # Default: show summary
        print_summary(human_annotations, llm_annotations)
        print()
        print("Use --chunk-id, --index, or --all to see detailed comparisons.")
        print("Use --summary for summary only.")


if __name__ == "__main__":
    main()
