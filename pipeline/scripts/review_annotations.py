#!/usr/bin/env python3
"""Interactive annotation review tool.

Displays chunk text alongside human and LLM variant annotations with difference
highlighting for easy comparison and judgment.

Usage:
    python scripts/review_annotations.py --dir ../data/classifier_dev --variants v0
    python scripts/review_annotations.py --dir ../data/classifier_dev --variants v0,full,limited_reasoning
    python scripts/review_annotations.py --dir ../data/classifier_dev --variants all --chunk 7
    python scripts/review_annotations.py --dir ../data/classifier_dev --variants v0 --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DATA_DIR = PIPELINE_ROOT.parent / "data"

# ANSI color codes
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"

# Colors for different variants (cycle through these)
VARIANT_COLORS = [C.BLUE, C.MAGENTA, C.YELLOW, C.RED, C.GREEN]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review annotations for classifier dev chunks.")
    parser.add_argument("--dir", type=Path,
                        default=DATA_DIR / "classifier_dev",
                        help="Directory containing test_chunks.jsonl, human.jsonl, and llm_*.jsonl files.")
    parser.add_argument("--variants", type=str, default="v0",
                        help="Comma-separated variant labels to compare (e.g. v0,full,limited_reasoning). "
                             "Use 'all' to auto-discover all available variants.")
    parser.add_argument("--chunk", type=int, default=None,
                        help="Jump to chunk number (1-indexed).")
    parser.add_argument("--all", action="store_true",
                        help="Print all chunks without paging.")
    parser.add_argument("--max-text", type=int, default=1500,
                        help="Max chars of chunk text to display.")
    return parser.parse_args()


def load_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    records = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                cid = obj.get("chunk_id")
                if cid:
                    records[cid] = obj
    return records


def discover_variants(dir_path: Path) -> List[str]:
    """Find all llm_*.jsonl files and return sorted variant labels."""
    variants = []
    for p in sorted(dir_path.glob("llm_*.jsonl")):
        label = p.stem[4:]  # strip "llm_" prefix
        variants.append(label)
    return variants


def fmt_list(items: List[str], color: str = "") -> str:
    if not items:
        return f"{C.DIM}(none){C.RESET}"
    return ", ".join(f"{color}{item}{C.RESET}" for item in sorted(items))


def fmt_conf(conf: Dict[str, Any], threshold: float = 0.1) -> str:
    if not conf:
        return f"{C.DIM}(none){C.RESET}"
    parts = []
    for k, v in sorted(conf.items()):
        if isinstance(v, (int, float)):
            if v >= 0.7:
                color = C.GREEN
            elif v >= threshold:
                color = C.YELLOW
            else:
                color = C.DIM
            parts.append(f"{color}{k}={v:.2f}{C.RESET}")
    return "  ".join(parts) if parts else f"{C.DIM}(none){C.RESET}"


def highlight_diff(human_items: List[str], llm_items: List[str]) -> str:
    """Show items with color: green=agree, red=LLM-only, magenta=human-only."""
    h_set = set(human_items)
    l_set = set(llm_items)
    agree = h_set & l_set
    human_only = h_set - l_set
    llm_only = l_set - h_set

    parts = []
    for item in sorted(agree):
        parts.append(f"{C.GREEN}{item}{C.RESET}")
    for item in sorted(human_only):
        parts.append(f"{C.MAGENTA}-{item}{C.RESET}")
    for item in sorted(llm_only):
        parts.append(f"{C.RED}+{item}{C.RESET}")
    return ", ".join(parts) if parts else f"{C.DIM}(none){C.RESET}"


def print_chunk_review(
    idx: int,
    total: int,
    chunk: Dict[str, Any],
    human: Optional[Dict[str, Any]],
    variants: Dict[str, Optional[Dict[str, Any]]],
    max_text: int,
) -> None:
    cid = chunk["chunk_id"]
    company = chunk.get("company_name", "Unknown")
    year = chunk.get("report_year", "?")
    keywords = chunk.get("matched_keywords", [])
    sections = chunk.get("report_sections", [])
    text = chunk.get("chunk_text", "")

    # Header
    print(f"\n{'='*80}")
    print(f"{C.BOLD}[{idx}/{total}] {company} ({year}){C.RESET}")
    print(f"{'='*80}")
    print(f"{C.DIM}chunk_id: {cid}{C.RESET}")
    print(f"{C.DIM}keywords: {', '.join(keywords)}  |  sections: {', '.join(sections[:2])}{C.RESET}")

    # Chunk text
    print(f"\n{C.BOLD}--- CHUNK TEXT ---{C.RESET}")
    display_text = text[:max_text]
    if len(text) > max_text:
        display_text += f"\n{C.DIM}[...truncated, {len(text)} chars total]{C.RESET}"
    # Highlight keyword matches in text
    for kw in keywords:
        search = kw.replace("_", " ")
        lower_text = display_text.lower()
        pos = lower_text.find(search)
        if pos >= 0:
            original = display_text[pos:pos+len(search)]
            display_text = display_text[:pos] + f"{C.BG_YELLOW}{C.BOLD}{original}{C.RESET}" + display_text[pos+len(search):]
    print(display_text)

    # Annotations comparison
    variant_labels = list(variants.keys())
    variant_color_map = {label: VARIANT_COLORS[i % len(VARIANT_COLORS)] for i, label in enumerate(variant_labels)}

    print(f"\n{C.BOLD}--- ANNOTATIONS ---{C.RESET}")
    legend_parts = [f"{C.GREEN}agree{C.RESET}", f"{C.MAGENTA}-human_only{C.RESET}", f"{C.RED}+llm_only{C.RESET}"]
    print(f"{'Legend:'} {'  '.join(legend_parts)}")
    variant_legend = "  ".join(f"{variant_color_map[l]}{l}{C.RESET}" for l in variant_labels)
    print(f"{'Variants:'} {variant_legend}")
    print()

    h = human or {}
    h_mention = h.get("mention_types", [])
    h_adoption = h.get("adoption_types", [])
    h_risk = h.get("risk_taxonomy", [])
    h_vendor = h.get("vendor_tags", [])
    h_adopt_conf = h.get("adoption_signals") or h.get("adoption_confidences", {})
    if isinstance(h_adopt_conf, list):
        h_adopt_conf = {
            str(e.get("type")): float(e.get("signal"))
            for e in h_adopt_conf
            if isinstance(e, dict) and isinstance(e.get("signal"), (int, float))
        }
    h_risk_conf = h.get("risk_confidences", {})

    # Mention types
    print(f"  {C.BOLD}Mention Types:{C.RESET}")
    print(f"    Human: {fmt_list(h_mention, C.CYAN)}")
    for label in variant_labels:
        v = variants[label] or {}
        v_mention = v.get("mention_types", [])
        color = variant_color_map[label]
        print(f"    {color}{label:16s}{C.RESET} {fmt_list(v_mention, C.CYAN)}")
        v_mention_conf = v.get("mention_confidences", {})
        if v_mention_conf:
            print(f"    {'':16s} conf: {fmt_conf(v_mention_conf)}")
        print(f"    {'':16s} diff: {highlight_diff(h_mention, v_mention)}")

    # Adoption types
    print(f"\n  {C.BOLD}Adoption Types:{C.RESET}")
    print(f"    Human: {fmt_list(h_adoption, C.CYAN)}  conf={fmt_conf(h_adopt_conf)}")
    for label in variant_labels:
        v = variants[label] or {}
        v_adoption = v.get("adoption_types", [])
        v_adopt_conf = v.get("adoption_signals") or v.get("adoption_confidences", {})
        if isinstance(v_adopt_conf, list):
            v_adopt_conf = {
                str(e.get("type")): float(e.get("signal"))
                for e in v_adopt_conf
                if isinstance(e, dict) and isinstance(e.get("signal"), (int, float))
            }
        color = variant_color_map[label]
        print(f"    {color}{label:16s}{C.RESET} {fmt_list(v_adoption, C.CYAN)}  conf={fmt_conf(v_adopt_conf)}")
        print(f"    {'':16s} diff: {highlight_diff(h_adoption, v_adoption)}")

    # Risk taxonomy
    print(f"\n  {C.BOLD}Risk Taxonomy:{C.RESET}")
    h_subst = h.get("risk_substantiveness")
    subst_str = f"  subst={h_subst}" if h_subst is not None else ""
    print(f"    Human: {fmt_list(h_risk, C.CYAN)}  conf={fmt_conf(h_risk_conf)}{subst_str}")
    for label in variant_labels:
        v = variants[label] or {}
        v_risk = v.get("risk_taxonomy", [])
        v_risk_conf = v.get("risk_confidences", {})
        v_subst = v.get("risk_substantiveness")
        color = variant_color_map[label]
        subst_str = f"  subst={v_subst}" if v_subst is not None else ""
        print(f"    {color}{label:16s}{C.RESET} {fmt_list(v_risk, C.CYAN)}  conf={fmt_conf(v_risk_conf)}{subst_str}")
        print(f"    {'':16s} diff: {highlight_diff(h_risk, v_risk)}")

    # Vendor
    print(f"\n  {C.BOLD}Vendor Tags:{C.RESET}")
    h_vendor_other = h.get("vendor_other")
    other_str = f"  other={h_vendor_other}" if h_vendor_other else ""
    print(f"    Human: {fmt_list(h_vendor, C.CYAN)}{other_str}")
    for label in variant_labels:
        v = variants[label] or {}
        v_vendor = v.get("vendor_tags", [])
        v_vendor_other = v.get("vendor_other")
        color = variant_color_map[label]
        other_str = f"  other={v_vendor_other}" if v_vendor_other else ""
        print(f"    {color}{label:16s}{C.RESET} {fmt_list(v_vendor, C.CYAN)}{other_str}")
        print(f"    {'':16s} diff: {highlight_diff(h_vendor, v_vendor)}")

    print()


def main() -> None:
    args = parse_args()

    chunks_path = args.dir / "test_chunks.jsonl"
    human_path = args.dir / "human.jsonl"

    if not chunks_path.exists():
        print(f"Error: {chunks_path} not found")
        sys.exit(1)

    # Load chunks
    chunks = []
    with chunks_path.open() as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    # Load human annotations
    human = load_jsonl(human_path) if human_path.exists() else {}

    # Resolve variant labels
    if args.variants.strip().lower() == "all":
        variant_labels = discover_variants(args.dir)
    else:
        variant_labels = [v.strip() for v in args.variants.split(",")]

    # Load variant annotations
    variant_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for label in variant_labels:
        variant_path = args.dir / f"llm_{label}.jsonl"
        if variant_path.exists():
            variant_data[label] = load_jsonl(variant_path)
        else:
            print(f"{C.YELLOW}Warning: {variant_path} not found, skipping variant '{label}'{C.RESET}")
            variant_data[label] = {}

    # Remove empty variants
    variant_labels = [l for l in variant_labels if l in variant_data]

    if not variant_labels:
        print("Error: No variant files found.")
        available = discover_variants(args.dir)
        if available:
            print(f"Available variants: {', '.join(available)}")
        sys.exit(1)

    print(f"{C.BOLD}Annotation Review Tool{C.RESET}")
    print(f"Dir: {args.dir}")
    print(f"Chunks: {len(chunks)} | Human annotations: {len(human)} | Variants: {', '.join(variant_labels)}")

    if args.chunk:
        # Show a single chunk
        idx = args.chunk
        if idx < 1 or idx > len(chunks):
            print(f"Invalid chunk number. Must be 1-{len(chunks)}")
            return
        chunk = chunks[idx - 1]
        cid = chunk["chunk_id"]
        chunk_variants = {label: variant_data[label].get(cid) for label in variant_labels}
        print_chunk_review(idx, len(chunks), chunk, human.get(cid), chunk_variants, args.max_text)
        return

    if args.all:
        # Print all without paging
        for i, chunk in enumerate(chunks, 1):
            cid = chunk["chunk_id"]
            chunk_variants = {label: variant_data[label].get(cid) for label in variant_labels}
            print_chunk_review(i, len(chunks), chunk, human.get(cid), chunk_variants, args.max_text)
        return

    # Interactive paging
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        cid = chunk["chunk_id"]
        chunk_variants = {label: variant_data[label].get(cid) for label in variant_labels}
        print_chunk_review(i + 1, len(chunks), chunk, human.get(cid), chunk_variants, args.max_text)

        if i < len(chunks) - 1:
            try:
                resp = input(f"{C.DIM}[Enter=next, q=quit, N=jump to #N]{C.RESET} ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if resp == "q":
                break
            elif resp.isdigit():
                target = int(resp)
                if 1 <= target <= len(chunks):
                    i = target - 1
                    continue
            i += 1
        else:
            print(f"\n{C.DIM}End of chunks.{C.RESET}")
            break


if __name__ == "__main__":
    main()
