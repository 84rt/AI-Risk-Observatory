#!/usr/bin/env python3
"""CLI tool for human annotation of AI-mention chunks."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


MENTION_TYPES = [
    ("adoption", "Adoption"),
    ("risk", "Risk"),
    ("harm", "Harm"),
    ("vendor", "Vendor"),
    ("general_ambiguous", "General/Ambiguous"),
    ("none", "None"),
]

ADOPTION_TYPES = [
    ("non_llm", "Non-LLM"),
    ("llm", "LLM"),
    ("agentic", "Agentic"),
    ("none", "None/Unclear"),
]

RISK_TAXONOMY = [
    ("strategic_market", "Strategic & Market (failure to adopt, displacement)"),
    ("operational_technical", "Operational & Technical (failures, bias, reliability, errors)"),
    ("cybersecurity", "Cybersecurity (AI attacks, breaches, vulnerabilities)"),
    ("workforce", "Workforce Impacts (jobs, skills, automation)"),
    ("regulatory", "Regulatory & Compliance (liability, AI regs)"),
    ("information_integrity", "Information Integrity (misinfo, deepfakes)"),
    ("reputational_ethical", "Reputational & Ethical (trust, ethics, rights)"),
    ("third_party_supply_chain", "Third-Party & Supply Chain (vendor reliance, misuse)"),
    ("environmental", "Environmental Impact (energy, emissions)"),
    ("national_security", "National Security (geopolitical, export, adversarial)"),
    ("none", "None/Unclear"),
]

VENDOR_TAGS = [
    ("google", "Google"),
    ("microsoft", "Microsoft"),
    ("openai", "OpenAI"),
    ("internal", "Internal"),
    ("undisclosed", "Undisclosed"),
    ("other", "Other (free text)"),
]


ANSI_HIGHLIGHT = "\x1b[93m"
ANSI_RESET = "\x1b[0m"


def use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def colorize(text: str, color: str) -> str:
    if not use_color():
        return text
    return f"{color}{text}{ANSI_RESET}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate AI-mention chunks (human).")
    parser.add_argument(
        "--chunks",
        type=str,
        required=True,
        help="Path to chunks.jsonl",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id override (defaults to inferred from chunks path).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/golden_set/human",
        help="Output directory for annotations and progress files.",
    )
    return parser.parse_args()


def infer_run_id(chunks_path: Path) -> str:
    parts = chunks_path.as_posix().split("/")
    if "processed" in parts:
        idx = parts.index("processed")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown-run"


def load_annotated_chunk_ids(path: Path) -> set:
    if not path.exists():
        return set()
    chunk_ids = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = obj.get("chunk_id")
            if chunk_id:
                chunk_ids.add(chunk_id)
    return chunk_ids


def highlight_matches(text: str, matches: Iterable[str]) -> str:
    highlighted = text
    for match in sorted(set(matches), key=len, reverse=True):
        if not match:
            continue
        pattern = re.escape(match)
        tagged = f"`{match}`"
        highlighted = re.sub(pattern, colorize(tagged, ANSI_HIGHLIGHT), highlighted)
    return highlighted


def display_chunk(chunk: Dict) -> None:
    text = chunk.get("chunk_text", "")
    keyword_matches = chunk.get("keyword_matches") or []
    match_texts = [m.get("text") for item in keyword_matches for m in item.get("matches", [])]
    display_text = highlight_matches(text, match_texts)
    print("\n" + "=" * 80)
    print(f"Chunk ID: {chunk.get('chunk_id')}")
    print(f"Document: {chunk.get('document_id')} | {chunk.get('company_name')} | {chunk.get('report_year')}")
    print(f"Sections: {', '.join(chunk.get('report_sections') or [])}")
    print("-" * 80)
    print(display_text)
    print("=" * 80 + "\n")


def prompt_multi_select(
    title: str,
    options: List[Tuple[str, str]],
    allow_none: bool = True,
    multiline: bool = False,
) -> List[str]:
    if multiline:
        option_str = "\n".join([f"{i + 1}={label}" for i, (_, label) in enumerate(options)])
    else:
        option_str = " ".join([f"{i + 1}={label}" for i, (_, label) in enumerate(options)])
    while True:
        prompt = f"{title}\n{option_str}\n> "
        raw = input(colorize(prompt, ANSI_HIGHLIGHT)).strip()
        if raw.lower() in {"q", "quit"}:
            raise KeyboardInterrupt
        if raw.lower() in {"s", "skip"}:
            return ["__skip__"]
        if not raw:
            print("Please enter one or more numbers.")
            continue
        parts = re.split(r"[,\s]+", raw)
        selections = []
        invalid = False
        for part in parts:
            if not part:
                continue
            if not part.isdigit():
                invalid = True
                break
            idx = int(part) - 1
            if idx < 0 or idx >= len(options):
                invalid = True
                break
            selections.append(options[idx][0])
        if invalid or not selections:
            print("Invalid selection. Try again.")
            continue
        if "none" in selections and len(selections) > 1:
            print("If you select None, no other options are allowed.")
            continue
        if not allow_none and "none" in selections:
            print("None is not allowed here.")
            continue
        return selections


def prompt_float(title: str, min_val: float = 0.0, max_val: float = 1.0) -> Optional[float]:
    while True:
        prompt = f"{title} ({min_val}-{max_val})\n> "
        raw = input(colorize(prompt, ANSI_HIGHLIGHT)).strip()
        if raw.lower() in {"q", "quit"}:
            raise KeyboardInterrupt
        if raw.lower() in {"s", "skip"}:
            return None
        try:
            val = float(raw)
        except ValueError:
            print("Enter a number.")
            continue
        if val < min_val or val > max_val:
            print("Out of range.")
            continue
        return val


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def append_jsonl(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def annotate_chunk(
    chunk: Dict,
    run_id: str,
    output_dir: Path,
    in_progress_dir: Path,
) -> Optional[Dict]:
    record = {
        "annotation_id": f"hum-{run_id}-{chunk.get('chunk_id')}",
        "run_id": run_id,
        "chunk_id": chunk.get("chunk_id"),
        "document_id": chunk.get("document_id"),
        "company_id": chunk.get("company_id"),
        "company_name": chunk.get("company_name"),
        "report_year": chunk.get("report_year"),
        "report_sections": chunk.get("report_sections"),
        "chunk_text": chunk.get("chunk_text"),
        "matched_keywords": chunk.get("matched_keywords"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mention_types": [],
        "adoption_types": [],
        "adoption_confidence": {},  # per-type confidence: {"llm": 0.8, "non_llm": 0.6}
        "risk_taxonomy": [],
        "risk_confidence": {},  # per-risk confidence: {"cybersecurity": 0.9}
        "risk_substantiveness": None,  # overall substantiveness for the mention
        "vendor_tags": [],
        "vendor_other": None,
    }

    in_progress_path = in_progress_dir / f"{chunk.get('chunk_id')}.json"
    write_json(in_progress_path, record)

    mention_types = prompt_multi_select(
        "Select mention types (multi-select).",
        MENTION_TYPES,
    )
    if mention_types == ["__skip__"]:
        return None
    record["mention_types"] = mention_types
    write_json(in_progress_path, record)

    if "adoption" in mention_types:
        adoption_types = prompt_multi_select(
            "Adoption type (multi-select).",
            ADOPTION_TYPES,
        )
        if adoption_types == ["__skip__"]:
            return None
        record["adoption_types"] = adoption_types
        write_json(in_progress_path, record)

        # Ask for confidence on each selected adoption type
        adoption_confidence = {}
        for atype in adoption_types:
            if atype == "none":
                continue
            label = next((lbl for key, lbl in ADOPTION_TYPES if key == atype), atype)
            conf = prompt_float(f"Confidence for '{label}'", 0.0, 1.0)
            if conf is not None:
                adoption_confidence[atype] = conf
            write_json(in_progress_path, record)
        record["adoption_confidence"] = adoption_confidence
        write_json(in_progress_path, record)

    if "risk" in mention_types:
        risk_types = prompt_multi_select(
            "Risk taxonomy (multi-select).",
            RISK_TAXONOMY,
            multiline=True,
        )
        if risk_types == ["__skip__"]:
            return None
        record["risk_taxonomy"] = risk_types
        write_json(in_progress_path, record)

        if risk_types != ["none"]:
            # Ask for confidence on each selected risk type
            risk_confidence = {}
            for rtype in risk_types:
                if rtype == "none":
                    continue
                label = next((lbl for key, lbl in RISK_TAXONOMY if key == rtype), rtype)
                # Truncate label for display
                short_label = label.split("(")[0].strip() if "(" in label else label
                conf = prompt_float(f"Confidence for '{short_label}'", 0.0, 1.0)
                if conf is not None:
                    risk_confidence[rtype] = conf
                write_json(in_progress_path, record)
            record["risk_confidence"] = risk_confidence
            write_json(in_progress_path, record)

            # Ask for overall substantiveness of the risk mention
            substantiveness = prompt_float("Overall risk substantiveness", 0.0, 1.0)
            record["risk_substantiveness"] = substantiveness
            write_json(in_progress_path, record)

    if "vendor" in mention_types:
        vendor_tags = prompt_multi_select(
            "Vendor tags (multi-select).",
            VENDOR_TAGS,
        )
        if vendor_tags == ["__skip__"]:
            return None
        record["vendor_tags"] = vendor_tags
        write_json(in_progress_path, record)

        if "other" in vendor_tags:
            prompt = "Vendor free text (required for Other):\n> "
            vendor_other = input(colorize(prompt, ANSI_HIGHLIGHT)).strip()
            if vendor_other.lower() in {"q", "quit"}:
                raise KeyboardInterrupt
            record["vendor_other"] = vendor_other
            write_json(in_progress_path, record)

    return record


def load_chunks(path: Path) -> Iterable[Dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    chunks_path = Path(args.chunks)
    run_id = args.run_id or infer_run_id(chunks_path)
    output_dir = Path(args.output_dir)
    annotations_path = output_dir / "annotations.jsonl"
    progress_path = output_dir / "progress.json"
    in_progress_dir = output_dir / "in_progress"

    annotated_ids = load_annotated_chunk_ids(annotations_path)
    total = 0
    annotated = 0
    skipped_existing = 0
    skipped_user = 0

    try:
        for chunk in load_chunks(chunks_path):
            total += 1
            chunk_id = chunk.get("chunk_id")
            if chunk_id in annotated_ids:
                skipped_existing += 1
                continue

            display_chunk(chunk)
            record = annotate_chunk(
                chunk=chunk,
                run_id=run_id,
                output_dir=output_dir,
                in_progress_dir=in_progress_dir,
            )
            if record is None:
                skipped_user += 1
                progress = {
                    "run_id": run_id,
                    "last_chunk_id": chunk_id,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                write_json(progress_path, progress)
                continue

            append_jsonl(annotations_path, record)
            annotated_ids.add(chunk_id)
            annotated += 1

            progress = {
                "run_id": run_id,
                "last_chunk_id": chunk_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json(progress_path, progress)

            in_progress_path = in_progress_dir / f"{chunk_id}.json"
            if in_progress_path.exists():
                in_progress_path.unlink()

    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved.")

    print(
        "Finished. "
        f"Annotated {annotated} new chunks; "
        f"skipped existing {skipped_existing}; "
        f"skipped by user {skipped_user}."
    )


if __name__ == "__main__":
    main()
