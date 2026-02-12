#!/usr/bin/env python3
"""Interactive tool to reconcile human vs LLM annotations.

The reviewer can pick the human annotation, the LLM annotation, or create
custom annotations (same flow as the human gold set tool).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from annotate_golden_set import (
    ADOPTION_TYPES,
    MENTION_TYPES,
    RISK_TAXONOMY,
    VENDOR_TAGS,
)
from src.utils.normalization import normalize_risk_label as normalize_risk_label_shared


ANSI_HIGHLIGHT = "\x1b[93m"
ANSI_RESET = "\x1b[0m"
ANSI_BOLD = "\x1b[1m"
ANSI_DIM = "\x1b[2m"
ANSI_CYAN = "\x1b[36m"
ANSI_GREEN = "\x1b[32m"


def use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def colorize(text: str, color: str) -> str:
    if not use_color():
        return text
    return f"{color}{text}{ANSI_RESET}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile human vs LLM annotations with interactive choices."
    )
    parser.add_argument(
        "--human",
        type=Path,
        default=Path("data/golden_set/human/annotations.jsonl"),
        help="Path to human annotations.jsonl",
    )
    parser.add_argument(
        "--llm",
        type=Path,
        default=Path("data/golden_set/llm/annotations.jsonl"),
        help="Path to LLM annotations.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/golden_set/reconciled"),
        help="Output directory for reconciled annotations.",
    )
    parser.add_argument(
        "--only-disagreements",
        action="store_true",
        help="Only show chunks where human and LLM mention_types differ.",
    )
    parser.add_argument(
        "--include-subtype-disagreements",
        action="store_true",
        help="When used with --only-disagreements, also treat adoption/risk/vendor differences as disagreements.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Limit chunks to review (0 = no limit).",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include chunks missing in either human or LLM annotations.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip chunks already reconciled in output file.",
    )
    parser.add_argument(
        "--llm-confidence-threshold",
        type=float,
        default=0.0,
        help="Ignore LLM labels below this confidence when comparing/displaying.",
    )
    parser.add_argument(
        "--show-llm-reasoning",
        action="store_true",
        help="Display LLM reasoning text when available.",
    )
    parser.add_argument(
        "--focus-field",
        choices=["all", "mention_types", "adoption", "risk", "vendor"],
        default="all",
        help=(
            "Only compare and reconcile a single field; other fields are preserved "
            "from the base record (human if available)."
        ),
    )
    return parser.parse_args()


def infer_run_id(records: Iterable[Dict[str, Any]]) -> str:
    for record in records:
        run_id = record.get("run_id")
        if run_id:
            return str(run_id)
    return "unknown-run"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def index_by_chunk_id(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        cid = record.get("chunk_id")
        if cid:
            indexed[cid] = record
    return indexed


def load_reconciled_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("chunk_id")
            if cid:
                ids.add(cid)
    return ids


def normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        return [value]
    return []


def normalize_risk_label(value: str) -> str:
    return normalize_risk_label_shared(value)


def normalize_labels_for_field(field: str, labels: List[str]) -> List[str]:
    if field == "risk_taxonomy":
        return [normalize_risk_label(label) for label in labels]
    return labels


def normalize_conf_for_field(field: str, conf: Dict[str, float]) -> Dict[str, float]:
    if field != "risk_taxonomy":
        return conf
    return {normalize_risk_label(k): v for k, v in conf.items()}


def _signals_list_to_map(value: Any) -> Dict[str, float]:
    if isinstance(value, list):
        out: Dict[str, float] = {}
        for entry in value:
            if isinstance(entry, dict):
                key = entry.get("type") or entry.get("label")
                sig = entry.get("signal")
                if key is not None and isinstance(sig, (int, float)):
                    out[str(key)] = float(sig)
        return out
    return {}


def extract_conf_map(record: Optional[Dict[str, Any]], keys: List[str]) -> Dict[str, float]:
    if not record:
        return {}
    for key in keys:
        value = record.get(key)
        if isinstance(value, dict):
            return {
                str(k): float(v)
                for k, v in value.items()
                if v is not None and isinstance(v, (int, float))
            }
        if isinstance(value, list):
            mapped = _signals_list_to_map(value)
            if mapped:
                return mapped
    return {}


def extract_llm_conf_map(record: Optional[Dict[str, Any]], key: str) -> Dict[str, float]:
    if not record:
        return {}
    details = record.get("llm_details") or {}
    value = details.get(key)
    if isinstance(value, dict):
        return {
            str(k): float(v)
            for k, v in value.items()
            if v is not None and isinstance(v, (int, float))
        }
    if isinstance(value, list):
        return _signals_list_to_map(value)
    return {}


def filter_llm_labels(
    record: Optional[Dict[str, Any]],
    field: str,
    threshold: float,
) -> List[str]:
    if not record:
        return []
    labels = normalize_list(record.get(field))
    if threshold <= 0:
        return labels
    details = record.get("llm_details") or {}
    if field == "mention_types":
        confs = details.get("mention_confidences") or {}
    elif field == "adoption_types":
        confs = details.get("adoption_signals") or details.get("adoption_confidences") or {}
        if isinstance(confs, list):
            confs = _signals_list_to_map(confs)
    elif field == "risk_taxonomy":
        confs = details.get("risk_signals") or details.get("risk_confidences") or {}
        if isinstance(confs, list):
            confs = _signals_list_to_map(confs)
    elif field == "vendor_tags":
        confs = details.get("vendor_confidences") or {}
    else:
        confs = {}
    filtered = [label for label in labels if float(confs.get(label, 0.0)) >= threshold]
    return normalize_labels_for_field(field, filtered)


def filter_llm_conf_map(conf: Dict[str, float], threshold: float) -> Dict[str, float]:
    if threshold <= 0:
        return conf
    return {k: v for k, v in conf.items() if float(v) >= threshold}


def highlight_keywords(text: str, keywords: List[str]) -> str:
    if not keywords:
        return text
    lowered = text.lower()
    out = text
    for kw in keywords:
        search = kw.replace("_", " ").lower()
        if not search:
            continue
        idx = lowered.find(search)
        if idx >= 0:
            original = out[idx:idx + len(search)]
            out = out[:idx] + colorize(original, ANSI_HIGHLIGHT + ANSI_BOLD) + out[idx + len(search):]
            lowered = out.lower()
    return out


def display_chunk(chunk: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(colorize(f"Chunk ID: {chunk.get('chunk_id')}", ANSI_BOLD))
    print(
        colorize(
            f"Document: {chunk.get('document_id')} | {chunk.get('company_name')} | {chunk.get('report_year')}",
            ANSI_DIM,
        )
    )
    print(colorize(f"Sections: {', '.join(chunk.get('report_sections') or [])}", ANSI_DIM))
    print("-" * 80)
    text = chunk.get("chunk_text", "")
    keywords = chunk.get("matched_keywords") or []
    print(highlight_keywords(text, keywords))
    print("=" * 80 + "\n")


def display_chunk_header(index: int, total: int, diff_label: Optional[str]) -> None:
    progress = f"[{index}/{total}]"
    print(colorize(f"{progress}", ANSI_BOLD))
    if diff_label:
        print(colorize(f"Mismatch: {diff_label}", ANSI_HIGHLIGHT + ANSI_BOLD))


def fmt_list(items: List[str]) -> str:
    if not items:
        return "(none)"
    return ", ".join(sorted(items))


def fmt_conf(conf: Dict[str, float]) -> str:
    if not conf:
        return "(none)"
    return ", ".join(f"{k}={v:.2f}" for k, v in sorted(conf.items()))


def fmt_risk_signals_raw(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "(none)"
    parts: List[str] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        t = entry.get("type")
        s = entry.get("signal")
        if t is None:
            continue
        if isinstance(s, (int, float)):
            parts.append(f"{t}:{int(s)}")
        else:
            parts.append(str(t))
    return ", ".join(parts) if parts else "(none)"


def fmt_substantiveness(value: Any) -> str:
    if value is None:
        return "(none)"
    return str(value)


def fmt_vendor_other(value: Optional[str]) -> str:
    if not value:
        return ""
    return f" (other_vendor: {value})"


def normalize_vendor_other(value: Optional[str]) -> str:
    return (value or "").strip()


def extract_labels(
    record: Optional[Dict[str, Any]],
    field: str,
    is_llm: bool,
    llm_threshold: float,
) -> List[str]:
    if not record:
        return []
    if is_llm:
        return normalize_labels_for_field(
            field, filter_llm_labels(record, field, llm_threshold)
        )
    return normalize_labels_for_field(field, normalize_list(record.get(field)))


def extract_confidence(
    record: Optional[Dict[str, Any]],
    keys: List[str],
    llm_detail_key: str,
    field: str,
    is_llm: bool,
    llm_threshold: float,
) -> Dict[str, float]:
    if not record:
        return {}
    conf = extract_conf_map(record, keys)
    if is_llm and not conf:
        conf = extract_llm_conf_map(record, llm_detail_key)
    if is_llm:
        conf = filter_llm_conf_map(conf, llm_threshold)
    return normalize_conf_for_field(field, conf)


def apply_focus_override(
    record: Dict[str, Any],
    selected: Dict[str, Any],
    selected_is_llm: bool,
    focus_field: str,
    llm_threshold: float,
) -> None:
    if focus_field == "mention_types":
        record["mention_types"] = extract_labels(
            selected, "mention_types", selected_is_llm, llm_threshold
        )
        return
    if focus_field == "adoption":
        record["adoption_types"] = extract_labels(
            selected, "adoption_types", selected_is_llm, llm_threshold
        )
        record["adoption_confidence"] = extract_confidence(
            selected,
            ["adoption_confidence", "adoption_confidences", "adoption_signals"],
            "adoption_signals",
            "adoption_types",
            selected_is_llm,
            llm_threshold,
        )
        return
    if focus_field == "risk":
        record["risk_taxonomy"] = extract_labels(
            selected, "risk_taxonomy", selected_is_llm, llm_threshold
        )
        record["risk_confidence"] = extract_confidence(
            selected,
            ["risk_confidence", "risk_confidences", "risk_signals"],
            "risk_signals",
            "risk_taxonomy",
            selected_is_llm,
            llm_threshold,
        )
        record["risk_substantiveness"] = selected.get("risk_substantiveness")
        return
    if focus_field == "vendor":
        vendor_tags = extract_labels(selected, "vendor_tags", selected_is_llm, llm_threshold)
        record["vendor_tags"] = vendor_tags
        record["vendor_other"] = (
            selected.get("vendor_other") if "other" in set(vendor_tags) else None
        )


def extract_llm_reasoning(record: Optional[Dict[str, Any]]) -> str:
    if not record:
        return ""
    details = record.get("llm_details") or {}
    for key in ("mention_reasoning", "reasoning", "rationale"):
        value = details.get(key)
        if value:
            return str(value).strip()
    for key in ("reasoning", "rationale"):
        value = record.get(key)
        if value:
            return str(value).strip()
    return ""


def indent_lines(text: str, prefix: str = "    ") -> str:
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def print_annotations(
    human: Optional[Dict[str, Any]],
    llm: Optional[Dict[str, Any]],
    llm_threshold: float,
    show_llm_reasoning: bool,
    focus_field: str,
) -> None:
    h = human or {}
    l = llm or {}

    h_mention = normalize_list(h.get("mention_types"))
    l_mention = filter_llm_labels(l, "mention_types", llm_threshold)

    h_adoption = normalize_list(h.get("adoption_types"))
    l_adoption = filter_llm_labels(l, "adoption_types", llm_threshold)

    h_risk = normalize_labels_for_field("risk_taxonomy", normalize_list(h.get("risk_taxonomy")))
    l_risk = filter_llm_labels(l, "risk_taxonomy", llm_threshold)

    h_vendor = normalize_list(h.get("vendor_tags"))
    l_vendor = filter_llm_labels(l, "vendor_tags", llm_threshold)
    h_vendor_other = (h.get("vendor_other") or "").strip()
    l_vendor_other = (l.get("vendor_other") or "").strip()

    h_adopt_conf = extract_conf_map(h, ["adoption_confidence", "adoption_confidences", "adoption_signals"])
    h_risk_conf = normalize_conf_for_field(
        "risk_taxonomy",
        extract_conf_map(h, ["risk_confidence", "risk_confidences", "risk_signals"]),
    )
    h_risk_sub = h.get("risk_substantiveness")

    l_adopt_conf = filter_llm_conf_map(
        extract_llm_conf_map(l, "adoption_signals") or extract_llm_conf_map(l, "adoption_confidences"),
        llm_threshold,
    )
    l_risk_conf = normalize_conf_for_field(
        "risk_taxonomy",
        filter_llm_conf_map(
            extract_llm_conf_map(l, "risk_signals") or extract_llm_conf_map(l, "risk_confidences"),
            llm_threshold,
        ),
    )
    l_mention_conf = filter_llm_conf_map(
        extract_llm_conf_map(l, "mention_confidences"),
        llm_threshold,
    )
    l_risk_sub = (l.get("llm_details") or {}).get("risk_substantiveness")
    if l_risk_sub is None:
        l_risk_sub = l.get("risk_substantiveness")
    l_risk_signals_raw = (l.get("llm_details") or {}).get("risk_signals")

    print(colorize("Annotations:", ANSI_BOLD))
    if focus_field == "all":
        print(colorize(f"  HUMAN mention_types: {fmt_list(h_mention)}", ANSI_GREEN))
        print(
            colorize(
                f"  HUMAN adoption_types: {fmt_list(h_adoption)}  conf: {fmt_conf(h_adopt_conf)}",
                ANSI_GREEN,
            )
        )
        print(
            colorize(
                f"  HUMAN risk_taxonomy: {fmt_list(h_risk)}  conf: {fmt_conf(h_risk_conf)}",
                ANSI_GREEN,
            )
        )
        print(
            colorize(
                f"  HUMAN risk_substantiveness: {fmt_substantiveness(h_risk_sub)}",
                ANSI_GREEN,
            )
        )
        print(
            colorize(
                f"  HUMAN vendor_tags: {fmt_list(h_vendor)}{fmt_vendor_other(h_vendor_other)}",
                ANSI_GREEN,
            )
        )
        print("-")
        print(
            colorize(
                f"  LLM   mention_types: {fmt_list(l_mention)}  conf: {fmt_conf(l_mention_conf)}",
                ANSI_CYAN,
            )
        )
        print(
            colorize(
                f"  LLM   adoption_types: {fmt_list(l_adoption)}  conf: {fmt_conf(l_adopt_conf)}",
                ANSI_CYAN,
            )
        )
        print(
            colorize(
                f"  LLM   risk_taxonomy: {fmt_list(l_risk)}  conf: {fmt_conf(l_risk_conf)}",
                ANSI_CYAN,
            )
        )
        print(
            colorize(
                f"  LLM   risk_signals(raw): {fmt_risk_signals_raw(l_risk_signals_raw)}",
                ANSI_CYAN,
            )
        )
        print(
            colorize(
                f"  LLM   risk_substantiveness: {fmt_substantiveness(l_risk_sub)}",
                ANSI_CYAN,
            )
        )
        print(
            colorize(
                f"  LLM   vendor_tags: {fmt_list(l_vendor)}{fmt_vendor_other(l_vendor_other)}",
                ANSI_CYAN,
            )
        )
        print()
        return

    if focus_field == "mention_types":
        print(colorize(f"  HUMAN mention_types: {fmt_list(h_mention)}", ANSI_GREEN))
        print(
            colorize(
                f"  LLM   mention_types: {fmt_list(l_mention)}  conf: {fmt_conf(l_mention_conf)}",
                ANSI_CYAN,
            )
        )
        print()
        return

    if focus_field == "adoption":
        print(
            colorize(
                f"  HUMAN adoption_types: {fmt_list(h_adoption)}  conf: {fmt_conf(h_adopt_conf)}",
                ANSI_GREEN,
            )
        )
        print(
            colorize(
                f"  LLM   adoption_types: {fmt_list(l_adoption)}  conf: {fmt_conf(l_adopt_conf)}",
                ANSI_CYAN,
            )
        )
        print()
        return

    if focus_field == "risk":
        print(
            colorize(
                f"  HUMAN risk_taxonomy: {fmt_list(h_risk)}  conf: {fmt_conf(h_risk_conf)}",
                ANSI_GREEN,
            )
        )
        print(
            colorize(
                f"  HUMAN risk_substantiveness: {fmt_substantiveness(h_risk_sub)}",
                ANSI_GREEN,
            )
        )
        print(
            colorize(
                f"  LLM   risk_taxonomy: {fmt_list(l_risk)}  conf: {fmt_conf(l_risk_conf)}",
                ANSI_CYAN,
            )
        )
        print(
            colorize(
                f"  LLM   risk_signals(raw): {fmt_risk_signals_raw(l_risk_signals_raw)}",
                ANSI_CYAN,
            )
        )
        print(
            colorize(
                f"  LLM   risk_substantiveness: {fmt_substantiveness(l_risk_sub)}",
                ANSI_CYAN,
            )
        )
        print()
        return

    if focus_field == "vendor":
        print(
            colorize(
                f"  HUMAN vendor_tags: {fmt_list(h_vendor)}{fmt_vendor_other(h_vendor_other)}",
                ANSI_GREEN,
            )
        )
        print(
            colorize(
                f"  LLM   vendor_tags: {fmt_list(l_vendor)}{fmt_vendor_other(l_vendor_other)}",
                ANSI_CYAN,
            )
        )
        print()
        return


def display_llm_reasoning(llm: Optional[Dict[str, Any]], show: bool) -> None:
    if not show or not llm:
        return
    reasoning = extract_llm_reasoning(llm)
    if not reasoning:
        return
    print(colorize("LLM reasoning:", ANSI_HIGHLIGHT + ANSI_BOLD))
    print(colorize(indent_lines(reasoning, "  "), ANSI_HIGHLIGHT))


def build_diff_label(
    human: Optional[Dict[str, Any]],
    llm: Optional[Dict[str, Any]],
    llm_threshold: float,
    focus_field: str,
) -> Optional[str]:
    if not human or not llm:
        return None
    field_map = {
        "all": "mention_types",
        "mention_types": "mention_types",
        "adoption": "adoption_types",
        "risk": "risk_taxonomy",
        "vendor": "vendor_tags",
    }
    field = field_map.get(focus_field, "mention_types")
    h_set = set(normalize_labels_for_field(field, normalize_list(human.get(field))))
    l_set = set(filter_llm_labels(llm, field, llm_threshold))
    if h_set == l_set:
        if focus_field == "vendor" and "other" in h_set:
            h_other = normalize_vendor_other(human.get("vendor_other"))
            l_other = normalize_vendor_other(llm.get("vendor_other"))
            if h_other != l_other:
                return "other_vendor differs"
        return None
    if len(h_set) == 1 and len(l_set) == 1:
        return f"{next(iter(sorted(h_set)))} -> {next(iter(sorted(l_set)))}"
    added = sorted(l_set - h_set)
    removed = sorted(h_set - l_set)
    parts = []
    if added:
        parts.append(f"Added {', '.join(added)}")
    if removed:
        parts.append(f"Removed {', '.join(removed)}")
    return "; ".join(parts) if parts else None


def summarize_disagreements(
    ids: List[str],
    human_by_id: Dict[str, Dict[str, Any]],
    llm_by_id: Dict[str, Dict[str, Any]],
    llm_threshold: float,
    focus_field: str,
) -> Dict[str, Counter]:
    added_counts: Counter = Counter()
    removed_counts: Counter = Counter()
    transition_counts: Counter = Counter()

    field_map = {
        "all": "mention_types",
        "mention_types": "mention_types",
        "adoption": "adoption_types",
        "risk": "risk_taxonomy",
        "vendor": "vendor_tags",
    }
    field = field_map.get(focus_field, "mention_types")

    for cid in ids:
        human = human_by_id.get(cid)
        llm = llm_by_id.get(cid)
        if not human or not llm:
            continue
        h_set = set(normalize_labels_for_field(field, normalize_list(human.get(field))))
        l_set = set(filter_llm_labels(llm, field, llm_threshold))
        if h_set == l_set:
            continue
        added = l_set - h_set
        removed = h_set - l_set
        for label in added:
            added_counts[label] += 1
        for label in removed:
            removed_counts[label] += 1
        if len(h_set) == 1 and len(l_set) == 1:
            transition_counts[f"{next(iter(h_set))} -> {next(iter(l_set))}"] += 1

    return {
        "added": added_counts,
        "removed": removed_counts,
        "transitions": transition_counts,
    }


def print_diff_summary(summary: Dict[str, Counter], focus_field: str) -> None:
    def print_block(title: str, counter: Counter, prefix: str = "") -> None:
        print(title)
        print("-" * len(title))
        if not counter:
            print("(none)\n")
            return
        for label, count in counter.most_common():
            print(f"{prefix}{label} x{count}")
        print()

    title_field = focus_field if focus_field != "all" else "mention_types"
    print("\n" + "=" * 60)
    print(f"DIFF SUMMARY ({title_field})")
    print("=" * 60)
    print_block("Added Labels", summary["added"], prefix="Added ")
    print_block("Removed Labels", summary["removed"], prefix="Removed ")
    print_block("Single-Label Transitions", summary["transitions"], prefix="")


def annotations_match(
    human: Optional[Dict[str, Any]],
    llm: Optional[Dict[str, Any]],
    llm_threshold: float,
    include_subtypes: bool,
    focus_field: str,
) -> bool:
    if not human or not llm:
        return False
    if focus_field != "all":
        field_map = {
            "mention_types": "mention_types",
            "adoption": "adoption_types",
            "risk": "risk_taxonomy",
            "vendor": "vendor_tags",
        }
        field = field_map.get(focus_field, "mention_types")
        human_set = set(normalize_labels_for_field(field, normalize_list(human.get(field))))
        llm_set = set(filter_llm_labels(llm, field, llm_threshold))
        if human_set != llm_set:
            return False
        if focus_field == "vendor" and "other" in human_set:
            h_other = normalize_vendor_other(human.get("vendor_other"))
            l_other = normalize_vendor_other(llm.get("vendor_other"))
            if h_other != l_other:
                return False
        return True

    fields = ["mention_types"]
    if include_subtypes:
        fields.extend(["adoption_types", "risk_taxonomy", "vendor_tags"])
    for field in fields:
        human_set = set(normalize_labels_for_field(field, normalize_list(human.get(field))))
        llm_set = set(filter_llm_labels(llm, field, llm_threshold))
        if human_set != llm_set:
            return False
    return True


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


def build_custom_annotation(chunk: Dict[str, Any], run_id: str) -> Optional[Dict[str, Any]]:
    record = {
        "annotation_id": f"review-{run_id}-{chunk.get('chunk_id')}",
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
        "adoption_confidence": {},
        "risk_taxonomy": [],
        "risk_confidence": {},
        "risk_substantiveness": None,
        "vendor_tags": [],
        "vendor_other": None,
    }

    mention_types = prompt_multi_select("Select mention types (multi-select).", MENTION_TYPES)
    if mention_types == ["__skip__"]:
        return None
    record["mention_types"] = mention_types

    if "adoption" in mention_types:
        adoption_types = prompt_multi_select("Adoption type (multi-select).", ADOPTION_TYPES)
        if adoption_types == ["__skip__"]:
            return None
        record["adoption_types"] = adoption_types

        adoption_confidence: Dict[str, float] = {}
        for atype in adoption_types:
            if atype == "none":
                continue
            label = next((lbl for key, lbl in ADOPTION_TYPES if key == atype), atype)
            conf = prompt_float(f"Confidence for '{label}'", 0.0, 1.0)
            if conf is not None:
                adoption_confidence[atype] = conf
        record["adoption_confidence"] = adoption_confidence

    if "risk" in mention_types:
        risk_types = prompt_multi_select(
            "Risk taxonomy (multi-select).",
            RISK_TAXONOMY,
            multiline=True,
        )
        if risk_types == ["__skip__"]:
            return None
        record["risk_taxonomy"] = risk_types

        if risk_types != ["none"]:
            risk_confidence: Dict[str, float] = {}
            for rtype in risk_types:
                if rtype == "none":
                    continue
                label = next((lbl for key, lbl in RISK_TAXONOMY if key == rtype), rtype)
                short_label = label.split("(")[0].strip() if "(" in label else label
                conf = prompt_float(f"Confidence for '{short_label}'", 0.0, 1.0)
                if conf is not None:
                    risk_confidence[rtype] = conf
            record["risk_confidence"] = risk_confidence

            substantiveness = prompt_float("Overall risk substantiveness", 0.0, 1.0)
            record["risk_substantiveness"] = substantiveness

    if "vendor" in mention_types:
        vendor_tags = prompt_multi_select("Vendor tags (multi-select).", VENDOR_TAGS)
        if vendor_tags == ["__skip__"]:
            return None
        record["vendor_tags"] = vendor_tags

        if "other" in vendor_tags:
            prompt = "Vendor free text (required for Other):\n> "
            vendor_other = input(colorize(prompt, ANSI_HIGHLIGHT)).strip()
            if vendor_other.lower() in {"q", "quit"}:
                raise KeyboardInterrupt
            record["vendor_other"] = vendor_other

    return record


def build_selected_annotation(
    selected: Dict[str, Any],
    run_id: str,
    source: str,
    llm_threshold: float,
) -> Dict[str, Any]:
    is_llm = source == "llm"
    mention_types = normalize_list(selected.get("mention_types"))
    adoption_types = normalize_list(selected.get("adoption_types"))
    risk_taxonomy = normalize_labels_for_field(
        "risk_taxonomy", normalize_list(selected.get("risk_taxonomy"))
    )
    vendor_tags = normalize_list(selected.get("vendor_tags"))

    if is_llm and llm_threshold > 0:
        mention_types = filter_llm_labels(selected, "mention_types", llm_threshold)
        adoption_types = filter_llm_labels(selected, "adoption_types", llm_threshold)
        risk_taxonomy = normalize_labels_for_field(
            "risk_taxonomy",
            filter_llm_labels(selected, "risk_taxonomy", llm_threshold),
        )
        vendor_tags = filter_llm_labels(selected, "vendor_tags", llm_threshold)

    record = {
        "annotation_id": f"review-{run_id}-{selected.get('chunk_id')}",
        "run_id": run_id,
        "chunk_id": selected.get("chunk_id"),
        "document_id": selected.get("document_id"),
        "company_id": selected.get("company_id"),
        "company_name": selected.get("company_name"),
        "report_year": selected.get("report_year"),
        "report_sections": selected.get("report_sections"),
        "chunk_text": selected.get("chunk_text"),
        "matched_keywords": selected.get("matched_keywords"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mention_types": mention_types,
        "adoption_types": adoption_types,
        "risk_taxonomy": risk_taxonomy,
        "risk_substantiveness": selected.get("risk_substantiveness"),
        "vendor_tags": vendor_tags,
        "vendor_other": selected.get("vendor_other"),
        "adoption_confidence": extract_conf_map(
            selected, ["adoption_confidence", "adoption_confidences", "adoption_signals"]
        ),
        "risk_confidence": normalize_conf_for_field(
            "risk_taxonomy",
            extract_conf_map(selected, ["risk_confidence", "risk_confidences", "risk_signals"]),
        ),
        "review_source": source,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_annotation_id": selected.get("annotation_id"),
    }

    if source == "llm":
        llm_details = selected.get("llm_details") or {}
        record["llm_details"] = llm_details
        if not record["adoption_confidence"]:
            record["adoption_confidence"] = extract_llm_conf_map(selected, "adoption_signals") or extract_llm_conf_map(selected, "adoption_confidences")
        if not record["risk_confidence"]:
            record["risk_confidence"] = normalize_conf_for_field(
                "risk_taxonomy",
                extract_llm_conf_map(selected, "risk_signals")
                or extract_llm_conf_map(selected, "risk_confidences"),
            )
        if llm_threshold > 0:
            record["adoption_confidence"] = filter_llm_conf_map(
                record["adoption_confidence"], llm_threshold
            )
            record["risk_confidence"] = filter_llm_conf_map(
                record["risk_confidence"], llm_threshold
            )

    return record


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()

    human_records = load_jsonl(args.human)
    llm_records = load_jsonl(args.llm)
    human_by_id = index_by_chunk_id(human_records)
    llm_by_id = index_by_chunk_id(llm_records)

    run_id = infer_run_id(human_records) or infer_run_id(llm_records)

    output_dir = args.output_dir
    output_path = output_dir / "annotations.jsonl"
    progress_path = output_dir / "progress.json"

    reconciled_ids = load_reconciled_ids(output_path) if args.resume else set()

    ids = set(human_by_id) | set(llm_by_id) if args.include_missing else set(human_by_id) & set(llm_by_id)
    sorted_ids = sorted(ids)

    reviewed = 0
    skipped = 0

    try:
        review_ids: List[str] = []
        for cid in sorted_ids:
            if args.max_chunks and len(review_ids) >= args.max_chunks:
                break
            if cid in reconciled_ids:
                skipped += 1
                continue
            human = human_by_id.get(cid)
            llm = llm_by_id.get(cid)
            if args.only_disagreements and annotations_match(
                human,
                llm,
                args.llm_confidence_threshold,
                args.include_subtype_disagreements,
                args.focus_field,
            ):
                skipped += 1
                continue
            if not (human or llm):
                skipped += 1
                continue
            review_ids.append(cid)

        if args.only_disagreements and review_ids:
            summary = summarize_disagreements(
                review_ids,
                human_by_id,
                llm_by_id,
                args.llm_confidence_threshold,
                args.focus_field,
            )
            print_diff_summary(summary, args.focus_field)

        total = len(review_ids)
        for idx, cid in enumerate(review_ids, start=1):
            human = human_by_id.get(cid)
            llm = llm_by_id.get(cid)
            chunk = human or llm
            if not chunk:
                skipped += 1
                continue

            display_chunk_header(
                idx,
                total,
                build_diff_label(human, llm, args.llm_confidence_threshold, args.focus_field),
            )
            display_llm_reasoning(llm, args.show_llm_reasoning)
            display_chunk(chunk)
            print_annotations(
                human,
                llm,
                args.llm_confidence_threshold,
                args.show_llm_reasoning,
                args.focus_field,
            )

            if args.focus_field == "all":
                prompt = "Select: 1=human 2=llm 3=custom s=skip q=quit\n> "
            else:
                prompt = (
                    f"Select (focus={args.focus_field}): 1=human 2=llm 3=custom s=skip q=quit\n> "
                )
            choice = input(colorize(prompt, ANSI_HIGHLIGHT)).strip().lower()
            if choice in {"q", "quit"}:
                raise KeyboardInterrupt
            if choice in {"s", "skip", ""}:
                skipped += 1
                continue

            if choice == "1":
                if not human:
                    print("No human annotation available; skipping.")
                    skipped += 1
                    continue
                if args.focus_field == "all":
                    record = build_selected_annotation(
                        human, run_id, "human", args.llm_confidence_threshold
                    )
                else:
                    base = human or llm
                    base_source = "human" if human else "llm"
                    record = build_selected_annotation(
                        base, run_id, base_source, args.llm_confidence_threshold
                    )
                    apply_focus_override(
                        record, human, False, args.focus_field, args.llm_confidence_threshold
                    )
            elif choice == "2":
                if not llm:
                    print("No LLM annotation available; skipping.")
                    skipped += 1
                    continue
                if args.focus_field == "all":
                    record = build_selected_annotation(
                        llm, run_id, "llm", args.llm_confidence_threshold
                    )
                else:
                    base = human or llm
                    base_source = "human" if human else "llm"
                    record = build_selected_annotation(
                        base, run_id, base_source, args.llm_confidence_threshold
                    )
                    apply_focus_override(
                        record, llm, True, args.focus_field, args.llm_confidence_threshold
                    )
                    if base_source != "llm":
                        record["review_source"] = "custom"
                        record["reviewed_at"] = datetime.now(timezone.utc).isoformat()
                        record["source_annotation_id"] = None
            elif choice == "3":
                record = build_custom_annotation(chunk, run_id)
                if record is None:
                    skipped += 1
                    continue
                record["review_source"] = "custom"
                record["reviewed_at"] = datetime.now(timezone.utc).isoformat()
                record["source_annotation_id"] = None
            else:
                print("Invalid choice, skipping.")
                skipped += 1
                continue

            append_jsonl(output_path, record)
            reconciled_ids.add(cid)
            reviewed += 1

            progress = {
                "run_id": run_id,
                "last_chunk_id": cid,
                "reviewed": reviewed,
                "skipped": skipped,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json(progress_path, progress)

    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved.")

    print(
        "Finished. "
        f"Reviewed {reviewed} chunks; "
        f"skipped {skipped} chunks; "
        f"output: {output_path}"
    )


if __name__ == "__main__":
    main()
