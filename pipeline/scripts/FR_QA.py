#!/usr/bin/env python3
"""QA audit for FR markdown vs our iXBRL->markdown on Golden Set companies.

This script compares, per company-year:
1) Document parity and raw text size
2) Chunk count and chunk-length distributions
3) Keyword/section distributions from chunking
4) Coverage against reconciled human AI mentions (precision/recall proxy)
5) Cross-source chunk overlap (our->FR and FR->our)

Outputs:
- per_document_metrics.csv
- keyword_metrics.csv
- mention_type_coverage.csv
- summary.json
- report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from src.markdown_chunker import chunk_markdown  # noqa: E402


NON_WORD_RE = re.compile(r"[^a-z0-9]+")
WORD_RE = re.compile(r"[a-z0-9]{3,}")
TITLE_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


@dataclass
class DocRef:
    lei: str
    company_name: str
    year: int
    cni_sector: str
    document_id: str
    markdown_path: Path
    source_note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FR markdown vs iXBRL markdown quality on Golden Set."
    )
    parser.add_argument(
        "--golden-companies-csv",
        type=Path,
        default=Path("data/reference/golden_set_companies_with_lei.csv"),
        help="CSV with Golden Set company_name/lei/sector.",
    )
    parser.add_argument(
        "--human-annotations",
        type=Path,
        default=Path("data/golden_set/human_reconciled/annotations.jsonl"),
        help="Reconciled human annotations for reference mention coverage.",
    )
    parser.add_argument(
        "--our-run-id",
        type=str,
        default="",
        help="Processed run id under data/processed (auto-inferred from annotations if empty).",
    )
    parser.add_argument(
        "--our-metadata-dir",
        type=Path,
        default=None,
        help="Override for our metadata dir (contains per-document .json files).",
    )
    parser.add_argument(
        "--fr-queue-json",
        type=Path,
        default=Path("data/processing_queue.json"),
        help="Preferred FR mapping source (if exists): lei/year -> markdown file.",
    )
    parser.add_argument(
        "--fr-metadata-csv",
        type=Path,
        default=Path("data/FinancialReports_downloaded/metadata.csv"),
        help="Fallback FR metadata CSV if queue json is absent.",
    )
    parser.add_argument(
        "--fr-markdown-dir",
        type=Path,
        default=Path("data/FinancialReports_downloaded/markdown"),
        help="FR markdown directory.",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2023,2024",
        help="Comma-separated report years to compare.",
    )
    parser.add_argument(
        "--context-before",
        type=int,
        default=2,
        help="chunk_markdown context_before.",
    )
    parser.add_argument(
        "--context-after",
        type=int,
        default=2,
        help="chunk_markdown context_after.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.35,
        help="Similarity threshold for chunk matching.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/fr_qa"),
        help="Output directory for QA artifacts.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_years(raw: str) -> list[int]:
    years: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        years.append(int(part))
    return sorted(set(years))


def normalize_text(text: str) -> str:
    text = text.lower()
    text = NON_WORD_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(text: str) -> set[str]:
    return set(WORD_RE.findall(normalize_text(text)))


def safe_ratio(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(round((len(xs) - 1) * q))
    return float(xs[idx])


def infer_run_id_from_annotations(path: Path) -> str:
    counts: Counter[str] = Counter()
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = str(obj.get("run_id") or "").strip()
            if run_id:
                counts[run_id] += 1
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def load_golden_companies(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Golden companies CSV not found: {path}")
    by_lei: dict[str, dict[str, str]] = {}
    by_name_norm: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lei = str(row.get("lei") or "").strip()
            company_name = str(row.get("company_name") or "").strip()
            sector = str(row.get("sector") or row.get("cni_sector") or "").strip()
            if not lei or not company_name:
                continue
            by_lei[lei] = {
                "company_name": company_name,
                "cni_sector": sector,
            }
            by_name_norm[normalize_text(company_name)] = lei
    return by_lei, by_name_norm


def load_human_reference_chunks(
    path: Path,
    name_to_lei: dict[str, str],
    years: set[int],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Human annotations file not found: {path}")

    latest_by_chunk_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = str(obj.get("chunk_id") or "").strip()
            if chunk_id:
                latest_by_chunk_id[chunk_id] = obj

    refs: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for obj in latest_by_chunk_id.values():
        company_name = str(obj.get("company_name") or "").strip()
        year_raw = obj.get("report_year")
        if company_name == "" or year_raw is None:
            continue
        try:
            year = int(year_raw)
        except (TypeError, ValueError):
            continue
        if year not in years:
            continue

        lei = name_to_lei.get(normalize_text(company_name))
        if not lei:
            continue

        mention_types_raw = obj.get("mention_types") or []
        if isinstance(mention_types_raw, str):
            mention_types = [mention_types_raw]
        else:
            mention_types = [str(x) for x in mention_types_raw if x is not None]
        mention_types = [m.strip() for m in mention_types if m and m.strip() and m.strip() != "none"]
        if not mention_types:
            continue

        refs[(lei, year)].append(
            {
                "chunk_id": str(obj.get("chunk_id") or ""),
                "chunk_text": str(obj.get("chunk_text") or ""),
                "mention_types": sorted(set(mention_types)),
            }
        )
    return refs


def load_our_documents(
    metadata_dir: Path,
    golden_by_lei: dict[str, dict[str, str]],
    years: set[int],
) -> dict[tuple[str, int], DocRef]:
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Our metadata dir not found: {metadata_dir}")

    out: dict[tuple[str, int], DocRef] = {}
    for meta_path in sorted(metadata_dir.glob("*.json")):
        try:
            obj = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        lei = str(obj.get("lei") or "").strip()
        if lei not in golden_by_lei:
            continue
        year_raw = obj.get("year") if obj.get("year") is not None else obj.get("report_year")
        try:
            year = int(year_raw)
        except (TypeError, ValueError):
            continue
        if year not in years:
            continue
        md_path_raw = str(obj.get("markdown_path") or "").strip()
        if md_path_raw:
            md_path = Path(md_path_raw)
            if not md_path.is_absolute():
                md_path = resolve_repo_path(md_path)
        else:
            doc_id = str(obj.get("document_id") or meta_path.stem)
            md_path = metadata_dir.parent / "documents" / f"{doc_id}.md"
        key = (lei, year)
        out[key] = DocRef(
            lei=lei,
            company_name=str(obj.get("company_name") or golden_by_lei[lei]["company_name"]),
            year=year,
            cni_sector=str(obj.get("cni_sector") or golden_by_lei[lei]["cni_sector"]),
            document_id=str(obj.get("document_id") or meta_path.stem),
            markdown_path=md_path,
            source_note="our_ixbrl_markdown",
        )
    return out


def _infer_fr_report_year(row: dict[str, str], target_years: set[int]) -> int | None:
    title = str(row.get("title") or "")
    years_in_title = [int(y) for y in TITLE_YEAR_RE.findall(title)]
    for y in years_in_title:
        if y in target_years:
            return y
    release = str(row.get("release_datetime") or "").strip()
    if release and len(release) >= 4:
        try:
            y = int(release[:4]) - 1
            if y in target_years:
                return y
        except ValueError:
            return None
    return None


def load_fr_documents_from_queue(
    queue_path: Path,
    golden_by_lei: dict[str, dict[str, str]],
    years: set[int],
) -> dict[tuple[str, int], DocRef]:
    if not queue_path.exists():
        return {}
    data = json.loads(queue_path.read_text(encoding="utf-8"))
    out: dict[tuple[str, int], DocRef] = {}
    for item in data:
        lei = str(item.get("lei") or "").strip()
        if lei not in golden_by_lei:
            continue
        try:
            year = int(item.get("year"))
        except (TypeError, ValueError):
            continue
        if year not in years:
            continue
        file_path = Path(str(item.get("file_path") or ""))
        if not file_path:
            continue
        if not file_path.is_absolute():
            file_path = resolve_repo_path(file_path)
        key = (lei, year)
        out[key] = DocRef(
            lei=lei,
            company_name=str(item.get("company_name") or golden_by_lei[lei]["company_name"]),
            year=year,
            cni_sector=golden_by_lei[lei]["cni_sector"],
            document_id=f"fr-{item.get('pk')}",
            markdown_path=file_path,
            source_note=f"fr_queue_pk={item.get('pk')}",
        )
    return out


def load_fr_documents_from_metadata(
    metadata_csv: Path,
    fr_markdown_dir: Path,
    golden_by_lei: dict[str, dict[str, str]],
    years: set[int],
) -> dict[tuple[str, int], DocRef]:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"FR metadata CSV not found: {metadata_csv}")

    best_rows: dict[tuple[str, int], dict[str, str]] = {}
    best_scores: dict[tuple[str, int], tuple[int, str]] = {}

    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lei = str(row.get("company__lei") or "").strip()
            if lei not in golden_by_lei:
                continue
            report_year = _infer_fr_report_year(row, years)
            if report_year is None:
                continue
            filing_type = str(row.get("filing_type__name") or "").lower()
            priority = 0
            if "esef" in filing_type:
                priority += 100
            if "annual report" in filing_type:
                priority += 10
            release = str(row.get("release_datetime") or "")
            key = (lei, report_year)
            score = (priority, release)
            if key not in best_scores or score > best_scores[key]:
                best_scores[key] = score
                best_rows[key] = row

    out: dict[tuple[str, int], DocRef] = {}
    for (lei, year), row in best_rows.items():
        pk = str(row.get("pk") or "").strip()
        md_path = fr_markdown_dir / f"{pk}.md"
        out[(lei, year)] = DocRef(
            lei=lei,
            company_name=golden_by_lei[lei]["company_name"],
            year=year,
            cni_sector=golden_by_lei[lei]["cni_sector"],
            document_id=f"fr-{pk}",
            markdown_path=md_path,
            source_note=f"fr_metadata_pk={pk}",
        )
    return out


def similarity_score(text_a: str, text_b: str) -> float:
    a = normalize_text(text_a)
    b = normalize_text(text_b)
    if not a or not b:
        return 0.0
    a_tokens = token_set(a)
    b_tokens = token_set(b)
    if a_tokens and b_tokens:
        dice = 2.0 * len(a_tokens & b_tokens) / (len(a_tokens) + len(b_tokens))
    else:
        dice = 0.0
    seq = SequenceMatcher(None, a[:6000], b[:6000]).ratio()
    return 0.65 * dice + 0.35 * seq


def best_match_indices(
    source_chunks: list[dict[str, Any]],
    target_chunks: list[dict[str, Any]],
    threshold: float,
) -> tuple[dict[int, tuple[int, float]], dict[int, list[tuple[int, float]]]]:
    assignments: dict[int, tuple[int, float]] = {}
    reverse: dict[int, list[tuple[int, float]]] = defaultdict(list)
    if not source_chunks or not target_chunks:
        return assignments, reverse
    for i, s in enumerate(source_chunks):
        s_text = str(s.get("chunk_text") or "")
        best_j = -1
        best_score = 0.0
        for j, t in enumerate(target_chunks):
            score = similarity_score(s_text, str(t.get("chunk_text") or ""))
            if score > best_score:
                best_score = score
                best_j = j
        if best_j >= 0 and best_score >= threshold:
            assignments[i] = (best_j, best_score)
            reverse[best_j].append((i, best_score))
    return assignments, reverse


def chunk_stats(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    char_lengths = [len(str(c.get("chunk_text") or "")) for c in chunks]
    word_lengths = [len(str(c.get("chunk_text") or "").split()) for c in chunks]
    unique_norm_chunks = len({normalize_text(str(c.get("chunk_text") or "")) for c in chunks})
    duplicates = max(0, len(chunks) - unique_norm_chunks)

    keyword_presence = Counter()
    keyword_hits = Counter()
    sections = Counter()

    for chunk in chunks:
        for kw in chunk.get("matched_keywords") or []:
            keyword_presence[str(kw)] += 1
        for group in chunk.get("keyword_matches") or []:
            for m in group.get("matches") or []:
                kw = str(m.get("keyword") or "")
                if kw:
                    keyword_hits[kw] += 1
        for sec in chunk.get("report_sections") or []:
            sec = str(sec).strip()
            if sec:
                sections[sec] += 1

    total = len(chunks)
    return {
        "chunk_count": total,
        "char_mean": mean(char_lengths) if char_lengths else 0.0,
        "char_p10": pctl(char_lengths, 0.10),
        "char_p50": pctl(char_lengths, 0.50),
        "char_p90": pctl(char_lengths, 0.90),
        "char_max": max(char_lengths) if char_lengths else 0,
        "word_mean": mean(word_lengths) if word_lengths else 0.0,
        "word_p50": pctl(word_lengths, 0.50),
        "short_rate": safe_ratio(sum(1 for x in char_lengths if x < 300), total),
        "long_rate": safe_ratio(sum(1 for x in char_lengths if x > 3000), total),
        "duplicate_rate": safe_ratio(duplicates, total),
        "keyword_presence": keyword_presence,
        "keyword_hits": keyword_hits,
        "keyword_hits_total": int(sum(keyword_hits.values())),
        "section_counter": sections,
        "section_set": set(sections.keys()),
    }


def mention_type_coverage(
    source_chunks: list[dict[str, Any]],
    ref_chunks: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    assignments, reverse = best_match_indices(source_chunks, ref_chunks, threshold)

    type_totals: Counter[str] = Counter()
    type_hits: Counter[str] = Counter()
    for j, ref in enumerate(ref_chunks):
        mention_types = [str(x) for x in ref.get("mention_types") or [] if str(x).strip()]
        for mt in mention_types:
            type_totals[mt] += 1
            if j in reverse:
                type_hits[mt] += 1

    matched_scores = [score for _, score in assignments.values()]
    return {
        "source_matched_chunks": len(assignments),
        "ref_covered_chunks": len(reverse),
        "ref_recall": safe_ratio(len(reverse), len(ref_chunks)),
        "precision_proxy": safe_ratio(len(assignments), len(source_chunks)),
        "avg_match_score": mean(matched_scores) if matched_scores else 0.0,
        "type_totals": type_totals,
        "type_hits": type_hits,
    }


def cross_source_overlap(
    chunks_a: list[dict[str, Any]],
    chunks_b: list[dict[str, Any]],
    threshold: float,
) -> dict[str, float]:
    ab, _ = best_match_indices(chunks_a, chunks_b, threshold)
    ba, _ = best_match_indices(chunks_b, chunks_a, threshold)
    ab_scores = [score for _, score in ab.values()]
    ba_scores = [score for _, score in ba.values()]
    return {
        "a_to_b_overlap": safe_ratio(len(ab), len(chunks_a)),
        "b_to_a_overlap": safe_ratio(len(ba), len(chunks_b)),
        "a_to_b_avg_score": mean(ab_scores) if ab_scores else 0.0,
        "b_to_a_avg_score": mean(ba_scores) if ba_scores else 0.0,
    }


def section_jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return safe_ratio(len(a & b), len(a | b))


def load_markdown(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    years = set(parse_years(args.years))

    golden_csv = resolve_repo_path(args.golden_companies_csv)
    annotations_path = resolve_repo_path(args.human_annotations)
    fr_queue_path = resolve_repo_path(args.fr_queue_json)
    fr_metadata_csv = resolve_repo_path(args.fr_metadata_csv)
    fr_markdown_dir = resolve_repo_path(args.fr_markdown_dir)
    output_root = resolve_repo_path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / f"fr-qa-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    golden_by_lei, golden_name_to_lei = load_golden_companies(golden_csv)
    human_refs = load_human_reference_chunks(annotations_path, golden_name_to_lei, years)

    run_id = args.our_run_id.strip() or infer_run_id_from_annotations(annotations_path)
    if args.our_metadata_dir is not None:
        our_metadata_dir = resolve_repo_path(args.our_metadata_dir)
    else:
        if not run_id:
            raise RuntimeError("Could not infer --our-run-id from annotations; please pass it explicitly.")
        our_metadata_dir = REPO_ROOT / "data" / "processed" / run_id / "metadata"

    our_docs = load_our_documents(our_metadata_dir, golden_by_lei, years)
    fr_docs = load_fr_documents_from_queue(fr_queue_path, golden_by_lei, years)
    if not fr_docs:
        fr_docs = load_fr_documents_from_metadata(
            fr_metadata_csv, fr_markdown_dir, golden_by_lei, years
        )

    all_keys = sorted(
        {(lei, y) for lei in golden_by_lei for y in years},
        key=lambda x: (golden_by_lei[x[0]]["company_name"].lower(), x[1]),
    )

    per_doc_rows: list[dict[str, Any]] = []
    keyword_rows: list[dict[str, Any]] = []
    mention_rows: list[dict[str, Any]] = []

    for lei, year in all_keys:
        company_name = golden_by_lei[lei]["company_name"]
        cni_sector = golden_by_lei[lei]["cni_sector"]
        our_ref = our_docs.get((lei, year))
        fr_ref = fr_docs.get((lei, year))

        our_text = load_markdown(our_ref.markdown_path) if our_ref else ""
        fr_text = load_markdown(fr_ref.markdown_path) if fr_ref else ""

        our_chunks = (
            chunk_markdown(
                our_text,
                document_id=our_ref.document_id,
                company_id=lei,
                company_name=company_name,
                report_year=year,
                context_before=args.context_before,
                context_after=args.context_after,
            )
            if our_text
            else []
        )
        fr_chunks = (
            chunk_markdown(
                fr_text,
                document_id=fr_ref.document_id,
                company_id=lei,
                company_name=company_name,
                report_year=year,
                context_before=args.context_before,
                context_after=args.context_after,
            )
            if fr_text
            else []
        )

        our_stats = chunk_stats(our_chunks)
        fr_stats = chunk_stats(fr_chunks)
        refs = human_refs.get((lei, year), [])
        our_cov = mention_type_coverage(our_chunks, refs, args.match_threshold)
        fr_cov = mention_type_coverage(fr_chunks, refs, args.match_threshold)
        overlap = cross_source_overlap(our_chunks, fr_chunks, args.match_threshold)

        row = {
            "company_name": company_name,
            "lei": lei,
            "year": year,
            "cni_sector": cni_sector,
            "our_doc_found": int(bool(our_text)),
            "fr_doc_found": int(bool(fr_text)),
            "our_doc_path": str(our_ref.markdown_path) if our_ref else "",
            "fr_doc_path": str(fr_ref.markdown_path) if fr_ref else "",
            "our_text_chars": len(our_text),
            "fr_text_chars": len(fr_text),
            "fr_vs_our_text_ratio": safe_ratio(len(fr_text), len(our_text)),
            "our_chunk_count": our_stats["chunk_count"],
            "fr_chunk_count": fr_stats["chunk_count"],
            "fr_minus_our_chunks": fr_stats["chunk_count"] - our_stats["chunk_count"],
            "fr_vs_our_chunk_ratio": safe_ratio(fr_stats["chunk_count"], our_stats["chunk_count"]),
            "our_chunk_char_p50": round(our_stats["char_p50"], 2),
            "our_chunk_char_p90": round(our_stats["char_p90"], 2),
            "fr_chunk_char_p50": round(fr_stats["char_p50"], 2),
            "fr_chunk_char_p90": round(fr_stats["char_p90"], 2),
            "our_short_chunk_rate": round(our_stats["short_rate"], 4),
            "fr_short_chunk_rate": round(fr_stats["short_rate"], 4),
            "our_long_chunk_rate": round(our_stats["long_rate"], 4),
            "fr_long_chunk_rate": round(fr_stats["long_rate"], 4),
            "our_duplicate_chunk_rate": round(our_stats["duplicate_rate"], 4),
            "fr_duplicate_chunk_rate": round(fr_stats["duplicate_rate"], 4),
            "our_keyword_hits_total": our_stats["keyword_hits_total"],
            "fr_keyword_hits_total": fr_stats["keyword_hits_total"],
            "our_keyword_types": len(our_stats["keyword_hits"]),
            "fr_keyword_types": len(fr_stats["keyword_hits"]),
            "our_section_count": len(our_stats["section_set"]),
            "fr_section_count": len(fr_stats["section_set"]),
            "section_jaccard": round(section_jaccard(our_stats["section_set"], fr_stats["section_set"]), 4),
            "ref_positive_chunks": len(refs),
            "our_ref_recall": round(our_cov["ref_recall"], 4),
            "fr_ref_recall": round(fr_cov["ref_recall"], 4),
            "our_precision_proxy": round(our_cov["precision_proxy"], 4),
            "fr_precision_proxy": round(fr_cov["precision_proxy"], 4),
            "our_ref_avg_match_score": round(our_cov["avg_match_score"], 4),
            "fr_ref_avg_match_score": round(fr_cov["avg_match_score"], 4),
            "our_to_fr_overlap": round(overlap["a_to_b_overlap"], 4),
            "fr_to_our_overlap": round(overlap["b_to_a_overlap"], 4),
            "our_to_fr_avg_score": round(overlap["a_to_b_avg_score"], 4),
            "fr_to_our_avg_score": round(overlap["b_to_a_avg_score"], 4),
        }
        per_doc_rows.append(row)

        for source_name, stats in (("our", our_stats), ("fr", fr_stats)):
            keys = sorted(set(stats["keyword_presence"].keys()) | set(stats["keyword_hits"].keys()))
            if not keys:
                keyword_rows.append(
                    {
                        "company_name": company_name,
                        "lei": lei,
                        "year": year,
                        "source": source_name,
                        "keyword": "",
                        "chunks_with_keyword": 0,
                        "keyword_hit_count": 0,
                    }
                )
            for kw in keys:
                keyword_rows.append(
                    {
                        "company_name": company_name,
                        "lei": lei,
                        "year": year,
                        "source": source_name,
                        "keyword": kw,
                        "chunks_with_keyword": stats["keyword_presence"].get(kw, 0),
                        "keyword_hit_count": stats["keyword_hits"].get(kw, 0),
                    }
                )

        all_types = sorted(set(our_cov["type_totals"].keys()) | set(fr_cov["type_totals"].keys()))
        for mt in all_types:
            total = max(our_cov["type_totals"].get(mt, 0), fr_cov["type_totals"].get(mt, 0))
            mention_rows.append(
                {
                    "company_name": company_name,
                    "lei": lei,
                    "year": year,
                    "mention_type": mt,
                    "ref_total": total,
                    "our_covered": our_cov["type_hits"].get(mt, 0),
                    "fr_covered": fr_cov["type_hits"].get(mt, 0),
                    "our_recall": round(safe_ratio(our_cov["type_hits"].get(mt, 0), total), 4),
                    "fr_recall": round(safe_ratio(fr_cov["type_hits"].get(mt, 0), total), 4),
                }
            )

    write_csv(
        out_dir / "per_document_metrics.csv",
        per_doc_rows,
        [
            "company_name",
            "lei",
            "year",
            "cni_sector",
            "our_doc_found",
            "fr_doc_found",
            "our_doc_path",
            "fr_doc_path",
            "our_text_chars",
            "fr_text_chars",
            "fr_vs_our_text_ratio",
            "our_chunk_count",
            "fr_chunk_count",
            "fr_minus_our_chunks",
            "fr_vs_our_chunk_ratio",
            "our_chunk_char_p50",
            "our_chunk_char_p90",
            "fr_chunk_char_p50",
            "fr_chunk_char_p90",
            "our_short_chunk_rate",
            "fr_short_chunk_rate",
            "our_long_chunk_rate",
            "fr_long_chunk_rate",
            "our_duplicate_chunk_rate",
            "fr_duplicate_chunk_rate",
            "our_keyword_hits_total",
            "fr_keyword_hits_total",
            "our_keyword_types",
            "fr_keyword_types",
            "our_section_count",
            "fr_section_count",
            "section_jaccard",
            "ref_positive_chunks",
            "our_ref_recall",
            "fr_ref_recall",
            "our_precision_proxy",
            "fr_precision_proxy",
            "our_ref_avg_match_score",
            "fr_ref_avg_match_score",
            "our_to_fr_overlap",
            "fr_to_our_overlap",
            "our_to_fr_avg_score",
            "fr_to_our_avg_score",
        ],
    )

    write_csv(
        out_dir / "keyword_metrics.csv",
        keyword_rows,
        [
            "company_name",
            "lei",
            "year",
            "source",
            "keyword",
            "chunks_with_keyword",
            "keyword_hit_count",
        ],
    )

    write_csv(
        out_dir / "mention_type_coverage.csv",
        mention_rows,
        [
            "company_name",
            "lei",
            "year",
            "mention_type",
            "ref_total",
            "our_covered",
            "fr_covered",
            "our_recall",
            "fr_recall",
        ],
    )

    docs_with_both = [r for r in per_doc_rows if r["our_doc_found"] and r["fr_doc_found"]]
    summary = {
        "run_id_inferred": run_id,
        "years": sorted(years),
        "golden_companies": len(golden_by_lei),
        "expected_company_year_pairs": len(golden_by_lei) * len(years),
        "pairs_with_our_doc": sum(r["our_doc_found"] for r in per_doc_rows),
        "pairs_with_fr_doc": sum(r["fr_doc_found"] for r in per_doc_rows),
        "pairs_with_both_docs": len(docs_with_both),
        "our_total_chunks": sum(r["our_chunk_count"] for r in docs_with_both),
        "fr_total_chunks": sum(r["fr_chunk_count"] for r in docs_with_both),
        "avg_our_ref_recall": round(mean([r["our_ref_recall"] for r in docs_with_both]), 4) if docs_with_both else 0.0,
        "avg_fr_ref_recall": round(mean([r["fr_ref_recall"] for r in docs_with_both]), 4) if docs_with_both else 0.0,
        "avg_our_precision_proxy": round(mean([r["our_precision_proxy"] for r in docs_with_both]), 4) if docs_with_both else 0.0,
        "avg_fr_precision_proxy": round(mean([r["fr_precision_proxy"] for r in docs_with_both]), 4) if docs_with_both else 0.0,
        "avg_our_to_fr_overlap": round(mean([r["our_to_fr_overlap"] for r in docs_with_both]), 4) if docs_with_both else 0.0,
        "avg_fr_to_our_overlap": round(mean([r["fr_to_our_overlap"] for r in docs_with_both]), 4) if docs_with_both else 0.0,
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    biggest_chunk_deltas = sorted(
        docs_with_both,
        key=lambda r: abs(r["fr_minus_our_chunks"]),
        reverse=True,
    )[:10]
    report_lines = [
        "# FR vs iXBRL Markdown QA (Golden Set)",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Years: {', '.join(str(y) for y in sorted(years))}",
        f"- Golden companies: {len(golden_by_lei)}",
        f"- Company-year pairs expected: {len(golden_by_lei) * len(years)}",
        f"- Pairs with both docs: {len(docs_with_both)}",
        "",
        "## Aggregate",
        "",
        f"- Total chunks (our): {summary['our_total_chunks']}",
        f"- Total chunks (fr): {summary['fr_total_chunks']}",
        f"- Avg reference recall (our): {summary['avg_our_ref_recall']}",
        f"- Avg reference recall (fr): {summary['avg_fr_ref_recall']}",
        f"- Avg precision proxy (our): {summary['avg_our_precision_proxy']}",
        f"- Avg precision proxy (fr): {summary['avg_fr_precision_proxy']}",
        f"- Avg overlap our->fr: {summary['avg_our_to_fr_overlap']}",
        f"- Avg overlap fr->our: {summary['avg_fr_to_our_overlap']}",
        "",
        "## Largest Chunk Count Deltas (FR - Our)",
        "",
        "| Company | Year | Our Chunks | FR Chunks | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in biggest_chunk_deltas:
        report_lines.append(
            f"| {r['company_name']} | {r['year']} | {r['our_chunk_count']} | {r['fr_chunk_count']} | {r['fr_minus_our_chunks']} |"
        )
    report_lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `per_document_metrics.csv`",
            "- `keyword_metrics.csv`",
            "- `mention_type_coverage.csv`",
            "- `summary.json`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote QA outputs to: {out_dir}")
    print(
        "Pairs with both docs: "
        f"{summary['pairs_with_both_docs']}/{summary['expected_company_year_pairs']}"
    )
    print(
        "Total chunks -> "
        f"our: {summary['our_total_chunks']} | fr: {summary['fr_total_chunks']}"
    )
    print(
        "Avg reference recall -> "
        f"our: {summary['avg_our_ref_recall']} | fr: {summary['avg_fr_ref_recall']}"
    )


if __name__ == "__main__":
    main()

