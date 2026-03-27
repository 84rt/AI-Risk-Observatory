"""Chunk markdown reports by AI keyword mentions."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .utils.keywords import AI_KEYWORD_PATTERNS, KeywordPattern, compile_keyword_patterns

_TABLE_ROW_RE = re.compile(r"\|.*\|")
_TABLE_RULE_ROW_RE = re.compile(r"^\|?\s*:?-{2,}.*\|.*$")
_KEEP_ROW_RE = re.compile(
    r"\b(ai|artificial intelligence|machine learning|\bml\b|gen(?:erative)?\s*ai|llm|algorithm|"
    r"risk|principal risk|mitigation|control|governance|opportunit(?:y|ies)|threat|compliance|"
    r"regulat(?:ion|ory))\b",
    re.I,
)
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?%")
_YEAR_RE = re.compile(r"\b20\d{2}\b")
_POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.I)
_ENTITY_TERMS = [
    "limited",
    "ltd",
    "llp",
    "plc",
    "holdco",
    "holdings",
    "nominee",
    "ordinary",
    "spv",
    "fund",
    "oeic",
    "sicav",
    "company",
    "group",
    "partners",
    "partnership",
    "trust",
]
_ADDRESS_TERMS = [
    "road",
    "street",
    "st ",
    "avenue",
    "drive",
    "house",
    "court",
    "park",
    "building",
    "floor",
    "unit",
    "united kingdom",
    "london",
    "glasgow",
    "edinburgh",
    "reading",
    "cardiff",
    "plymouth",
    "warrington",
    "york",
]


@dataclass
class MarkdownBlock:
    """A markdown block with offsets and section context."""

    section: Optional[str]
    text: str
    index: int
    start_offset: int
    end_offset: int
    is_heading: bool


@dataclass
class MarkdownParagraph:
    """A paragraph block with a mapping to the original block index."""

    section: Optional[str]
    text: str
    index: int
    block_index: int


@dataclass(frozen=True)
class CharWindow:
    """A character-span window centered on one or more mention matches."""

    start: int
    end: int
    matches: Tuple[dict, ...]


def _parse_markdown_blocks(markdown: str) -> List[MarkdownBlock]:
    blocks: List[MarkdownBlock] = []
    current_section = None
    buffer: List[str] = []
    block_start: Optional[int] = None
    last_nonblank_pos: Optional[int] = None
    index = 0

    def flush():
        nonlocal index, block_start, last_nonblank_pos
        if buffer and block_start is not None and last_nonblank_pos is not None:
            raw_text = "\n".join(buffer).strip()
            if raw_text:
                blocks.append(
                    MarkdownBlock(
                        section=current_section,
                        text=raw_text,
                        index=index,
                        start_offset=block_start,
                        end_offset=last_nonblank_pos,
                        is_heading=False,
                    )
                )
                index += 1
        buffer.clear()
        block_start = None
        last_nonblank_pos = None

    pos = 0
    for line in markdown.splitlines(keepends=True):
        stripped = line.strip()
        line_len = len(line)
        line_no_nl = line.rstrip("\n")
        line_no_nl_len = len(line_no_nl)

        if stripped.startswith("#"):
            flush()
            heading_text = line_no_nl.strip()
            if heading_text:
                blocks.append(
                    MarkdownBlock(
                        section=current_section,
                        text=heading_text,
                        index=index,
                        start_offset=pos,
                        end_offset=pos + line_no_nl_len,
                        is_heading=True,
                    )
                )
                index += 1
            current_section = stripped.lstrip("#").strip()
            pos += line_len
            continue

        if not stripped:
            flush()
            pos += line_len
            continue

        if block_start is None:
            block_start = pos
        buffer.append(line_no_nl)
        last_nonblank_pos = pos + line_no_nl_len
        pos += line_len

    flush()
    return blocks


def _extract_paragraphs(blocks: List[MarkdownBlock]) -> List[MarkdownParagraph]:
    paragraphs: List[MarkdownParagraph] = []
    paragraph_index = 0
    for block in blocks:
        if block.is_heading:
            continue
        paragraphs.append(
            MarkdownParagraph(
                section=block.section,
                text=block.text,
                index=paragraph_index,
                block_index=block.index,
            )
        )
        paragraph_index += 1
    return paragraphs


def _compile_patterns(patterns: Optional[List[KeywordPattern]] = None):
    patterns = patterns or AI_KEYWORD_PATTERNS
    return compile_keyword_patterns(patterns)


def _find_matches(text: str, patterns: List[tuple]) -> List[dict]:
    matches: List[dict] = []
    for name, regex in patterns:
        for match in regex.finditer(text):
            matches.append(
                {
                    "keyword": name,
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    return matches


def _merge_windows(windows: List[Tuple[int, int, int]]) -> List[Tuple[int, int, List[int]]]:
    if not windows:
        return []
    windows.sort(key=lambda w: (w[0], w[1]))
    merged: List[Tuple[int, int, List[int]]] = []
    current_start, current_end, mention_indices = windows[0][0], windows[0][1], [windows[0][2]]
    for start, end, mention_index in windows[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
            mention_indices.append(mention_index)
        else:
            merged.append((current_start, current_end, mention_indices))
            current_start, current_end, mention_indices = start, end, [mention_index]
    merged.append((current_start, current_end, mention_indices))
    return merged


def _dedupe_paragraph_windows(
    windows: List[Tuple[int, int, int]],
) -> List[Tuple[int, int, List[int]]]:
    deduped: Dict[Tuple[int, int], List[int]] = {}
    for start_idx, end_idx, mention_index in sorted(windows, key=lambda w: (w[0], w[1], w[2])):
        deduped.setdefault((start_idx, end_idx), []).append(mention_index)
    return [
        (start_idx, end_idx, mention_indices)
        for (start_idx, end_idx), mention_indices in deduped.items()
    ]


def _stable_chunk_id(document_id: str, start_idx: int, end_idx: int) -> str:
    payload = f"{document_id}:{start_idx}:{end_idx}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"{document_id}-chunk-{digest[:12]}"


def _stable_subchunk_id(
    document_id: str,
    start_idx: int,
    end_idx: int,
    subchunk_index: int,
) -> str:
    payload = f"{document_id}:{start_idx}:{end_idx}:{subchunk_index}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"{document_id}-chunk-{digest[:12]}"


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _is_malformed_for_paragraph_chunking(
    markdown: str,
    paragraphs: List[MarkdownParagraph],
) -> bool:
    if markdown.count("\n") <= 1:
        return True
    if len(paragraphs) <= 1:
        return True
    longest_paragraph_chars = max((len(p.text) for p in paragraphs), default=0)
    return longest_paragraph_chars >= 20000


def _term_hits(text: str, terms: List[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


def _is_listing_signature_row(line: str) -> bool:
    """Detect low-information entity/register rows that are safe to drop by default."""
    if not _TABLE_ROW_RE.search(line):
        return False
    if _TABLE_RULE_ROW_RE.match(line):
        return False
    if _KEEP_ROW_RE.search(line):
        return False

    pipe_count = line.count("|")
    entity_hits = _term_hits(line, _ENTITY_TERMS)
    address_hits = _term_hits(line, _ADDRESS_TERMS)
    has_postcode = bool(_POSTCODE_RE.search(line))
    has_percent = bool(_PERCENT_RE.search(line))
    year_count = len(_YEAR_RE.findall(line))

    metricish = (year_count >= 2 or has_percent) and entity_hits < 2 and address_hits == 0 and not has_postcode
    if metricish:
        return False

    score = 0
    if pipe_count >= 14:
        score += 2
    elif pipe_count >= 10:
        score += 1

    char_count = len(line)
    if char_count >= 900:
        score += 3
    elif char_count >= 500:
        score += 2
    elif char_count >= 300:
        score += 1

    if entity_hits >= 4:
        score += 3
    elif entity_hits >= 2:
        score += 2
    elif entity_hits >= 1:
        score += 1

    if address_hits >= 2:
        score += 2
    elif address_hits >= 1:
        score += 1
    if has_postcode:
        score += 2

    comma_count = line.count(",")
    if comma_count >= 12:
        score += 2
    elif comma_count >= 8:
        score += 1

    return score >= 6


def _clean_chunk_lines(
    text: str,
    drop_table_rule_lines: bool,
    drop_listing_rows: bool,
    patterns: List[tuple],
) -> tuple[str, dict]:
    removed_table_rule_lines = 0
    removed_listing_rows = 0
    kept_lines: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Never drop a line that directly carries an AI mention.
        if any(regex.search(stripped) for _, regex in patterns):
            kept_lines.append(line)
            continue
        if drop_table_rule_lines and _TABLE_RULE_ROW_RE.match(stripped):
            removed_table_rule_lines += 1
            continue
        if drop_listing_rows and _is_listing_signature_row(stripped):
            removed_listing_rows += 1
            continue
        kept_lines.append(line)

    cleaned = "\n".join(kept_lines).strip()
    return cleaned, {
        "removed_table_rule_lines": removed_table_rule_lines,
        "removed_listing_rows": removed_listing_rows,
    }


def _find_sentence_start(text: str, target: int) -> int:
    if target <= 0:
        return 0

    blank_break = text.rfind("\n\n", 0, target)
    punctuation_breaks = [
        text.rfind(marker, 0, target)
        for marker in (". ", "! ", "? ", ".\n", "!\n", "?\n")
    ]
    boundary = max([blank_break, *punctuation_breaks])
    if boundary < 0:
        return 0
    if text.startswith("\n\n", boundary):
        return boundary + 2
    return min(boundary + 2, len(text))


def _find_sentence_end(text: str, target: int) -> int:
    if target >= len(text):
        return len(text)

    candidates = []
    blank_break = text.find("\n\n", target)
    if blank_break >= 0:
        candidates.append(blank_break)
    for marker in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        idx = text.find(marker, target)
        if idx >= 0:
            candidates.append(idx + 1)
    if not candidates:
        return len(text)
    return min(candidates) + 1


def _build_char_windows_from_matches(
    text: str,
    matches: List[dict],
    char_radius: int,
) -> List[CharWindow]:
    if not matches:
        return []

    seeded: List[Tuple[int, int, dict]] = []
    for match in sorted(matches, key=lambda m: (m["start"], m["end"])):
        start = max(0, match["start"] - char_radius)
        end = min(len(text), match["end"] + char_radius)
        start = _find_sentence_start(text, start)
        end = _find_sentence_end(text, end)
        if end <= start:
            continue
        seeded.append((start, end, match))

    if not seeded:
        return []

    merged: List[CharWindow] = []
    current_start, current_end, current_matches = seeded[0][0], seeded[0][1], [seeded[0][2]]
    for start, end, match in seeded[1:]:
        if start < current_end:
            current_end = max(current_end, end)
            current_matches.append(match)
        else:
            merged.append(
                CharWindow(
                    start=current_start,
                    end=current_end,
                    matches=tuple(current_matches),
                )
            )
            current_start, current_end, current_matches = start, end, [match]
    merged.append(
        CharWindow(
            start=current_start,
            end=current_end,
            matches=tuple(current_matches),
        )
    )
    return merged


def _split_text_by_word_limit(
    text: str,
    max_words: int,
    overlap_sentences: int = 1,
) -> List[str]:
    """Split long text by natural boundaries while preserving content."""
    if max_words <= 0 or _word_count(text) <= max_words:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    def split_long_sentence(sentence: str) -> List[str]:
        words = sentence.split()
        if len(words) <= max_words:
            return [sentence.strip()]
        out: List[str] = []
        cursor = 0
        overlap_words = min(40, max_words // 10)
        while cursor < len(words):
            end = min(cursor + max_words, len(words))
            out.append(" ".join(words[cursor:end]).strip())
            if end == len(words):
                break
            cursor = max(end - overlap_words, cursor + 1)
        return [x for x in out if x]

    def split_long_paragraph(paragraph: str) -> List[str]:
        if _word_count(paragraph) <= max_words:
            return [paragraph.strip()]

        sentence_candidates = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", paragraph)
            if s.strip()
        ]
        if not sentence_candidates:
            return split_long_sentence(paragraph)

        segments: List[str] = []
        current: List[str] = []
        current_words = 0

        for sent in sentence_candidates:
            sent_words = _word_count(sent)
            sent_parts = split_long_sentence(sent) if sent_words > max_words else [sent]

            for part in sent_parts:
                part_words = _word_count(part)
                if current and current_words + part_words > max_words:
                    segments.append(" ".join(current).strip())
                    if overlap_sentences > 0 and current:
                        overlap = current[-overlap_sentences:]
                        current = overlap.copy()
                        current_words = _word_count(" ".join(current))
                    else:
                        current = []
                        current_words = 0
                    if current_words + part_words > max_words:
                        current = []
                        current_words = 0
                current.append(part)
                current_words += part_words

        if current:
            segments.append(" ".join(current).strip())
        return [s for s in segments if s]

    units: List[str] = []
    for para in paragraphs:
        units.extend(split_long_paragraph(para))

    segments: List[str] = []
    current_parts: List[str] = []
    current_words = 0
    for unit in units:
        unit_words = _word_count(unit)
        if current_parts and current_words + unit_words > max_words:
            segments.append("\n\n".join(current_parts).strip())
            current_parts = [unit]
            current_words = unit_words
        else:
            current_parts.append(unit)
            current_words += unit_words

    if current_parts:
        segments.append("\n\n".join(current_parts).strip())
    return [s for s in segments if s]


def _emit_chunk_variants(
    *,
    source_text: str,
    document_id: str,
    company_id: str,
    company_name: str,
    report_year: int,
    chunk_text: str,
    start_offset: int,
    end_offset: int,
    parent_identity: str,
    paragraph_start: Optional[int],
    paragraph_end: Optional[int],
    block_start: Optional[int],
    block_end: Optional[int],
    context_before: int,
    context_after: int,
    report_sections: List[str],
    max_chunk_words: Optional[int],
    overlap_sentences: int,
    patterns: List[tuple],
    drop_table_rule_lines: bool,
    drop_listing_rows: bool,
) -> List[Dict]:
    parent_chunk_id = _stable_chunk_id(document_id, start_offset, end_offset)
    sub_texts = _split_text_by_word_limit(
        chunk_text,
        max_words=max_chunk_words or 0,
        overlap_sentences=overlap_sentences,
    )

    chunks: List[Dict] = []
    search_pos = 0
    for sub_idx, raw_sub_text in enumerate(sub_texts):
        if len(sub_texts) == 1:
            chunk_id = parent_chunk_id
        else:
            chunk_id = _stable_subchunk_id(document_id, start_offset, end_offset, sub_idx)

        local_start = chunk_text.find(raw_sub_text, search_pos)
        if local_start >= 0:
            local_end = local_start + len(raw_sub_text)
            search_pos = max(local_start + 1, local_end - 1)
        else:
            local_start = 0
            local_end = len(raw_sub_text)

        cleaned_sub_text, cleaning_meta = _clean_chunk_lines(
            raw_sub_text,
            drop_table_rule_lines=drop_table_rule_lines,
            drop_listing_rows=drop_listing_rows,
            patterns=patterns,
        )
        if not cleaned_sub_text:
            continue

        sub_matches = _find_matches(cleaned_sub_text, patterns)
        if not sub_matches:
            continue

        chunks.append(
            {
                "chunk_id": chunk_id,
                "parent_chunk_id": parent_chunk_id if len(sub_texts) > 1 else None,
                "parent_identity": parent_identity,
                "subchunk_index": sub_idx,
                "subchunk_count": len(sub_texts),
                "parent_chunk_char_len": len(chunk_text),
                "parent_chunk_word_len": _word_count(chunk_text),
                "chunk_char_len": len(cleaned_sub_text),
                "chunk_word_len": _word_count(cleaned_sub_text),
                "has_direct_keyword_match": True,
                "chunk_cleaned": bool(
                    cleaning_meta["removed_table_rule_lines"] + cleaning_meta["removed_listing_rows"]
                ),
                "removed_table_rule_lines": cleaning_meta["removed_table_rule_lines"],
                "removed_listing_rows": cleaning_meta["removed_listing_rows"],
                "document_id": document_id,
                "company_id": company_id,
                "company_name": company_name,
                "report_year": report_year,
                "report_sections": report_sections,
                "paragraph_start": paragraph_start,
                "paragraph_end": paragraph_end,
                "block_start": block_start,
                "block_end": block_end,
                "char_start": start_offset + local_start,
                "char_end": min(end_offset, start_offset + local_end),
                "context_before": context_before,
                "context_after": context_after,
                "chunk_text": cleaned_sub_text,
                "matched_keywords": sorted({m["keyword"] for m in sub_matches}),
                "keyword_matches": [
                    {
                        "paragraph_index": paragraph_start,
                        "matches": sub_matches,
                    }
                ],
            }
        )

    return chunks


def chunk_markdown(
    markdown: str,
    document_id: str,
    company_id: str,
    company_name: str,
    report_year: int,
    context_before: int = 2,
    context_after: int = 2,
    max_chunk_words: Optional[int] = None,
    overlap_sentences: int = 1,
    keyword_patterns: Optional[List[KeywordPattern]] = None,
    drop_table_rule_lines: bool = True,
    drop_listing_rows: bool = True,
) -> List[Dict]:
    """Chunk markdown into AI keyword spans with paragraph context."""
    blocks = _parse_markdown_blocks(markdown)
    paragraphs = _extract_paragraphs(blocks)
    patterns = _compile_patterns(keyword_patterns)
    chunks: List[Dict] = []

    if _is_malformed_for_paragraph_chunking(markdown, paragraphs):
        char_radius = max(1200, min(4000, (max_chunk_words or 600) * 3))
        windows = _build_char_windows_from_matches(
            markdown,
            _find_matches(markdown, patterns),
            char_radius=char_radius,
        )
        for window in windows:
            chunk_text = markdown[window.start:window.end].strip()
            if not chunk_text:
                continue
            report_sections = sorted(
                {
                    paragraph.section
                    for paragraph in paragraphs
                    if paragraph.section
                }
            )
            chunks.extend(
                _emit_chunk_variants(
                    source_text=markdown,
                    document_id=document_id,
                    company_id=company_id,
                    company_name=company_name,
                    report_year=report_year,
                    chunk_text=chunk_text,
                    start_offset=window.start,
                    end_offset=window.end,
                    parent_identity="mention_char_window",
                    paragraph_start=0 if paragraphs else None,
                    paragraph_end=max(len(paragraphs) - 1, 0) if paragraphs else None,
                    block_start=paragraphs[0].block_index if paragraphs else None,
                    block_end=paragraphs[-1].block_index if paragraphs else None,
                    context_before=context_before,
                    context_after=context_after,
                    report_sections=report_sections,
                    max_chunk_words=max_chunk_words,
                    overlap_sentences=overlap_sentences,
                    patterns=patterns,
                    drop_table_rule_lines=drop_table_rule_lines,
                    drop_listing_rows=drop_listing_rows,
                )
            )
        return chunks

    mention_windows: List[Tuple[int, int, int]] = []
    matches_by_paragraph: Dict[int, List[dict]] = {}
    for paragraph in paragraphs:
        matches = _find_matches(paragraph.text, patterns)
        if not matches:
            continue
        matches_by_paragraph[paragraph.index] = matches
        start_idx = max(paragraph.index - context_before, 0)
        end_idx = min(paragraph.index + context_after, len(paragraphs) - 1)
        mention_windows.append((start_idx, end_idx, paragraph.index))

    for start_idx, end_idx, mention_indices in _dedupe_paragraph_windows(mention_windows):
        start_block_idx = paragraphs[start_idx].block_index
        end_block_idx = paragraphs[end_idx].block_index
        block_slice = blocks[start_block_idx:end_block_idx + 1]
        chunk_text = "\n\n".join(block.text for block in block_slice).strip()
        if not chunk_text:
            continue

        report_sections = sorted(
            {
                paragraphs[mention_index].section
                for mention_index in sorted(set(mention_indices))
                if paragraphs[mention_index].section
            }
        )
        chunks.extend(
            _emit_chunk_variants(
                source_text=markdown,
                document_id=document_id,
                company_id=company_id,
                company_name=company_name,
                report_year=report_year,
                chunk_text=chunk_text,
                start_offset=blocks[start_block_idx].start_offset,
                end_offset=blocks[end_block_idx].end_offset,
                parent_identity="paragraph_window",
                paragraph_start=start_idx,
                paragraph_end=end_idx,
                block_start=start_block_idx,
                block_end=end_block_idx,
                context_before=context_before,
                context_after=context_after,
                report_sections=report_sections,
                max_chunk_words=max_chunk_words,
                overlap_sentences=overlap_sentences,
                patterns=patterns,
                drop_table_rule_lines=drop_table_rule_lines,
                drop_listing_rows=drop_listing_rows,
            )
        )

    return chunks
