"""Chunk markdown reports by AI keyword mentions."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .utils.keywords import AI_KEYWORD_PATTERNS, KeywordPattern, compile_keyword_patterns


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
) -> List[Dict]:
    """Chunk markdown into AI keyword spans with paragraph context."""
    blocks = _parse_markdown_blocks(markdown)
    paragraphs = _extract_paragraphs(blocks)
    patterns = _compile_patterns(keyword_patterns)

    chunks: List[Dict] = []
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

    merged_windows = _merge_windows(mention_windows)

    for start_idx, end_idx, mention_indices in merged_windows:
        start_block_idx = paragraphs[start_idx].block_index
        end_block_idx = paragraphs[end_idx].block_index
        block_slice = blocks[start_block_idx:end_block_idx + 1]
        chunk_text = "\n\n".join(block.text for block in block_slice).strip()

        matched_keywords = set()
        keyword_matches = []
        report_sections = set()
        for mention_index in sorted(set(mention_indices)):
            matches = matches_by_paragraph.get(mention_index, [])
            if matches:
                keyword_matches.append(
                    {
                        "paragraph_index": mention_index,
                        "matches": matches,
                    }
                )
                matched_keywords.update({m["keyword"] for m in matches})
            section = paragraphs[mention_index].section
            if section:
                report_sections.add(section)

        parent_chunk_id = _stable_chunk_id(document_id, start_idx, end_idx)
        sub_texts = _split_text_by_word_limit(
            chunk_text,
            max_words=max_chunk_words or 0,
            overlap_sentences=overlap_sentences,
        )

        search_pos = 0
        for sub_idx, sub_text in enumerate(sub_texts):
            if len(sub_texts) == 1:
                chunk_id = parent_chunk_id
            else:
                chunk_id = _stable_subchunk_id(
                    document_id, start_idx, end_idx, sub_idx
                )

            local_start = chunk_text.find(sub_text, search_pos)
            if local_start >= 0:
                local_end = local_start + len(sub_text)
                search_pos = max(local_start + 1, local_end - 1)
            else:
                local_start = 0
                local_end = len(sub_text)

            sub_matches = _find_matches(sub_text, patterns)
            if sub_matches:
                sub_keyword_matches = [
                    {
                        "paragraph_index": start_idx,
                        "matches": sub_matches,
                    }
                ]
                sub_matched_keywords = sorted({m["keyword"] for m in sub_matches})
            else:
                # Keep traceability for context-only subchunks produced by splitting.
                sub_keyword_matches = keyword_matches
                sub_matched_keywords = sorted(matched_keywords)

            chunk = {
                "chunk_id": chunk_id,
                "parent_chunk_id": parent_chunk_id if len(sub_texts) > 1 else None,
                "subchunk_index": sub_idx,
                "subchunk_count": len(sub_texts),
                "document_id": document_id,
                "company_id": company_id,
                "company_name": company_name,
                "report_year": report_year,
                "report_sections": sorted(report_sections),
                "paragraph_start": start_idx,
                "paragraph_end": end_idx,
                "block_start": start_block_idx,
                "block_end": end_block_idx,
                "char_start": blocks[start_block_idx].start_offset + local_start,
                "char_end": blocks[start_block_idx].start_offset + local_end,
                "context_before": context_before,
                "context_after": context_after,
                "chunk_text": sub_text,
                "matched_keywords": sub_matched_keywords,
                "keyword_matches": sub_keyword_matches,
            }
            chunks.append(chunk)

    return chunks
