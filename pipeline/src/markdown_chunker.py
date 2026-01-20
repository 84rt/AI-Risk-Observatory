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


def chunk_markdown(
    markdown: str,
    document_id: str,
    company_id: str,
    company_name: str,
    report_year: int,
    context_before: int = 2,
    context_after: int = 2,
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

        chunk_id = _stable_chunk_id(document_id, start_idx, end_idx)
        chunk = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "company_id": company_id,
            "company_name": company_name,
            "report_year": report_year,
            "report_sections": sorted(report_sections),
            "paragraph_start": start_idx,
            "paragraph_end": end_idx,
            "block_start": start_block_idx,
            "block_end": end_block_idx,
            "char_start": blocks[start_block_idx].start_offset,
            "char_end": blocks[end_block_idx].end_offset,
            "context_before": context_before,
            "context_after": context_after,
            "chunk_text": chunk_text,
            "matched_keywords": sorted(matched_keywords),
            "keyword_matches": keyword_matches,
        }
        chunks.append(chunk)

    return chunks
