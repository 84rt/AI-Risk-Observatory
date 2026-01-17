"""Chunk markdown reports by AI keyword mentions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .utils.keywords import AI_KEYWORD_PATTERNS, KeywordPattern


@dataclass
class MarkdownParagraph:
    """A paragraph extracted from markdown with section context."""

    section: Optional[str]
    text: str
    index: int


def _parse_markdown_paragraphs(markdown: str) -> List[MarkdownParagraph]:
    paragraphs: List[MarkdownParagraph] = []
    current_section = None
    buffer: List[str] = []
    index = 0

    def flush():
        nonlocal index
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                paragraphs.append(MarkdownParagraph(section=current_section, text=text, index=index))
                index += 1
        buffer.clear()

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            current_section = stripped.lstrip("#").strip()
            continue
        if not stripped:
            flush()
            continue
        buffer.append(stripped)

    flush()
    return paragraphs


def _compile_patterns(patterns: Optional[List[KeywordPattern]] = None):
    patterns = patterns or AI_KEYWORD_PATTERNS
    return [(kp.name, re.compile(kp.pattern, re.IGNORECASE)) for kp in patterns]


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


def chunk_markdown(
    markdown: str,
    document_id: str,
    company_id: str,
    company_name: str,
    report_year: int,
    context_before: int = 1,
    context_after: int = 1,
    keyword_patterns: Optional[List[KeywordPattern]] = None,
) -> List[Dict]:
    """Chunk markdown into AI keyword spans with paragraph context."""
    paragraphs = _parse_markdown_paragraphs(markdown)
    patterns = _compile_patterns(keyword_patterns)

    chunks: List[Dict] = []
    chunk_index = 0

    for idx, paragraph in enumerate(paragraphs):
        matches = _find_matches(paragraph.text, patterns)
        if not matches:
            continue

        start_idx = max(idx - context_before, 0)
        end_idx = min(idx + context_after, len(paragraphs) - 1)
        context_paragraphs = [p.text for p in paragraphs[start_idx:end_idx + 1]]
        chunk_text = "\n\n".join(context_paragraphs).strip()

        chunk_index += 1
        chunk_id = f"{document_id}-chunk-{chunk_index:04d}"

        chunks.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "company_id": company_id,
                "company_name": company_name,
                "report_year": report_year,
                "report_section": paragraph.section,
                "paragraph_index": paragraph.index,
                "context_before": context_before,
                "context_after": context_after,
                "chunk_text": chunk_text,
                "keyword_matches": matches,
            }
        )

    return chunks
