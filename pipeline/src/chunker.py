"""Text chunking module to generate candidate spans for LLM processing."""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import nltk

from .utils.keywords import AI_KEYWORD_PATTERNS, KeywordPattern, compile_keyword_patterns
# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)


@dataclass
class CandidateSpan:
    """A candidate text span for LLM analysis."""

    span_id: str
    firm_id: str
    firm_name: str
    sector: str
    report_year: int
    report_section: Optional[str]
    text: str
    page_number: Optional[int]
    context_before: str = ""  # Previous sentence for context
    context_after: str = ""   # Next sentence for context
    keyword: Optional[str] = None
    keyword_text: Optional[str] = None
    match_start: Optional[int] = None
    match_end: Optional[int] = None


@dataclass
class ChunkingStats:
    """Summary statistics for a chunking pass."""

    total_mentions: int
    keyword_counts: Dict[str, int]
    total_chunks: int
    total_sections: int
    total_paragraphs: int


class TextChunker:
    """Chunk extracted report text into candidate spans."""

    def __init__(
        self,
        min_chunk_length: int = 50,
        max_chunk_length: int = 1000,
        chunk_by: str = "paragraph",
        keyword_patterns: Optional[List[KeywordPattern]] = None,
    ):
        """Initialize the chunker.

        Args:
            min_chunk_length: Minimum chunk length in characters
            max_chunk_length: Maximum chunk length in characters
            chunk_by: Chunking strategy: "sentence" or "paragraph"
        """
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.chunk_by = chunk_by
        patterns = keyword_patterns or AI_KEYWORD_PATTERNS
        self.keyword_patterns = compile_keyword_patterns(patterns)

    def chunk_report(
        self,
        extracted_report,
        firm_id: str,
        firm_name: str,
        sector: str,
        report_year: int,
        return_stats: bool = False,
    ) -> Union[List[CandidateSpan], Tuple[List[CandidateSpan], ChunkingStats]]:
        """Chunk a report into candidate spans.

        Args:
            extracted_report: ExtractedReport object from pdf_extractor
            firm_id: Company identifier (ISIN or ticker)
            firm_name: Company name
            sector: Sector classification
            report_year: Report year

        Returns:
            List of CandidateSpan objects
        """
        logger.info(f"Chunking report for {firm_name} ({report_year})")

        candidates = []
        span_counter = 0
        total_mentions = 0
        keyword_counts: Dict[str, int] = {}
        total_paragraphs = 0

        # Process each section
        for section_name, section_spans in extracted_report.sections.items():
            # Filter out headings and combine into text
            text_spans = [
                s for s in section_spans
                if not s.is_heading and len(s.text.strip()) > 0
            ]

            if not text_spans:
                continue

            # Get page numbers (use most common page in section)
            page_numbers = [s.page_number for s in text_spans]
            most_common_page = max(set(page_numbers), key=page_numbers.count)

            # Build paragraph candidates from spans
            paragraphs = [s.text.strip() for s in text_spans if s.text.strip()]
            total_paragraphs += len(paragraphs)

            for paragraph in paragraphs:
                if len(paragraph) < self.min_chunk_length:
                    continue

                matches = self._find_keyword_matches(paragraph)
                if not matches:
                    continue

                for match in matches:
                    total_mentions += 1
                    keyword_counts[match["keyword"]] = keyword_counts.get(
                        match["keyword"], 0
                    ) + 1

                    span_counter += 1
                    span_id = f"{firm_id}-{report_year}-{span_counter:04d}"

                    candidates.append(CandidateSpan(
                        span_id=span_id,
                        firm_id=firm_id,
                        firm_name=firm_name,
                        sector=sector,
                        report_year=report_year,
                        report_section=section_name,
                        text=paragraph,
                        page_number=most_common_page,
                        keyword=match["keyword"],
                        keyword_text=match["text"],
                        match_start=match["start"],
                        match_end=match["end"],
                    ))

        logger.info(
            f"Generated {len(candidates)} candidate spans "
            f"from {len(extracted_report.sections)} sections"
        )

        stats = ChunkingStats(
            total_mentions=total_mentions,
            keyword_counts=keyword_counts,
            total_chunks=len(candidates),
            total_sections=len(extracted_report.sections),
            total_paragraphs=total_paragraphs,
        )

        if return_stats:
            return candidates, stats
        return candidates

    def _chunk_by_paragraph(self, text: str) -> List[str]:
        """Chunk text by paragraphs.

        Args:
            text: Input text

        Returns:
            List of paragraph chunks
        """
        # Split by double newlines or multiple spaces
        paragraphs = re.split(r'\n\s*\n|\s{3,}', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_length = len(para)

            # If adding this paragraph exceeds max length, save current chunk
            if current_length + para_length > self.max_chunk_length and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            # If single paragraph is too long, split by sentences
            if para_length > self.max_chunk_length:
                sentences = self._split_sentences(para)
                for sent in sentences:
                    if current_length + len(sent) > self.max_chunk_length and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_length = 0

                    current_chunk.append(sent)
                    current_length += len(sent) + 1
            else:
                current_chunk.append(para)
                current_length += para_length + 1

        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _chunk_by_sentence(self, text: str) -> List[str]:
        """Chunk text by sentences.

        Args:
            text: Input text

        Returns:
            List of sentence chunks
        """
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sent in sentences:
            sent_length = len(sent)

            if current_length + sent_length > self.max_chunk_length and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(sent)
            current_length += sent_length + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences using NLTK.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Use NLTK's sentence tokenizer
        sentences = nltk.sent_tokenize(text)

        # Clean up sentences
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            if sent:
                cleaned.append(sent)

        return cleaned

    def _find_keyword_matches(self, text: str) -> List[Dict[str, Optional[str]]]:
        """Find keyword matches in text and return match metadata."""
        matches = []
        for keyword_name, pattern in self.keyword_patterns:
            for match in pattern.finditer(text):
                matches.append({
                    "keyword": keyword_name,
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                })
        return matches


def chunk_report(
    extracted_report,
    firm_id: str,
    firm_name: str,
    sector: str,
    report_year: int,
    chunk_by: str = "paragraph",
    return_stats: bool = False,
) -> Union[List[CandidateSpan], Tuple[List[CandidateSpan], ChunkingStats]]:
    """Convenience function to chunk a report.

    Args:
        extracted_report: ExtractedReport from pdf_extractor
        firm_id: Company identifier
        firm_name: Company name
        sector: Sector classification
        report_year: Report year
        chunk_by: Chunking strategy ("paragraph" or "sentence")

    Returns:
        List of CandidateSpan objects
    """
    chunker = TextChunker(chunk_by=chunk_by)
    return chunker.chunk_report(
        extracted_report=extracted_report,
        firm_id=firm_id,
        firm_name=firm_name,
        sector=sector,
        report_year=report_year,
        return_stats=return_stats,
    )
