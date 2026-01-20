"""Preprocessing module for filtering and formatting extracted reports.

This module provides different strategies for preprocessing annual reports:
1. risk_only: Extract only risk-related sections
2. keyword: Filter paragraphs containing AI/ML/risk keywords
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Set, Optional

from .utils.keywords import AI_KEYWORD_PATTERNS, compile_keyword_patterns
from .text_repair import TextRepairService

logger = logging.getLogger(__name__)

class PreprocessingStrategy(Enum):
    """Available preprocessing strategies."""
    FULL = "full"  # No filtering (baseline)
    RISK_ONLY = "risk_only"  # Only risk sections
    KEYWORD = "keyword"  # Keyword-based filtering


# Risk-related keywords
RISK_KEYWORDS = [
    r"\brisk\b",
    r"\brisks\b",
    r"\buncertainty\b",
    r"\buncertainties\b",
    r"\bthreat\b",
    r"\bthreats\b",
    r"\bchallenge\b",
    r"\bchallenges\b",
    r"\bconcern\b",
    r"\bconcerns\b",
    r"\bvulnerabilit(?:y|ies)\b",
    r"\bexposure\b",
    r"\bimpact\b",
    r"\bdisrupt(?:ion|ive)\b",
    r"\badverse(?:ly)?\b",
]

# Risk section identifiers (normalized from SECTION_PATTERNS in extractors)
RISK_SECTION_IDENTIFIERS = [
    "principal_risk",
    "risk_management",
    "risk_factor",
    "risk_review",
    "risk_and_uncertaint",  # Matches "risks and uncertainties"
]

# Qualitative sections (exclude pure financial sections)
QUALITATIVE_SECTIONS = [
    "strategic_report",
    "directors_report",
    "governance",
    "sustainability",
    "esg",
    "environmental",
    "operational_review",
    "business_review",
]


@dataclass
class PreprocessedReport:
    """Container for preprocessed report content."""
    strategy: PreprocessingStrategy
    markdown_content: str
    metadata: dict
    stats: dict  # Statistics about filtering


class Preprocessor:
    """Preprocessor for filtering and formatting extracted reports."""

    def __init__(
        self,
        strategy: PreprocessingStrategy = PreprocessingStrategy.FULL,
        include_context: bool = True
    ):
        """Initialize preprocessor.

        Args:
            strategy: Preprocessing strategy to use
            include_context: Whether to include surrounding context for keyword matches
        """
        self.strategy = strategy
        self.include_context = include_context

        # Compile regex patterns for efficiency
        self.ai_patterns = compile_keyword_patterns(AI_KEYWORD_PATTERNS)
        self.risk_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in RISK_KEYWORDS]
        self.all_keyword_patterns = [p[1] for p in self.ai_patterns] + self.risk_patterns
        self.text_repair = TextRepairService()

    def process(self, extracted_report, firm_name: str) -> PreprocessedReport:
        """Process an extracted report according to the strategy.

        Args:
            extracted_report: ExtractedReport object from extractor
            firm_name: Company name for metadata

        Returns:
            PreprocessedReport with filtered markdown content
        """
        logger.info(f"Preprocessing {firm_name} using strategy: {self.strategy.value}")

        extracted_report.spans = self._repair_spans(extracted_report.spans)

        if self.strategy == PreprocessingStrategy.FULL:
            filtered_spans, stats = self._filter_full(extracted_report)
        elif self.strategy == PreprocessingStrategy.RISK_ONLY:
            filtered_spans, stats = self._filter_risk_only(extracted_report)
        elif self.strategy == PreprocessingStrategy.KEYWORD:
            filtered_spans, stats = self._filter_by_keywords(extracted_report)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Convert to markdown
        markdown = self._to_markdown(filtered_spans)

        # Add metadata
        metadata = {
            "firm_name": firm_name,
            "strategy": self.strategy.value,
            "original_spans": len(extracted_report.spans),
            "filtered_spans": len(filtered_spans),
            "original_sections": len(extracted_report.sections),
        }

        logger.info(
            f"  Filtered {len(extracted_report.spans)} → {len(filtered_spans)} spans "
            f"({stats.get('retention_pct', 0):.1f}% retained)"
        )

        return PreprocessedReport(
            strategy=self.strategy,
            markdown_content=markdown,
            metadata=metadata,
            stats=stats
        )

    def _repair_spans(self, spans: List) -> List:
        """Repair spans if they include raw iXBRL fragments."""
        if not spans:
            return spans
        if not hasattr(spans[0], "raw_text"):
            return spans
        return self.text_repair.repair_spans(spans)

    def _filter_full(self, report) -> tuple[List, dict]:
        """No filtering - return all spans."""
        stats = {
            "retention_pct": 100.0,
            "method": "full_document"
        }
        return report.spans, stats

    def _filter_risk_only(self, report) -> tuple[List, dict]:
        """Extract only risk-related sections."""
        filtered_spans = []
        sections_included = []

        for section_name, spans in report.sections.items():
            section_lower = section_name.lower().replace(" ", "_")

            # Check if section matches risk identifiers
            is_risk_section = any(
                identifier in section_lower
                for identifier in RISK_SECTION_IDENTIFIERS
            )

            if is_risk_section:
                filtered_spans.extend(spans)
                sections_included.append(section_name)

        retention_pct = (len(filtered_spans) / len(report.spans) * 100) if report.spans else 0

        stats = {
            "retention_pct": retention_pct,
            "sections_included": sections_included,
            "num_sections": len(sections_included),
            "method": "risk_sections_only"
        }

        logger.info(f"  Found {len(sections_included)} risk sections: {sections_included}")

        return filtered_spans, stats

    def _filter_by_keywords(self, report) -> tuple[List, dict]:
        """Filter spans containing AI/ML/risk keywords.

        Keeps:
        - All headings (for structure)
        - Paragraphs matching keywords
        - Optionally: context before/after matches
        """
        filtered_spans = []
        matched_keywords = set()
        ai_matches = 0
        risk_matches = 0

        for i, span in enumerate(report.spans):
            should_include = False

            # Always include headings for structure
            if span.is_heading:
                should_include = True
            else:
                # Check for keyword matches
                # Check AI keywords
                for name, pattern in self.ai_patterns:
                    if pattern.search(span.text):
                        should_include = True
                        ai_matches += 1
                        matched_keywords.add(name)
                        break

                # Check risk keywords
                if not should_include:
                    for pattern in self.risk_patterns:
                        if pattern.search(span.text):
                            should_include = True
                            risk_matches += 1
                            matched_keywords.add(pattern.pattern)
                            break

            if should_include:
                # Optionally include context (previous span if not already included)
                if self.include_context and filtered_spans:
                    # Check if previous span is already included
                    if i > 0 and report.spans[i-1] not in filtered_spans:
                        prev_span = report.spans[i-1]
                        if not prev_span.is_heading:  # Don't duplicate headings
                            filtered_spans.append(prev_span)

                filtered_spans.append(span)

        retention_pct = (len(filtered_spans) / len(report.spans) * 100) if report.spans else 0

        stats = {
            "retention_pct": retention_pct,
            "ai_keyword_matches": ai_matches,
            "risk_keyword_matches": risk_matches,
            "total_keyword_matches": ai_matches + risk_matches,
            "unique_keywords_matched": len(matched_keywords),
            "method": "keyword_filtering"
        }

        logger.info(
            f"  Keyword matches: {ai_matches} AI/ML, {risk_matches} risk terms "
            f"({len(matched_keywords)} unique patterns)"
        )

        return filtered_spans, stats

    def _to_markdown(self, spans: List) -> str:
        """Convert spans to markdown format.

        Args:
            spans: List of TextSpan objects

        Returns:
            Markdown-formatted text
        """
        if not spans:
            return ""

        markdown_parts = []
        last_section = None

        for span in spans:
            # Track section changes
            if span.section and span.section != last_section:
                last_section = span.section

            if span.is_heading:
                # Determine heading level based on font size if available
                if span.font_size and span.font_size > 16:
                    heading_marker = "#"
                elif span.font_size and span.font_size > 14:
                    heading_marker = "##"
                else:
                    heading_marker = "##"  # Default to h2

                markdown_parts.append(f"\n{heading_marker} {span.text}\n")
            else:
                # Regular paragraph
                # Clean up excessive whitespace
                text = " ".join(span.text.split())
                if text:
                    markdown_parts.append(f"{text}\n")

        return "\n".join(markdown_parts)

    def save_to_file(
        self,
        preprocessed: PreprocessedReport,
        output_path: Path,
        include_header: bool = True,
    ):
        """Save preprocessed report to markdown file.

        Args:
            preprocessed: PreprocessedReport object
            output_path: Path to save markdown file
            include_header: Whether to include a metadata header in markdown
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        content = preprocessed.markdown_content
        if include_header:
            header = f"""---
firm: {preprocessed.metadata['firm_name']}
strategy: {preprocessed.strategy.value}
original_spans: {preprocessed.metadata['original_spans']}
filtered_spans: {preprocessed.metadata['filtered_spans']}
retention: {preprocessed.stats.get('retention_pct', 0):.1f}%
---

"""
            content = header + content

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Saved preprocessed report to {output_path}")


def preprocess_report(
    extracted_report,
    firm_name: str,
    strategy: PreprocessingStrategy = PreprocessingStrategy.FULL,
    output_path: Optional[Path] = None
) -> PreprocessedReport:
    """Convenience function to preprocess a report.

    Args:
        extracted_report: ExtractedReport from extractor
        firm_name: Company name
        strategy: Preprocessing strategy
        output_path: Optional path to save markdown file

    Returns:
        PreprocessedReport object
    """
    preprocessor = Preprocessor(strategy=strategy)
    result = preprocessor.process(extracted_report, firm_name)

    if output_path:
        preprocessor.save_to_file(result, output_path)

    return result
