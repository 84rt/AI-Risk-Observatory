"""iXBRL/XHTML text extraction for annual reports."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser
from html import unescape

logger = logging.getLogger(__name__)


# Common section headings in UK annual reports
SECTION_PATTERNS = [
    r"principal\s+risks?(?:\s+and\s+uncertainties)?",
    r"risk\s+(?:management|factors|review)",
    r"strategic\s+report",
    r"directors'?\s+report",
    r"governance\s+report",
    r"sustainability\s+report",
    r"esg\s+report",
    r"environmental,?\s+social\s+and\s+governance",
    r"operational\s+review",
    r"business\s+review",
    r"financial\s+review",
    r"notes\s+to\s+(?:the\s+)?financial\s+statements",
]


@dataclass
class TextSpan:
    """A span of text extracted from an iXBRL/XHTML document."""

    text: str
    page_number: Optional[int] = None  # iXBRL doesn't have pages, but we keep for compatibility
    section: Optional[str] = None
    is_heading: bool = False
    font_size: Optional[float] = None


@dataclass
class ExtractedReport:
    """Container for extracted report content."""

    spans: List[TextSpan]
    metadata: Dict
    full_text: str

    @property
    def sections(self) -> Dict[str, List[TextSpan]]:
        """Group spans by section."""
        result = {}
        current_section = "other"
        
        section_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in SECTION_PATTERNS
        ]

        for span in self.spans:
            if span.section:
                current_section = span.section
            elif span.is_heading:
                # Try to detect section from heading
                text_lower = span.text.lower()
                for pattern in section_patterns:
                    match = pattern.search(text_lower)
                    if match:
                        current_section = match.group(0).replace(" ", "_")
                        break

            if current_section not in result:
                result[current_section] = []

            result[current_section].append(span)

        return result


class iXBRLParser(HTMLParser):
    """HTML parser to extract text from iXBRL/XHTML documents."""

    def __init__(self):
        super().__init__()
        self.spans = []
        self.current_text = []
        self.current_tag = None
        self.in_script = False
        self.in_style = False
        self.section_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in SECTION_PATTERNS
        ]

    def handle_starttag(self, tag, attrs):
        """Handle opening tags."""
        self.current_tag = tag.lower()
        
        if tag.lower() in ['script', 'style']:
            if tag.lower() == 'script':
                self.in_script = True
            elif tag.lower() == 'style':
                self.in_style = True
            return

        # Check if this is a heading tag
        if tag.lower() in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            # Save any accumulated text before the heading
            if self.current_text:
                text = ' '.join(self.current_text).strip()
                if text:
                    self._add_span(text, is_heading=False)
                self.current_text = []

    def handle_endtag(self, tag):
        """Handle closing tags."""
        tag_lower = tag.lower()
        
        if tag_lower == 'script':
            self.in_script = False
            return
        elif tag_lower == 'style':
            self.in_style = False
            return

        # When we close a heading, save it as a heading span
        if tag_lower in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            if self.current_text:
                text = ' '.join(self.current_text).strip()
                if text:
                    self._add_span(text, is_heading=True)
                self.current_text = []
        # When we close block-level elements, save accumulated text
        elif tag_lower in ['p', 'div', 'li', 'td', 'th', 'section', 'article']:
            if self.current_text:
                text = ' '.join(self.current_text).strip()
                if text and len(text) > 10:  # Minimum length filter
                    self._add_span(text, is_heading=False)
                self.current_text = []

    def handle_data(self, data):
        """Handle text data."""
        if self.in_script or self.in_style:
            return

        # Clean and accumulate text
        text = data.strip()
        if text:
            self.current_text.append(text)

    def _add_span(self, text: str, is_heading: bool):
        """Add a text span."""
        # Clean up excessive whitespace (iXBRL often has spaces between characters)
        # This fixes issues like "risk s" -> "risks"
        cleaned_text = self._clean_text(text)

        # Detect section from heading (or from any span that matches)
        section = None
        for pattern in self.section_patterns:
            if pattern.search(cleaned_text.lower()):
                match = pattern.search(cleaned_text.lower())
                if match:
                    section = match.group(0).replace(" ", "_")
                    # Mark as heading if it matches a section pattern
                    if not is_heading and len(cleaned_text) < 100:
                        is_heading = True
                    break

        self.spans.append(TextSpan(
            text=unescape(cleaned_text),
            section=section,
            is_heading=is_heading
        ))

    def _clean_text(self, text: str) -> str:
        """Clean up text spacing issues common in iXBRL.

        iXBRL files often have spaces between characters or within words.
        This method normalizes the spacing while preserving intentional word breaks.
        """
        import re

        # First, normalize all whitespace to single spaces
        text = ' '.join(text.split())

        # Common spacing issue in iXBRL: characters/syllables are split
        # "risk s" -> "risks"
        # "principal risk s" -> "principal risks"

        # Be conservative: only fix obvious single-character fragments
        # Pattern 1: Single letter at start of word: "c ust" -> "cust"
        # Exclude "a" and "i" to avoid merging articles/pronouns ("a company" -> "acompany").
        # Avoid common stopwords to prevent "s and" -> "sand".
        text = re.sub(
            r'(?<=\s)([b-hj-z])\s+(?!(?:and|the|for|with|to|of|in|on|at)\b)([a-z]{3,})',
            r'\1\2',
            text
        )

        # Pattern 2: Single letter suffix at end: "risk s" -> "risks"
        # Exclude "a" and "i" to avoid merging articles/pronouns ("uses a" -> "usesa").
        text = re.sub(r'([a-z]{3,})\s+([b-hj-z])(?=\s|[,.;:\)]|$)', r'\1\2', text)

        return text

    def get_spans(self) -> List[TextSpan]:
        """Get all extracted spans."""
        # Flush any remaining text
        if self.current_text:
            text = ' '.join(self.current_text).strip()
            if text and len(text) > 10:
                self._add_span(text, is_heading=False)
        
        return self.spans


class iXBRLExtractor:
    """Extract text and structure from iXBRL/XHTML annual reports."""

    def __init__(self):
        """Initialize the iXBRL extractor."""
        self.section_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in SECTION_PATTERNS
        ]

    def extract_report(self, file_path: Path) -> ExtractedReport:
        """Extract text and structure from an iXBRL/XHTML file.

        Args:
            file_path: Path to iXBRL/XHTML file

        Returns:
            ExtractedReport with structured content
        """
        logger.info(f"Extracting text from iXBRL/XHTML: {file_path}")

        # Read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                html_content = f.read()

        # Parse HTML
        parser = iXBRLParser()
        parser.feed(html_content)
        spans = parser.get_spans()
        spans = self._rebuild_paragraphs(spans)

        # Build full text
        full_text = "\n\n".join(
            span.text for span in spans
            if not span.is_heading or len(span.text) > 20  # Include substantial headings
        )

        metadata = {
            "filename": file_path.name,
            "num_spans": len(spans),
            "format": "ixbrl"
        }

        logger.info(
            f"Extracted {len(spans)} spans from iXBRL/XHTML document"
        )

        return ExtractedReport(
            spans=spans,
            metadata=metadata,
            full_text=full_text
        )

    def _rebuild_paragraphs(self, spans: List[TextSpan]) -> List[TextSpan]:
        """Merge line-like spans into paragraphs using simple heuristics."""
        rebuilt = []
        buffer_parts = []
        buffer_section = None

        for span in spans:
            text = span.text.strip()
            if not text:
                continue

            if span.is_heading:
                self._flush_paragraph(buffer_parts, buffer_section, rebuilt)
                buffer_parts = []
                buffer_section = None
                rebuilt.append(span)
                continue

            if not buffer_parts:
                buffer_parts = [text]
                buffer_section = span.section
                continue

            prev_text = buffer_parts[-1]
            if self._should_join(prev_text, text):
                buffer_parts[-1] = self._merge_fragments(prev_text, text)
            else:
                self._flush_paragraph(buffer_parts, buffer_section, rebuilt)
                buffer_parts = [text]
                buffer_section = span.section

        self._flush_paragraph(buffer_parts, buffer_section, rebuilt)
        return rebuilt

    def _flush_paragraph(
        self,
        parts: List[str],
        section: Optional[str],
        rebuilt: List[TextSpan]
    ) -> None:
        if not parts:
            return
        merged = " ".join(part.strip() for part in parts if part.strip())
        if merged:
            rebuilt.append(TextSpan(text=merged, section=section, is_heading=False))

    def _should_join(self, prev_text: str, next_text: str) -> bool:
        if self._looks_like_heading(next_text):
            return False
        if re.match(r'^[•\\-–]\\s', next_text):
            return False

        prev_stripped = prev_text.rstrip()
        if prev_stripped.endswith(('.', '!', '?')):
            return False

        if prev_stripped.endswith((':', ';')):
            return True

        if re.match(r'^[a-z0-9(]', next_text):
            return True

        if len(prev_stripped) < 60:
            return False

        return True

    def _merge_fragments(self, prev_text: str, next_text: str) -> str:
        if prev_text.endswith('-') and next_text and next_text[0].islower():
            return prev_text[:-1] + next_text
        return f"{prev_text} {next_text}"

    def _looks_like_heading(self, text: str) -> bool:
        if len(text) > 60:
            return False
        words = text.split()
        if not words:
            return False
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(1, sum(1 for c in text if c.isalpha()))
        if uppercase_ratio > 0.6 and len(words) <= 8:
            return True
        if len(words) <= 6 and all(w[:1].isupper() for w in words if w[:1].isalpha()):
            return True
        return False


def extract_text_from_ixbrl(file_path: Path) -> ExtractedReport:
    """Convenience function to extract text from iXBRL/XHTML file.

    Args:
        file_path: Path to iXBRL/XHTML file

    Returns:
        ExtractedReport with structured content
    """
    extractor = iXBRLExtractor()
    return extractor.extract_report(file_path)
