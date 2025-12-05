"""PDF text extraction with section detection for annual reports."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

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
    """A span of text extracted from a PDF."""

    text: str
    page_number: int
    section: Optional[str] = None
    is_heading: bool = False
    font_size: Optional[float] = None


@dataclass
class ExtractedReport:
    """Container for extracted report content."""

    spans: List[TextSpan]
    metadata: Dict[str, any]
    full_text: str

    @property
    def sections(self) -> Dict[str, List[TextSpan]]:
        """Group spans by section."""
        result = {}
        current_section = "Unknown"

        for span in self.spans:
            if span.is_heading and span.section:
                current_section = span.section
            if current_section not in result:
                result[current_section] = []
            result[current_section].append(span)

        return result


class PDFExtractor:
    """Extract text and structure from PDF annual reports."""

    def __init__(self):
        """Initialize the PDF extractor."""
        self.section_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in SECTION_PATTERNS
        ]

    def extract_report(self, pdf_path: Path) -> ExtractedReport:
        """Extract text and structure from a PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            ExtractedReport with structured content
        """
        logger.info(f"Extracting text from {pdf_path}")

        doc = fitz.open(pdf_path)
        spans = []
        full_text_parts = []
        
        # Get page count BEFORE processing (needed for metadata)
        num_pages = len(doc)

        # Get average font size for heading detection
        avg_font_size = self._get_average_font_size(doc)

        for page_num, page in enumerate(doc, start=1):
            page_spans = self._extract_page(page, page_num, avg_font_size)
            spans.extend(page_spans)

            # Build full text
            page_text = " ".join(s.text for s in page_spans if not s.is_heading)
            full_text_parts.append(page_text)

        doc.close()

        full_text = "\n\n".join(full_text_parts)

        metadata = {
            "filename": pdf_path.name,
            "num_pages": num_pages,
            "num_spans": len(spans),
        }

        logger.info(
            f"Extracted {len(spans)} spans from {num_pages} pages"
        )

        return ExtractedReport(
            spans=spans,
            metadata=metadata,
            full_text=full_text
        )

    def _extract_page(
        self,
        page: fitz.Page,
        page_num: int,
        avg_font_size: float
    ) -> List[TextSpan]:
        """Extract structured text from a single page.

        Args:
            page: PyMuPDF page object
            page_num: Page number (1-indexed)
            avg_font_size: Average font size in document (for heading detection)

        Returns:
            List of TextSpan objects
        """
        spans = []
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in blocks.get("blocks", []):
            if block.get("type") != 0:  # Skip non-text blocks
                continue

            for line in block.get("lines", []):
                line_text_parts = []
                font_sizes = []

                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        line_text_parts.append(text)
                        font_sizes.append(span.get("size", 0))

                if not line_text_parts:
                    continue

                line_text = " ".join(line_text_parts)
                avg_line_font = sum(font_sizes) / len(font_sizes) if font_sizes else 0

                # Detect if this is a heading
                is_heading = self._is_heading(line_text, avg_line_font, avg_font_size)

                # Detect section
                section = None
                if is_heading:
                    section = self._detect_section(line_text)

                spans.append(TextSpan(
                    text=line_text,
                    page_number=page_num,
                    section=section,
                    is_heading=is_heading,
                    font_size=avg_line_font
                ))

        return spans

    def _get_average_font_size(self, doc: fitz.Document) -> float:
        """Calculate average font size across document.

        Args:
            doc: PyMuPDF document

        Returns:
            Average font size
        """
        font_sizes = []

        # Sample first 10 pages for efficiency
        for page in list(doc)[:10]:
            blocks = page.get_text("dict")
            for block in blocks.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            size = span.get("size", 0)
                            if size > 0:
                                font_sizes.append(size)

        return sum(font_sizes) / len(font_sizes) if font_sizes else 12.0

    def _is_heading(
        self,
        text: str,
        font_size: float,
        avg_font_size: float
    ) -> bool:
        """Detect if a line is a heading.

        Args:
            text: Line text
            font_size: Font size of this line
            avg_font_size: Average font size in document

        Returns:
            True if likely a heading
        """
        # Check font size (headings are typically larger)
        if font_size > avg_font_size * 1.2:
            return True

        # Check for all caps short text (common heading pattern)
        if text.isupper() and len(text.split()) <= 8:
            return True

        # Check for numbered headings (e.g., "1. Introduction", "2.3 Risk Factors")
        if re.match(r"^\d+\.(\d+\.?)?\s+[A-Z]", text):
            return True

        return False

    def _detect_section(self, text: str) -> Optional[str]:
        """Detect if text matches a known section heading.

        Args:
            text: Heading text

        Returns:
            Section name if matched, None otherwise
        """
        for pattern in self.section_patterns:
            if pattern.search(text):
                # Normalize section name
                return self._normalize_section_name(text)

        return None

    @staticmethod
    def _normalize_section_name(text: str) -> str:
        """Normalize section heading to canonical form.

        Args:
            text: Raw heading text

        Returns:
            Normalized section name
        """
        # Remove numbering
        text = re.sub(r"^\d+\.(\d+\.?)?\s+", "", text)

        # Normalize whitespace
        text = " ".join(text.split())

        # Title case
        text = text.title()

        return text


def extract_text_from_pdf(pdf_path: Path) -> ExtractedReport:
    """Convenience function to extract text from a PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        ExtractedReport with structured content
    """
    extractor = PDFExtractor()
    return extractor.extract_report(pdf_path)
