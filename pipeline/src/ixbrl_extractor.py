"""iXBRL/XHTML text extraction for annual reports."""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set
from html.parser import HTMLParser
from html import unescape

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dictionary-based word repair
# ---------------------------------------------------------------------------

def _load_word_dictionary() -> Set[str]:
    """Load English dictionary plus financial domain terms for word repair."""
    words = set()

    # Try to load NLTK words corpus
    try:
        import nltk
        try:
            from nltk.corpus import words as nltk_words
            words.update(w.lower() for w in nltk_words.words())
        except LookupError:
            # Download if not present
            nltk.download('words', quiet=True)
            from nltk.corpus import words as nltk_words
            words.update(w.lower() for w in nltk_words.words())
    except Exception as e:
        logger.warning(f"Could not load NLTK words corpus: {e}")

    # Add financial/business domain terms not in standard dictionary
    domain_terms = {
        # Financial terms
        "shareholders", "stakeholders", "shareholder", "stakeholder",
        "ebitda", "ebit", "roce", "rote", "roe", "eps", "nav", "aum",
        "capex", "opex", "forex", "libor", "sonia", "euribor",
        "refinancing", "deleveraging", "derisking", "ringfencing",
        # Business terms
        "digitalisation", "digitalization", "decarbonisation", "decarbonization",
        "sustainability", "esg", "tcfd", "ifrs", "gaap", "xbrl", "ixbrl",
        "cybersecurity", "fintech", "regtech", "proptech", "insurtech",
        "omnichannel", "multichannel", "blockchain", "cryptocurrency",
        # UK specific
        "programme", "programmes", "organisation", "organisations",
        "recognised", "recognising", "analysed", "analysing", "capitalised",
        "utilised", "utilising", "optimised", "optimising", "prioritised",
        # Common report words
        "governance", "remuneration", "diversification", "transformation",
        "infrastructure", "operations", "operational", "strategically",
        "profitability", "liquidity", "solvency", "provisioning",
        "impairment", "amortisation", "depreciation", "consolidation",
        "subsidiaries", "undertakings", "disclosures", "regulatory",
        # Compound-prone words
        "customers", "colleagues", "communities", "businesses", "services",
        "performance", "management", "development", "environment", "investment",
        "opportunities", "challenges", "initiatives", "objectives", "priorities",
        "sustainable", "financial", "commercial", "technology", "innovation",
        "manufacturing", "engineering", "procurement", "compliance", "cyber",
        "renewables", "sustainably", "sustainability", "digital", "digitally",
        "supplychain", "supply", "chain", "workforce", "stakeholder",
        "stakeholders", "shareholder", "shareholders", "remediation",
        "materiality", "material", "mitigation", "mitigate", "mitigated",
        "resilience", "resilient", "decarbonise", "decarbonised",
        "decarbonising", "decarbonization", "decarbonisation",
        "governance", "governed", "govern", "governing",
    }
    words.update(domain_terms)

    # Load system wordlist if available (macOS/Linux)
    for wordlist_path in (Path("/usr/share/dict/words"), Path("/usr/dict/words")):
        if wordlist_path.exists():
            try:
                with wordlist_path.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        word = line.strip()
                        if word:
                            words.add(word.lower())
            except Exception as exc:
                logger.warning(f"Failed to load wordlist {wordlist_path}: {exc}")

    # Load optional project wordlist if present
    project_root = Path(__file__).resolve().parents[2]
    project_wordlist = project_root / "data" / "reference" / "wordlist.txt"
    if project_wordlist.exists():
        try:
            with project_wordlist.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith("#"):
                        words.add(word.lower())
        except Exception as exc:
            logger.warning(f"Failed to load project wordlist: {exc}")

    return words


# Module-level dictionary (loaded once)
_WORD_DICT: Optional[Set[str]] = None


def _get_word_dict() -> Set[str]:
    """Get the word dictionary, loading it if necessary."""
    global _WORD_DICT
    if _WORD_DICT is None:
        _WORD_DICT = _load_word_dictionary()
    return _WORD_DICT


@lru_cache(maxsize=10000)
def _is_valid_word(word: str) -> bool:
    """Check if a word is in the dictionary."""
    return word.lower() in _get_word_dict()


def _repair_fragmented_words(text: str) -> str:
    """Use dictionary lookup to merge incorrectly split words.

    Handles cases like "cus to mers" -> "customers" by trying to merge
    adjacent tokens when the result is a valid dictionary word.
    Also handles "collea guesand" -> "colleagues and" by recognizing
    that tokens may contain concatenated suffixes/prefixes.
    """
    words = _get_word_dict()
    # Common words that get concatenated as suffixes
    concat_words = ["and", "the", "for", "are", "was", "has", "its", "our", "all",
                    "can", "may", "will", "not", "but", "also", "more", "most"]

    def try_split_token(token: str, require_valid_base: bool = True):
        """Try to split a token that might have concatenated words at start or end.

        Args:
            token: The token to try splitting
            require_valid_base: If True, only return split if base is a valid word.
                               If False, return any valid suffix/prefix split.
        """
        token_lower = token.lower()

        # Try stripping common words from the END
        for word in concat_words:
            if token_lower.endswith(word) and len(token) > len(word) + 2:
                base = token[:-len(word)]
                remainder = token[-len(word):]
                # Check if base is a valid word (if required)
                if not require_valid_base or base.lower() in words:
                    return (base, remainder)
                # Try stripping another word from the end of base
                for word2 in concat_words:
                    if base.lower().endswith(word2) and len(base) > len(word2) + 2:
                        base2 = base[:-len(word2)]
                        if not require_valid_base or base2.lower() in words:
                            return (base2, base[-len(word2):] + " " + remainder)

        # Try stripping common words from the START
        for word in concat_words:
            if token_lower.startswith(word) and len(token) > len(word) + 2:
                prefix = token[:len(word)]
                remainder = token[len(word):]
                if not require_valid_base or remainder.lower() in words:
                    return (prefix, remainder)

        return None

    tokens = text.split()
    if not tokens:
        return text

    result = []
    i = 0

    while i < len(tokens):
        current = tokens[i]

        # Try merging with next 0-4 tokens (0 = just process current token)
        merged = False
        for lookahead in range(4, -1, -1):  # 4, 3, 2, 1, 0
            if lookahead > 0 and i + lookahead >= len(tokens):
                continue

            # Build candidate by merging tokens
            candidate_tokens = tokens[i:i + lookahead + 1]
            candidate = ''.join(candidate_tokens)

            # Check if merged version is a valid word
            if (candidate.lower() in words and
                len(candidate) >= 4 and
                not all(t.lower() in words and len(t) > 2 for t in candidate_tokens)):

                # Preserve case from first token
                if current[0].isupper():
                    candidate = candidate[0].upper() + candidate[1:]

                result.append(candidate)
                i += lookahead + 1
                merged = True
                break

            # Special case: try splitting the last token (don't require valid base
            # since we'll check if the merged result is valid)
            if lookahead >= 1:
                last_token = candidate_tokens[-1]
                split_result = try_split_token(last_token, require_valid_base=False)
                if split_result:
                    base, remainder = split_result
                    # Try merging without the concatenated part
                    candidate_without = ''.join(candidate_tokens[:-1]) + base
                    if (candidate_without.lower() in words and
                        len(candidate_without) >= 4):

                        # Preserve case
                        if current[0].isupper():
                            candidate_without = candidate_without[0].upper() + candidate_without[1:]

                        result.append(candidate_without)
                        # Add the remainder (might be "and" or "returns and")
                        for part in remainder.split():
                            result.append(part)
                        i += lookahead + 1
                        merged = True
                        break

            # Special case: first token might have prefix from previous word split
            # e.g., in "ereturnsand", the "e" might be a suffix from previous word
            if lookahead == 0 and len(current) > 3:
                split_result = try_split_token(current)
                if split_result:
                    prefix, remainder = split_result
                    result.append(prefix)
                    result.append(remainder)
                    i += 1
                    merged = True
                    break

        if not merged:
            result.append(current)
            i += 1

    return ' '.join(result)


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
        # First, normalize all whitespace to single spaces
        text = ' '.join(text.split())

        # -----------------------------------------------------------------------
        # Quick win #1: Remove LEI codes + date patterns (XBRL context identifiers)
        # These appear at document start: "549300PPXHEU2JF0AM85 2024-01-01 2024-12-31"
        # -----------------------------------------------------------------------
        # Remove LEI code (20 alphanumeric chars) followed by optional date ranges
        text = re.sub(
            r'\b[A-Z0-9]{18,20}\b(?:\s+\d{4}-\d{2}-\d{2}(?:\s+\d{4}-\d{2}-\d{2})?)*',
            '',
            text
        )

        # Also remove standalone ISO date patterns that may remain
        text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', text)

        # Remove currency/unit codes from XBRL (iso4217:GBP, xbrli:shares)
        text = re.sub(r'\b(?:iso4217|xbrli):[A-Za-z]+\b', '', text)

        # Merge spelled-out words where each letter is separated by spaces
        # Example: "S T R A T E G I C" -> "STRATEGIC"
        def merge_spelled(match: re.Match) -> str:
            letters = re.findall(r"[A-Za-z]", match.group(0))
            return "".join(letters)

        text = re.sub(
            r"(?:\b[A-Za-z]\b(?:\s+|&nbsp;)+){2,}\b[A-Za-z]\b",
            merge_spelled,
            text,
        )

        # Merge short token runs when single-letter fragments appear
        # Example: "STR A T EGIC" -> "STRATEGIC"
        def merge_short_run(match: re.Match) -> str:
            tokens = match.group(0).split()
            if any(len(t) == 1 for t in tokens):
                return "".join(tokens)
            return match.group(0)

        text = re.sub(
            r"(?:\b[A-Za-z]{1,4}\b\s+){1,}\b[A-Za-z]{1,4}\b",
            merge_short_run,
            text,
        )

        # Remove XBRL namespace tokens (e.g., ifrs-full:RetainedEarningsMember)
        text = re.sub(
            r"\b[a-z0-9]{2,}(?:-[a-z0-9]{2,})?:[A-Za-z][A-Za-z0-9_-]*\b",
            "",
            text,
        )

        # Remove dotted leaders and trailing page numbers in TOC-like lines
        text = re.sub(r"\s*\.{3,}\s*\d+\b", "", text)

        # Fix digit spacing and numeric punctuation
        text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)
        text = re.sub(r"(\d)\s+,\s+(\d)", r"\1,\2", text)
        text = re.sub(r"(\d)\s+\.\s+(\d)", r"\1.\2", text)
        text = re.sub(r"(\d)\s+%","\\1%", text)
        text = re.sub(r"([£$€])\s+(\d)", r"\1\2", text)

        # Fix hyphenation across line breaks/spans
        text = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", text)

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

        # Insert word breaks on camel-case boundaries (e.g., "theStrategic" -> "the Strategic")
        text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

        # Remove Unicode replacement characters introduced by bad glyph decoding
        if "\uFFFD" in text:
            text = text.replace("\uFFFD", "")

        # Normalize whitespace
        text = ' '.join(text.split())

        # -----------------------------------------------------------------------
        # Step 1: Dictionary-based word repair (handles "cus to mers" -> "customers")
        # Run FIRST to merge fragments before splitting concatenations
        # -----------------------------------------------------------------------
        text = _repair_fragmented_words(text)

        # -----------------------------------------------------------------------
        # Step 2: Split concatenated words like "colleaguesand" -> "colleagues and"
        # -----------------------------------------------------------------------
        words_dict = _get_word_dict()

        def split_concatenated_with_dict(token: str) -> str:
            """Split concatenated words using dictionary validation."""
            if len(token) < 6:
                return token

            # If already a valid word, don't split
            if token.lower() in words_dict:
                return token

            token_lower = token.lower()

            # Safe connectors that rarely appear inside words
            safe_connectors = ["and", "the", "for", "our", "are", "has", "was",
                               "with", "from", "been", "this", "that", "will"]

            # Try safe connectors first (no dictionary check needed)
            for connector in safe_connectors:
                idx = token_lower.find(connector, 2)
                if idx != -1 and idx < len(token) - len(connector) - 1:
                    left = token[:idx]
                    right = token[idx + len(connector):]
                    if len(left) >= 2 and len(right) >= 2:
                        # Recursively process parts
                        left_split = split_concatenated_with_dict(left)
                        right_split = split_concatenated_with_dict(right)
                        return f"{left_split} {connector} {right_split}"

            # Try risky connectors only if they produce valid dictionary words
            risky_connectors = ["of", "to", "in", "on", "at", "as", "or", "by", "its"]
            for connector in risky_connectors:
                idx = token_lower.find(connector, 2)
                while idx != -1 and idx < len(token) - len(connector) - 1:
                    left = token[:idx]
                    right = token[idx + len(connector):]

                    # Only split if BOTH parts are valid words or can be further split
                    left_valid = left.lower() in words_dict or len(left) < 4
                    right_valid = right.lower() in words_dict

                    if len(left) >= 2 and len(right) >= 3 and left_valid and right_valid:
                        left_split = split_concatenated_with_dict(left)
                        right_split = split_concatenated_with_dict(right)
                        return f"{left_split} {connector} {right_split}"

                    idx = token_lower.find(connector, idx + 1)

            return token

        # Apply to tokens 7+ chars that aren't already valid words
        def apply_split(match: re.Match) -> str:
            token = match.group(0)
            if token.lower() in words_dict:
                return token
            return split_concatenated_with_dict(token)

        for _ in range(2):
            updated = re.sub(r"\b[a-zA-Z]{7,}\b", apply_split, text)
            if updated == text:
                break
            text = updated
            text = _repair_fragmented_words(text)

        # -----------------------------------------------------------------------
        # Step 3: Run dictionary repair again (splitting may have created new fragments)
        # -----------------------------------------------------------------------
        text = _repair_fragmented_words(text)

        # Final whitespace normalization
        text = ' '.join(text.split())

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
        spans = self._filter_repeated_spans(spans)
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

    def _filter_repeated_spans(self, spans: List[TextSpan]) -> List[TextSpan]:
        """Drop short lines that repeat across the document (headers/footers)."""
        def normalize(text: str) -> str:
            text = text.lower()
            text = re.sub(r"\d+", "", text)
            text = re.sub(r"[^\w\s]", " ", text)
            return " ".join(text.split())

        counts: Dict[str, int] = {}
        for span in spans:
            if span.is_heading:
                continue
            if len(span.text) < 10 or len(span.text) > 120:
                continue
            key = normalize(span.text)
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1

        filtered: List[TextSpan] = []
        for span in spans:
            if span.is_heading:
                filtered.append(span)
                continue
            if len(span.text) < 10 or len(span.text) > 120:
                filtered.append(span)
                continue
            key = normalize(span.text)
            if key and counts.get(key, 0) >= 8:
                continue
            filtered.append(span)

        return filtered

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
