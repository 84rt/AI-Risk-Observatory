"""iXBRL/XHTML text extraction for annual reports."""

import logging
import re
from dataclasses import dataclass, field
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
        # Place names commonly split in iXBRL (to prevent "Shangh ai" → AI false positive)
        "shanghai", "chairman", "curtain", "ertain", "maintain", "contain",
        "fountain", "mountain", "captain", "britain", "villain", "domain",
        "porcelain", "terrain", "campaign", "bargain", "complain", "explain",
        "obtain", "remain", "retain", "sustain", "attain", "detain", "strain",
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
_SYMSPELL = None


def _get_word_dict() -> Set[str]:
    """Get the word dictionary, loading it if necessary."""
    global _WORD_DICT
    if _WORD_DICT is None:
        _WORD_DICT = _load_word_dictionary()
    return _WORD_DICT


def _get_symspell():
    """Get a SymSpell instance if available, otherwise None."""
    global _SYMSPELL
    if _SYMSPELL is not None:
        return _SYMSPELL

    try:
        from symspellpy import SymSpell
    except Exception:
        _SYMSPELL = None
        return None

    symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    try:
        for word in _get_word_dict():
            symspell.create_dictionary_entry(word, 1)
    except Exception as exc:
        logger.warning(f"Failed to build SymSpell dictionary: {exc}")
        _SYMSPELL = None
        return None

    _SYMSPELL = symspell
    return _SYMSPELL


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


def _basic_normalize_text(text: str) -> str:
    """Apply conservative normalization without word repairs."""
    text = ' '.join(text.split())

    text = re.sub(
        r'\b[A-Z0-9]{18,20}\b(?:\s+\d{4}-\d{2}-\d{2}(?:\s+\d{4}-\d{2}-\d{2})?)*',
        '',
        text
    )
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', text)
    text = re.sub(r'\b(?:iso4217|xbrli):[A-Za-z]+\b', '', text)

    def merge_spelled(match: re.Match) -> str:
        letters = re.findall(r"[A-Za-z]", match.group(0))
        return "".join(letters)

    text = re.sub(
        r"(?:\b[A-Za-z]\b(?:\s+|&nbsp;)+){2,}\b[A-Za-z]\b",
        merge_spelled,
        text,
    )

    text = re.sub(
        r"\b[a-z0-9]{2,}(?:-[a-z0-9]{2,})?:[A-Za-z][A-Za-z0-9_-]*\b",
        "",
        text,
    )

    text = re.sub(r"\s*\.{3,}\s*\d+\b", "", text)
    text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)
    text = re.sub(r"(\d)\s+,\s+(\d)", r"\1,\2", text)
    text = re.sub(r"(\d)\s+\.\s+(\d)", r"\1.\2", text)
    text = re.sub(r"(\d)\s+%","\\1%", text)
    text = re.sub(r"([£$€])\s+(\d)", r"\1\2", text)
    text = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

    if "\uFFFD" in text:
        text = text.replace("\uFFFD", "")

    return ' '.join(text.split())


def _merge_single_letter_fragments(text: str) -> str:
    """Merge obvious single-letter fragments into adjacent words."""
    text = re.sub(
        r'(?<=\s)([b-hj-z])\s+(?!(?:and|the|for|with|to|of|in|on|at)\b)([a-z]{3,})',
        r'\1\2',
        text
    )

    text = re.sub(r'([a-z]{3,})\s+([b-hj-z])(?=\s|[,.;:\)]|$)', r'\1\2', text)

    return text


def _calculate_quality_metrics(text: str) -> Dict[str, float]:
    """Compute heuristic quality metrics for routing."""
    tokens = [t for t in text.split() if t]
    alpha_chars = sum(c.isalpha() for c in text)
    alpha_ratio = alpha_chars / max(1, len(text))
    single_letters = [t for t in tokens if len(t) == 1 and t.isalpha()]
    single_letter_rate = len(single_letters) / max(1, len(tokens))
    long_tokens = [t for t in tokens if len(t) >= 12 and t.isalpha()]
    long_token_rate = len(long_tokens) / max(1, len(tokens))
    space_ratio = text.count(" ") / max(1, len(text))

    return {
        "alpha_ratio": alpha_ratio,
        "single_letter_rate": single_letter_rate,
        "long_token_rate": long_token_rate,
        "space_ratio": space_ratio,
        "token_count": float(len(tokens)),
    }


def _count_unknown_tokens(text: str) -> int:
    """Count alphabetic tokens not found in the dictionary."""
    words = _get_word_dict()
    tokens = [t for t in text.split() if t.isalpha()]
    return sum(1 for t in tokens if t.lower() not in words)


def _count_suspicious_single_letters(text: str) -> int:
    """Count single-letter tokens excluding 'a' and 'i'."""
    tokens = text.split()
    return sum(
        1 for t in tokens
        if len(t) == 1 and t.isalpha() and t.lower() not in ("a", "i")
    )


def _repair_fragmented_words_with_symspell(
    text: str,
    *,
    per_sentence: bool = False,
    aggressive: bool = False,
) -> str:
    """Use SymSpell compound lookup to repair spaced fragments."""
    symspell = _get_symspell()
    if symspell is None:
        return text

    def apply_symspell(segment: str) -> str:
        try:
            suggestions = symspell.lookup_compound(segment, max_edit_distance=2)
        except Exception as exc:
            logger.warning(f"SymSpell lookup failed: {exc}")
            return segment

        if not suggestions:
            return segment

        candidate = suggestions[0].term
        if not candidate or candidate == segment:
            return segment

        unknown_before = _count_unknown_tokens(segment)
        unknown_after = _count_unknown_tokens(candidate)
        single_before = _count_suspicious_single_letters(segment)
        single_after = _count_suspicious_single_letters(candidate)

        if aggressive:
            if single_after < single_before and unknown_after <= unknown_before:
                return candidate
            return segment

        if unknown_after < unknown_before:
            return candidate
        if unknown_after == unknown_before and single_after < single_before:
            return candidate

        return segment

    if per_sentence:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(apply_symspell(part) for part in parts if part)

    return apply_symspell(text)


def _split_concatenated_with_wordninja(
    token: str,
    words_dict: Set[str],
    allowed_connectors: Set[str],
    *,
    aggressive: bool = False,
) -> Optional[str]:
    """Split concatenated words using wordninja with dictionary validation."""
    try:
        import wordninja
    except Exception:
        return None

    parts = wordninja.split(token)
    if len(parts) <= 1:
        return None

    validated = []
    for part in parts:
        part_lower = part.lower()
        if len(part) == 1 and part_lower not in ("a", "i"):
            return None
        if part_lower not in words_dict and part_lower not in allowed_connectors:
            if not aggressive or len(part) < 3 or not part.isalpha():
                return None
        validated.append(part)

    return " ".join(validated)


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
    raw_text: Optional[str] = None
    page_number: Optional[int] = None  # iXBRL doesn't have pages, but we keep for compatibility
    section: Optional[str] = None
    is_heading: bool = False
    font_size: Optional[float] = None
    quality: Dict[str, float] = field(default_factory=dict)
    repaired: bool = False


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
        self._hidden_stack = []
        self._hidden_depth = 0
        self.section_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in SECTION_PATTERNS
        ]

    def handle_starttag(self, tag, attrs):
        """Handle opening tags."""
        self.current_tag = tag.lower()
        tag_lower = self.current_tag
        attrs_dict = {k.lower(): v for k, v in attrs}
        
        if tag_lower in ['script', 'style']:
            if tag_lower == 'script':
                self.in_script = True
            elif tag_lower == 'style':
                self.in_style = True
            return

        is_hidden_tag = tag_lower in {
            'ix:header', 'ix:hidden', 'ix:resources',
            'xbrli:context', 'xbrli:unit', 'xbrli:entity',
        }
        style = (attrs_dict.get('style') or '').lower()
        is_hidden_style = (
            'display:none' in style or
            'display: none' in style or
            'visibility:hidden' in style or
            'visibility: hidden' in style
        )
        is_hidden_attr = (
            'hidden' in attrs_dict or
            (attrs_dict.get('aria-hidden') or '').lower() == 'true'
        )
        is_hidden = is_hidden_tag or is_hidden_style or is_hidden_attr
        self._hidden_stack.append(is_hidden)
        if is_hidden:
            self._hidden_depth += 1
            return

        # Check if this is a heading tag
        if tag_lower in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            # Save any accumulated text before the heading
            if self.current_text:
                text = self._join_text_segments()
                if text:
                    self._add_span(text, is_heading=False)
                self.current_text = []
            return

        if tag_lower == 'br':
            if self.current_text:
                text = self._join_text_segments()
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

        if self._hidden_stack:
            was_hidden = self._hidden_stack.pop()
            if was_hidden:
                self._hidden_depth = max(0, self._hidden_depth - 1)
                return

        # When we close a heading, save it as a heading span
        if tag_lower in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            if self.current_text:
                text = self._join_text_segments()
                if text:
                    self._add_span(text, is_heading=True)
                self.current_text = []
        # When we close block-level elements, save accumulated text
        elif tag_lower in ['p', 'div', 'li', 'td', 'th', 'section', 'article']:
            if self.current_text:
                text = self._join_text_segments()
                if text:
                    self._add_span(text, is_heading=False)
                self.current_text = []

    def handle_data(self, data):
        """Handle text data."""
        if self.in_script or self.in_style or self._hidden_depth > 0:
            return

        # Clean and accumulate text
        if data and data.strip():
            self.current_text.append(data)

    def _join_text_segments(self) -> str:
        """Join accumulated text segments with proper spacing.

        Ensures spaces between segments from different HTML elements
        while preserving document structure.
        """
        if not self.current_text:
            return ""
        stitched = self._stitch_adjacent_segments(self.current_text)
        return ' '.join(stitched)

    def _stitch_adjacent_segments(self, segments: List[str]) -> List[str]:
        """Merge adjacent segments that look like split words."""
        stitched: List[str] = []
        for segment in segments:
            part = segment.strip()
            if not part:
                continue
            if not stitched:
                stitched.append(part)
                continue

            prev = stitched[-1]
            if self._should_stitch(prev, part):
                stitched[-1] = prev + part
            else:
                stitched.append(part)

        return stitched

    def _should_stitch(self, left: str, right: str) -> bool:
        """Heuristic for merging two adjacent fragments."""
        if not left or not right:
            return False
        if left[-1].isspace() or right[0].isspace():
            return False
        if not left[-1].isalnum() or not right[0].isalnum():
            return False
        if left[-1].isdigit() and right[0].isdigit():
            return True

        if left[-1].isalpha() and right[0].isalpha():
            if len(left) <= 3 or len(right) <= 3:
                return True
            if left.islower() and right.islower() and len(right) <= 4:
                return True

        return False

    def _add_span(self, text: str, is_heading: bool):
        """Add a text span."""
        raw_text = unescape(text)
        section_text = _basic_normalize_text(raw_text)
        quality = _calculate_quality_metrics(raw_text)

        # Detect section from heading (or from any span that matches)
        section = None
        for pattern in self.section_patterns:
            if pattern.search(section_text.lower()):
                match = pattern.search(section_text.lower())
                if match:
                    section = match.group(0).replace(" ", "_")
                    # Mark as heading if it matches a section pattern
                    if not is_heading and len(section_text) < 100:
                        is_heading = True
                    break

        self.spans.append(TextSpan(
            text=raw_text,
            raw_text=raw_text,
            section=section,
            is_heading=is_heading,
            quality=quality,
        ))

    def _clean_text(self, text: str) -> str:
        """Clean up text spacing issues common in iXBRL.

        iXBRL files often have spaces between characters or within words.
        This method normalizes the spacing while preserving intentional word breaks.
        """
        text = _basic_normalize_text(text)

        # Merge short token runs when single-letter fragments appear
        # Example: "STR A T EGIC" -> "STRATEGIC"
        def merge_short_run(match: re.Match) -> str:
            tokens = match.group(0).split()
            single_letter_count = sum(1 for t in tokens if len(t) == 1)
            if single_letter_count < 2:
                return match.group(0)

            stopwords = {
                "a", "an", "and", "the", "for", "with", "to", "of", "in", "on",
                "at", "as", "or", "by", "is", "are", "was", "were", "be", "been",
                "being",
            }
            lower_tokens = [t.lower() for t in tokens if len(t) > 1]
            if any(t in stopwords for t in lower_tokens):
                return match.group(0)

            return "".join(tokens)

        text = re.sub(
            r"(?:\b[A-Za-z]{1,4}\b\s+){1,}\b[A-Za-z]{1,4}\b",
            merge_short_run,
            text,
        )

        # Common spacing issue in iXBRL: characters/syllables are split
        # "risk s" -> "risks"
        # "principal risk s" -> "principal risks"

        text = _merge_single_letter_fragments(text)

        # -----------------------------------------------------------------------
        # Step 1: Dictionary-based word repair (handles "cus to mers" -> "customers")
        # Run FIRST to merge fragments before splitting concatenations
        # -----------------------------------------------------------------------
        tokens = text.split()
        single_letters = [t for t in tokens if len(t) == 1 and t.isalpha()]
        suspicious_letters = [t for t in single_letters if t.lower() not in ('a', 'i')]
        has_spelled = re.search(r"(?:\b[A-Za-z]\b\s+){2,}\b[A-Za-z]\b", text)
        suspicious_ratio = len(suspicious_letters) / max(1, len(tokens))
        needs_repair = bool(has_spelled) or (suspicious_ratio > 0.05)
        if needs_repair:
            text = _repair_fragmented_words(text)
        if suspicious_ratio > 0.02:
            text = _repair_fragmented_words_with_symspell(
                text,
                per_sentence=True,
                aggressive=True,
            )
        elif needs_repair:
            text = _repair_fragmented_words_with_symspell(text)

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
            risky_connectors = ["of", "to", "in", "on", "at", "as", "or", "by", "its"]
            allowed_connectors = set(safe_connectors + risky_connectors)

            wordninja_split = _split_concatenated_with_wordninja(
                token,
                words_dict,
                allowed_connectors,
                aggressive=True,
            )
            if wordninja_split:
                return wordninja_split

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
            updated = re.sub(r"\b[a-zA-Z]{6,}\b", apply_split, text)
            if updated == text:
                break
            text = updated
            if needs_repair:
                text = _repair_fragmented_words(text)

        # -----------------------------------------------------------------------
        # Step 3: Run dictionary repair again (splitting may have created new fragments)
        # -----------------------------------------------------------------------
        if needs_repair:
            text = _repair_fragmented_words(text)

        # Final whitespace normalization
        text = ' '.join(text.split())

        return text

    def get_spans(self) -> List[TextSpan]:
        """Get all extracted spans."""
        # Flush any remaining text
        if self.current_text:
            text = self._join_text_segments()
            if text:
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
        spans = self._repair_spans(spans)

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

    def _repair_spans(self, spans: List[TextSpan]) -> List[TextSpan]:
        """Repair spans via the text repair service if available."""
        if not spans:
            return spans
        try:
            from .text_repair import TextRepairService
        except Exception as exc:
            logger.warning(f"TextRepairService unavailable: {exc}")
            return spans

        return TextRepairService().repair_spans(spans)

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
        if re.match(r'^[•\-–]\s', next_text):
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
