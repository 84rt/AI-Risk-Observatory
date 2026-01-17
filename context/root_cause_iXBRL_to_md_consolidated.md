# iXBRL → Markdown Conversion: Consolidated Root Cause Analysis & Implementation Guide

> **Purpose**: This document consolidates findings from 5 independent analyses of the iXBRL extraction pipeline. It serves as the definitive guide for implementing a robust fix.

---

## Executive Summary

The current iXBRL → markdown conversion produces artifacts (broken words, concatenated text, missing headings, metadata leakage) due to fundamental flaws in the extraction approach. The source files are **PDF2HTML conversions with embedded XBRL tagging**, not native text-based iXBRL. This requires a DOM-aware extraction strategy rather than the current HTMLParser + regex cleanup approach.

**Core Problem**: The pipeline introduces spacing errors during extraction, then attempts to fix them with aggressive regex/dictionary heuristics—which often creates new errors.

**Solution**: Replace HTMLParser with a DOM-based parser (lxml/BeautifulSoup), filter hidden elements structurally, and preserve original whitespace from the HTML source.

---

## Table of Contents

1. [Source File Format](#1-source-file-format)
2. [Root Causes (Ranked by Impact)](#2-root-causes-ranked-by-impact)
3. [Artifact Examples](#3-artifact-examples)
4. [Proposed Solution Architecture](#4-proposed-solution-architecture)
5. [Implementation Plan](#5-implementation-plan)
6. [Files to Modify](#6-files-to-modify)
7. [QA & Validation](#7-qa--validation)
8. [Risks & Edge Cases](#8-risks--edge-cases)

---

## 1. Source File Format

### Critical Discovery: Files Are PDF2HTML Conversions

The iXBRL files from `filings.xbrl.org` are **not native text-based iXBRL**. They are PDF annual reports converted to HTML via `pdf2htmlEX` (or similar), with XBRL tagging overlaid.

**Evidence** (from Johnson Matthey 2023):
```html
<div id="pf1" class="pf w0 h0" style="content-visibility: auto;">
  <div class="pc pc1 w0 h0">
    <img src="data:image/png;base64,..."/>  <!-- Page image -->
    <span class="_ _3">text</span>           <!-- CSS-positioned text overlay -->
    <span class="c x0 y2184 w29 h155">positioned text</span>
  </div>
</div>
```

**Implications**:
- Text is split into absolutely-positioned spans (character/syllable level)
- No natural word boundaries—spacing must be inferred from CSS positions
- Base64 images are embedded (should be ignored)
- Hidden `<ix:header>` blocks contain XBRL metadata

---

## 2. Root Causes (Ranked by Impact)

### RC-1: Whitespace Injection via `strip()` + `join(' ')` [CRITICAL]

**Location**: `ixbrl_extractor.py` lines 374-383, 343

**Current Code**:
```python
def handle_data(self, data):
    text = data.strip()          # Strips each text node
    if text:
        self.current_text.append(text)

# Later...
text = ' '.join(self.current_text).strip()  # Joins with space
```

**Problem**: When HTML has `<span>cus</span><span>to</span><span>mers</span>`, this produces `"cus to mers"` instead of `"customers"`.

**Impact**: This is the **root source** of all spacing artifacts. Everything downstream is damage control.

---

### RC-2: Aggressive Regex/Dictionary Cleanup Creates New Artifacts [HIGH]

**Location**: `ixbrl_extractor.py` lines 408-575

**Problematic Patterns**:

| Pattern | Purpose | Failure Mode |
|---------|---------|--------------|
| `merge_short_run` | Merge `S T R A T E G I C` → `STRATEGIC` | Also merges `"ever y par t of the"` → `"everypartofthe"` |
| `_repair_fragmented_words()` | Dictionary-based word repair | Corrupts proper nouns, acronyms, financial terms |
| Camel-case splitting | Insert space at `aA` | `"iXBRL"` → `"i XBRL"` |
| Connector splitting | Split `colleaguesand` | Creates false splits in valid words |

**Evidence**:
```python
# This regex merges ANY sequence of short tokens if one is single-letter
text = re.sub(
    r"(?:\b[A-Za-z]{1,4}\b\s+){1,}\b[A-Za-z]{1,4}\b",
    merge_short_run,
    text,
)
```

---

### RC-3: Hidden XBRL Metadata Not Filtered [HIGH]

**Location**: `ixbrl_extractor.py` lines 328-337

**Current Code** (only skips `script`/`style`):
```python
def handle_starttag(self, tag, attrs):
    if tag.lower() in ['script', 'style']:
        self.in_script = True  # etc.
```

**Missing Filters**:
- `<div style="display:none">` (contains `<ix:header>`)
- `<ix:header>`, `<ix:resources>`, `<ix:hidden>`
- `<xbrli:context>`, `<xbrli:unit>` blocks
- `#sidebar`, `#outline` navigation elements

**Artifact**: First line of output contains:
```
2138001AVBSD1HSC6Z10 2021-04-01 2022-03-31 iso4217:GBP ifrs-full:IssuedCapitalMember...
```

---

### RC-4: Block Flushing Logic Is Incomplete [MEDIUM]

**Location**: `ixbrl_extractor.py` lines 366-372

**Current Code**:
```python
elif tag_lower in ['p', 'div', 'li', 'td', 'th', 'section', 'article']:
    if self.current_text:
        text = ' '.join(self.current_text).strip()
        if text and len(text) > 10:  # Drops short content!
            self._add_span(text, is_heading=False)
```

**Problems**:
1. **`<br>` not handled** as line break
2. **Minimum 10-char filter** drops valid headings like "Risk", "AI", "Note"
3. **PDF2HTML has deeply nested divs**—flushing on every `</div>` fragments content

---

### RC-5: Heading Detection Too Narrow [MEDIUM]

**Location**: `ixbrl_extractor.py` lines 339-366, 727-738

**Current Logic**:
- Only recognizes `<h1>`–`<h6>` tags
- Falls back to regex matching section names

**Problem**: iXBRL headings are typically `<div>` or `<p>` with:
- CSS classes (`heading`, `title`, `section-header`)
- Inline styles (`font-weight: bold`, `font-size: 18pt`)
- Visual characteristics (ALL CAPS, short length)

---

### RC-6: iXBRL Continuation Chains Ignored [MEDIUM]

**Not Handled**: `<ix:nonNumeric continuedAt="...">` + `<ix:continuation id="...">`

**Problem**: Long narrative text in iXBRL is often split across multiple `<ix:continuation>` elements. Without resolving these chains, text appears fragmented or out of order.

---

### RC-7: Bullet Detection Regex Is Broken [LOW]

**Location**: `ixbrl_extractor.py` line 704

**Current Code**:
```python
if re.match(r'^[•\\-–]\\s', next_text):  # BUG: matches backslash literal
    return False
```

**Should Be**:
```python
if re.match(r'^[•\-–]\s', next_text):  # Correct: matches bullet chars
    return False
```

---

## 3. Artifact Examples

| Artifact | Raw HTML | Current Output | Expected |
|----------|----------|----------------|----------|
| Split word | `<span>cus</span><span>to</span><span>mers</span>` | `cus to mers` | `customers` |
| Merged words | Adjacent spans without space | `onClimate-related` | `on Climate-related` |
| Over-merged | Short tokens merged | `everypartofthe` | `every part of the` |
| Metadata leak | Hidden `<ix:header>` content | `2138001AVBSD1HSC6Z10 iso4217:GBP...` | (filtered) |
| Missing heading | `<div class="heading">Risk</div>` | (dropped or inline) | `## Risk` |
| Fragmented syllables | Per-character positioning | `cur ren tly` | `currently` |

---

## 4. Proposed Solution Architecture

### Two-Phase Extraction

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Structural Pre-processing                              │
│ ─────────────────────────────────────────────────────────────── │
│ • Parse with lxml/BeautifulSoup (DOM-aware)                     │
│ • Detect file format (PDF2HTML vs native iXBRL)                 │
│ • Remove hidden elements (display:none, ix:header, sidebar)     │
│ • Resolve ix:continuation chains                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Intelligent Text Extraction                            │
│ ─────────────────────────────────────────────────────────────── │
│ • For PDF2HTML: Parse CSS positions, merge by y-coordinate      │
│ • For native iXBRL: Walk DOM preserving block boundaries        │
│ • Detect headings via CSS/style analysis                        │
│ • Minimal whitespace normalization (no dictionary heuristics)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Output: Clean TextSpan list → Markdown                          │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Fix at the source**: Don't inject spacing errors, then repair—preserve original structure
2. **DOM over regex**: Use structural parsing, not text pattern matching
3. **Format detection**: PDF2HTML and native iXBRL need different strategies
4. **Minimal normalization**: Only collapse whitespace, don't guess word boundaries
5. **Preserve structure**: Tables → table format, lists → bullets, headings → markdown `##`

---

## 5. Implementation Plan

### Step 1: Add Dependencies

```bash
# Add to pipeline/requirements.txt
beautifulsoup4>=4.12.0
lxml>=5.0.0
```

### Step 2: Create New Extractor Class

**File**: `pipeline/src/ixbrl_extractor.py`

```python
from bs4 import BeautifulSoup, NavigableString
from typing import List, Dict, Optional
import re

class iXBRLExtractorV2:
    """DOM-based iXBRL extractor replacing HTMLParser approach."""
    
    HIDDEN_SELECTORS = [
        'script', 'style', 'noscript',
        '[style*="display:none"]',
        '[style*="display: none"]',
        '[style*="visibility:hidden"]',
        'ix\\:header',
        'ix\\:hidden', 
        'ix\\:resources',
        '#sidebar',
        '#outline',
    ]
    
    BLOCK_TAGS = {'p', 'div', 'section', 'article', 'li', 'tr', 'td', 'th', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
    
    def extract_report(self, file_path: Path) -> ExtractedReport:
        html = self._read_file(file_path)
        soup = BeautifulSoup(html, 'lxml')
        
        # Phase 1: Structural cleanup
        self._remove_hidden_elements(soup)
        self._resolve_continuations(soup)
        
        # Phase 2: Format-specific extraction
        file_format = self._detect_format(soup)
        
        if file_format == 'pdf2html':
            spans = self._extract_pdf2html(soup)
        else:
            spans = self._extract_native_ixbrl(soup)
        
        return ExtractedReport(spans=spans, ...)
    
    def _remove_hidden_elements(self, soup: BeautifulSoup) -> None:
        """Remove all hidden/metadata elements from DOM."""
        for selector in self.HIDDEN_SELECTORS:
            for elem in soup.select(selector):
                elem.decompose()
    
    def _detect_format(self, soup: BeautifulSoup) -> str:
        """Detect PDF2HTML vs native iXBRL format."""
        if soup.select('.pf.w0.h0'):
            return 'pdf2html'
        return 'native_ixbrl'
    
    def _extract_pdf2html(self, soup: BeautifulSoup) -> List[TextSpan]:
        """Extract text from PDF2HTML format using CSS positions."""
        pages = soup.select('.pf') or [soup.body]
        all_spans = []
        
        for page in pages:
            # Collect positioned text elements
            elements = self._collect_positioned_elements(page)
            # Sort by position (top-to-bottom, left-to-right)
            elements.sort(key=lambda e: (e['y'], e['x']))
            # Merge into lines based on y-coordinate
            lines = self._merge_into_lines(elements)
            # Convert to TextSpans
            all_spans.extend(self._lines_to_spans(lines))
        
        return all_spans
    
    def _extract_native_ixbrl(self, soup: BeautifulSoup) -> List[TextSpan]:
        """Extract text from native iXBRL, respecting block structure."""
        spans = []
        current_text = []
        
        for node in soup.body.descendants:
            if isinstance(node, NavigableString):
                text = str(node)
                if text.strip():
                    current_text.append(text)
            elif node.name in self.BLOCK_TAGS:
                # Flush accumulated text
                if current_text:
                    merged = ''.join(current_text)
                    normalized = ' '.join(merged.split())  # Minimal cleanup
                    if normalized:
                        is_heading = self._is_heading(node, normalized)
                        spans.append(TextSpan(text=normalized, is_heading=is_heading))
                    current_text = []
        
        return spans
```

### Step 3: Position-Aware Line Merging (for PDF2HTML)

```python
def _collect_positioned_elements(self, page) -> List[Dict]:
    """Extract text with CSS position info."""
    elements = []
    for elem in page.descendants:
        if isinstance(elem, NavigableString) and elem.strip():
            parent = elem.parent
            pos = self._parse_position_class(parent.get('class', []))
            elements.append({
                'text': str(elem),
                'x': pos.get('x', 0),
                'y': pos.get('y', 0),
            })
    return elements

def _parse_position_class(self, classes: List[str]) -> Dict[str, int]:
    """Extract x/y position from class like 'c x100 y200'."""
    pos = {'x': 0, 'y': 0}
    for cls in classes:
        if cls.startswith('x') and cls[1:].isdigit():
            pos['x'] = int(cls[1:], 16)  # Hex encoding common in pdf2htmlEX
        elif cls.startswith('y') and cls[1:].isdigit():
            pos['y'] = int(cls[1:], 16)
    return pos

def _merge_into_lines(self, elements: List[Dict], y_tolerance: int = 5) -> List[str]:
    """Merge elements into lines based on y-coordinate proximity."""
    if not elements:
        return []
    
    lines = []
    current_line = [elements[0]]
    current_y = elements[0]['y']
    
    for elem in elements[1:]:
        if abs(elem['y'] - current_y) <= y_tolerance:
            # Same line - check for word gap
            if current_line:
                x_gap = elem['x'] - current_line[-1]['x']
                if x_gap > 15:  # Significant gap = word boundary
                    current_line.append({'text': ' ', 'x': 0, 'y': 0})
            current_line.append(elem)
        else:
            # New line
            lines.append(''.join(e['text'] for e in current_line))
            current_line = [elem]
            current_y = elem['y']
    
    if current_line:
        lines.append(''.join(e['text'] for e in current_line))
    
    return lines
```

### Step 4: Heading Detection via CSS/Style

```python
def _is_heading(self, element, text: str) -> bool:
    """Detect headings using visual/structural cues."""
    if element.name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
        return True
    
    # Check inline style
    style = element.get('style', '').lower()
    if 'font-weight' in style and ('bold' in style or '700' in style):
        if len(text) < 100:
            return True
    if 'font-size' in style:
        # Extract size, compare to threshold
        match = re.search(r'font-size:\s*(\d+)', style)
        if match and int(match.group(1)) >= 14:
            if len(text) < 100:
                return True
    
    # Check class names
    classes = ' '.join(element.get('class', [])).lower()
    if any(kw in classes for kw in ('heading', 'title', 'header', 'section')):
        return True
    
    # Heuristic: short ALL CAPS text
    if len(text) < 60 and text.isupper() and len(text.split()) <= 8:
        return True
    
    return False
```

### Step 5: Continuation Chain Resolution

```python
def _resolve_continuations(self, soup: BeautifulSoup) -> None:
    """Resolve ix:continuation chains by appending text to source elements."""
    continuations = {}
    
    # Collect all continuations
    for cont in soup.select('ix\\:continuation'):
        cont_id = cont.get('id')
        if cont_id:
            continuations[cont_id] = cont.get_text()
    
    # Attach to source elements
    for elem in soup.select('[continuedAt]'):
        cont_id = elem.get('continuedAt')
        if cont_id in continuations:
            # Append continuation text
            elem.append(continuations[cont_id])
    
    # Remove standalone continuation elements
    for cont in soup.select('ix\\:continuation'):
        cont.decompose()
```

---

## 6. Files to Modify

| File | Changes |
|------|---------|
| `pipeline/src/ixbrl_extractor.py` | Add `iXBRLExtractorV2` class; keep `iXBRLExtractor` for fallback |
| `pipeline/src/ixbrl_dom.py` (new) | Helper functions: `is_hidden()`, `parse_position()`, `is_block()` |
| `pipeline/src/preprocessor.py` | Simplify `_to_markdown()`—less cleanup needed |
| `pipeline/scripts/golden_set_phase1.py` | Use new extractor class |
| `pipeline/requirements.txt` | Add `beautifulsoup4`, `lxml` |

---

## 7. QA & Validation

### New Unit Tests

**File**: `pipeline/tests/qa_suite/test_ixbrl_extraction_v2.py`

```python
import pytest
import re
from src.ixbrl_extractor import iXBRLExtractorV2

class TestExtractionQuality:
    
    def test_no_xbrl_metadata_leakage(self, sample_ixbrl):
        """Verify hidden XBRL metadata is filtered."""
        report = iXBRLExtractorV2().extract_report(sample_ixbrl)
        text = report.full_text
        
        # No LEI codes (20 alphanumeric chars)
        assert not re.search(r'\b[A-Z0-9]{20}\b', text)
        # No namespace prefixes
        assert 'ifrs-full:' not in text
        assert 'iso4217:' not in text
        assert 'xbrli:' not in text
    
    def test_word_fragmentation_rate(self, sample_ixbrl):
        """Check single-letter token ratio is acceptable."""
        report = iXBRLExtractorV2().extract_report(sample_ixbrl)
        words = report.full_text.split()
        
        single_letters = [w for w in words if len(w) == 1 and w.isalpha()]
        # Exclude legitimate single letters (a, I)
        suspicious = [w for w in single_letters if w.lower() not in 'ai']
        rate = len(suspicious) / max(1, len(words))
        
        assert rate < 0.02, f"Too many single-letter tokens: {rate:.2%}"
    
    def test_word_concatenation_rate(self, sample_ixbrl):
        """Check for common concatenation artifacts."""
        report = iXBRLExtractorV2().extract_report(sample_ixbrl)
        text = report.full_text.lower()
        
        concat_patterns = [
            r'\b\w{4,}andthe\b',
            r'\b\w{4,}tothe\b',
            r'\btheir\w{6,}\b',
        ]
        
        matches = sum(len(re.findall(p, text)) for p in concat_patterns)
        word_count = len(text.split())
        rate = matches / max(1, word_count)
        
        assert rate < 0.001, f"Too many concatenation artifacts: {rate:.4%}"
    
    def test_headings_detected(self, sample_ixbrl):
        """Verify reasonable number of headings found."""
        report = iXBRLExtractorV2().extract_report(sample_ixbrl)
        headings = [s for s in report.spans if s.is_heading]
        
        assert len(headings) >= 10, f"Too few headings detected: {len(headings)}"
    
    def test_short_headings_preserved(self, sample_ixbrl):
        """Verify short headings like 'Risk' are not dropped."""
        report = iXBRLExtractorV2().extract_report(sample_ixbrl)
        headings = [s.text.lower() for s in report.spans if s.is_heading]
        
        # At least one short heading should exist
        short_headings = [h for h in headings if len(h) < 15]
        assert len(short_headings) > 0, "No short headings found—min length filter may be too aggressive"
```

### Validation Workflow

1. **A/B Comparison**: Run old vs new extractor on same files
2. **Spot Checks**: Verify known artifacts are fixed:
   - No LEI codes in output
   - "on Climate-related" has space
   - "currently" not split
3. **Metrics**: Compare word count, heading count, sentence length distribution
4. **Regression**: Ensure no substantive content is lost

---

## 8. Risks & Edge Cases

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| **Different PDF2HTML versions** | Medium | Format detection + fallback to generic DOM extraction |
| **Native iXBRL (rare)** | Low | Separate extraction path for non-PDF2HTML files |
| **External CSS (font-size in stylesheet)** | Medium | Parse `<style>` blocks; fallback to text heuristics |
| **`ix:continuation` ordering** | Low | Resolve chains before extraction |
| **Large files (30MB+)** | Medium | Skip base64 image data early; stream processing |
| **Encoding issues** | Low | Use `errors='replace'`; detect charset from XML declaration |
| **Tables losing structure** | Medium | Emit cells with separators; consider markdown table format |

---

## Implementation Sequence

1. ☐ Add `beautifulsoup4`, `lxml` to `requirements.txt`
2. ☐ Create `iXBRLExtractorV2` class in `ixbrl_extractor.py`
3. ☐ Implement hidden element filtering
4. ☐ Implement format detection (PDF2HTML vs native)
5. ☐ Implement position-aware span merging for PDF2HTML
6. ☐ Implement continuation chain resolution
7. ☐ Implement heading detection via CSS/style
8. ☐ Add QA tests
9. ☐ Update `golden_set_phase1.py` to use V2 extractor
10. ☐ Run validation on golden set
11. ☐ Deprecate V1 extractor after confirmation

---

## Appendix: Quick Reference

### Artifact → Root Cause → Fix

| Artifact | Root Cause | Fix |
|----------|------------|-----|
| `cus to mers` | RC-1: strip+join | Don't inject spaces between inline spans |
| `onClimate` | RC-1: no space between spans | Add space based on CSS position gap |
| `everypartofthe` | RC-2: merge_short_run | Remove aggressive regex merging |
| LEI codes in output | RC-3: no hidden filter | Filter `display:none`, `ix:header` |
| "Risk" heading dropped | RC-4: min 10 chars | Remove or lower minimum length filter |
| Missing headings | RC-5: only h1-h6 | Detect via CSS classes/styles |
| `cur ren tly` | PDF2HTML syllable spans | Position-aware merging |

---

*Document Version: 1.0*  
*Last Updated: 2026-01-17*  
*Contributors: Consolidated from 5 independent analyses*
