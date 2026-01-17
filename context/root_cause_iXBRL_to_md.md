# Grok Summary
I've conducted a comprehensive investigation of the iXBRL → markdown conversion artifacts and identified the root causes with concrete evidence. The issues stem primarily from inadequate handling of syllable-level and character-level spacing artifacts in iXBRL files.
Key Findings:
Primary Issue: iXBRL files contain text where spaces are inserted between syllables/characters (e.g., "Llo yds", "re sults", "mana gement")
Secondary Issues: Fragmented word repair logic is too conservative, missing many real-world patterns
Evidence: Concrete examples from processed Lloyd's Banking Group report showing systematic spacing errors
Solution:
A robust, deterministic text reconstruction approach using:
Aggressive syllable-aware merging for patterns like "man ag ement" → "management"
Enhanced fragmented word repair with better heuristics
Multi-pass cleaning pipeline with validation
Pattern-based reconstruction for common iXBRL artifacts
Implementation:
New module: ixbrl_text_reconstructor.py with specialized reconstruction logic
Enhanced cleaning: Improved _clean_text() method with better pattern recognition
Quality assurance: New QA tests and validation metrics
Backward compatibility: Preserves existing functionality while fixing artifacts
The solution is practical, deterministic, and specifically designed to handle the heterogeneous nature of iXBRL sources without losing substantive content or requiring per-company tuning.

# GPT-5.2 Codex

Here’s the deep-dive based on the current code path and extractor behavior. I could not access /data/raw or /data/processed in this worktree (the data/ directory is absent), so I can’t show raw‑vs‑markdown diffs directly yet. I’ve still traced the full pipeline and identified concrete, code‑level causes of the artifacts, plus a deterministic, repo‑appropriate fix with a validation plan. If you want me to run the raw/processed comparisons, drop the data/ tree into this worktree or point me to its location and I’ll produce the evidence.
Current Pipeline (end‑to‑end)
scripts/golden_set_phase1.py runs download → extraction → preprocessing → markdown save. For iXBRL, it uses iXBRLExtractor.extract_report, then Preprocessor._to_markdown for output.
(See pipeline/scripts/golden_set_phase1.py and pipeline/src/preprocessor.py.)
Extraction flow: iXBRLParser (HTMLParser) accumulates text nodes → _clean_text → TextSpan list → _rebuild_paragraphs merges spans heuristically → Preprocessor turns spans into markdown.
The extraction logic is in pipeline/src/ixbrl_extractor.py.
Root cause analysis (code‑level, with evidence)
1) Whitespace insertion before cleaning creates artifacts
handle_data strips every text node and then joins nodes with spaces, which inserts spaces that never existed in the HTML, especially when iXBRL uses per‑character or per‑syllable spans.
This is the direct source of “r i s k s”‑style breaks that are later “fixed” with heuristics.
374:408:pipeline/src/ixbrl_extractor.py
    def handle_data(self, data):
        ...
        text = data.strip()
        if text:
            self.current_text.append(text)
...
    text = ' '.join(self.current_text).strip()
374:408:pipeline/src/ixbrl_extractor.py    def handle_data(self, data):        ...        text = data.strip()        if text:            self.current_text.append(text)...    text = ' '.join(self.current_text).strip()
2) Aggressive cleanup introduces false merges/splits
_clean_text performs broad regex merges, dictionary‑based repairs, and concatenation splitting that can corrupt correct words or merge across token boundaries.
408:574:pipeline/src/ixbrl_extractor.py
 Merge short token runs when single-letter fragments appear
    text = re.sub(
        r"(?:\b[A-Za-z]{1,4}\b\s+){1,}\b[A-Za-z]{1,4}\b",
        merge_short_run,
        text,
    )
...
    text = _repair_fragmented_words(text)
...
    text = re.sub(r"\b[a-zA-Z]{7,}\b", apply_split, text)
408:574:pipeline/src/ixbrl_extractor.py    # Merge short token runs when single-letter fragments appear    text = re.sub(        r"(?:\b[A-Za-z]{1,4}\b\s+){1,}\b[A-Za-z]{1,4}\b",        merge_short_run,        text,    )...    text = _repair_fragmented_words(text)...    text = re.sub(r"\b[a-zA-Z]{7,}\b", apply_split, text)
This design is compensating for artifacts introduced earlier by strip() + ' '.join() rather than preserving the original spacing from the HTML.
3) Block boundaries are incomplete (no <br>, limited tags)
Text is flushed only on a narrow set of tags, and <br> isn’t treated as a line break. This causes run‑ons, broken list structure, and malformed headings.
366:372:pipeline/src/ixbrl_extractor.py
elif tag_lower in ['p', 'div', 'li', 'td', 'th', 'section', 'article']:
    ...
366:372:pipeline/src/ixbrl_extractor.pyelif tag_lower in ['p', 'div', 'li', 'td', 'th', 'section', 'article']:    ...
4) Minimum‑length filter drops short but meaningful spans
Any block shorter than 10 chars is discarded, which removes headings like “Risk”, “Note”, “AI” or short list items and breaks continuity.
369:371:pipeline/src/ixbrl_extractor.py
if text and len(text) > 10:
    self._add_span(...)
369:371:pipeline/src/ixbrl_extractor.pyif text and len(text) > 10:    self._add_span(...)
5) Heuristic paragraph reconstruction merges unrelated blocks
_rebuild_paragraphs merges spans based on surface heuristics and length, which can combine table rows, list items, and multi‑column text into a single paragraph.
655:725:pipeline/src/ixbrl_extractor.py
if self._should_join(prev_text, text):
    buffer_parts[-1] = self._merge_fragments(prev_text, text)
655:725:pipeline/src/ixbrl_extractor.pyif self._should_join(prev_text, text):    buffer_parts[-1] = self._merge_fragments(prev_text, text)
6) Heading detection is too narrow
Headings are only recognized via <h1>–<h6> tags or regex matches inside any span. iXBRL headings are often div/p with CSS classes or inline styles, so many headings are missed or misclassified.
339:366:pipeline/src/ixbrl_extractor.py
if tag.lower() in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
...
339:366:pipeline/src/ixbrl_extractor.pyif tag.lower() in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:...
7) iXBRL‑specific structure is ignored
iXBRL has <ix:nonNumeric> and continuedAt chains; without resolving these, text is split or reordered. The current parser is unaware of ix: namespaces and ignores continuation structure entirely.
Proposed robust, deterministic fix
A. Replace HTMLParser with DOM‑aware extraction (lxml)
Why: HTMLParser loses structure and forces spacing fixes.
Fix: Use lxml.html or lxml.etree to parse XHTML + namespaces; walk the DOM to preserve block/inline structure deterministically.
New module: pipeline/src/ixbrl_dom.py
Key features:
Parse with lxml (recover=True) and namespace awareness.
Build a block model: block tags + CSS display:block + <br> → explicit line breaks.
Inline text nodes are concatenated without injecting extra spaces unless the source contains them.
B. Resolve iXBRL continuations and exclusions
Why: iXBRL splits narrative across ix:continuation.
Fix: Collect <ix:continuation id="..."> and attach to <ix:nonNumeric continuedAt="..."> in order. Skip nodes in ix:hidden or style="display:none".
C. Token repair becomes structural, not dictionary‑guessing
Why: dictionary‑based merge/split creates new errors.
Fix: Remove or heavily restrict _repair_fragmented_words. Instead:
Merge sequences of single‑character nodes only when no whitespace exists in the HTML and the original text nodes were adjacent.
Keep whitespace exactly as represented by the HTML text/tail.
Optionally keep a minimal fix for hyphenation at explicit line breaks.
D. Heading detection based on DOM + style hints
Why: headings are typically styled, not tagged.
Fix: infer heading spans based on:
class names matching heading|title|section|subhead
inline style (font-size, font-weight)
short text in all‑caps or title‑case
location in hierarchy (e.g., direct child of body or div.reportSection)
Add font_size into TextSpan when found to drive markdown levels.
E. Replace heuristic paragraph rebuilding with block‑level grouping
Why: _rebuild_paragraphs merges across unrelated spans.
Fix: Use the block model directly. Each block becomes a span; no secondary heuristics needed.
F. Preserve list/table structure deterministically
<li> becomes a bullet line (- ...).
<table> rows become structured lines (e.g., col1 | col2 | col3) or kept as separate paragraphs with cell separators.
This avoids word smearing across columns.
Implementation plan (repo‑specific)
Files to change / add
pipeline/src/ixbrl_extractor.py
Replace HTMLParser pipeline with a DOM‑based extractor.
Deprecate _clean_text or reduce to minimal normalization.
Remove _rebuild_paragraphs or make it a no‑op.
pipeline/src/ixbrl_dom.py (new)
parse_ixbrl_html(bytes) -> lxml_root
extract_blocks(root) -> List[Block]
resolve_ixbrl_continuations(root) -> Dict[id, text]
normalize_inline_text(text, was_adjacent, had_whitespace) -> str
pipeline/src/preprocessor.py
Use new span.font_size and span.is_heading (computed from DOM).
Reduce redundant whitespace normalization (" ".join(span.text.split())) to avoid re‑introducing artifacts.
Tests / QA additions
New fixtures under pipeline/tests/fixtures/ixbrl_samples/
Mini iXBRL snippets with:
per‑letter spans
continuation chains
<br> line breaks
tables + lists
New tests
test_ixbrl_continuations.py: verifies continuation merging.
test_ixbrl_block_boundaries.py: verifies <br>, <p>, <li> boundaries preserved.
Extend test_ixbrl_spacing.py to ensure no dictionary‑based “over‑repair”.
Validation plan (real data)
A/B extraction on 4–6 companies across two years (different layouts).
Compare:
word count, heading count
ratio of single‑letter tokens
average sentence length and run‑on ratio
Diff sampling: per‑document, sample 10 random paragraphs and compare raw HTML region → extracted markdown.
QA gating: run scripts/qa_preprocessing.py and ensure fewer many_long_sentences / few_headings.
Why the fix resolves the root causes
It removes the root source of spacing artifacts: injecting spaces via strip() + ' '.join() on text nodes.
It preserves actual HTML structure, so there’s no need for dictionary repair to undo damage.
It respects iXBRL’s continuation logic, preventing broken words/sections.
It uses deterministic block boundaries rather than heuristic paragraph merges, reducing run‑ons and malformed headings.
Risks / edge cases
External CSS stylesheets: heading inference from class names may miss font sizes defined externally. Mitigation: use heuristic heading detection based on text shape and DOM location.
Layout tables: preserving tables could keep noise; consider skipping tables with mostly numeric cells or no alpha tokens.
Inline display styles: iXBRL can be inconsistent; fallback to block tag list + <br> + list items should still improve.
What I need to complete the “evidence” deliverable
I need access to data/raw/ixbrl/... and data/processed/... in this worktree to extract real snippets and show the raw→markdown mismatches for multiple companies/years. Once the data is available, I’ll run a focused comparison and include concrete excerpts.
If you want, I can start implementing the DOM‑based extractor and tests now, or first build a one‑off diagnostic script to produce the raw/markdown diffs for the evidence section

# My Analysis

I reviewed the iXBRL extractor + preprocessor flow and QA checks in this worktree. I wasn’t able to locate any actual raw/processed data under data/raw or data/processed in this worktree, so I can’t yet do side‑by‑side raw‑to‑markdown comparisons on real filings. Everything below is grounded in the current code and QA harness, plus a concrete plan to generate the raw/markdown evidence once data is present.
Pipeline Overview (raw → spans → markdown)
Download + preprocess for the golden set runs through pipeline/scripts/golden_set_phase1.py, which calls iXBRLExtractor.extract_report(...) and then Preprocessor.process(...) to emit markdown in data/processed/<run_id>/documents/*.md.
The core conversion is in pipeline/src/ixbrl_extractor.py (HTMLParser extraction, aggressive text cleanup, paragraph rebuild) and pipeline/src/preprocessor.py (heading selection + markdown formatting).
Root Cause Analysis (with evidence in code)
1) HTML parsing strips and re‑spaces text nodes
The extractor uses HTMLParser and strips each text node before concatenation, then joins with spaces. This is a classic source of broken words and odd spacing when iXBRL splits words across inline spans or uses letter‑spaced spans.
ixbrl_extractor.py
Lines 374-383
    def handle_data(self, data):
        """Handle text data."""
        if self.in_script or self.in_style:
            return

        # Clean and accumulate text
        text = data.strip()
        if text:
            self.current_text.append(text)
    def handle_data(self, data):        """Handle text data."""        if self.in_script or self.in_style:            return        # Clean and accumulate text        text = data.strip()        if text:            self.current_text.append(text)
Impact: If the source uses <span>cus</span><span>to</span><span>mers</span>, stripping + space‑joining yields cus to mers before cleanup. That cascades into regex/dictionary repairs that sometimes over‑correct.
2) Aggressive regex + dictionary “repair” can introduce artifacts
_clean_text applies multiple normalization passes (merging, splitting, camel‑case, dict repair). These are strong heuristics that can fix some artifacts but also create new ones when content is non‑standard (tables, abbreviations, entity names, financial terms).
ixbrl_extractor.py
Lines 408-575
    def _clean_text(self, text: str) -> str:
        """Clean up text spacing issues common in iXBRL."""
        # normalize whitespace
        text = ' '.join(text.split())
        ...
        # Merge spelled-out letters
        text = re.sub(r"(?:\b[A-Za-z]\b(?:\s+|&nbsp;)+){2,}\b[A-Za-z]\b", merge_spelled, text)
        ...
        # Merge short token runs if any 1-letter token exists
        text = re.sub(r"(?:\b[A-Za-z]{1,4}\b\s+){1,}\b[A-Za-z]{1,4}\b", merge_short_run, text)
        ...
        # Insert word breaks on camel-case
        text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
        ...
        # Dictionary merge/split (two passes)
        text = _repair_fragmented_words(text)
        text = re.sub(r"\b[a-zA-Z]{7,}\b", apply_split, text)
        text = _repair_fragmented_words(text)
    def _clean_text(self, text: str) -> str:        """Clean up text spacing issues common in iXBRL."""        # normalize whitespace        text = ' '.join(text.split())        ...        # Merge spelled-out letters        text = re.sub(r"(?:\b[A-Za-z]\b(?:\s+|&nbsp;)+){2,}\b[A-Za-z]\b", merge_spelled, text)        ...        # Merge short token runs if any 1-letter token exists        text = re.sub(r"(?:\b[A-Za-z]{1,4}\b\s+){1,}\b[A-Za-z]{1,4}\b", merge_short_run, text)        ...        # Insert word breaks on camel-case        text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)        ...        # Dictionary merge/split (two passes)        text = _repair_fragmented_words(text)        text = re.sub(r"\b[a-zA-Z]{7,}\b", apply_split, text)        text = _repair_fragmented_words(text)
Impact:
A run like in Q 3 results can be incorrectly merged into inQ3results (short‑run merge).
Company names, product codes, and acronyms can be split/merged incorrectly.
Camel‑case splitting can degrade iXBRL‑specific tokens (iXBRL → i XBRL).
This explains “broken words” and inconsistent quality across companies.
3) Block/paragraph reconstruction is too heuristic and sometimes wrong
Paragraph rebuild merges spans based on punctuation and length rather than DOM structure. That can join unrelated lines (especially tables) or split continuous paragraphs.
ixbrl_extractor.py
Lines 655-725
    def _rebuild_paragraphs(self, spans: List[TextSpan]) -> List[TextSpan]:
        ...
        if self._should_join(prev_text, text):
            buffer_parts[-1] = self._merge_fragments(prev_text, text)
        else:
            self._flush_paragraph(...)
    def _rebuild_paragraphs(self, spans: List[TextSpan]) -> List[TextSpan]:        ...        if self._should_join(prev_text, text):            buffer_parts[-1] = self._merge_fragments(prev_text, text)        else:            self._flush_paragraph(...)
ixbrl_extractor.py
Lines 701-721
    def _should_join(self, prev_text: str, next_text: str) -> bool:
        if self._looks_like_heading(next_text):
            return False
        if re.match(r'^[•\\-–]\\s', next_text):
            return False
        ...
        if re.match(r'^[a-z0-9(]', next_text):
            return True
    def _should_join(self, prev_text: str, next_text: str) -> bool:        if self._looks_like_heading(next_text):            return False        if re.match(r'^[•\\-–]\\s', next_text):            return False        ...        if re.match(r'^[a-z0-9(]', next_text):            return True
Impact:
Bullet detection regex is wrong (r'^[•\\-–]\s' matches a backslash literal).
Lines from table rows/cells are merged into prose.
Short lines are dropped by the “min length > 10” filter at flush time.
4) Missing visibility and iXBRL‑specific structures
The parser ignores visibility and iXBRL constructs:
No suppression of hidden text (display:none, ix:hidden, etc.)
No handling for ix:continuation (split facts)
No explicit handling of ix:header, ix:resources, xbrli:* blocks
This allows hidden or non‑human‑readable text into output and can reorder or omit fact text.
5) Headings are often missed or misclassified
Headings are detected only via <h1..h6> or heuristics; iXBRL rarely uses heading tags. The parser also auto‑marks any short line that matches a section regex as a heading, even if it’s inline text.
ixbrl_extractor.py
Lines 339-406
        if tag.lower() in ['h1', 'h2', ...]:
            ...
    def _add_span(self, text: str, is_heading: bool):
        ...
        for pattern in self.section_patterns:
            if pattern.search(cleaned_text.lower()):
                ...
                if not is_heading and len(cleaned_text) < 100:
                    is_heading = True
        if tag.lower() in ['h1', 'h2', ...]:            ...    def _add_span(self, text: str, is_heading: bool):        ...        for pattern in self.section_patterns:            if pattern.search(cleaned_text.lower()):                ...                if not is_heading and len(cleaned_text) < 100:                    is_heading = True
Impact: Inconsistent heading structure in markdown; headings are missing for some filings and falsely inserted for others.
Proposed Robust Fix (deterministic, generalizable)
Core extraction changes (replace HTMLParser path)
1) Use a DOM parser (lxml or BeautifulSoup with lxml backend)
Parse as XHTML, recover malformed markup.
Traverse DOM nodes in order.
2) Visibility filtering (deterministic)
Skip nodes with inline style display:none, visibility:hidden, font-size:0, color:transparent.
Skip within ix:hidden, ix:header, ix:resources, xbrli:context, xbrli:unit.
3) Block/line segmentation by DOM, not heuristics
Treat p, div, section, article, li, tr, td, th, br as structural boundaries.
Preserve line breaks from <br> and table rows.
4) Text normalization: minimal, targeted
Convert NBSP to spaces.
Fix letter‑spaced headings only when a line is mostly single letters or all‑caps short heading; avoid dictionary heuristics by default.
Hyphenation repair only on line breaks with lowercase continuation.
Avoid dictionary‑based merging/splitting unless a per‑document heuristic indicates high split‑token ratio.
5) Heading detection using visual clues
Extract font-size from inline styles; compute quantiles and mark top sizes as headings.
Fallback: all‑caps short lines + numbering patterns.
6) Preserve lists and tables
List items should become markdown bullets.
Table rows: flatten to “cell1 | cell2 | …” or emit per‑cell lines.
Why this resolves root causes
DOM‑aware segmentation preserves original boundaries instead of guessing.
Visibility filters remove hidden iXBRL scaffolding.
Minimal normalization avoids over‑correction and preserves original words.
Structured heading detection replaces regex‑only heuristics.
Tables/lists are represented, not merged into paragraphs.
Implementation Plan (practical in this repo)
Modules
1) pipeline/src/ixbrl_extractor.py
Replace iXBRLParser with a DOM-based extractor.
Add a TextBuilder that collects text nodes with explicit block/line boundaries.
Make normalization passes configurable (e.g., repair_mode="light|legacy").
Drop dictionary-based repair from default path; keep as optional.
2) New helper module: pipeline/src/ixbrl_dom.py
DOM walker; visibility checks; style parsing; block detection.
Utilities: is_hidden(node), is_block(node), get_font_size(node).
3) pipeline/src/preprocessor.py
Preserve list items in markdown (prefix - ).
Use heading levels based on computed font size ranks.
Tests / QA
Add fixture HTML in pipeline/tests/fixtures/ixbrl/ to cover:
letter‑spaced headings
hidden elements
list/table structure
ix:continuation fragments
Add tests:
test_ixbrl_visibility_filters_hidden_text
test_ixbrl_preserves_list_items
test_ixbrl_table_rows_do_not_merge
Extend pipeline/tests/qa_preprocessing.py to include:
single‑letter token ratio
average token length
markdown list count
table line heuristic
Evidence generation (once data exists)
Add script pipeline/scripts/ixbrl_artifact_audit.py:
For N documents, emit paired “raw snippet → cleaned span → markdown line” samples.
Save before/after examples per company/year for inspection.
Risks / Edge Cases
Inline CSS class styles: if font sizes are only in external CSS, inline parsing might miss them. Mitigation: parse <style> blocks for class selectors (simple regex).
Continuations: ix:continuation handling must preserve ordering; this can be tricky if IDs are reused.
Tables: Flattening can still lose nuance (row headers vs cells). Better than current accidental merges.
Non‑UTF encodings: add lightweight charset detection using XML declaration or <meta charset>.
Validation Plan
Run the existing QA suite (pipeline/tests/qa_suite) + qa_preprocessing.py on a run ID.
Add a “diff review set” from ixbrl_artifact_audit.py to compare before/after on at least 3 sectors and 2 years.
Track metrics: single‑letter ratio, average sentence length, headings count, and list/table preservation rate.
If you want me to implement the DOM-based extractor + QA script now, I can do that in this worktree. Once you point me to a data directory (or provide a small sample iXBRL file), I can produce the concrete raw‑to‑markdown evidence the request asks for and validate the fixes end‑to‑end.


# Gemini 3

I have completed a deep investigation into the iXBRL to Markdown conversion artifacts. Below is the root cause analysis, evidence-backed findings, and a comprehensive implementation plan for a robust fix.
1. Root Cause Analysis (Evidence-Backed)
The investigation revealed that artifacts are primarily introduced by two stages of the pipeline: the Extraction Stage and the Cleaning Stage.
A. The "Purp ose" Artifact (Indiscriminate Spacing)
Root Cause: The iXBRLParser (subclass of HTMLParser) joins every text fragment within a block-level element using a space: ' '.join(self.current_text).
Evidence: In the raw iXBRL for Tesco 2024, the word "purpose" is split across a span: our purp<span class="_ _4"></span>ose. The parser extracts ["our purp", "ose"], which becomes our purp ose.
Impact: Broken words throughout the document, making keyword search and LLM analysis unreliable.
B. The "everypartofthe" Artifact (Over-Aggressive Merging)
Root Cause: A regex pattern named merge_short_run in _clean_text merges any sequence of tokens if they are all $\le 4$ characters long and at least one is a single character.
Evidence: The phrase ever y par t of the (resulting from split syllables) matches this regex because ever (4), y (1), par (3), t (1), of (2), the (3) are all short. The regex mashes them into everypartofthe.
Impact: Common English words (and, the, of, in) are concatenated with nearby fragments, creating noisy text.
C. The "Missing Heading" Artifact (Simplistic Detection)
Root Cause: Headings in iXBRL often don't use <h1>-<h6> tags but are instead <div> elements with specific CSS classes or styles. The current pipeline relies on tag names or exact regex matches.
Evidence: A section like Principal risks and uncertainties was missed as a heading because the cleaning logic had already turned it into Principalrisks, which failed the regex check.
2. Proposed Extraction Pipeline Changes
I propose a "structural-first" approach that prioritizes the DOM tree's block/inline distinction.
Step 1: Structural Buffer Extraction
Modify iXBRLParser to distinguish between Block tags (div, p, tr, h1-h6) and Inline tags (span, b, i, a, and all ix: tags).
Change: Instead of a list of text parts, maintain a single current_buffer string.
Logic:
handle_data: Append raw text to current_buffer without stripping.
handle_starttag/endtag: If the tag is a Block tag, "flush" the buffer as a new TextSpan and reset it. If it's an Inline tag, do nothing (keep appending).
Step 2: Intelligent Block Normalization
When a block is flushed:
Perform whitespace normalization: ' '.join(buffer.split()). This collapses all internal newlines and spaces into single spaces, exactly as a browser renders it. This naturally fixes the purp ose issue.
Step 3: Conservative Word Repair
Replace the current _clean_text regexes with a refined WordRepair class:
Remove: merge_short_run and merge_spelled_out regexes.
Add: A dictionary-based merge that only joins token_a + token_b if:
The result is a valid word (e.g., Report).
token_a or token_b is a single character OR is not a valid word on its own.
Step 4: Visual Header Detection
Enhance TextSpan detection by checking the class or style attribute for:
font-weight: bold or font-size > [threshold].
Classes commonly used for headers in iXBRL (e.g., header, heading, title).
3. Implementation Plan
Phase 1: Dependency Update
Add beautifulsoup4 and lxml to pipeline/requirements.txt to enable more robust DOM traversal than html.parser.
Phase 2: Core Module Refactor (pipeline/src/ixbrl_extractor.py)
iXBRLParser Class:
Replace HTMLParser with a BeautifulSoup tree walker.
Implement is_block_element(tag) check.
Update text collection to preserve intra-word continuity across spans.
iXBRLExtractor._clean_text Method:
Delete aggressive re.sub calls (lines 440-460).
Refine _repair_fragmented_words to protect common stopwords (and, the, of).
Phase 3: Validation & QA (pipeline/tests/)
Unit Tests: Create tests/test_ixbrl_word_boundaries.py with known problematic snippets.
Regression Check: Run tests/qa_suite/test_ixbrl_quality.py to ensure the single_ratio (ratio of 1-char tokens) remains below 0.12 without merging valid words.
Visual Diff: Run the pipeline on Tesco 2024 and diff the old vs. new Markdown to verify the removal of artifacts.
4. Risks & Edge Cases
Encoding: Some iXBRL files use non-standard glyph mapping. We should ensure the lxml parser handles latin-1 fallbacks as the current code does.
Tables: Aggressive merging in tables can break column alignment. The fix must ensure <td> tags are always treated as block boundaries.
ML-based Cleanup: While I recommend deterministic fixes first, a "Repair LLM" (like repair_markdown_with_llm.py) could be used as a final pass for high-importance "Golden Set" documents, but should not be the primary tool.
Summary of "Why"
This solution resolves the root causes by aligning the parser's behavior with standard HTML rendering rules. By treating <span> as transparent to word boundaries and only normalizing whitespace at the <div> or <p> level, we eliminate the primary source of broken words ("purp ose") and the primary reason we needed dangerous "merging" regexes in the first place.

# Opus 4.5 

XBRL → Markdown Conversion: Root Cause Analysis & Solution
Executive Summary
The iXBRL files in this pipeline are not native text-based iXBRL but rather PDF2HTML conversions (via pdf2htmlEX or similar) with embedded XBRL tagging. This creates a complex extraction challenge because:
The files contain both base64-encoded page images AND absolutely-positioned text spans
XBRL metadata is embedded in hidden <div style="display:none"> containers
Text spans use CSS positioning classes (class="_ _3") that fragment words at arbitrary boundaries
No explicit whitespace exists between adjacent inline elements
Root Cause Analysis
Root Cause 1: Hidden XBRL Metadata Leakage
Evidence (from grep on Johnson Matthey file):
<body><div style="display:none"><ix:header>
  <ix:resources>
    <xbrli:context id="c-1">
      <xbrli:entity>
        <xbrli:identifier scheme="...">2138001AVBSD1HSC6Z10</xbrli:identifier>
      </xbrli:entity>
      <xbrli:period>
        <xbrli:startDate>2021-04-01</xbrli:startDate>
        <xbrli:endDate>2022-03-31</xbrli:endDate>
      </xbrli:period>
    </xbrli:context>
    <xbrli:unit id="u-1"><xbrli:measure>iso4217:GBP</xbrli:measure></xbrli:unit>
    ...
  </ix:resources>
</ix:header></div>
<body><div style="display:none"><ix:header>  <ix:resources>    <xbrli:context id="c-1">      <xbrli:entity>        <xbrli:identifier scheme="...">2138001AVBSD1HSC6Z10</xbrli:identifier>      </xbrli:entity>      <xbrli:period>        <xbrli:startDate>2021-04-01</xbrli:startDate>        <xbrli:endDate>2022-03-31</xbrli:endDate>      </xbrli:period>    </xbrli:context>    <xbrli:unit id="u-1"><xbrli:measure>iso4217:GBP</xbrli:measure></xbrli:unit>    ...  </ix:resources></ix:header></div>
Problem: The current iXBRLParser class (lines 328-337 in ixbrl_extractor.py) only skips <script> and <style> tags:
def handle_starttag(self, tag, attrs):
    if tag.lower() in ['script', 'style']:
        # skip
def handle_starttag(self, tag, attrs):    if tag.lower() in ['script', 'style']:        # skip
It does not detect or skip display:none containers or <ix:header> blocks.
Artifact: First line of markdown contains 2138001AVBSD1HSC6Z10 2021-04-01 2022-03-31 iso4217:GBP ifrs-full:IssuedCapitalMember...
Root Cause 2: PDF2HTML Layout Structure Not Handled
Evidence: The file structure shows:
<div id="pf1" class="pf w0 h0" style="content-visibility: auto;">
  <div class="pc pc1 w0 h0">
    <img src="data:image/png;base64,..."/>
    <!-- Text overlay spans with CSS positioning -->
    <span class="_ _3">text</span>
    <span class="c x0 y2184 w29 h155">positioned text</span>
  </div>
</div>
<div id="pf1" class="pf w0 h0" style="content-visibility: auto;">  <div class="pc pc1 w0 h0">    <img src="data:image/png;base64,..."/>    <!-- Text overlay spans with CSS positioning -->    <span class="_ _3">text</span>    <span class="c x0 y2184 w29 h155">positioned text</span>  </div></div>
Problem: The HTMLParser processes ALL text nodes including:
Navigation/sidebar elements (<div id="sidebar">)
Outline elements (<div id="outline">)
CSS-positioned text fragments that need intelligent merging
Root Cause 3: Missing Whitespace Between Adjacent Elements
Evidence from test output:
onClimate-related → "on Climate-related"
toprovide → "to provide"  
Pleasesee → "Please see"
wearedeveloping → "we are developing"
onClimate-related → "on Climate-related"toprovide → "to provide"  Pleasesee → "Please see"wearedeveloping → "we are developing"
Problem: In the HTML:
<span>on</span><span>Climate</span>
<span>on</span><span>Climate</span>
Python's HTMLParser.handle_data() is called separately for each text node. The current implementation:
def handle_data(self, data):
    text = data.strip()
    if text:
        self.current_text.append(text)  # Line 382
def handle_data(self, data):    text = data.strip()    if text:        self.current_text.append(text)  # Line 382
Then joins with space only at block end (line 343):
text = ' '.join(self.current_text).strip()
text = ' '.join(self.current_text).strip()
But between inline elements, no space is added.
Root Cause 4: Word Fragmentation from PDF Layout
Evidence:
"cur ren tly" should be "currently"
"cur ren tly" should be "currently"
Problem: PDF2HTML preserves exact PDF glyph positions. If letters are positioned individually (common in headlines or styled text), each becomes a separate <span>:
<span class="c x100">c</span><span class="c x105">u</span><span class="c x110">r</span>
<span class="_ _3"> </span>  <!-- tiny spacing span -->
<span class="c x120">r</span><span class="c x125">e</span><span class="c x130">n</span>
<span class="c x100">c</span><span class="c x105">u</span><span class="c x110">r</span><span class="_ _3"> </span>  <!-- tiny spacing span --><span class="c x120">r</span><span class="c x125">e</span><span class="c x130">n</span>
The current _clean_text() method attempts repair but runs AFTER structural damage is done.
Root Cause 5: Block Flushing Logic Fragments Content
Current logic (lines 367-372):
elif tag_lower in ['p', 'div', 'li', 'td', 'th', 'section', 'article']:
    if self.current_text:
        text = ' '.join(self.current_text).strip()
        if text and len(text) > 10:  # Minimum length filter
            self._add_span(text, is_heading=False)
elif tag_lower in ['p', 'div', 'li', 'td', 'th', 'section', 'article']:    if self.current_text:        text = ' '.join(self.current_text).strip()        if text and len(text) > 10:  # Minimum length filter            self._add_span(text, is_heading=False)
Problem: PDF2HTML files have deeply nested divs for layout. Flushing on every </div> creates excessive fragmentation.
Proposed Solution
Architecture: Two-Phase Extraction
Phase 1: Structural Pre-processing
Parse HTML once to identify document structure
Detect and skip hidden/metadata containers
Identify content regions vs navigation/chrome
Phase 2: Intelligent Text Extraction
Use lxml or BeautifulSoup (not HTMLParser) for better DOM manipulation
Process text with awareness of CSS positioning
Apply heuristics to merge adjacent positioned spans
Implementation Plan
Step 1: Replace HTMLParser with lxml/BeautifulSoup
File: pipeline/src/ixbrl_extractor.py
from bs4 import BeautifulSoup, NavigableString

class iXBRLExtractorV2:
    """Improved iXBRL extractor using BeautifulSoup for DOM manipulation."""
    
    SKIP_SELECTORS = [
        'style',
        'script', 
        '[style*="display:none"]',
        '[style*="display: none"]',
        'ix\\:header',  # Note: namespace colon needs escaping
        '#sidebar',
        '#outline',
    ]
    
    def extract_report(self, file_path: Path) -> ExtractedReport:
        html = file_path.read_text(encoding='utf-8', errors='replace')
        soup = BeautifulSoup(html, 'lxml')
        
        # Step 1: Remove unwanted elements
        for selector in self.SKIP_SELECTORS:
            for element in soup.select(selector):
                element.decompose()
        
        # Step 2: Process page containers
        pages = soup.select('.pf') or [soup.body]
        
        spans = []
        for page in pages:
            page_spans = self._extract_page_content(page)
            spans.extend(page_spans)
        
        return ExtractedReport(spans=spans, ...)
from bs4 import BeautifulSoup, NavigableStringclass iXBRLExtractorV2:    """Improved iXBRL extractor using BeautifulSoup for DOM manipulation."""        SKIP_SELECTORS = [        'style',        'script',         '[style*="display:none"]',        '[style*="display: none"]',        'ix\\:header',  # Note: namespace colon needs escaping        '#sidebar',        '#outline',    ]        def extract_report(self, file_path: Path) -> ExtractedReport:        html = file_path.read_text(encoding='utf-8', errors='replace')        soup = BeautifulSoup(html, 'lxml')                # Step 1: Remove unwanted elements        for selector in self.SKIP_SELECTORS:            for element in soup.select(selector):                element.decompose()                # Step 2: Process page containers        pages = soup.select('.pf') or [soup.body]                spans = []        for page in pages:            page_spans = self._extract_page_content(page)            spans.extend(page_spans)                return ExtractedReport(spans=spans, ...)
Step 2: Smart Span Merging Based on CSS Position
def _extract_page_content(self, page_element) -> List[TextSpan]:
    """Extract text from a PDF2HTML page container."""
    
    # Find all text-containing elements
    text_elements = []
    for elem in page_element.descendants:
        if isinstance(elem, NavigableString) and elem.strip():
            parent = elem.parent
            # Get CSS position from class or style
            pos = self._get_element_position(parent)
            text_elements.append({
                'text': str(elem),
                'x': pos.get('x', 0),
                'y': pos.get('y', 0),
            })
    
    # Sort by position (top-to-bottom, left-to-right)
    text_elements.sort(key=lambda e: (e['y'], e['x']))
    
    # Merge elements on same line (similar y-coordinate)
    lines = self._merge_into_lines(text_elements)
    
    return self._lines_to_paragraphs(lines)
def _extract_page_content(self, page_element) -> List[TextSpan]:    """Extract text from a PDF2HTML page container."""        # Find all text-containing elements    text_elements = []    for elem in page_element.descendants:        if isinstance(elem, NavigableString) and elem.strip():            parent = elem.parent            # Get CSS position from class or style            pos = self._get_element_position(parent)            text_elements.append({                'text': str(elem),                'x': pos.get('x', 0),                'y': pos.get('y', 0),            })        # Sort by position (top-to-bottom, left-to-right)    text_elements.sort(key=lambda e: (e['y'], e['x']))        # Merge elements on same line (similar y-coordinate)    lines = self._merge_into_lines(text_elements)        return self._lines_to_paragraphs(lines)
Step 3: Improved Line Merging
def _merge_into_lines(self, elements: List[dict], y_tolerance=5) -> List[str]:
    """Merge positioned elements into lines based on y-coordinate proximity."""
    if not elements:
        return []
    
    lines = []
    current_line = [elements[0]]
    current_y = elements[0]['y']
    
    for elem in elements[1:]:
        if abs(elem['y'] - current_y) <= y_tolerance:
            # Same line - check x gap for whitespace insertion
            if current_line:
                x_gap = elem['x'] - (current_line[-1]['x'] + 
                                     len(current_line[-1]['text']) * 6)  # approx char width
                if x_gap > 10:  # Significant gap = word break
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
def _merge_into_lines(self, elements: List[dict], y_tolerance=5) -> List[str]:    """Merge positioned elements into lines based on y-coordinate proximity."""    if not elements:        return []        lines = []    current_line = [elements[0]]    current_y = elements[0]['y']        for elem in elements[1:]:        if abs(elem['y'] - current_y) <= y_tolerance:            # Same line - check x gap for whitespace insertion            if current_line:                x_gap = elem['x'] - (current_line[-1]['x'] +                                      len(current_line[-1]['text']) * 6)  # approx char width                if x_gap > 10:  # Significant gap = word break                    current_line.append({'text': ' ', 'x': 0, 'y': 0})            current_line.append(elem)        else:            # New line            lines.append(''.join(e['text'] for e in current_line))            current_line = [elem]            current_y = elem['y']        if current_line:        lines.append(''.join(e['text'] for e in current_line))        return lines
Step 4: Fallback for Non-PDF2HTML Files
def _detect_file_format(self, soup: BeautifulSoup) -> str:
    """Detect whether file is PDF2HTML or native iXBRL."""
    # PDF2HTML files have characteristic structure
    if soup.select('.pf.w0.h0'):
        return 'pdf2html'
    elif soup.select('ix\\:nonFraction, ix\\:nonNumeric'):
        return 'native_ixbrl'
    else:
        return 'generic_html'
def _detect_file_format(self, soup: BeautifulSoup) -> str:    """Detect whether file is PDF2HTML or native iXBRL."""    # PDF2HTML files have characteristic structure    if soup.select('.pf.w0.h0'):        return 'pdf2html'    elif soup.select('ix\\:nonFraction, ix\\:nonNumeric'):        return 'native_ixbrl'    else:        return 'generic_html'
Step 5: Enhanced Hidden Content Detection
def _is_hidden(self, element) -> bool:
    """Check if element is hidden via CSS or semantic tags."""
    if element.name in ['script', 'style', 'noscript']:
        return True
    
    # Check inline style
    style = element.get('style', '')
    if 'display:none' in style.replace(' ', '').lower():
        return True
    if 'visibility:hidden' in style.replace(' ', '').lower():
        return True
    
    # Check for XBRL header containers
    if element.name and element.name.startswith('ix:'):
        if 'header' in element.name.lower():
            return True
    
    return False
def _is_hidden(self, element) -> bool:    """Check if element is hidden via CSS or semantic tags."""    if element.name in ['script', 'style', 'noscript']:        return True        # Check inline style    style = element.get('style', '')    if 'display:none' in style.replace(' ', '').lower():        return True    if 'visibility:hidden' in style.replace(' ', '').lower():        return True        # Check for XBRL header containers    if element.name and element.name.startswith('ix:'):        if 'header' in element.name.lower():            return True        return False
Changes to Existing Files
pipeline/src/ixbrl_extractor.py
Line	Change
1-10	Add from bs4 import BeautifulSoup import
313-595	Replace iXBRLParser(HTMLParser) with new iXBRLExtractorV2
408-430	Remove regex-based metadata cleanup (handled structurally)
New	Add _detect_file_format(), _is_hidden(), _merge_into_lines()
pipeline/src/preprocessor.py
Line	Change
254-291	Simplify _to_markdown() - less cleanup needed
QA Checks to Add
pipeline/tests/qa_suite/test_ixbrl_extraction.py
import pytest
from src.ixbrl_extractor import iXBRLExtractorV2

class TestExtractionQuality:
    
    def test_no_lei_codes_in_output(self, sample_ixbrl):
        """Verify XBRL metadata is filtered."""
        report = iXBRLExtractorV2().extract_report(sample_ixbrl)
        # LEI codes are 20 alphanumeric characters
        assert not re.search(r'\b[A-Z0-9]{20}\b', report.full_text)
    
    def test_no_ifrs_namespace_tokens(self, sample_ixbrl):
        """Verify XBRL namespace prefixes are filtered."""
        report = iXBRLExtractorV2().extract_report(sample_ixbrl)
        assert 'ifrs-full:' not in report.full_text
        assert 'iso4217:' not in report.full_text
    
    def test_word_concatenation_rate(self, sample_ixbrl):
        """Check for common word concatenation artifacts."""
        report = iXBRLExtractorV2().extract_report(sample_ixbrl)
        text = report.full_text.lower()
        
        # Count obvious concatenation patterns
        concat_patterns = [
            r'\b\w+andthe\b',
            r'\b\w+tothe\b', 
            r'\btheir\w{4,}\b',  # "their" + word
        ]
        
        total_matches = sum(
            len(re.findall(p, text)) for p in concat_patterns
        )
        
        word_count = len(text.split())
        concat_rate = total_matches / max(1, word_count)
        
        assert concat_rate < 0.001, "Too many word concatenation artifacts"
    
    def test_fragmentation_rate(self, sample_ixbrl):
        """Check for excessive single-letter tokens."""
        report = iXBRLExtractorV2().extract_report(sample_ixbrl)
        words = report.full_text.split()
        single_letters = [w for w in words if len(w) == 1 and w.isalpha()]
        
        # Allow for legitimate single letters (a, I)
        suspicious = [w for w in single_letters if w.lower() not in 'ai']
        rate = len(suspicious) / max(1, len(words))
        
        assert rate < 0.02, "Too many fragmented single-letter tokens"
import pytestfrom src.ixbrl_extractor import iXBRLExtractorV2class TestExtractionQuality:        def test_no_lei_codes_in_output(self, sample_ixbrl):        """Verify XBRL metadata is filtered."""        report = iXBRLExtractorV2().extract_report(sample_ixbrl)        # LEI codes are 20 alphanumeric characters        assert not re.search(r'\b[A-Z0-9]{20}\b', report.full_text)        def test_no_ifrs_namespace_tokens(self, sample_ixbrl):        """Verify XBRL namespace prefixes are filtered."""        report = iXBRLExtractorV2().extract_report(sample_ixbrl)        assert 'ifrs-full:' not in report.full_text        assert 'iso4217:' not in report.full_text        def test_word_concatenation_rate(self, sample_ixbrl):        """Check for common word concatenation artifacts."""        report = iXBRLExtractorV2().extract_report(sample_ixbrl)        text = report.full_text.lower()                # Count obvious concatenation patterns        concat_patterns = [            r'\b\w+andthe\b',            r'\b\w+tothe\b',             r'\btheir\w{4,}\b',  # "their" + word        ]                total_matches = sum(            len(re.findall(p, text)) for p in concat_patterns        )                word_count = len(text.split())        concat_rate = total_matches / max(1, word_count)                assert concat_rate < 0.001, "Too many word concatenation artifacts"        def test_fragmentation_rate(self, sample_ixbrl):        """Check for excessive single-letter tokens."""        report = iXBRLExtractorV2().extract_report(sample_ixbrl)        words = report.full_text.split()        single_letters = [w for w in words if len(w) == 1 and w.isalpha()]                # Allow for legitimate single letters (a, I)        suspicious = [w for w in single_letters if w.lower() not in 'ai']        rate = len(suspicious) / max(1, len(words))                assert rate < 0.02, "Too many fragmented single-letter tokens"
Validation Approach
Diff-based Validation
Process same files with old vs new extractor
Compare output quality metrics (word count, sentence coherence)
Spot-Check Known Artifacts
Verify specific artifacts from test output are fixed:
Line 6: No LEI/date strings
Line 8: "on Climate-related" (with space)
Line 22: "currently" (not "cur ren tly")
Regression Testing
Ensure no substantive content is lost
Compare heading detection rates
Risks and Edge Cases
Risk	Mitigation
Different PDF2HTML versions	Detect format dynamically; maintain multiple parsers
Native iXBRL files (rare)	Fallback to simpler DOM-based extraction
Position coordinate parsing	Use heuristics + dictionary repair as backup
Large file performance	Stream processing; skip image data early
Encoding issues	Always use errors='replace'; normalize Unicode
Implementation Sequence
Add BeautifulSoup/lxml to requirements.txt
Create iXBRLExtractorV2 class alongside existing iXBRLExtractor
Add format detection logic
Implement hidden element filtering
Add position-aware span merging
Add QA tests
Switch pipeline to use V2 extractor
Deprecate V1 after validation
This completes my root cause analysis and solution proposal. Would you like me to proceed with implementing any part of this solution?