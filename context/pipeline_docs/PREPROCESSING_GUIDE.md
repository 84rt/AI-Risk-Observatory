# Preprocessing Guide

## Overview

This guide explains the preprocessing strategies implemented to filter and format annual reports before analysis. The goal is to reduce noise and focus on relevant qualitative content about AI risks.

Path note: legacy `output/...` examples now live under `/data` (`data/raw`, `data/processed`, `data/results`).

## Implemented Strategies

### 1. **Risk Sections Only** (`risk_only`)
**Approach:** Extract only sections explicitly labeled as risk-related.

**Pros:**
- Most focused on relevant content
- Lowest processing cost
- Targets principal risk disclosures

**Cons:**
- May miss AI mentions in other sections (strategy, operations, innovation)
- **Current limitation:** iXBRL documents don't always have clear section boundaries
- Test result: 0% retention for Shell (no risk sections detected in iXBRL)

**Use case:** Best for PDF reports with clear section headings.

---

### 2. **Keyword-Based Filtering** (`keyword`)
**Approach:** Keep any paragraph mentioning AI/ML/risk keywords from the entire document.

**Pros:**
- Comprehensive coverage - won't miss AI mentions anywhere
- Works well with iXBRL/XHTML format
- Captures context around AI discussions
- More robust to structural variations

**Cons:**
- More text to process (higher LLM costs)
- Some false positives possible

**Test results (Shell plc):**
- Retained 8.6% of content (2,322 spans)
- Found 10 AI/ML mentions
- Found 1,343 risk-related mentions
- Successfully captured: *"Shell and Space Intelligence are maturing an artificial intelligence (AI)... of AI can improve how Shell's nature-based solutions (NBS) business"*

**Recommendation:** **Use this strategy** for the golden dataset analysis.

---

## Keywords Used

### AI/ML Keywords (25 patterns)
```
- artificial intelligence, ai
- machine learning, ml
- deep learning, neural network
- large language model, llm
- generative ai, gen ai
- natural language processing, nlp
- computer vision, image recognition
- intelligent automation, robotic process automation, rpa
- predictive analytics, data analytics
- chatbot, virtual assistant
- recommendation engine/system/algorithm
- autonomous
- ai-powered/driven/enabled/based
- algorithmic (trading/decision/bias)
```

### Risk Keywords (12 patterns)
```
- risk, risks
- uncertainty, uncertainties
- threat, threats
- challenge, challenges
- concern, concerns
- vulnerability, vulnerabilities
- exposure
- impact
- disruption, disruptive
- adverse, adversely
```

---

## Usage

### Quick Comparison Test (Single Company)

Test both strategies on Shell plc:

```bash
source venv/bin/activate
python test_preprocessing_comparison.py
```

**Output:**
- `/data/processed/preprocessed/risk_only/04366849_risk_only.md`
- `/data/processed/preprocessed/keyword/04366849_keyword.md`
- Comparison table showing retention rates and statistics

---

### Full Pipeline (All 20 Companies)

Run the main pipeline with keyword-based strategy:

```bash
source venv/bin/activate
python pipeline_test.py
```

This will:
1. Download reports (iXBRL preferred, PDF fallback)
2. Extract text from documents
3. Apply keyword-based preprocessing
4. Save markdown files to `/data/processed/preprocessed/keyword/`
5. Continue with chunking and classification

To use risk-only strategy instead, edit `pipeline_test.py` line 688:
```python
main(preprocessing_strategy=PreprocessingStrategy.RISK_ONLY)
```

---

## Output Format

Preprocessed reports are saved as markdown files with metadata headers:

```markdown
---
firm: Shell plc
strategy: keyword
original_spans: 26861
filtered_spans: 2322
retention: 8.6%
---

## Risk factors

performance and compliance with the Shell General Business Principles,
Code of Conduct, Statement on Risk Management and Risk Manual...

[Content continues...]
```

---

## Architecture

### Files

1. **`src/preprocessor.py`** - Core preprocessing module
   - `PreprocessingStrategy` enum
   - `Preprocessor` class with filtering logic
   - Markdown conversion
   - Keyword pattern matching

2. **`test_preprocessing_comparison.py`** - Comparison test script
   - Downloads single report
   - Applies both strategies
   - Generates side-by-side comparison

3. **`pipeline_test.py`** - Updated main pipeline
   - Added Step 4: Preprocessing & Filtering
   - Renumbered Step 5: Chunking
   - Configurable strategy selection

### Integration Flow

```
Step 1: Filing History
    ↓
Step 2: Download Documents (iXBRL/PDF)
    ↓
Step 3: Extract Text (raw spans)
    ↓
Step 4: Preprocess & Filter ← NEW STEP
    ↓
Step 5: Chunking (candidate spans)
    ↓
Step 6: LLM Classification
    ↓
Step 7: Aggregation
```

---

## Evaluation Criteria

To determine which strategy works better, compare:

### Quantitative Metrics
- **Retention rate:** % of original text kept
- **Coverage:** Number of AI mentions captured
- **Precision:** Relevance of filtered content

### Qualitative Assessment
1. Review sample markdown files
2. Check if AI risk discussions are captured
3. Verify minimal false positives (irrelevant content)
4. Ensure sufficient context around mentions

### LLM Classification Performance
After running full pipeline:
- Compare F1 scores between strategies
- Measure false positive/negative rates
- Assess confidence scores

---

## Findings & Recommendations

### Test Results Summary

| Strategy | Retention | AI Matches | Risk Matches | Status |
|----------|-----------|------------|--------------|--------|
| Risk Only | 0.0% | N/A | 0 sections | ❌ Failed (iXBRL) |
| Keyword | 8.6% | 10 | 1,343 | ✅ Success |

### Issues Identified

**Risk-only strategy limitations:**
- iXBRL documents lack clear section boundaries
- Extractor grouped entire document as "other" section
- 0 risk sections detected despite relevant content existing

**Potential fix:**
- Improve section detection in `src/ixbrl_extractor.py`
- Look for heading patterns in text content, not just HTML tags
- Use keyword-based section identification as fallback

### Recommendation

**Use keyword-based strategy** for the golden dataset because:

1. ✅ **Works reliably** with both iXBRL and PDF formats
2. ✅ **High recall** - captures AI mentions anywhere in document
3. ✅ **Maintains context** - includes surrounding paragraphs
4. ✅ **Reasonable retention** - ~8-10% of content (manageable for LLM)
5. ✅ **Proven effectiveness** - found real AI mentions in Shell report

The trade-off of slightly higher processing costs is worth the improved coverage and reliability.

---

## Next Steps

1. ✅ Test comparison script (completed)
2. **Run full pipeline** with keyword strategy on all 20 companies
3. **Review markdown outputs** in `/data/processed/preprocessed/keyword/`
4. **Measure LLM classification accuracy** and adjust if needed
5. **Consider hybrid approach** if results suggest combining strategies

---

## Future Improvements

### Short-term
- Add more AI-related keywords (e.g., "predictive modeling", "data science")
- Tune context window (currently includes previous span)
- Add exclusion patterns for false positives

### Medium-term
- Improve iXBRL section detection
- Implement hybrid strategy (risk sections + keyword enrichment)
- Add section-level statistics to output

### Long-term
- Use embeddings to find semantically similar content
- Train custom classifier for "relevant paragraph" detection
- Experiment with LLM-based filtering (zero-shot classification)

---

## Contact & Support

For questions or issues:
- Check logs in `logs/pipeline_test.log`
- Review sample outputs in `data/processed/preprocessed/`
- Modify keywords in `src/preprocessor.py` lines 35-69
