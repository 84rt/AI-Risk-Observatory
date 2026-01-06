# Gemini Text Cleaning Status

## Summary
Gemini text cleaning has been **implemented with parallel processing** but is **temporarily disabled** due to summarization issues.

## What Was Implemented

### 1. Parallel Processing Infrastructure ✅
- **ThreadPoolExecutor** with configurable workers (default: 20)
- **Rate limiting**: 4 requests/second (safe under 5 req/sec limit)
- **Large chunks**: 100K characters (reduced API calls by 5x)
- **Tier 1 API limits**: 300 RPM, 1M TPM fully supported

Location: `src/text_cleaner.py`

### 2. Command-Line Flag ✅
- `--no-clean-text` flag added to `run_pipeline.py`
- Default: cleaning **enabled** (but currently returns original text)
- Can be disabled for faster processing

### 3. Performance Results

**Test on RELX report (100K chars after filtering):**
- Extraction: 0.4s
- Preprocessing without cleaning: 0.0s
- Preprocessing with parallel Gemini: 60.0s (1 chunk @ 100K)
- **Overhead**: 60s for 100K chars

## The Problem ❌

**Gemini 2.0 Flash keeps summarizing text** despite explicit prompts:

```
Cleaned text length changed too much: 100,652 → 42,585 (ratio: 0.42)
```

Even with prompts containing:
- "CRITICAL RULES"
- "Keep EVERY sentence, paragraph, number, date, and word"
- "Do NOT remove any content"
- "Return ALL the text with ONLY spacing fixed"

**Gemini reduces text by ~60%**, likely removing "redundant" content.

## Current Solution ✅

**Using regex-based cleanup in `ixbrl_extractor.py:176-199`:**

```python
# Pattern 1: Single letter suffix at end: "risk s" → "risks"
text = re.sub(r'([a-z]{3,})\s+([a-z])(?=\s|[,.;:\)]|$)', r'\1\2', text)

# Pattern 2: Single letter at start of word: "c ust" → "cust"
text = re.sub(r'(?<=\s)([a-z])\s+([a-z]{3,})', r'\1\2', text)
```

**Result**: From testing, extracted text had **"No obvious spacing issues"**

## What To Do Next

### Option 1: Keep Gemini Disabled (Current) ✅
- Regex cleanup is working well
- Much faster (0.4s vs 60.4s total)
- No risk of content loss
- **Recommended for now**

### Option 2: Fix Gemini Prompting 🔧
Try different approaches:
1. Use examples in prompt showing before/after with same length
2. Try `gemini-2.0-flash-thinking-exp` model for better instruction following
3. Add output format constraints (e.g., "Output must be similar length to input")
4. Use few-shot examples of acceptable vs unacceptable cleaning

### Option 3: Hybrid Approach 🔄
1. Use regex for common patterns (fast, reliable)
2. Only use Gemini for edge cases or validation
3. Sample checking instead of full document cleaning

## Files Modified

1. `src/text_cleaner.py` - Parallel processing infrastructure (ready but disabled)
2. `run_pipeline.py` - Added `--no-clean-text` flag
3. `src/pipeline.py` - Added `clean_text` parameter
4. `pipeline_test.py` - Default `clean_text=True` (currently no-op)
5. `src/preprocessor.py` - Accepts `clean_text` parameter

## API Tier Information

**Tier 1 Limits (Current):**
- 300 RPM (Requests Per Minute)
- 1,000,000 TPM (Tokens Per Minute)
- 1,000 RPD (Requests Per Day)

**Capacity:** Could process ~300 reports/minute with parallel processing if Gemini worked correctly.

## Recommendations

1. **Short term**: Keep Gemini disabled, use regex cleanup ✅
2. **Medium term**: Investigate better prompting strategies or alternative models
3. **Long term**: Consider fine-tuning a smaller model specifically for spacing fixes

---

**Status**: Infrastructure complete, feature temporarily disabled due to quality issues.
**Last updated**: 2025-12-08
