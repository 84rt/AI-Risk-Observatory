# Migration from Anthropic Claude to Google Gemini

## Changes Made

The pipeline has been updated to use **Google Gemini 2.0 Flash** instead of Anthropic Claude.

## Key Benefits

✅ **FREE tier:** 15 requests/minute, 1500 requests/day
✅ **Lower cost:** ~$2-10 for 20 companies (vs $40-100 with Claude)
✅ **Same quality:** Gemini 2.0 Flash is excellent for structured classification tasks
✅ **JSON mode:** Native JSON output for reliable parsing

## Files Updated

### 1. Dependencies (`requirements.txt`)
- **Removed:** `anthropic>=0.39.0`
- **Added:** `google-generativeai>=0.3.0`

### 2. Configuration (`src/config.py`)
- **Changed:** `anthropic_api_key` → `gemini_api_key`
- **Changed:** `anthropic_model` → `gemini_model`
- **Default model:** `gemini-2.0-flash-exp`
- **Now checks:** `.env.local` first, then `.env`

### 3. LLM Classifier (`src/llm_classifier.py`)
- **Complete rewrite** to use Google Generative AI SDK
- Uses `genai.GenerativeModel()` instead of `Anthropic()`
- Enables JSON response mode for structured output
- Same classification logic and prompt structure

### 4. Environment Template
- **Created:** `.env.template` with Gemini keys
- **Updated setup.sh** to reference Gemini

### 5. Documentation
- **README.md:** Updated to reference Gemini throughout
- **QUICKSTART.md:** Updated API key instructions
- **Cost estimates:** Adjusted for Gemini pricing

## Setup Instructions

### 1. Install new dependencies
```bash
cd pipeline
source venv/bin/activate  # if already created
pip install -r requirements.txt
```

### 2. Get Gemini API Key
1. Go to https://aistudio.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the key

### 3. Update your `.env.local` file
Your file should have:
```bash
GEMINI_API_KEY=AIza...your_key_here
COMPANIES_HOUSE_API_KEY=your_key_here
```

### 4. Get Companies House API Key
1. Register at https://developer.company-information.service.gov.uk/
2. When asked, select **REST API**
3. Create API key
4. Add to `.env.local`

## Testing

Test with a single company first:
```bash
python scripts/test_single_company.py \
  --company-number 00489800 \
  --company-name "Barclays PLC" \
  --ticker BARC \
  --sector Financials \
  --max-candidates 5
```

This will cost $0 (within free tier) and confirm everything works.

## Rate Limits

Gemini free tier limits:
- **15 requests per minute**
- **1500 requests per day**

The pipeline automatically handles rate limiting with retry logic. For 20 companies with ~1000 spans each:
- Total requests: ~20,000
- Time needed: ~13-15 days if processing continuously
- **OR** process in batches to stay under daily limit

**Recommendation:** Process 1-2 companies at a time for testing, then scale up.

## Cost Comparison

| Scenario | Claude Sonnet 4.5 | Gemini 2.0 Flash |
|----------|-------------------|------------------|
| Single company test (5 spans) | ~$0.50 | **FREE** |
| Single company full (1000 spans) | ~$2-5 | **FREE** or ~$0.10-0.50 |
| 20 companies | ~$40-100 | **FREE** or ~$2-10 |

## Troubleshooting

### Rate limit errors
If you hit rate limits:
- Pipeline will automatically retry
- Process fewer companies at once
- Space out processing over multiple days

### API key errors
Make sure `.env.local` or `.env` exists with:
```bash
GEMINI_API_KEY=AIza...
COMPANIES_HOUSE_API_KEY=...
```

### Import errors
Reinstall dependencies:
```bash
pip install -r requirements.txt
```

## Notes

- Classification quality should be comparable to Claude
- Gemini 2.0 Flash is optimized for structured outputs
- The prompt structure remains identical
- All taxonomy categories and logic unchanged
- Database schema unchanged
