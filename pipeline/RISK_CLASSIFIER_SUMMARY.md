# AI Risk Type Classifier - Implementation Summary

## What We Built

You now have a complete AI Risk Type classification system for your golden set of annual reports. Here's what was created:

### 1. Core Classifier Module
**File**: `src/risk_type_classifier.py`

Features:
- Multi-label classification (0-8 risk types per report)
- Conservative approach (only tags when evidence is clear)
- Evidence extraction (quotes supporting each risk type)
- Confidence scoring (0.0-1.0 per risk type)
- Automatic retry logic for API failures

### 2. Test Script
**File**: `test_risk_type_classifier.py`

Usage:
```bash
python test_risk_type_classifier.py [--input-dir DIR] [--detailed] [--log-level LEVEL]
```

Outputs:
- JSON results file
- Console summary with statistics
- Detailed evidence quotes (optional)
- Log file for debugging

### 3. Visualization Tool
**File**: `visualize_risk_types.py`

Usage:
```bash
python visualize_risk_types.py [--input FILE] [--show-legend]
```

Visualizations:
- ASCII bar chart of risk type distribution
- Heatmap: companies × risk types
- Confidence score analysis
- Companies ranked by number of risk types

### 4. Quick Test Script
**File**: `test_single_risk_classification.py`

For quick testing on a single report (RELX PLC) to verify setup.

### 5. Documentation
**File**: `RISK_TYPE_CLASSIFIER_GUIDE.md`

Complete guide covering:
- Quick start instructions
- Architecture overview
- Programmatic usage examples
- Customization guide
- Troubleshooting tips

## Risk Categories Implemented

| Risk Type | Description |
|-----------|-------------|
| Operational & Technical Risk | Model failures, bias, cybersecurity, reliability |
| Workforce Impacts | Job displacement |
| Regulatory & Compliance Risk | Legal liability, compliance costs |
| Information Integrity | Misinformation, content authenticity, deepfakes |
| Reputational & Ethical Risk | Public trust, ethical concerns, human rights |
| Third-Party & Supply Chain Risk | Vendor reliance, downstream misuse |
| Environmental Impact | Energy use, carbon footprint, sustainability |
| National Security Risk | Geopolitical, export controls, adversarial use |

## Next Steps to Run the Classifier

### Step 1: Update API Key

The Gemini API key needs to be renewed. Update your `.env` or `.env.local` file:

```bash
# Get a new API key from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_new_api_key_here
```

### Step 2: Test on Single Report

```bash
source venv/bin/activate
python test_single_risk_classification.py
```

This should show results like:
```
AI Mentioned: True
Number of Risk Types: 2-3
Risk Types Identified:
  [Operational & Technical Risk]
  Confidence: 0.85
  Evidence: "Quote from report..."
```

### Step 3: Run on Golden Set

```bash
python test_risk_type_classifier.py
```

Expected output:
- Processes 20 reports in ~2-4 minutes
- Creates `output/risk_classifications/golden_set_results.json`
- Displays summary statistics

### Step 4: Visualize Results

```bash
python visualize_risk_types.py --show-legend
```

## Output Structure

The classifier produces a JSON file with this structure:

```json
[
  {
    "firm_name": "RELX PLC",
    "company_number": "00077536",
    "ticker": "REL",
    "ai_mentioned": true,
    "risk_types": ["operational_technical", "regulatory_compliance"],
    "evidence": {
      "operational_technical": [
        "LexisNexis Risk Solutions continues to develop sophisticated AI and ML techniques...",
        "We apply advanced algorithms such as machine learning and natural language processing..."
      ],
      "regulatory_compliance": [
        "Compliance with evolving AI regulations including GDPR and proposed EU AI Act..."
      ]
    },
    "confidence_scores": {
      "operational_technical": 0.85,
      "regulatory_compliance": 0.75
    },
    "reasoning": "The report contains clear mentions of AI/ML technologies in risk management context..."
  },
  ...
]
```

## Key Design Decisions

### 1. Multi-Label Classification
- Companies can have 0 to 8 risk types
- Each risk type is independent
- Only assigned when clear evidence exists

### 2. Evidence-Based
- Every risk type tag includes supporting quotes
- Maximum 200 characters per quote
- Stored for validation and analysis

### 3. Confidence Scores
- Per risk type (not global)
- Helps identify classifications needing review
- Can filter by confidence threshold

### 4. Conservative Approach
- Better to miss a mention than false positive
- "Clear and evident" standard
- Reduces manual review burden

## Future Enhancements

### Adding Examples (Recommended Next)
To improve accuracy, add examples for each risk category in the prompt:

```python
# In src/risk_type_classifier.py, _build_prompt() method
# Add an EXAMPLES section like:

## EXAMPLES

**operational_technical**:
"Our AI-powered fraud detection system experienced a 15% false positive rate,
requiring manual review and causing customer friction."

**regulatory_compliance**:
"We are monitoring the EU AI Act and assessing which of our products may be
classified as high-risk systems requiring certification."

# ... add examples for each risk category
```

### Sector-Specific Analysis
Compare risk type prevalence across sectors:
- Financial services vs. Energy vs. Technology
- Requires sector metadata in golden set

### Time-Series Analysis
Track how risk mentions evolve over time:
- Requires multiple years of reports
- Can identify emerging vs. declining risk concerns

### Co-Occurrence Analysis
Identify which risk types appear together:
- Network analysis of risk relationships
- Cluster companies by risk profile

## Testing Checklist

Before running on full golden set, verify:

- [ ] Updated Gemini API key in `.env`
- [ ] Virtual environment activated
- [ ] Single report test passes (`test_single_risk_classification.py`)
- [ ] Output directory exists (`output/risk_classifications/`)
- [ ] Logs directory exists (`logs/`)

## Expected Performance

With valid API key:

- **Processing time**: ~5-10 seconds per report
- **Total time for 20 reports**: ~2-4 minutes
- **API calls**: 20 (well within free tier: 1,500/day)
- **Cost**: $0 (within free tier)
- **Success rate**: Should be 95%+ with retry logic

## Troubleshooting

### Common Issues

1. **API Key Expired** ✓ You're here
   - Get new key: https://aistudio.google.com/app/apikey
   - Update `.env` file
   - Test with `test_single_risk_classification.py`

2. **No preprocessed files**
   - Run: `python test_preprocessing_comparison.py`
   - Or check: `ls output/preprocessed/keyword/`

3. **Rate limiting**
   - Classifier has automatic retry
   - If persistent, process in smaller batches

4. **Unexpected classifications**
   - Check confidence scores (low = uncertain)
   - Review evidence quotes
   - Enable `--detailed` flag
   - Use `--log-level DEBUG`

## Files Created

```
pipeline/
├── src/
│   └── risk_type_classifier.py           (NEW)
├── test_risk_type_classifier.py          (NEW)
├── visualize_risk_types.py               (NEW)
├── test_single_risk_classification.py    (NEW)
├── RISK_TYPE_CLASSIFIER_GUIDE.md         (NEW)
└── RISK_CLASSIFIER_SUMMARY.md            (NEW - this file)
```

## Questions?

Refer to:
- `RISK_TYPE_CLASSIFIER_GUIDE.md` - Complete usage guide
- `logs/risk_type_classifier.log` - Detailed execution log
- `README.md` - Main pipeline documentation

---

**Status**: ✅ System built and ready to use
**Blocker**: API key needs renewal (easy fix)
**Next step**: Update `.env` with valid Gemini API key and run tests
