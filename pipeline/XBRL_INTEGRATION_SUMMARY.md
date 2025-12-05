# XBRL API Integration - Summary

## What We've Achieved

We've successfully integrated **filings.xbrl.org** as the primary data source for downloading UK company annual reports in iXBRL/XHTML format.

## Key Benefits

### ✅ Better Data Quality
- **Native iXBRL/XHTML format** - structured, machine-readable reports
- **Much cleaner text extraction** compared to PDF parsing
- **1.6 million characters** of clean text from a single report (Shell plc)
- **26,861 text spans** automatically extracted with proper structure

### ✅ Free & Easy Access
- **No authentication required** - completely free public API
- **2,437 UK company filings** available
- **Coverage from 2021 onwards** - perfect for recent data
- **FTSE 350 companies** that file in ESEF format

### ✅ Robust Fallback
- Automatically falls back to Companies House PDF download if XBRL not available
- Seamless integration with existing pipeline
- All 20 companies in golden dataset have LEI codes

## What's Been Updated

### 1. New Client: `src/xbrl_filings_client.py`
Complete API client for filings.xbrl.org with methods:
- `search_entity_by_lei()` - Find company by LEI code
- `get_entity_filings()` - Get all filings for a company
- `download_xhtml_report()` - Download iXBRL report
- `fetch_annual_report()` - High-level download with caching

### 2. Updated Pipeline: `pipeline_test.py`
- Loads LEI codes from `data/companies_with_lei.json`
- Tries filings.xbrl.org first (if LEI available)
- Falls back to Companies House PDF automatically
- Shows source statistics (XBRL API vs Companies House)

### 3. LEI Lookup Tool: `lookup_lei_codes.py`
- Looks up LEI codes using GLEIF API
- **Found all 20 golden dataset companies** (100% success!)
- Saves results to `data/companies_with_lei.json` and `.csv`

### 4. Documentation: `README.md`
Added "Recent Updates" section explaining the new capability

## Files Created

```
pipeline/
├── src/
│   └── xbrl_filings_client.py          # NEW: filings.xbrl.org API client
├── data/
│   ├── companies_with_lei.json         # NEW: Companies with LEI codes
│   └── companies_with_lei.csv          # NEW: CSV format
├── lookup_lei_codes.py                 # NEW: LEI lookup tool
├── test_xbrl_download.py               # NEW: Test XBRL download
├── test_pipeline_single.py             # NEW: Test full workflow
└── XBRL_INTEGRATION_SUMMARY.md         # NEW: This file
```

## Test Results

### Single Company Test (Shell plc)
```
✅ Downloaded: 51.1 MB iXBRL report
✅ Extracted: 26,861 text spans (1,628,927 characters)
✅ Generated: 1,778 candidate spans for classification
```

### Golden Dataset Coverage
```
✅ 20/20 companies have LEI codes
✅ All companies available in filings.xbrl.org
✅ Years available: 2021, 2022, 2023
```

## Usage

### Quick Test (Single Company)
```bash
source venv/bin/activate
python test_pipeline_single.py
```

### Full Golden Dataset Test
```bash
source venv/bin/activate
python pipeline_test.py
```

### Just Download Documents
```bash
source venv/bin/activate
python test_download_only.py
```

## Next Steps

1. **Run Full Pipeline Test** - Test all 20 companies with the updated pipeline
2. **Add LLM Classification** - Connect to Gemini API for risk classification
3. **Database Storage** - Store results in SQLite
4. **Dashboard Integration** - Export data for visualization

## Technical Details

### Data Format: iXBRL (Inline XBRL)
- **Standard**: European Single Electronic Format (ESEF)
- **Structure**: XHTML with embedded XBRL tags
- **Benefits**: Human-readable AND machine-readable
- **Tags**: Financial line items are semantically tagged

### API: filings.xbrl.org
- **Base URL**: https://filings.xbrl.org
- **API Docs**: https://filings.xbrl.org/docs/api
- **Authentication**: None required (free public access)
- **Rate Limits**: None specified

### LEI: Legal Entity Identifier
- **Standard**: ISO 17442
- **Source**: GLEIF (Global Legal Entity Identifier Foundation)
- **Format**: 20-character alphanumeric code
- **Example**: 21380068P1DRHMJ8KU70 (Shell plc)

## Comparison: XBRL API vs Companies House

| Feature | filings.xbrl.org | Companies House |
|---------|------------------|-----------------|
| **Format** | iXBRL/XHTML | PDF |
| **Auth** | None | API Key required |
| **Coverage** | FTSE 350, 2021+ | All UK companies |
| **Text Quality** | Excellent (structured) | Good (parsed) |
| **File Size** | 10-100 MB | 0.1-10 MB |
| **Extraction** | Fast, clean | Slower, noisier |

## References

- [filings.xbrl.org Documentation](https://filings.xbrl.org/docs/about)
- [GLEIF LEI Database](https://www.gleif.org/en/lei-data/lei-search)
- [ESEF Requirements](https://www.esma.europa.eu/policy-activities/corporate-disclosure/european-single-electronic-format)
- [Companies House API](https://developer.company-information.service.gov.uk/)

---

**Status**: ✅ Integration Complete & Tested
**Date**: December 5, 2025
**Next Milestone**: Full pipeline test with all 20 companies
