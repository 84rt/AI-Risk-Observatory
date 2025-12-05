# Companies House Document API Issue

## Problem

The Companies House Document API (used to download PDFs) is returning 404 errors even with valid API credentials. This is a known limitation.

## Why This Happens

Companies House has two separate APIs:
1. **REST API** (`api.company-information.service.gov.uk`) - ✅ Working
   - Company profiles
   - Filing history
   - **FREE tier available**

2. **Document API** (`document-api.company-information.service.gov.uk`) - ❌ Not accessible
   - PDF downloads
   - May require **premium/paid access**
   - Or different authentication method

## Current Status

Our API key works for:
- ✅ Getting company information
- ✅ Listing filing history
- ✅ Getting document metadata
- ❌ **Downloading actual PDFs**

## Workarounds

### Option 1: Manual PDF Collection (Recommended for v1)
1. Use Companies House website to manually download PDFs
2. Place them in `output/pdfs/` folder
3. Run pipeline with `--skip-download` flag

```bash
# After manually downloading PDFs:
python run_pipeline.py --companies data/companies.csv --skip-download
```

### Option 2: Use Companies House Website
Download reports from: https://find-and-update.company-information.service.gov.uk/

### Option 3: Request Document API Access
Contact Companies House to request Document API access:
- Email: enquiries@companieshouse.gov.uk
- May require paid tier

### Option 4: Web Scraping (Advanced)
Use `selenium` or similar to automate downloads from the website.

## For Testing v1

**Recommended approach:**
1. Manually download 2-3 company annual reports as PDFs
2. Place in `output/pdfs/` with naming: `{company_number}_{name}_{year}.pdf`
3. Test rest of pipeline (PDF extraction → LLM → Database)

Example:
```
output/pdfs/01026167_Barclays_Bank_PLC_2024.pdf
output/pdfs/00102498_BP_PLC_2024.pdf
```

Then run:
```bash
python run_pipeline.py --skip-download
```

This will test the entire pipeline except the download step!
