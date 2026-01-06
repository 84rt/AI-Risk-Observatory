# Pipeline Test Script - Golden Dataset

## Overview

`pipeline_test.py` is a comprehensive step-by-step testing script for the AIRO pipeline using the **Golden Dataset** of the top 20 UK public companies.

## Golden Dataset

The script includes a hardcoded list of the 20 largest UK public companies by market capitalization:

1. AstraZeneca plc (02723534)
2. Shell plc (04366849)
3. HSBC Holdings plc (00617987)
4. Unilever PLC (00041424)
5. BP p.l.c. (00102498)
6. GSK plc (03888792)
7. RELX PLC (00077536)
8. Diageo plc (00023307)
9. Rio Tinto plc (00719885)
10. British American Tobacco p.l.c. (03407696)
11. London Stock Exchange Group plc (05369106)
12. National Grid plc (04031152)
13. Compass Group PLC (04083914)
14. Barclays PLC (00048839)
15. Lloyds Banking Group plc (SC095000)
16. BAE Systems plc (01470151)
17. Reckitt Benckiser Group plc (06270876)
18. Rolls-Royce Holdings plc (07524813)
19. Anglo American plc (03564138)
20. Tesco PLC (00445790)

## Test Steps

The script runs 4 sequential test steps:

### Step 1: Fetch Filing History & Check Formats
- Fetches filing history for each company
- Filters for Annual Accounts (AA type)
- Checks available formats (PDF, XBRL, iXBRL)
- Logs detailed information about each filing

### Step 2: Download Documents
- Downloads PDF documents from Companies House
- Saves to `../data/raw/pdfs/`
- Logs file sizes and download status

### Step 3: Extract Text
- Extracts text from downloaded PDFs using PyMuPDF
- Identifies sections (risk sections, governance, etc.)
- Logs extraction statistics (pages, spans, text length)

### Step 4: Chunk Text
- Chunks extracted text into candidate spans
- Prepares text for LLM processing
- Logs chunking statistics

## Usage

```bash
cd pipeline
source venv/bin/activate
python pipeline_test.py
```

## Logging

The script provides **heavy logging** at DEBUG level:

- **Console output**: Real-time progress with tqdm progress bars
- **Log file**: Detailed logs saved to `../data/logs/pipeline/pipeline_test.log`
- **Format**: Includes timestamps, module names, line numbers, and detailed messages

## Output

After running, you'll see:

1. **Step-by-step progress** for each company
2. **Summary statistics** after each step
3. **Final summary** showing success rates across all steps
4. **Detailed log file** at `../data/logs/pipeline/pipeline_test.log`

## Example Output

```
================================================================================
STEP 1: FETCHING FILING HISTORY & CHECKING FORMATS
================================================================================

============================================================
Processing: AstraZeneca plc (02723534)
============================================================
✅ Found 25 total account filings
✅ Found 23 annual account filings
📄 Latest filing:
   Type: AA
   Description: accounts-with-accounts-type-group
   Date: 2024-03-31
📦 Available formats: PDF, iXBRL
   Links: ['self', 'document_metadata', 'xbrl', 'ixbrl']

...

================================================================================
STEP 1 SUMMARY
================================================================================
✅ Successfully processed: 20/20
⚠️  No annual accounts: 0
❌ Errors: 0
```

## Next Steps

After testing, you can:

1. **Review logs** to identify any issues
2. **Check downloaded PDFs** in `../data/raw/pdfs/`
3. **Modify the script** to test specific companies or steps
4. **Add LLM classification step** (Step 5) once text extraction is validated

## Notes

- The script currently focuses on **PDF downloads** (as per current pipeline design)
- Future enhancement: Check for iXBRL/electronic formats first, fall back to PDFs
- All company numbers are verified UK-incorporated entities (not Jersey/Ireland subsidiaries)

