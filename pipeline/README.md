# AIRO Data Pipeline

The AIRO (AI Risk Observatory) pipeline processes UK company annual reports to identify and classify AI-related risk disclosures.

## Overview

The pipeline:
1. **Downloads** annual reports in iXBRL/XHTML format from filings.xbrl.org API (for FTSE 350 companies from 2021+) or PDF from Companies House API
2. **Extracts** text with section detection (using PyMuPDF for PDFs or HTML parsing for iXBRL)
3. **Chunks** text into candidate spans
4. **Classifies** spans using Google Gemini (relevance detection + risk taxonomy classification)
5. **Stores** results in SQLite database
6. **Aggregates** mention-level data to firm-year metrics

## Recent Updates (December 2025)

### ✅ filings.xbrl.org Integration
We've integrated the **filings.xbrl.org** API as the primary data source for FTSE 350 companies:

- **2,437 UK company filings** available in native iXBRL/XHTML format
- **No authentication required** - free public API
- **Coverage**: FTSE 350 companies filing in ESEF format (2021 onwards)
- **Direct XHTML access** - much cleaner data extraction than PDF parsing

This solves the issue where Companies House only provides PDFs for most large companies. For companies not in filings.xbrl.org, the pipeline falls back to Companies House PDF downloads.

**Key benefit**: iXBRL/XHTML format provides structured, machine-readable financial reports with better text extraction quality compared to PDFs.

## Setup

### 1. Install Dependencies

```bash
cd pipeline
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` or `.env.local` file in the `pipeline/` directory:

```bash
cp .env.template .env
```

Edit `.env` and add your API keys:
```
GEMINI_API_KEY=your_gemini_api_key_here
COMPANIES_HOUSE_API_KEY=your_companies_house_api_key_here
```

**Getting API Keys:**
- **Google Gemini API Key:** Get yours at https://aistudio.google.com/app/apikey (free tier available)
- **Companies House API Key:** Register at https://developer.company-information.service.gov.uk/ (select **REST API**)

### 3. Prepare Company List

Edit `data/companies_template.csv` with your 20 companies:

```csv
ticker,company_number,company_name,sector
BARC,00489800,Barclays PLC,Financials
HSBC,00617987,HSBC Holdings PLC,Financials
BP,00102498,BP PLC,Energy
...
```

**Finding Company Numbers:**
- Search on Companies House: https://find-and-update.company-information.service.gov.uk/
- The company number is shown in the company profile (e.g., "00489800" for Barclays)

## Usage

### Basic Usage

Run the full pipeline:

```bash
python run_pipeline.py --companies data/companies_template.csv
```

This will:
- Download the latest annual reports for all companies
- Process them through the full pipeline
- Store results in `data/airo.db`

### Command-Line Options

```bash
# Specify a particular year
python run_pipeline.py --year 2023

# Skip download step (use existing PDFs)
python run_pipeline.py --skip-download

# Change log level
python run_pipeline.py --log-level DEBUG

# Custom companies file
python run_pipeline.py --companies my_companies.csv
```

### Output

The pipeline produces:

1. **Downloaded PDFs:** `output/pdfs/`
2. **SQLite Database:** `data/airo.db`
   - `mentions` table: AI-relevant text spans with classifications
   - `firms` table: Aggregated firm-year metrics
3. **Logs:** `logs/pipeline.log`

## Database Schema

### `mentions` Table

Each row represents one AI-relevant excerpt from an annual report:

| Column | Type | Description |
|--------|------|-------------|
| `mention_id` | TEXT | Unique ID (e.g., "BARC-2024-0001") |
| `firm_id` | TEXT | Company identifier (ticker) |
| `firm_name` | TEXT | Company name |
| `sector` | TEXT | Sector classification |
| `report_year` | INT | Fiscal year |
| `text_excerpt` | TEXT | The actual text span |
| `mention_type` | TEXT | risk_statement, adoption_use_case, governance_mitigation, etc. |
| `tier_1_category` | TEXT | operational_reliability, security_malicious_use, etc. |
| `tier_2_driver` | TEXT | hallucination_accuracy, cyber_enablement, etc. |
| `specificity_level` | TEXT | boilerplate, contextual, or concrete |
| `governance_maturity` | TEXT | none, basic, intermediate, or advanced |
| `confidence_score` | FLOAT | LLM confidence (0.0-1.0) |
| `reasoning_summary` | TEXT | LLM explanation |

### `firms` Table

Aggregated metrics for each firm-year:

| Column | Type | Description |
|--------|------|-------------|
| `firm_id` | TEXT | Company identifier |
| `report_year` | INT | Fiscal year |
| `total_ai_mentions` | INT | Count of all AI mentions |
| `total_ai_risk_mentions` | INT | Count of risk mentions |
| `dominant_tier_1_category` | TEXT | Most frequent risk category |
| `max_governance_maturity` | TEXT | Highest governance level observed |
| `specificity_ratio` | FLOAT | % of concrete mentions |
| `mitigation_gap_score` | FLOAT | High risk + low governance = gap |

## Querying Results

### Using SQLite

```bash
sqlite3 data/airo.db

# View all firms processed
SELECT firm_name, report_year, total_ai_mentions, total_ai_risk_mentions
FROM firms;

# View mentions for a specific firm
SELECT mention_type, tier_1_category, specificity_level, text_excerpt
FROM mentions
WHERE firm_id = 'BARC' AND report_year = 2024;

# Find high-risk firms with low governance
SELECT firm_name, total_ai_risk_mentions, max_governance_maturity, mitigation_gap_score
FROM firms
WHERE mitigation_gap_score > 0.5
ORDER BY mitigation_gap_score DESC;
```

### Using Python

```python
from src.database import Database

db = Database()
session = db.get_session()

# Get all mentions
from src.database import Mention
mentions = session.query(Mention).all()

# Get firm metrics
from src.database import Firm
firms = session.query(Firm).all()

session.close()
```

## Architecture

```
pipeline/
├── src/
│   ├── config.py           # Configuration management
│   ├── companies_house.py  # Companies House API client
│   ├── pdf_extractor.py    # PDF text extraction
│   ├── chunker.py          # Text chunking
│   ├── llm_classifier.py   # Claude-based classification
│   ├── database.py         # Database models and operations
│   ├── aggregator.py       # Firm-level aggregation
│   └── pipeline.py         # Main orchestrator
├── data/
│   ├── companies_template.csv  # Input: company list
│   └── airo.db                 # Output: SQLite database
├── output/
│   └── pdfs/               # Downloaded annual reports
├── logs/
│   └── pipeline.log        # Execution logs
├── run_pipeline.py         # CLI entry point
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Cost Estimation

For Google Gemini 2.0 Flash:
- **Free tier:** 15 requests per minute, 1500 requests per day
- **Input:** $0.075 per million tokens (if exceeds free tier)
- **Output:** $0.30 per million tokens (if exceeds free tier)

**Typical report processing:**
- 100-page annual report → ~1000 candidate spans
- After relevance filtering → ~50-100 relevant mentions
- Cost per report: **~$0.10-0.50** (or FREE within daily limits)

**For 20 companies:** **~$2-10 total** (likely FREE if processed within daily rate limits)

## Troubleshooting

### "Missing API keys" error

Make sure you've created a `.env` or `.env.local` file with valid API keys:
```bash
cp .env.template .env
# Edit .env with your actual keys
```

### "No PDF found for company" warning

The Companies House API may not have the document available. Possible reasons:
- Company number is incorrect
- No annual accounts filed for that year
- Document not yet available via API

Check the company manually: https://find-and-update.company-information.service.gov.uk/

### Rate limiting

Companies House API has rate limits. If you hit them:
- Add delays between requests (already implemented with retry logic)
- Process companies in smaller batches

### Low relevance rate

If very few spans are classified as relevant:
- Check that companies actually discuss AI in their reports
- Try a different report year
- Review the LLM classification prompt in `src/llm_classifier.py`

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
ruff check src/
```

### Adding Custom Classification Dimensions

To add new classification dimensions:

1. Update the prompt in `src/llm_classifier.py`
2. Add fields to `Classification` dataclass
3. Update database schema in `src/database.py`
4. Update aggregation logic in `src/aggregator.py`

## Next Steps (v2)

- Automate ticker → company number mapping
- Add support for multiple report years per company
- Implement sector-level aggregation views
- Add data export to CSV/JSON for dashboard
- Create quality control/review interface
- Add support for HTML annual reports (many FTSE companies publish HTML versions)

## References

- [AIRO Data Pipeline Specification](../AIRO_DATA_PIPELINE_SPECIFICATION.md)
- [Companies House API Documentation](https://developer.company-information.service.gov.uk/api/docs/)
- [Google Gemini API Documentation](https://ai.google.dev/api)
- [Gemini API Key](https://aistudio.google.com/app/apikey)

## Support

For issues or questions, check:
1. `logs/pipeline.log` for detailed error messages
2. Database using `sqlite3 data/airo.db`
3. Project specification in `../AIRO_DATA_PIPELINE_SPECIFICATION.md`
