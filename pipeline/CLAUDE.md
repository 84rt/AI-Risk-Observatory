# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIRO (AI Risk Observatory) is a data pipeline that processes UK company annual reports (FTSE 350) to identify and classify AI-related risk disclosures. The pipeline downloads reports in iXBRL/XHTML or PDF format, extracts text, preprocesses and filters relevant sections, chunks the text, classifies mentions using Google Gemini LLM, and stores structured results in a SQLite database.

## Common Commands

### Setup and Environment

```bash
# Initial setup
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create environment file
cp .env.template .env
# Edit .env to add GEMINI_API_KEY and COMPANIES_HOUSE_API_KEY
```

### Running the Pipeline

```bash
# Run full pipeline with default settings
python run_pipeline.py --companies data/companies_template.csv

# Run for specific year
python run_pipeline.py --year 2024

# Skip download step (use existing reports)
python run_pipeline.py --skip-download

# Enable debug logging
python run_pipeline.py --log-level DEBUG
```

### Testing and Development

```bash
# Run tests
pytest tests/

# Format code
black src/
ruff check src/

# Test preprocessing comparison
python test_preprocessing_comparison.py

# Test single company
python scripts/test_single_company.py
```

### Database Queries

```bash
# Access SQLite database
sqlite3 data/airo.db

# View all firms
SELECT firm_name, report_year, total_ai_mentions, total_ai_risk_mentions FROM firms;

# View mentions for specific firm
SELECT mention_type, tier_1_category, specificity_level, text_excerpt
FROM mentions
WHERE firm_id = 'BARC' AND report_year = 2024;
```

## Architecture Overview

### Pipeline Flow

The pipeline operates in 5 main stages:

```
1. Download → 2. Extract → 3. Preprocess → 4. Chunk → 5. Classify → Store → Aggregate
```

**Stage 1: Download Reports**
- Primary source: `filings.xbrl.org` API for iXBRL/XHTML files (2,437 UK company filings available)
- Fallback: Companies House API for PDF documents
- Implementation: `src/xbrl_filings_client.py` and `src/companies_house.py`
- Output: `output/reports/ixbrl/*.xhtml` or `output/reports/pdfs/*.pdf`

**Stage 2: Text Extraction**
- iXBRL/XHTML: `src/ixbrl_extractor.py` - HTML parser with automatic spacing cleanup
- PDF: `src/pdf_extractor.py` - PyMuPDF-based extraction (fallback)
- Key feature: Regex-based text cleaning happens during extraction (`_clean_text()` at line 176-199)
- Output: `ExtractedReport` object with `TextSpan` objects organized by section

**Stage 3: Preprocessing (NEW - December 2025)**
- Implementation: `src/preprocessor.py`
- Two strategies available:
  - `risk_only`: Extract only risk-related sections (~11% retention, works better with PDFs)
  - `keyword`: AI/ML and risk keyword filtering (~12% retention, **recommended for iXBRL**)
- Output: Markdown files in `output/preprocessed/{strategy}/`
- Note: Gemini text cleaning is currently disabled - regex cleanup in extraction is sufficient

**Stage 4: Chunking**
- Implementation: `src/chunker.py`
- Splits preprocessed text into candidate spans (paragraph-based by default)
- Output: List of `CandidateSpan` objects with context metadata

**Stage 5: LLM Classification**
- Implementation: `src/llm_classifier.py`
- Uses Google Gemini (gemini-2.0-flash-exp) with structured JSON output
- Classifies each span for:
  - Relevance (is_relevant boolean)
  - Mention type (risk_statement, adoption_use_case, governance_mitigation, etc.)
  - Risk taxonomy (tier_1_category, tier_2_driver)
  - Specificity level (boilerplate, contextual, concrete)
  - Governance maturity (none, basic, intermediate, advanced)
  - Confidence score (0.0-1.0)
- Includes retry logic with exponential backoff via `tenacity`

**Stage 6: Database Storage**
- Implementation: `src/database.py`
- SQLAlchemy ORM models: `Mention` and `Firm`
- `Mention`: Span-level data with full classification
- `Firm`: Aggregated firm-year metrics

**Stage 7: Aggregation**
- Implementation: `src/aggregator.py`
- Rolls up mention-level data to firm-year statistics
- Calculates: dominant risk categories, specificity ratios, mitigation gaps

### Key Architectural Decisions

**Data Source Strategy**
- filings.xbrl.org provides superior data quality (structured iXBRL) vs. Companies House PDFs
- Automatic fallback ensures coverage when iXBRL not available
- No authentication required for filings.xbrl.org

**Preprocessing Strategy**
- **Keyword-based filtering is recommended** (see PREPROCESSING_GUIDE.md)
- iXBRL documents lack clear section boundaries, making risk_only strategy unreliable
- Keyword approach: ~8-10% retention, high recall, works with both formats

**Text Cleaning Approach**
- Regex-based cleaning during extraction (fixes spacing issues like "risk s" → "risks")
- Gemini-based cleaning disabled due to unwanted summarization (reduces text by ~60%)
- Current approach is sufficient and preserves content fidelity

**LLM Design**
- Temperature set to 0.0 for deterministic classification
- JSON response mode enforced via `response_mime_type`
- Comprehensive prompt with step-by-step reasoning structure
- Retry logic handles API failures gracefully

## File Organization

```
pipeline/
├── src/                          # Core modules
│   ├── config.py                 # Settings from .env files
│   ├── pipeline.py               # Main orchestrator
│   ├── xbrl_filings_client.py    # filings.xbrl.org API client
│   ├── companies_house.py        # Companies House API client
│   ├── ixbrl_extractor.py        # iXBRL/XHTML extraction + text cleaning
│   ├── pdf_extractor.py          # PDF extraction (fallback)
│   ├── preprocessor.py           # Text filtering strategies
│   ├── chunker.py                # Text chunking logic
│   ├── llm_classifier.py         # Gemini classification
│   ├── database.py               # SQLAlchemy models
│   └── aggregator.py             # Firm-level aggregation
├── data/
│   ├── companies_template.csv    # Input: company list
│   └── airo.db                   # Output: SQLite database
├── output/
│   ├── reports/
│   │   ├── ixbrl/                # Downloaded iXBRL files
│   │   └── pdfs/                 # Downloaded PDF files
│   └── preprocessed/
│       ├── risk_only/            # Risk section extracts
│       └── keyword/              # Keyword-filtered extracts
├── logs/
│   └── pipeline.log              # Execution logs
├── run_pipeline.py               # CLI entry point
└── requirements.txt              # Python dependencies
```

## Configuration and Settings

Settings are managed via `pydantic-settings` in `src/config.py`:

**Required Environment Variables:**
- `GEMINI_API_KEY`: Google Gemini API key (get from https://aistudio.google.com/app/apikey)
- `COMPANIES_HOUSE_API_KEY`: Companies House REST API key

**Optional Environment Variables:**
- `GEMINI_MODEL`: Default is "gemini-2.0-flash-exp"
- `DATABASE_PATH`: Default is "./data/airo.db"
- `LOG_LEVEL`: Default is "INFO"

Environment files are checked in order: `.env.local`, then `.env`

## Database Schema

### `mentions` Table
Each row represents one AI-relevant excerpt from an annual report.

**Key columns:**
- `mention_id` (PK): Unique identifier (format: "TICKER-YEAR-####")
- `firm_id`, `firm_name`, `sector`, `report_year`: Context
- `text_excerpt`: The actual text span
- `report_section`, `page_number`: Source location
- `mention_type`: risk_statement | adoption_use_case | governance_mitigation | incident_event | regulatory_environment | strategy_opportunity
- `tier_1_category`: operational_reliability | security_malicious_use | legal_regulatory_compliance | workforce_human_capital | societal_ethical_reputational | frontier_systemic
- `tier_2_driver`: More specific risk driver (see llm_classifier.py lines 186-202)
- `specificity_level`: boilerplate | contextual | concrete
- `governance_maturity`: none | basic | intermediate | advanced
- `confidence_score`: LLM confidence (0.0-1.0)
- `model_version`, `extraction_date`: Provenance tracking

### `firms` Table
Aggregated metrics for each firm-year.

**Key columns:**
- `firm_id`, `report_year` (composite PK)
- `total_ai_mentions`, `total_ai_risk_mentions`: Counts
- `dominant_tier_1_category`: Most frequent risk category
- `max_governance_maturity`: Highest governance level observed
- `specificity_ratio`: Percentage of concrete mentions
- `mitigation_gap_score`: High risk + low governance = gap

## Important Implementation Details

### Text Extraction and Cleaning

Text spacing issues in iXBRL files are fixed automatically during extraction:

```python
# src/ixbrl_extractor.py:176-199
def _clean_text(self, text: str) -> str:
    # Fix "risk s" → "risks"
    text = re.sub(r'([a-z]{3,})\s+([a-z])(?=\s|[,.;:\)]|$)', r'\1\2', text)
    # Fix "c ust" → "cust"
    text = re.sub(r'(?<=\s)([a-z])\s+([a-z]{3,})', r'\1\2', text)
    return text
```

This happens for every text span before it's added to the ExtractedReport. Do not attempt additional text cleaning with Gemini unless specifically addressing a new issue.

### Preprocessing Strategies

When modifying preprocessing logic in `src/preprocessor.py`:

**risk_only strategy:**
- Filters spans where section name contains: "principal_risk", "risk_management", "risk_factor", "risk_review"
- Works best with PDFs that have clear section headings
- May fail on iXBRL documents with poor section detection

**keyword strategy (RECOMMENDED):**
- Always includes heading spans
- Includes paragraph spans matching AI keywords OR risk keywords
- AI keywords: artificial intelligence, machine learning, LLM, generative ai, etc. (25 patterns)
- Risk keywords: risk, uncertainty, threat, challenge, vulnerability, etc. (12 patterns)
- More robust across document formats

### LLM Classifier Prompt Structure

The classification prompt in `src/llm_classifier.py` is structured as a 10-step decision tree:
1. Is AI-relevant? (Yes/No gate)
2. Mention type classification
3. AI specificity level
4. Frontier tech flag
5. Risk taxonomy (tier 1 & 2)
6. Specificity level
7. Materiality signal
8. Governance & mitigation
9. Confidence score
10. Reasoning summary

When modifying the prompt or adding new classification dimensions:
1. Update the prompt in `_build_classification_prompt()`
2. Add fields to the `Classification` dataclass
3. Update the `Mention` model in `database.py`
4. Update aggregation logic in `aggregator.py`

### API Rate Limits and Costs

**Google Gemini 2.0 Flash (free tier):**
- 15 requests per minute
- 1,500 requests per day
- Pipeline automatically handles rate limiting via retry logic

**Typical costs per report:**
- 100-page report → ~1,000 candidate spans
- After relevance filtering → ~50-100 relevant mentions
- Cost: ~$0.10-0.50 per report (or FREE within daily limits)
- 20 companies: ~$2-10 total

**Companies House API:**
- Has rate limits (handled by retry logic with delays)
- Process companies in smaller batches if hitting limits

## Common Development Patterns

### Adding a New Classification Dimension

1. Add field to `Classification` dataclass in `src/llm_classifier.py`
2. Update the prompt in `_build_classification_prompt()` to extract it
3. Add column to `Mention` model in `src/database.py`
4. Update `_parse_response()` if special handling needed
5. Update aggregation logic in `src/aggregator.py` if firm-level rollup needed
6. Run database migration or recreate database

### Processing a Single Company for Testing

```python
from pathlib import Path
from src.xbrl_filings_client import XBRLFilingsClient
from src.ixbrl_extractor import extract_text_from_ixbrl
from src.preprocessor import Preprocessor, PreprocessingStrategy

# Download
client = XBRLFilingsClient()
result = client.search_filings(company_name="Shell plc", limit=1)
file_path = client.download_filing(result[0])

# Extract
extracted = extract_text_from_ixbrl(file_path)

# Preprocess
preprocessor = Preprocessor(strategy=PreprocessingStrategy.KEYWORD)
preprocessed = preprocessor.process(extracted)

# Save
preprocessor.save_to_file(preprocessed, output_dir=Path("output/test"))
```

### Debugging Classification Issues

Check logs at different levels:
- `logs/pipeline.log`: Full execution log
- Set `--log-level DEBUG` for detailed LLM interactions
- Review `confidence_score` in database - scores below 0.5 need human review
- Use `reasoning_summary` field to understand LLM decisions

### Querying Results Programmatically

```python
from src.database import Database, Mention, Firm
from sqlalchemy import func

db = Database()
session = db.get_session()

# Get high-risk firms
high_risk = session.query(Firm).filter(
    Firm.total_ai_risk_mentions > 10,
    Firm.max_governance_maturity.in_(['none', 'basic'])
).all()

# Get all operational reliability risks
op_risks = session.query(Mention).filter(
    Mention.tier_1_category == 'operational_reliability'
).all()

session.close()
```

## Known Limitations and Issues

1. **iXBRL Section Detection**: Section boundaries are not reliably detected in iXBRL format. Use keyword-based preprocessing as workaround.

2. **Gemini Text Cleaning Not Used**: Gemini-based text cleaning was attempted but abandoned because Gemini tends to summarize rather than just clean text (reduces text by ~60%). Regex-based cleaning during extraction is sufficient.

3. **PDF Quality**: Some PDFs from Companies House may have poor OCR quality. iXBRL format is strongly preferred.

4. **Rate Limiting**: Large batches may hit Gemini API rate limits. The pipeline has retry logic but may need manual intervention for very large runs.

## Important Data Flow Details

The data flows through distinct stages with clear boundaries:

1. **Raw HTML/PDF → ExtractedReport**: Text extraction happens here, including all text cleaning
2. **ExtractedReport → PreprocessedReport**: Filtering by strategy (risk_only or keyword)
3. **PreprocessedReport → CandidateSpans**: Chunking for LLM processing
4. **CandidateSpans → Classifications**: LLM analysis
5. **Classifications → Database**: Storage in mentions table
6. **Mentions → Firms**: Aggregation to firm-year level

Each stage has well-defined input/output types. When debugging, check the data at each stage boundary.

## References and Documentation

- Full pipeline specification: `../AIRO_DATA_PIPELINE_SPECIFICATION.md`
- Data flow explanation: `DATAFLOW_EXPLAINED.md`
- Preprocessing guide: `PREPROCESSING_GUIDE.md`
- Pipeline status: `PIPELINE_STATUS_SUMMARY.md`
- Companies House API: https://developer.company-information.service.gov.uk/api/docs/
- Google Gemini API: https://ai.google.dev/api
- filings.xbrl.org: Public XBRL filings database (no auth required)
