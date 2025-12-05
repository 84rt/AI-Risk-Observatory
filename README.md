# AI Risk Observatory (AIRO)

The AI Risk Observatory tracks how UK public companies are adopting, governing, and disclosing risks related to Artificial Intelligence.

**Components:**
- **Data Pipeline:** Ingests annual reports, extracts AI-relevant text, and uses LLMs to tag and classify mentions using a risk taxonomy.
- **Dashboard:** Next.js web application to visualize risk disclosure trends by sector, risk category, time, and governance maturity.

**Goals:**
- Identify emerging AI threats and disclosure patterns.
- Monitor gaps between AI adoption and governance.
- Enable regulators and stakeholders to focus on high-risk sectors.

**Key Questions:**
- How reliably can LLMs identify/classify AI risk disclosures in financial filings?
- What are the trends in AI risk severity, nature, and mitigation across sectors?

**Architecture:**
- **Pipeline:** Processes annual reports, extracts content, applies a multi-stage LLM filter for relevance and classification, outputs structured data to SQL (SQLite/PostgreSQL).
- **Dashboard:** Built with Next.js and Recharts/Tremor, offers visualizations (e.g., risk composition, governance maturity, sector trends).

**Repo Structure:**
- `/airo-dashboard`: Next.js frontend
- `/pipeline`: (Planned) Python ETL/classification scripts
- `/data`: Mock data and schemas
- `AIRO_DATA_PIPELINE_SPECIFICATION.md`: Data model specification

# Pipeline Status

✅ **v1 Pipeline Complete!** The data pipeline is now fully implemented and ready for testing.

## Quick Start

```bash
cd pipeline
./scripts/setup.sh                    # Install dependencies
# Edit .env with your API keys
# Edit data/companies_template.csv with 20 companies
python run_pipeline.py                # Run the pipeline
python scripts/query_db.py            # View results
```

See [`pipeline/README.md`](pipeline/README.md) for detailed documentation.

## What's Included in v1

- ✅ Companies House API integration for fetching annual reports
- ✅ PDF text extraction with section detection (PyMuPDF)
- ✅ Intelligent text chunking for LLM processing
- ✅ Two-stage LLM classification (Google Gemini 2.0 Flash):
  - Stage 1: Relevance detection
  - Stage 2: Risk taxonomy classification
- ✅ SQLite database with full schema (mentions + firms tables)
- ✅ Firm-level aggregation (specificity ratio, mitigation gap, etc.)
- ✅ Comprehensive error handling and logging
- ✅ CLI tools and utilities
- 💰 **Cost-effective:** FREE tier covers most usage!

## Next Steps

1. **Test the pipeline:** Run on your 20 companies
2. **Validate classifications:** Review sample mentions for accuracy
3. **Connect to dashboard:** Export data for visualization
4. **Iterate on v2:** Add automated ticker mapping, multi-year support 