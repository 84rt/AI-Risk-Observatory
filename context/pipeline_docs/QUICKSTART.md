# AIRO Pipeline Quick Start

## 1. Setup (5 minutes)

```bash
cd pipeline
./scripts/setup.sh
```

## 2. Get API Keys

### Google Gemini
1. Go to https://aistudio.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy key
5. **Free tier:** 15 requests/min, 1500/day

### Companies House
1. Go to https://developer.company-information.service.gov.uk/
2. Register for an account
3. Select **REST API** when asked
4. Create API key
5. Copy key

## 3. Configure

Edit `.env.local` or create `.env`:
```bash
GEMINI_API_KEY=AIza...
COMPANIES_HOUSE_API_KEY=...
```

## 4. Add Companies

Edit `data/companies_template.csv`:

Find company numbers at: https://find-and-update.company-information.service.gov.uk/

Example:
```csv
ticker,company_number,company_name,sector
BARC,00489800,Barclays PLC,Financials
HSBC,00617987,HSBC Holdings PLC,Financials
BP,00102498,BP PLC,Energy
SHEL,00140141,Shell plc,Energy
GSK,03888792,GSK plc,Healthcare
AZN,02723534,AstraZeneca PLC,Healthcare
VOD,01833679,Vodafone Group PLC,Telecommunications
BT,04190816,BT Group plc,Telecommunications
LLOY,00095000,Lloyds Banking Group plc,Financials
STAN,00889514,Standard Chartered PLC,Financials
```

## 5. Test with One Company

```bash
source venv/bin/activate

python scripts/test_single_company.py \
  --company-number 00489800 \
  --company-name "Barclays PLC" \
  --ticker BARC \
  --sector Financials \
  --max-candidates 5
```

This will:
- Download 1 annual report
- Process 5 text spans
- Show classifications
- Cost: ~$0.50

## 6. Run Full Pipeline

```bash
python run_pipeline.py --companies data/companies_template.csv
```

This will process all companies in your CSV.

Estimated time for 20 companies: 30-60 minutes
Estimated cost: **$2-10** (or **FREE** with Gemini's free tier!)

## 7. View Results

```bash
# Summary statistics
python scripts/query_db.py

# Export to CSV
python scripts/query_db.py --export

# Query directly
sqlite3 data/airo.db
> SELECT * FROM firms;
> SELECT * FROM mentions LIMIT 5;
```

## Common Commands

```bash
# Run with specific year
python run_pipeline.py --year 2023

# Skip download (use existing PDFs)
python run_pipeline.py --skip-download

# Debug mode
python run_pipeline.py --log-level DEBUG

# Test single company
python scripts/test_single_company.py --company-number 00489800 --company-name "Barclays" --ticker BARC --sector Financials
```

## Troubleshooting

**"Missing API keys" error**
→ Check `.env` file exists and has valid keys

**"No PDF found" warning**
→ Company number may be wrong, or no accounts filed for that year

**Rate limiting from Companies House**
→ Pipeline has built-in retry logic, it will wait and retry automatically

**Low relevance rate (<5%)**
→ Normal! Most reports have limited AI discussion. Try financial sector companies first.

**Out of memory**
→ Reduce `max_chunk_length` in `src/chunker.py` or process fewer companies at once

## File Locations

- **Config:** `.env`
- **Input:** `data/reference/companies_template.csv`
- **Database:** `data/db/airo.db`
- **PDFs:** `data/raw/pdfs/`
- **Logs:** `data/logs/pipeline/pipeline.log`
- **Exports:** `data/results/mentions_export.csv`

## Cost Breakdown

Google Gemini 2.0 Flash pricing:
- **FREE tier:** 15 RPM, 1500 requests/day
- $0.075 per 1M input tokens (paid)
- $0.30 per 1M output tokens (paid)

Typical usage per company:
- 100-page report → 1000 candidates
- After filtering → 50-100 classifications
- Cost per company: **$0.10-0.50** (or FREE!)

**Pro tip:** Process in batches to stay within free tier limits!

## Next Steps

1. Process your 20 companies
2. Review results with `scripts/query_db.py`
3. Export data for dashboard: `scripts/query_db.py --export`
4. Connect to Next.js dashboard in `/dashboard`

## Need Help?

- Check logs: `tail -f data/logs/pipeline/pipeline.log`
- Read full docs: `README.md`
- Review spec: `../AIRO_DATA_PIPELINE_SPECIFICATION.md`
