# AI Risk Observatory

The Observatory tracks AI-related risk, adoption, and third-party AI exposure across UK public-company annual reports — with a focus on Critical National Infrastructure sectors. Built for policymakers and resilience researchers, sponsored by the [AI Security Institute (AISI)](https://www.aisi.gov.uk/).

**Current coverage: 1,362 companies · 9,821 reports · 2020–2026**

---

## Why annual reports?

Annual reports are audited, legally mandated, and published on a consistent schedule — making them a reliable baseline for tracking real corporate AI exposure at scale. Unlike surveys or voluntary disclosures, they reflect what companies are prepared to stand behind in a regulated context.

## What we measure

Each report is processed end-to-end: iXBRL source files are downloaded, converted to structured text, and AI-related passages are extracted and classified across three core dimensions:

- **AI risk disclosure** — is AI named as a material risk to the business?
- **LLM adoption** — is the company reporting active use of large language models?
- **AI as a cybersecurity threat** — is AI-enabled threat specifically called out?

Additional signals captured include vendor exposure, workforce impact, and substantiveness of disclosure (distinguishing boilerplate from meaningful reporting).

## Repository structure

```
/dashboard   — Next.js visualization app
/pipeline    — end-to-end data processing and classification pipeline
/scripts     — batch orchestration and data-build scripts
/data        — raw inputs, processed outputs, and results (gitignored except reference files)
```

Pipeline and dashboard live together so the dashboard can track model and methodology changes quickly. Data is colocated and append-only to preserve full lineage from raw filing to final result.

## Quick start

**Run the dashboard locally**
```bash
cd dashboard
npm install
npm run dev
# open http://localhost:3000
```

**Run the pipeline**
```bash
cd pipeline
source venv/bin/activate        # Fish shell: source venv/bin/activate.fish
pip install -r requirements.txt
python run_pipeline.py --companies ../data/reference/companies_template.csv
```

See `pipeline/README.md` for the full operational guide.

## How the pipeline works

1. **Ingestion** — downloads iXBRL filings via [financialreports.eu](https://financialreports.eu/) and Companies House, organized by company and fiscal year
2. **Preprocessing** — converts iXBRL to clean markdown; produces per-document metadata sidecars
3. **Chunking** — extracts AI-mention passages with surrounding context windows and deduplicates overlapping spans
4. **Classification** — LLM classifiers label each chunk across risk type, adoption signals, vendor exposure, and substantiveness
5. **Aggregation** — results are rolled up and exported to `dashboard/data/dashboard-data.json`

Classifier prompts and the full taxonomy live in `pipeline/prompts/`. Human-annotated evaluation data lives in `data/golden_set/`.

## Data & quality

All pipeline outputs are append-only and versioned. Every record is traceable from raw source filing to final classification — including the model used, classifier version, confidence score, and full prompt/response logs.

Accuracy is evaluated against a human-annotated golden set with a target of ~90% agreement. Low-confidence or multi-model disagreement cases are flagged for human review.

## Sponsorship & data

- **Main sponsor:** [AI Security Institute (AISI)](https://www.aisi.gov.uk/)
- **Data provider:** [financialreports.eu](https://financialreports.eu/)
