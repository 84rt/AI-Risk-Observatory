## 1. Project Overview
AI Risk Observatory surfaces AI risk signals in UK annual reports to support policy and oversight. It targets UK AI safety and policy teams, producing two outputs: a traceable dataset and a dashboard that summarizes risk posture.

## 2. High-Level Architecture
- Data pipeline: ingest, clean, classify filings.
- Data storage: append-only, versioned `/data` tree.
- Visualization dashboard: Next.js app for exploration.

Diagram:
```
Data Sources -> Pipeline -> /data (raw/processed/results) -> Dashboard
```

## 3. Repository Structure
- `/dashboard` — Next.js app for risk visualization.
- `/pipeline` — end-to-end data processing and classification.
- `/data` — central store for raw, processed, results, logs.
- `/context` — scope, specs, taxonomy, LLM guidance.

Pipeline and dashboard live together so the dashboard can track model changes quickly. Data is colocated and versioned to preserve lineage, enable iteration, and support audits.

## 4. Data & Storage Model
`/data` holds raw inputs (iXBRL, PDFs), processed text, annotations (human + model), logs, and outputs. Everything is append-only and versioned; records remain traceable from raw source to final result.

## 5. Pipeline Stages
Execution order:
1) Ingestion — download filings (inputs: company list; outputs: iXBRL/PDF in `/data/raw`).
2) Preprocessing — OCR/clean/normalize (inputs: raw; outputs: markdown in `/data/processed`).
3) Processing — chunk AI mentions (inputs: processed; outputs: chunks in `/data/processed/chunks`).
4) Post-processing — LLM classification, enrichment, QA (inputs: chunks; outputs: DB + JSON in `/data/results`).
5) Aggregation & export — rollups and dashboard extracts (inputs: results; outputs: dashboard-ready views).

## 6. Classification Blueprint
Core dimensions: AI harms, AI adaptation, AI risk disclosures. Additional layers: severity and mitigation (“cowboy risk”), vendor extraction, workforce impact. Taxonomy lives in `/context`.

## 7. Evaluation & Quality Control
- Human-annotated golden set; accuracy target ~90%.
- Anti-error: multi-run consistency checks, multi-model agreement, confidence thresholds.
- Human review triggers on low-confidence or disagreement cases.

## 8. Versioning & Iteration Model
Classifiers are versioned independently. Each run logs classifier version, model used, and timestamp. Data is never deleted—new versions are added to allow comparison, rollback, and audits.

## 9. Logs & Auditability
Run-level logs plus per-record model I/O, confidence, and reasoning are stored for transparency, debugging, and policy trust.

## 10. Adding New Data or Years
Add new filings, years, or classifiers; rerun ingestion → preprocessing → processing → post-processing → exports. Prior runs stay preserved for comparison.
- Manual fallback: filings.xbrl.org is a reliable source to download individual iXBRL filings by LEI/company when automated downloads fail.

## 11. Running the Project
- Pipeline: `cd pipeline && python run_pipeline.py --companies ../data/reference/companies_template.csv`
- Dashboard: `cd dashboard && npm install && npm run dev`
See `/pipeline/QUICKSTART.md` for details.

## 12. Further Documentation
See `/context` for taxonomy, specs, experimental notes, and design decisions.
