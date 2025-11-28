# AI Risk Observatory (AIRO)

The AI Risk Observatory tracks how UK public companies are adopting, governing, and disclosing risks related to Artificial Intelligence.

This project consists of two main components:
1.  **Data Pipeline:** Ingests annual reports, extracts AI-relevant text, and uses LLMs to classify mentions according to a risk taxonomy.
2.  **Dashboard:** A Next.js web application to visualize these trends over time, by sector, and by risk category.

## 1. Project Goals

The objective is to create a consistent, data-driven methodology to:
*   **Identify Emerging Threats:** Track which specific AI risks (e.g., "Hallucination", "Cybersecurity") are rising.
*   **Monitor Adoption vs. Governance:** Spot gaps where adoption is high but governance maturity is low.
*   **Prioritize Interventions:** Enable regulators (like the AI Safety Institute) to focus on high-risk sectors.

**Core Research Questions:**
1.  Can LLMs reliably identify and classify AI risk disclosures in financial filings compared to human analysts?
2.  What are the trends in AI risk severity, nature, and mitigation across UK sectors?

## 2. Architecture Overview

### A. Data Processing Pipeline
*   **Source:** Annual Financial Reports (prioritizing XHTML/HTML, falling back to PDF).
*   **Ingestion:** Converts documents to Markdown to preserve structure (headers, tables).
*   **The "Funnel" Strategy:**
    1.  **Keyword Sieve:** Fast filtering of sections containing AI keywords.
    2.  **Relevance Check (LLM):** Determines if the context is truly about Risk, Adoption, or Governance.
    3.  **Deep Classification (LLM):** Applies the specific taxonomy (Tier 1 Risk Categories, Specificity Score, Governance Maturity).
*   **Output:** Structured data stored in a SQL database (SQLite/Postgres).

### B. Dashboard (Frontend)
*   **Tech Stack:** Next.js, Recharts/Tremor for visualization.
*   **Key Metrics:**
    *   **Risk Composition:** Stacked area charts of risk categories over time.
    *   **Maturity Assessment:** "Cowboy" detection (High Risk / Low Governance).
    *   **Frontier Signal:** Tracking specific mentions of Generative AI/LLMs.

## 3. Implementation Plan

### Phase 1: Prototype & Benchmarking
- [ ] Develop parsing script for PDF/XHTML to Markdown.
- [ ] Create a "Gold Standard" test dataset with human annotations.
- [ ] Benchmark LLM classification accuracy against human labels.
- [ ] Establish "Human-in-the-loop" workflow for ongoing validation.

### Phase 2: Pipeline Production
- [ ] Implement the 3-stage classification funnel.
- [ ] Process a pilot batch of FTSE 350 reports.
- [ ] Populate the SQL database.

### Phase 3: Visualization
- [ ] Build Sector Heatmaps and Risk Trend charts.
- [ ] Deploy dashboard for stakeholder review.

## 4. Repository Structure

*   `/airo-dashboard`: Next.js frontend application.
*   `/pipeline`: (Planned) Python scripts for ETL and Classification.
*   `/data`: Mock data and schemas.
*   `AIRO_DATA_PIPELINE_SPECIFICATION.md`: The authoritative technical spec for the data model.

