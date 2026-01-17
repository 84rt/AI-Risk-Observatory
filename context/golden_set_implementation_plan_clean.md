# Golden Set Implementation Plan (Clean)

## Aim
Deliver Phase II of the project by producing a reliable, human‑annotated golden set of AI mentions and running LLM classifiers against it for benchmarking.

## End State
- A curated golden set of company reports with AI‑mention chunks.
- Human gold‑standard annotations stored in a database.
- LLM classifier outputs stored per run with confidence scores.
- Prompts for all LLM classifiers kept in a separate YAML file for iteration.

## Process Overview
### 1) Ingestion
- Collect iXBRL/PDF filings for the target company list.
- Store raw files and download metadata in `/data/raw/`.

### 2) Preprocessing (documents)
- Convert iXBRL/PDF to human‑readable markdown.
- Run QA (regex + model‑assisted corrections) and optional manual review.
- Store markdown in `/data/processed/<run_id>/documents/`.
- Store per‑document JSON sidecars in `/data/processed/<run_id>/metadata/`.
- Maintain `documents_manifest.json` for each run.

### 3) Chunking (AI mentions)
- Identify mentions of AI keywords (AI, ML, LLM, Artificial Intelligence, etc.).
- Create context chunks per mention and save in `/data/processed/<run_id>/chunks/`.
- Store per‑document metadata about number of mentions/chunks (including zero mentions).
- Chunks are the boundary between preprocessing and classification.

### 4) Human Annotation (gold standard)
- Annotate each chunk to create the human gold standard.
- We need to build tooling for a human being to be able to annotate the database easily (following the same flow as the LLM classifiers will).
- Save human annotations separately from LLM runs.

### 5) LLM Classification (processing)
Run the following classifiers on each chunk. All assigned tags carry confidence scores.

#### 5.1 Mention Type Classifier (multi‑label)
Tags (not mutually exclusive):
- Adoption
- Risk
- Harm
- Vendor
- General/Ambiguous

#### 5.2 Adoption Classifier (if Adoption tag present)
Classify AI type (non‑exclusive, confidence per label):
- Non‑LLM
- LLM
- Agentic AI
Low confidence is allowed when the text is unclear.

#### 5.3 Risk Classifier (if Risk tag present)
Assign zero or more risk taxonomy tags with confidence plus a substantiveness score (0–1).

#### 5.4 Vendor Classifier (if Vendor tag present)
Assign vendor tags with confidence:
- Google
- Microsoft
- OpenAI
- Internal
- Undisclosed
- Other (specify)

#### 5.5 Harm + General/Ambiguous (if tags present)
Store the excerpt and tag confidence as‑is.

### 6) Confidence Thresholds
- Process all tags with confidence > 0 by default.
- Keep the threshold configurable (e.g., filter below 0.2 later).

## Risk Taxonomy (for Risk Classifier)
1. Strategic & Market
    Scope: competitive positioning, failure to adopt, business model disruption, demand shifts.
2. Product & Model Performance
    Scope: model accuracy, hallucinations, bias in outputs, brittleness, quality regressions.
3. Operational & Reliability
    Scope: integration failures, uptime, scalability, MLOps failures, monitoring gaps.
4. Cybersecurity (Model/AI‑Specific)
    Scope: prompt injection, model inversion, data poisoning, adversarial inputs.
5. Data Privacy & Governance
    Scope: data leakage, consent, retention, cross‑border transfer, training data provenance.
6. Legal & Regulatory
    Scope: AI Act compliance, IP infringement, liability exposure, consumer protection.
7. Trust, Reputation & Information Integrity
    Scope: misinformation/deepfakes, brand damage, loss of confidence.
8. Human & Societal Impacts
    Scope: workforce displacement, human rights, discrimination impacts (societal lens).
9. Supply Chain & Third‑Party Dependence
    Scope: vendor concentration, API outages, subcontractor risk, downstream misuse.
10. Financial & Reporting
    Scope: valuation impacts, revenue recognition, auditability, controls, disclosure risk.

## Data Layout
- `/data/raw/` for original iXBRL files and download metadata.
- `/data/processed/<run_id>/documents/` for markdown outputs.
- `/data/processed/<run_id>/metadata/` for JSON sidecars per document.
- `/data/processed/<run_id>/documents_manifest.json` for processed document index.
- `/data/processed/<run_id>/chunks/` for AI‑mention chunks (JSONL).
- `/data/results/` reserved for classifier outputs.
