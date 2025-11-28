# AIRO Data Processing Pipeline & Database Specification

## Document Purpose

This document provides the **authoritative specification** for the AI Risk Observatory (AIRO) data processing pipeline and database schema. It consolidates the analysis from multiple taxonomy proposals into a single, coherent design optimized for the proof-of-concept deliverables.

---

## 1. Executive Summary

### What AIRO Does

AIRO processes UK listed firms' annual reports to:

1. **Identify** AI-relevant mentions (sentences/paragraphs discussing AI in risk, adoption, or governance contexts)
2. **Classify** each mention along multiple dimensions (risk type, specificity, governance, frontier relevance)
3. **Aggregate** mention-level data to firm-year and sector-year views for dashboards and reporting

### Core Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Actionability over completeness** | Focus on dimensions that enable AISI to prioritize interventions |
| **Fluff detection** | Explicit specificity scoring to separate boilerplate from substantive disclosures |
| **Phased complexity** | MVP schema is lean; additional dimensions added in later phases |
| **Auditability** | Every classification links back to source text with LLM reasoning |

---

## 2. Data Processing Pipeline

### 2.1 Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Annual Report  │────▶│  Text Chunking  │────▶│  LLM Step 1:    │────▶│  LLM Step 2:    │
│  PDFs / HTMLs   │     │  & Extraction   │     │  Relevance      │     │  Classification │
└─────────────────┘     └─────────────────┘     │  Detection      │     │  (if relevant)  │
                                                └─────────────────┘     └─────────────────┘
                                                                                 │
                                                                                 ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Dashboard &    │◀────│  Sector-Year    │◀────│  Firm-Year      │◀────│  Mentions       │
│  Reports        │     │  Aggregates     │     │  Aggregates     │     │  Table          │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 2.2 Step 1: Text Extraction & Chunking

**Input**: Annual report files (PDF, HTML, or pre-extracted text)

**Process**:
- Extract text from reports, preserving section structure where possible
- Identify report sections (Principal Risks, Strategy, ESG, Operational Review, etc.)
- Chunk text into sentence or paragraph units suitable for LLM processing

**Output**: Candidate text spans with metadata:
```json
{
  "span_id": "BARC-2024-0047",
  "firm_id": "GB0031348658",
  "report_year": 2024,
  "report_section": "Principal Risks",
  "text": "The increasing sophistication of AI-enabled fraud..."
}
```

### 2.3 Step 2: LLM Relevance Detection

**Purpose**: Filter candidate spans to only AI-relevant mentions

**Relevance Criteria** (a span is relevant if it mentions AI/ML/algorithms AND relates to):
- Risk, threat, downside, or uncertainty
- Adoption, deployment, or use-case
- Governance, mitigation, or controls
- Incidents, failures, or regulatory actions
- Regulatory environment or compliance challenges

**Output**: Boolean `is_relevant` flag + brief reasoning

### 2.4 Step 3: LLM Classification

**Purpose**: For relevant spans, classify along the taxonomy dimensions

**Output**: Structured classification record (see Section 3 for schema)

### 2.5 Step 4: Aggregation

**Purpose**: Roll up mention-level data to firm-year and sector-year views

**Firm-Year Aggregates**:
- Total AI mentions, total risk mentions
- Dominant risk category (most frequent Tier 1)
- Maximum specificity level
- Governance maturity (highest observed)
- Frontier AI flag (any frontier mentions?)

**Sector-Year Aggregates**:
- Percentage of firms mentioning AI risks
- Distribution of Tier 1 risk categories
- Average specificity scores
- Percentage with governance mechanisms

---

## 3. Database Schema

### 3.1 Schema Overview

The database consists of three primary tables:

```
┌─────────────────────────────────────────────────────────────────────┐
│                           mentions                                   │
│  (One row per AI-relevant excerpt from annual reports)              │
├─────────────────────────────────────────────────────────────────────┤
│  mention_id (PK)                                                    │
│  firm_id (FK) ──────────────────────────────────┐                   │
│  report_year                                     │                   │
│  ... classification fields ...                   │                   │
└─────────────────────────────────────────────────┼───────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            firms                                     │
│  (One row per firm-year, with aggregated metrics)                   │
├─────────────────────────────────────────────────────────────────────┤
│  firm_id (PK)                                                       │
│  report_year (PK)                                                   │
│  ... aggregate fields ...                                           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       sector_year_stats                              │
│  (Derived view for dashboard aggregates)                            │
├─────────────────────────────────────────────────────────────────────┤
│  sector                                                             │
│  report_year                                                        │
│  ... sector-level metrics ...                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 `mentions` Table (Core)

This is the **primary analytical table**. Each row represents one AI-relevant excerpt.

#### Schema Definition

```sql
CREATE TABLE mentions (
    -- ═══════════════════════════════════════════════════════════════
    -- IDENTIFIERS & CONTEXT
    -- ═══════════════════════════════════════════════════════════════
    mention_id          TEXT PRIMARY KEY,       -- Unique ID (e.g., "BARC-2024-001")
    firm_id             TEXT NOT NULL,          -- Company identifier (ISIN or internal ID)
    firm_name           TEXT NOT NULL,          -- Human-readable company name
    sector              TEXT NOT NULL,          -- Sector classification (FTSE/GICS)
    sector_code         TEXT,                   -- Standardized sector code
    report_year         INTEGER NOT NULL,       -- Fiscal year of the report
    report_section      TEXT,                   -- Section location in report
    
    -- ═══════════════════════════════════════════════════════════════
    -- SOURCE TEXT & TRACEABILITY
    -- ═══════════════════════════════════════════════════════════════
    text_excerpt        TEXT NOT NULL,          -- The actual text span
    page_number         INTEGER,                -- Page in original document (if available)
    
    -- ═══════════════════════════════════════════════════════════════
    -- MENTION TYPE & AI CONTEXT
    -- ═══════════════════════════════════════════════════════════════
    mention_type        TEXT NOT NULL,          -- What is this mention about?
    ai_specificity      TEXT NOT NULL,          -- Is this clearly about AI?
    frontier_tech_flag  BOOLEAN DEFAULT FALSE,  -- Mentions frontier AI (LLMs, GenAI, etc.)?
    
    -- ═══════════════════════════════════════════════════════════════
    -- RISK CLASSIFICATION (Tiered)
    -- ═══════════════════════════════════════════════════════════════
    tier_1_category     TEXT,                   -- Broad risk category (5-6 options)
    tier_2_driver       TEXT,                   -- Specific risk driver (nullable)
    
    -- ═══════════════════════════════════════════════════════════════
    -- SEVERITY & SUBSTANCE
    -- ═══════════════════════════════════════════════════════════════
    specificity_level   TEXT NOT NULL,          -- Boilerplate / Contextual / Concrete
    materiality_signal  TEXT,                   -- Low / Medium / High / Unspecified
    
    -- ═══════════════════════════════════════════════════════════════
    -- GOVERNANCE & MITIGATION
    -- ═══════════════════════════════════════════════════════════════
    mitigation_mentioned BOOLEAN DEFAULT FALSE, -- Any mitigation discussed?
    governance_maturity TEXT,                   -- None / Basic / Intermediate / Advanced
    
    -- ═══════════════════════════════════════════════════════════════
    -- LLM METADATA & QUALITY CONTROL
    -- ═══════════════════════════════════════════════════════════════
    confidence_score    REAL,                   -- LLM confidence (0.0 - 1.0)
    reasoning_summary   TEXT,                   -- LLM explanation for classification
    model_version       TEXT,                   -- Which LLM/prompt version was used
    extraction_date     DATE,                   -- When this record was created
    review_status       TEXT DEFAULT 'unreviewed', -- unreviewed / validated / rejected
    reviewer_notes      TEXT,                   -- Human reviewer comments
    
    -- ═══════════════════════════════════════════════════════════════
    -- CONSTRAINTS
    -- ═══════════════════════════════════════════════════════════════
    FOREIGN KEY (firm_id, report_year) REFERENCES firms(firm_id, report_year)
);
```

#### Field Enumerations

**`mention_type`** — What is this mention primarily about?

| Value | Description | Example |
|-------|-------------|---------|
| `risk_statement` | Explicit discussion of AI-related risk or downside | "AI may introduce bias in lending decisions" |
| `adoption_use_case` | Description of AI deployment or planned use | "We deploy ML models for fraud detection" |
| `governance_mitigation` | Controls, policies, or risk management for AI | "Our AI ethics committee oversees model deployment" |
| `incident_event` | Concrete failure, breach, or regulatory action | "We experienced an outage in our AI trading system" |
| `regulatory_environment` | Discussion of AI regulation or compliance burden | "The EU AI Act will require significant investment" |
| `strategy_opportunity` | Strategic AI discussion with implicit risk context | "AI is central to our digital transformation" |

**`ai_specificity`** — How clearly is this about AI vs generic automation?

| Value | Description |
|-------|-------------|
| `ai_specific` | Explicit AI/ML/algorithm terms (AI, machine learning, neural network, LLM, etc.) |
| `automation_general` | Generic automation/digital/algorithm without clear AI reference |

**`tier_1_category`** — Broad risk category (for dashboard aggregation)

| Category | Description | Example Concerns |
|----------|-------------|------------------|
| `operational_reliability` | Risks to business continuity from AI systems | System failures, model errors, outages, hallucination |
| `security_malicious_use` | Risks from attacks on/with AI | Cyber attacks, deepfakes, fraud, adversarial ML |
| `legal_regulatory_compliance` | Risks from laws, regulations, litigation | AI Act, GDPR, IP infringement, liability |
| `workforce_human_capital` | Risks to/from workforce | Job displacement, skill gaps, talent competition |
| `societal_ethical_reputational` | Broader societal and trust risks | Bias, discrimination, misinformation, brand damage |
| `frontier_systemic` | Advanced AI and macro-level risks | Loss of control, systemic risk, concentration risk |

**`tier_2_driver`** — Specific risk driver (nullable, for research drill-down)

| Driver | Parent Tier 1 | Description |
|--------|---------------|-------------|
| `third_party_dependence` | operational_reliability | Reliance on external AI providers (OpenAI, Google, etc.) |
| `hallucination_accuracy` | operational_reliability | Model errors, incorrect outputs, reliability issues |
| `model_drift_degradation` | operational_reliability | Performance degradation over time |
| `cyber_enablement` | security_malicious_use | AI helping attackers (phishing, social engineering) |
| `adversarial_attacks` | security_malicious_use | Attacks targeting AI systems (poisoning, evasion) |
| `deepfakes_synthetic_media` | security_malicious_use | Synthetic content for fraud or manipulation |
| `data_privacy_leakage` | legal_regulatory_compliance | Data exposure through AI systems |
| `ip_copyright` | legal_regulatory_compliance | Training data rights, generated content IP |
| `regulatory_uncertainty` | legal_regulatory_compliance | Evolving AI regulation (AI Act, etc.) |
| `job_displacement` | workforce_human_capital | Automation of roles, redundancy |
| `skill_obsolescence` | workforce_human_capital | Workforce skills becoming outdated |
| `shadow_ai` | workforce_human_capital | Unauthorized AI use by employees |
| `bias_discrimination` | societal_ethical_reputational | Unfair outcomes, protected characteristics |
| `misinformation_content` | societal_ethical_reputational | AI-generated misleading content |
| `trust_reputation` | societal_ethical_reputational | Brand damage from AI incidents |
| `concentration_risk` | frontier_systemic | Dependence on few AI providers |
| `loss_of_control` | frontier_systemic | Autonomous systems, emergent behaviors |

**`specificity_level`** — How substantive is this disclosure?

| Level | Description | Signal Value |
|-------|-------------|--------------|
| `boilerplate` | Generic language applicable to any firm | Low — "AI presents emerging risks" |
| `contextual` | Sector or company-relevant but not specific | Medium — "AI may impact our underwriting" |
| `concrete` | Named systems, quantified, or detailed | High — "Our GPT-4 chatbot had 3 incidents" |

**`materiality_signal`** — How serious does the firm indicate this is?

| Level | Description |
|-------|-------------|
| `low` | Minor, speculative, or explicitly non-material |
| `medium` | Non-trivial but manageable impact |
| `high` | Explicitly material, principal, or critical |
| `unspecified` | No clear materiality indication |

**`governance_maturity`** — How developed is their AI governance?

| Level | Description |
|-------|-------------|
| `none` | No governance structure, policy, or process mentioned |
| `basic` | Generic policy or high-level commitment, little detail |
| `intermediate` | Clear processes, committees, some testing/validation |
| `advanced` | Systematic lifecycle management, audits, red-teaming, metrics |

**`review_status`** — Quality control status

| Status | Description |
|--------|-------------|
| `unreviewed` | LLM classification not yet validated |
| `validated` | Human reviewer confirmed classification |
| `rejected` | Human reviewer flagged as incorrect |

---

### 3.3 `firms` Table (Aggregated)

One row per firm per year, with aggregated metrics for dashboards.

```sql
CREATE TABLE firms (
    -- ═══════════════════════════════════════════════════════════════
    -- IDENTIFIERS
    -- ═══════════════════════════════════════════════════════════════
    firm_id             TEXT NOT NULL,
    firm_name           TEXT NOT NULL,
    sector              TEXT NOT NULL,
    sector_code         TEXT,
    report_year         INTEGER NOT NULL,
    
    -- ═══════════════════════════════════════════════════════════════
    -- AI ADOPTION INDICATORS
    -- ═══════════════════════════════════════════════════════════════
    ai_mentioned                BOOLEAN DEFAULT FALSE,  -- Any AI mention in report?
    ai_risk_mentioned           BOOLEAN DEFAULT FALSE,  -- Any AI risk mention?
    frontier_ai_mentioned       BOOLEAN DEFAULT FALSE,  -- Any frontier AI mention?
    total_ai_mentions           INTEGER DEFAULT 0,      -- Count of all AI mentions
    total_ai_risk_mentions      INTEGER DEFAULT 0,      -- Count of risk mentions
    
    -- ═══════════════════════════════════════════════════════════════
    -- RISK PROFILE (Aggregated)
    -- ═══════════════════════════════════════════════════════════════
    dominant_tier_1_category    TEXT,                   -- Most frequent Tier 1 category
    tier_1_distribution         JSONB,                  -- Count per Tier 1 category
    max_specificity_level       TEXT,                   -- Highest specificity observed
    max_materiality_signal      TEXT,                   -- Highest materiality observed
    
    -- ═══════════════════════════════════════════════════════════════
    -- GOVERNANCE PROFILE (Aggregated)
    -- ═══════════════════════════════════════════════════════════════
    has_ai_governance           BOOLEAN DEFAULT FALSE,  -- Any governance mention?
    max_governance_maturity     TEXT,                   -- Highest maturity observed
    ai_in_principal_risks       BOOLEAN DEFAULT FALSE,  -- AI in Principal Risks section?
    
    -- ═══════════════════════════════════════════════════════════════
    -- DERIVED METRICS
    -- ═══════════════════════════════════════════════════════════════
    specificity_ratio           REAL,                   -- % concrete / total mentions
    mitigation_gap_score        REAL,                   -- High risk + low governance = gap
    
    -- ═══════════════════════════════════════════════════════════════
    -- METADATA
    -- ═══════════════════════════════════════════════════════════════
    last_updated                TIMESTAMP,
    
    PRIMARY KEY (firm_id, report_year)
);
```

#### Derived Metrics Explained

**`specificity_ratio`**: Proportion of mentions that are `concrete` (vs boilerplate/contextual). Higher = more substantive disclosure.

**`mitigation_gap_score`**: Identifies firms that acknowledge high-severity risks but lack governance. Calculated as:
```
gap_score = (high_severity_mentions / total_risk_mentions) * (1 - governance_maturity_score)
```
Where `governance_maturity_score` maps none=0, basic=0.33, intermediate=0.66, advanced=1.0

---

### 3.4 `sector_year_stats` View (Dashboard)

Derived view for sector-level dashboard visualizations.

```sql
CREATE VIEW sector_year_stats AS
SELECT 
    sector,
    report_year,
    
    -- Adoption metrics
    COUNT(DISTINCT firm_id) AS total_firms,
    COUNT(DISTINCT CASE WHEN ai_mentioned THEN firm_id END) AS firms_mentioning_ai,
    COUNT(DISTINCT CASE WHEN ai_risk_mentioned THEN firm_id END) AS firms_mentioning_ai_risk,
    COUNT(DISTINCT CASE WHEN frontier_ai_mentioned THEN firm_id END) AS firms_mentioning_frontier_ai,
    
    -- Risk distribution (aggregated from mentions)
    -- ... JSON aggregation of tier_1_category counts ...
    
    -- Governance metrics
    COUNT(DISTINCT CASE WHEN has_ai_governance THEN firm_id END) AS firms_with_governance,
    AVG(specificity_ratio) AS avg_specificity_ratio,
    AVG(mitigation_gap_score) AS avg_mitigation_gap
    
FROM firms
GROUP BY sector, report_year;
```

---

## 4. LLM Classification Prompt

### 4.1 Prompt Template

```
You are an expert analyst for the UK AI Safety Institute, analyzing company annual 
reports for mentions of AI-related risks and adoption.

## INPUT
Company: {firm_name}
Sector: {sector}
Report Year: {report_year}
Report Section: {report_section}

Excerpt:
"""
{text_excerpt}
"""

## TASK
Analyze this excerpt and provide a structured classification.

### Step 1: Is this AI-relevant?
Is this excerpt about AI, machine learning, algorithms, or automated decision-making 
in a context of risk, adoption, governance, or incidents?

Answer: [Yes/No]
If No, stop here.

### Step 2: Mention Type
What is this excerpt primarily about?
- risk_statement: Explicit discussion of AI-related risk or downside
- adoption_use_case: Description of AI deployment or planned use
- governance_mitigation: Controls, policies, or risk management for AI
- incident_event: Concrete failure, breach, or regulatory action
- regulatory_environment: Discussion of AI regulation or compliance
- strategy_opportunity: Strategic AI discussion with implicit risk context

Answer: [mention_type]

### Step 3: AI Specificity
- ai_specific: Explicit AI/ML terms (AI, machine learning, LLM, neural network, etc.)
- automation_general: Generic automation without clear AI reference

Answer: [ai_specificity]

### Step 4: Frontier Technology
Does this mention frontier AI technologies (large language models, generative AI, 
foundation models, GPT, Claude, etc.)?

Answer: [true/false]

### Step 5: Risk Classification (if risk-related)
Tier 1 Category (select one):
- operational_reliability: System failures, model errors, outages
- security_malicious_use: Cyber attacks, deepfakes, fraud
- legal_regulatory_compliance: AI Act, GDPR, IP, liability
- workforce_human_capital: Job displacement, skills, talent
- societal_ethical_reputational: Bias, misinformation, trust
- frontier_systemic: Loss of control, systemic risk

Answer: [tier_1_category or null if not risk-related]

Tier 2 Driver (select one if applicable, or null):
[List of tier_2_driver options]

Answer: [tier_2_driver or null]

### Step 6: Specificity Level
- boilerplate: Generic language applicable to any firm
- contextual: Sector/company-relevant but not specific
- concrete: Named systems, quantified, or detailed

Answer: [specificity_level]

### Step 7: Materiality Signal
Based on language cues ("material", "significant", "principal", etc.):
- low / medium / high / unspecified

Answer: [materiality_signal]

### Step 8: Governance & Mitigation
Is mitigation or governance discussed?
Answer: [true/false]

If yes, what level of governance maturity is indicated?
- none / basic / intermediate / advanced

Answer: [governance_maturity]

### Step 9: Confidence
How confident are you in this classification? (0.0 - 1.0)
- 0.9-1.0: Very clear, unambiguous
- 0.7-0.89: Clear, minor ambiguity
- 0.5-0.69: Moderate uncertainty
- Below 0.5: High uncertainty, needs human review

Answer: [confidence_score]

### Step 10: Reasoning
Provide a 1-2 sentence explanation of your classification.

Answer: [reasoning_summary]

## OUTPUT FORMAT
Return your response as JSON:

```json
{
  "is_relevant": true,
  "mention_type": "risk_statement",
  "ai_specificity": "ai_specific",
  "frontier_tech_flag": false,
  "tier_1_category": "operational_reliability",
  "tier_2_driver": "hallucination_accuracy",
  "specificity_level": "contextual",
  "materiality_signal": "medium",
  "mitigation_mentioned": true,
  "governance_maturity": "basic",
  "confidence_score": 0.85,
  "reasoning_summary": "Clear mention of AI model accuracy risks in customer-facing context, with generic policy reference but no specific controls."
}
```
```

---

## 5. MVP vs Future Phases

### Phase 1: MVP (Current Scope)

**Included in MVP**:
- ✅ `mention_type` (6 values)
- ✅ `ai_specificity` (2 values)
- ✅ `frontier_tech_flag` (boolean)
- ✅ `tier_1_category` (6 values)
- ✅ `tier_2_driver` (~15 values, nullable)
- ✅ `specificity_level` (3 values)
- ✅ `materiality_signal` (4 values)
- ✅ `mitigation_mentioned` (boolean)
- ✅ `governance_maturity` (4 values)
- ✅ LLM metadata (confidence, reasoning, model version)

**Deferred to Phase 2**:
- ❌ `stakeholder_affected` (customers, employees, etc.)
- ❌ `impact_type` (financial, operational, reputational, etc.)
- ❌ `time_horizon` (current, short-term, long-term)
- ❌ `mitigation_type` (detailed breakdown: technical, policy, training, etc.)
- ❌ `ai_application_type` (LLM, computer vision, predictive ML, etc.)
- ❌ `business_function` (customer service, trading, HR, etc.)

### Phase 2: Enhanced Classification

Add when MVP is validated and there's demand for deeper analysis:
- Stakeholder and harm type dimensions
- Detailed mitigation type breakdown
- AI application and business function mapping
- Longitudinal tracking (did mitigation reduce subsequent risk mentions?)

### Phase 3: Advanced Analytics

Future enhancements:
- Regulatory trigger detection (AI Act, GDPR mentions)
- Third-party/ecosystem risk network mapping
- Sentiment trend analysis
- Effectiveness tracking (mitigation → outcomes)

---

## 6. Quality Control & Validation

### 6.1 Confidence Thresholds

| Confidence | Action |
|------------|--------|
| ≥ 0.85 | Auto-accept, spot-check sample |
| 0.70 - 0.84 | Accept with flag for review queue |
| 0.50 - 0.69 | Requires human review before inclusion |
| < 0.50 | Reject or re-classify with different prompt |

### 6.2 Human Review Process

1. **Initial calibration**: First 100-200 mentions manually reviewed to validate LLM accuracy
2. **Ongoing sampling**: 5-10% random sample reviewed each batch
3. **Edge case documentation**: Ambiguous cases documented for prompt refinement
4. **Inter-rater reliability**: Compare LLM vs human classifications, target >80% agreement

### 6.3 Version Tracking

Every classification record includes:
- `model_version`: Which LLM and prompt version was used
- `extraction_date`: When the classification was performed
- `review_status`: Current QC status

This enables:
- Reproducibility of results
- Re-classification if prompts improve
- Audit trail for AISI reporting

---

## 7. Dashboard Metrics (Priority Order)

### Priority 1: Core MVP Metrics

1. **AI Risk Mentions Over Time** (by sector)
   - Stacked area chart of Tier 1 categories by year
   - Shows evolution of disclosed risks

2. **Sector Heatmap**
   - Rows: Sectors | Columns: Tier 1 categories
   - Cell value: Count or percentage of firms
   - Quickly identifies "where is the fire?"

3. **Frontier AI Velocity**
   - Line chart of `frontier_tech_flag = true` mentions over time
   - Shows rate of frontier AI proliferation

4. **Specificity Distribution**
   - Pie/bar chart: boilerplate vs contextual vs concrete
   - Answers "how much is fluff?"

### Priority 2: Governance & Gaps

5. **Mitigation Gap Analysis**
   - Scatter plot: X = risk severity, Y = governance maturity
   - Quadrant view identifies "cowboys" (high risk, low governance)

6. **Talk vs Action Ratio**
   - By sector: % of risk mentions with governance_maturity ≥ intermediate
   - Highlights sectors that acknowledge but don't act

### Priority 3: Deep Dives

7. **Tier 2 Driver Breakdown** (within each Tier 1)
   - Drill-down from Tier 1 to specific drivers

8. **Evidence Explorer**
   - Searchable table of underlying mentions
   - Filter by sector, category, specificity, etc.

---

## 8. Example Classification

### Input Excerpt

> "We have begun deploying large language models in our customer service operations. 
> While this has improved response times, we are concerned about the potential for 
> these models to generate inaccurate information (hallucinations) that could mislead 
> customers and damage our reputation. We are developing monitoring protocols and 
> investing in human oversight mechanisms to mitigate these risks."

### Classification Output

```json
{
  "mention_id": "ACME-2024-017",
  "firm_id": "GB0012345678",
  "firm_name": "Acme Financial Services",
  "sector": "Financials",
  "report_year": 2024,
  "report_section": "Operational Review",
  "text_excerpt": "We have begun deploying large language models...",
  
  "mention_type": "risk_statement",
  "ai_specificity": "ai_specific",
  "frontier_tech_flag": true,
  
  "tier_1_category": "operational_reliability",
  "tier_2_driver": "hallucination_accuracy",
  
  "specificity_level": "concrete",
  "materiality_signal": "medium",
  
  "mitigation_mentioned": true,
  "governance_maturity": "intermediate",
  
  "confidence_score": 0.92,
  "reasoning_summary": "Clear mention of LLM hallucination risk in production customer service use. Company acknowledges risk and describes specific mitigation (monitoring + human oversight) with investment commitment. Concrete due to named technology and specific use case.",
  
  "model_version": "gpt-4-turbo-2024-04-09",
  "extraction_date": "2024-11-28",
  "review_status": "unreviewed"
}
```

### Why This Classification

| Field | Value | Reasoning |
|-------|-------|-----------|
| `mention_type` | risk_statement | Core concern is potential harm (hallucinations misleading customers) |
| `frontier_tech_flag` | true | Explicitly mentions "large language models" |
| `tier_1_category` | operational_reliability | Risk is about system outputs being incorrect |
| `tier_2_driver` | hallucination_accuracy | Specifically about hallucination/accuracy issues |
| `specificity_level` | concrete | Names technology (LLM), use case (customer service), and specific risk (hallucination) |
| `governance_maturity` | intermediate | Describes specific processes (monitoring, human oversight) with investment |

---

## 9. Success Criteria

The AIRO proof-of-concept will be considered successful if:

### Technical Feasibility
- [ ] LLM classification achieves >80% agreement with human reviewers
- [ ] Pipeline processes 100+ annual reports within budget
- [ ] Classifications are reproducible and auditable

### Actionable Insights
- [ ] Identifies ≥3 distinct patterns in AI risk disclosure by sector
- [ ] Distinguishes substantive disclosures from boilerplate (specificity_level distribution)
- [ ] Surfaces firms/sectors with high risk acknowledgment but low governance ("cowboys")
- [ ] Shows temporal trends (YoY changes in frontier AI mentions, risk categories)

### Stakeholder Value
- [ ] AISI finds dashboard useful for situational awareness
- [ ] Enables prioritization of which sectors/firms to investigate further
- [ ] Provides evidence trail (excerpts + reasoning) for any classification

---

## Appendix A: Sector Classification

Use FTSE Industry Classification Benchmark (ICB) or GICS for standardized sector codes:

| Sector | ICB Code | Example Companies |
|--------|----------|-------------------|
| Financials | 30 | Barclays, HSBC, Aviva |
| Technology | 10 | Sage, Micro Focus |
| Healthcare | 20 | AstraZeneca, GSK |
| Consumer Discretionary | 40 | Next, Whitbread |
| Industrials | 50 | BAE Systems, Rolls-Royce |
| Energy | 60 | BP, Shell |
| Utilities | 65 | National Grid, SSE |
| Basic Materials | 55 | Rio Tinto, Glencore |
| Consumer Staples | 45 | Unilever, Tesco |
| Telecommunications | 15 | Vodafone, BT |
| Real Estate | 35 | Land Securities, British Land |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **AI-relevant mention** | A sentence or paragraph in an annual report that discusses AI/ML in context of risk, adoption, governance, or incidents |
| **Boilerplate** | Generic risk language that could apply to any firm; low signal value |
| **Frontier AI** | Advanced AI systems with capabilities similar to current frontier models (LLMs, foundation models, generative AI) |
| **Mitigation gap** | When a firm acknowledges high-severity AI risks but describes little or no governance/controls |
| **Specificity** | How concrete and firm-specific a disclosure is (vs generic/boilerplate) |
| **Tier 1 category** | Broad risk classification for dashboard aggregation (6 categories) |
| **Tier 2 driver** | Specific risk driver within a Tier 1 category for research drill-down |

---

*Document Version: 1.0*  
*Last Updated: November 2024*  
*Author: AIRO Project Team*

