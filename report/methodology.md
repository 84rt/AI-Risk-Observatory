# About the Observatory

Source: copied from `../dashboard/src/app/about/page.tsx` and rendered with values from `../dashboard/data/dashboard-data.json` on 2026-04-15.

This page explains how we turn annual reports from UK-listed public companies into the data powering the dashboard, and the decisions behind each step in the pipeline. For a deeper dive, read our full technical report.

## Contents

- [Overview](#overview)
- [1. Data](#1-data)
- [Scope](#scope)
- [Decisions and Rationale](#decisions-and-rationale)
- [Data Provider Acknowledgment](#data-provider-acknowledgment)
- [2. Pre-processing](#2-pre-processing)
- [Chunking Approach](#chunking-approach)
- [Chunking Results](#chunking-results)
- [3. Processing](#3-processing)
- [Phase 1: Mention-Type Classification](#phase-1-mention-type-classification)
- [Phase 2: Deep-Taxonomy Classification](#phase-2-deep-taxonomy-classification)
- [4. Quality Assurance](#4-quality-assurance)
- [Footnotes](#footnotes)

## Overview

The AI Risk Observatory processes annual reports from UK-listed public companies through a two-stage AI classification pipeline. The dataset spans all annual reports published between 2020 and 2026 by 1,362 companies, totalling 9,821 filings. Of these, 4,637 filings contain at least one AI-relevant mention, and after quality filters 4,078 carry meaningful AI signal. Because annual reports can run to hundreds of pages, we extract only the relevant AI mentions and their surrounding context, giving us 24,189 annotated text chunks in total.

The pipeline follows three stages:

1. Extract all relevant AI mentions from each filing.
2. Broadly classify the type of AI mentioned into six categories: Adoption, Risk, Harm, Vendor, General or ambiguous, or False Positive.
3. For each Adoption, Risk, and Vendor mention, classify it into a detailed sub-taxonomy.

We also run a substantiveness classifier to measure the depth of each mention, rating it on a scale from boilerplate to substantive.

The pipeline is illustrated below. Phase 1 labels are not mutually exclusive, so those counts sum to more than the number of extracted reports.

### Extraction Gate

| Stage | Reports | Share |
|---|---:|---:|
| Filings examined | 9,821 | 100% |
| Reports that mention AI | 4,637 | 47% |
| Reports that do not mention AI | 5,184 | 53% |

### Phase 1: Classify the Type of AI Mention

AI mentions are classified into six categories.

| Category | Reports | Share of extracted reports |
|---|---:|---:|
| General or ambiguous | 2,884 | 62% |
| None (including false positive) | 559 | 12% |
| Harm | 2 | 0% |
| Adoption | 3,012 | 65% |
| Risk | 1,860 | 40% |
| Vendor | 877 | 19% |

### Phase 2: Detailed Taxonomies

From Phase 1, only Adoption, Risk, and Vendor are processed further into the following subcategories. Report counts are unique-report counts; AI mention counts are chunk-level label-assignment counts.

#### Adoption

| Label | Reports | AI mentions tagged |
|---|---:|---:|
| Traditional AI (non-LLM) | 2,869 | 9,772 |
| LLM | 1,631 | 5,023 |
| Agentic AI | 623 | 1,336 |

#### Risk

| Label | Reports | AI mentions tagged |
|---|---:|---:|
| Strategic/competitive | 1,241 | 2,203 |
| Cybersecurity | 1,048 | 2,065 |
| Operational/technical | 1,039 | 2,127 |
| Regulatory/compliance | 947 | 2,384 |
| Reputational/ethical | 746 | 1,594 |
| Third party/supply chain | 521 | 900 |
| Information integrity | 427 | 591 |
| Workforce impacts | 428 | 599 |
| Environmental impact | 91 | 109 |
| National security | 115 | 144 |

#### Vendor

| Label | Reports | AI mentions tagged |
|---|---:|---:|
| Other | 464 | 718 |
| Microsoft | 279 | 547 |
| Internal | 171 | 252 |
| OpenAI | 146 | 222 |
| Undisclosed | 136 | 178 |
| Google | 131 | 208 |
| Amazon | 97 | 168 |
| Meta | 37 | 47 |
| Anthropic | 11 | 22 |

## 1. Data

### Scope

To measure AI risk, adoption, and vendor dependence across the UK economy, we process all annual reports published by all public companies in the UK. There are 1,660 public companies listed on UK markets (LSE Main Market, AIM Market, and AQSE). After excluding companies not registered in the UK (e.g. Irish or Canadian companies listed on these exchanges) and firms without filings available via Companies House, our working universe is approximately 1,362 companies. Each company files, on average, one annual report per year.[^1]

### Decisions and Rationale

**Why annual reports?** Unlike earnings calls, press releases, or public media, annual reports are audited, structured, and published on a consistent cadence, making them a reliable, high-signal source of information. UK public companies must publish annual accounts, a strategic report, a directors' report, and an auditor's report under the Companies Act 2006. All listed companies share that statutory core, but Main Market issuers face tighter deadlines and more detailed disclosure rules than AIM and AQSE companies.[^5]

This makes annual reports well suited to tracking trends across the UK economy over time. There are two primary limitations: (1) they are inherently backward-looking, often with a significant delay; and (2) their highly regulated nature means many statements are boilerplate and contain little real information.[^2]

**Why 2020-2026?** We chose this window to capture a pre-ChatGPT baseline (before the late-2022 inflection) and the rapid adoption cycle that followed.

**How do we map to CNI?** The Critical National Infrastructure in the [UK has 13 distinct sectors](https://www.npsa.gov.uk/about-npsa/critical-national-infrastructure). Each company in our database has an [ISIC sector code](https://en.wikipedia.org/wiki/International_Standard_Industrial_Classification) that only partially maps to CNI sectors. We take a conservative approach, using an LLM classifier to assign CNI sectors to companies that do not map directly from ISIC; when no assignment can be made, we use an "Other" CNI category.[^3] A major limitation of CNI analysis via annual reports is that some sectors, such as Space, Emergency Services, or Civil Nuclear, have few public companies or suppliers represented.[^4]

### Data Provider Acknowledgment

Converting PDFs to clean, structured text is technically demanding, and doing so at that scale would have exceeded our compute budget. We partnered with [FinancialReports.eu](https://financialreports.eu), a third-party financial data provider, to obtain all annual reports in our scope in Markdown format. Their filings API and generous support made this project possible.

## 2. Pre-processing

### Chunking Approach

Once each annual report is in structured Markdown text, we split it into chunks using a sliding-window approach that respects paragraph and section boundaries, with generous padding around each AI mention. An AI keyword filter isolates sections that explicitly mention AI or closely related techniques; only those sections are retained for further annotation as AI mentions. Each chunk carries metadata: company identifier, reporting year, release month, report section (e.g. *Risk Factors*, *Strategy*), and a stable chunk ID for traceability.

### Chunking Results

The table below shows filings with AI mentions and the number of AI mentions extracted per year.

| Year | Number of filings | Filings with AI mention (% of total) | Count of AI mentions |
|---:|---:|---:|---:|
| 2020 | 1,007 | 196 (19%) | 493 |
| 2021 | 1,328 | 361 (27%) | 1,083 |
| 2022 | 1,853 | 526 (28%) | 1,848 |
| 2023 | 1,905 | 702 (37%) | 2,312 |
| 2024 | 1,828 | 1,008 (55%) | 5,070 |
| 2025 | 1,561 | 1,023 (66%) | 6,528 |
| 2026 | 339 | 262 (77%) | 3,226 |
| Total | 9,821 | 4,078 (42%) | 20,560 |

## 3. Processing

### Phase 1: Mention-Type Classification

First, each chunk is passed to an LLM classifier that decides whether the text contains a genuine AI mention and, if so, assigns one or more mention-type labels. Chunks assigned only the *None* label are filtered out as false positives before Phase 2.

The Phase 1 classifier uses the following taxonomy:

| Label | Definition |
|---|---|
| Adoption | Current use, rollout, pilot, implementation, or delivery of AI systems. |
| Risk | AI described as a downside or exposure for the company. |
| Harm | AI described as causing or enabling harm (misinformation, fraud, abuse, safety incidents). |
| Vendor reference | A named AI model, vendor, or platform provider is referenced. |
| General or ambiguous | AI mentioned but too high-level or vague for the above categories. |
| None | No real AI mention / false positive. Exclusive; cannot co-occur with others. |

#### Phase 1 Label Distribution Over Time

Distribution of Phase 1 mention-type labels across all AI-mentioning filings, by year. Labels are not mutually exclusive, so a single filing can contribute to multiple categories.

| Year | Adoption | Risk | Vendor | General or ambiguous | Harm | None |
|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 140 | 32 | 28 | 112 | 0 | 109 |
| 2021 | 285 | 59 | 41 | 195 | 0 | 195 |
| 2022 | 434 | 93 | 83 | 282 | 0 | 347 |
| 2023 | 511 | 194 | 119 | 427 | 0 | 326 |
| 2024 | 703 | 582 | 244 | 785 | 0 | 319 |
| 2025 | 721 | 674 | 271 | 853 | 2 | 324 |
| 2026 | 218 | 226 | 91 | 230 | 0 | 95 |

#### Phase 1 Classifier Prompt

All prompts and structured output schemas used in Phase 1 are visible in the [project repository](https://github.com/84rt/ai-risk-observatory). The prompt used for the Phase 1 classification of the data visible on the dashboard is:

```text
You are an expert analyst labeling AI mentions from company annual reports.

## TASK
Assign ALL mention types that apply to the excerpt. Types are NOT mutually exclusive except for "none".
If the excerpt contains no AI mention, return only "none". Only tag content that is explicitly about AI in the excerpt; ignore unrelated sentences.

## RULES
1. AI EXPLICITNESS GATE: If the excerpt does NOT explicitly mention AI/ML/LLM/GenAI or a clearly AI-specific technique (e.g., machine learning, neural networks, computer vision), assign the tag "none". Terms like "data analytics" or "digital tools" generally are NOT considered AI under our definition. The tag "none" is used when there is no AI mention, a false positive, or unrelated automation not clearly AI. Only consider terms like "autonomous or virtual assistant" as AI if it can be clearly attributed to AI. Otherwise, use the following tags in a non-mutually exclusive manner: adoption (the current usage of AI technology by the company), risk (AI as a risk: risks that are directly coming from AI), harm (past harms that were caused by AI), vendor (any mention of a provider of AI technology), general_ambiguous (any statement about AI that does not fit into the other tags). Here are more details on each tag:
   - adoption: must directly describe real current deployment, implementation, rollout, pilot, or use of AI by the company or for its clients. Generic statements about intent/strategy/roadmaps/plans (adoption in the future) are NOT considered adoption. Treat "exploring", "piloting", or "investigating" AI use as adoption ONLY when it refers to an initiative currently underway (e.g. "current trial resulted in..."). Delivering AI systems directly or indirectly for clients does count as adoption; pure consulting/advice without deployment does NOT.
   - risk: must directly attribute AI as the source of a risk or downside (i.e. strategic & competitive, operational & technical, cybersecurity, workforce impacts, regulatory & compliance, information integrity, reputational & ethical, third-party & supply chain, environmental impact, and national security etc.). The excerpt might contain a sentence on risk and a separate sentence on AI; make sure to only assign the "risk" tag if AI is mentioned as the source of the risk. Generic risk language without explicitly mentioning AI is NOT risk from AI. However, the risk section might outline downstream risks or effects from AI technologies in an indirect way; these should be classified as risk from AI.
   - harm: must describe past harms that were caused by AI (misinformation, fraud, abuse, safety incidents).
   - vendor: must explicitly name a third-party AI vendor/platform that provides the AI technology (i.e. Microsoft, Google, OpenAI, AWS, or explicitly developed in-house). We primarily want to tag text that mentions any information about what AI models are used by the company (i.e. GPT or Google Gemini).
   - general_ambiguous: vague AI strategy, high-level plans, or AI mentions that don't have enough context or are not specific enough. If AI is explicitly mentioned but does not meet adoption/risk/harm/vendor, use general_ambiguous. The tag general_ambiguous should only be added when the excerpt clearly talks about AI but does not meet the other tag definitions.
2. Assign confidence scores to each tag. Confidence scores always indicate how likely the label applies (including "none").

## CONFIDENCE GUIDANCE (0.0-1.0)
- 0.2: faint/implicit signal; could be this type but hard to tell
- 0.5: uncertain -- weak evidence
- 0.8: likely YES -- strong signal, but not fully explicit
- 0.95-1.0: confident YES -- explicit, unambiguous mention

## EXAMPLES
- "We deployed an AI chatbot for customer support." -> adoption ~0.9
- "We are exploring AI opportunities." -> general_ambiguous ~0.7
- "We are piloting AI to automate invoice processing." -> adoption ~0.8
- "We use data analytics and predictive analytics to optimize routing." -> none ~0.6 (unless AI/ML explicitly stated)
- "AI could increase misinformation risks." -> risk ~0.8
- "AI is a long-term megatrend, being widely adopted within the industry; we are evaluating any risks associated with it." -> risk ~0.7 (no "adoption" tag, as no evidence of adoption by company)
- "We partner with Microsoft for AI tooling." -> vendor ~0.9, adoption ~0.6
- "We partnered with OpenAI to deliver AI systems for clients in 2024." -> vendor ~0.9, adoption ~0.8
- "Automation of customer service tasks improved our..." -> general_ambiguous ~0.2 (not necessarily AI)
- "Address: FT-AI 4810 Shangh'ai', is where the..." -> none ~0.9 (a false positive)
- "AI-generated misinformation has damaged our brand reputation." -> harm ~0.9

## OUTPUT CONSTRAINTS
- mention_types must be non-empty.
- If "none" is present, it must be the only label.
- Provide a confidence score for EVERY label in mention_types.
- Do NOT include confidence scores for labels not in mention_types.

## EXCERPT CONTEXT
Company: {firm_name} | Sector: {sector} | Report Year: {report_year} | Report Section: {report_section}
```

### Phase 2: Deep-Taxonomy Classification

Chunks that passed Phase 1 are processed by dedicated classifiers depending on their mention types. We process three of the Phase 1 mention types - adoption, risk, and vendor - each through its own LLM classifier. Chunks tagged as Risk are also scored for substantiveness. The taxonomies used are as follows:

#### Adoption Taxonomy

| Label | Definition |
|---|---|
| Traditional AI/ML | Traditional AI/ML: predictive models, computer vision, detection/classification systems. |
| LLM/GenAI | Large language model/GenAI use (GPT, Gemini, Claude, Copilot-style deployments). |
| Agentic systems | Autonomous or agent-based workflows with limited human intervention. |

#### Risk Taxonomy

| Label | Definition |
|---|---|
| Strategic / competitive | Competitive disadvantage, disruption, or failure to adapt. |
| Operational / technical | Reliability/accuracy/model-risk failures degrading operations. |
| Cybersecurity | AI-enabled attacks, fraud, breach pathways, or adversarial abuse. |
| Workforce impacts | Displacement, skills gaps, or risky employee AI usage. |
| Regulatory / compliance | Legal, regulatory, privacy, or IP liability and compliance burden. |
| Information integrity | Misinformation, deepfakes, or authenticity manipulation. |
| Reputational / ethical | Trust, fairness, ethics, or rights concerns. |
| Third-party / supply chain | Dependency on external AI vendors and concentration exposure. |
| Environmental impact | Energy, carbon, or resource-burden risk. |
| National security | Geopolitical destabilisation or critical-systems exposure. |
| None | No attributable risk category (or too vague to assign one). |

#### Vendor Taxonomy

Vendors are tagged against a predefined list: Amazon, Google, Microsoft, OpenAI, Anthropic, Meta, internal (in-house), undisclosed (implied but unnamed), and other (named provider outside the predefined list, captured as free text).

#### Substantiveness

| Level | Definition |
|---|---|
| Boilerplate | Generic AI language; could appear in many reports unchanged. |
| Moderate | A specific area is identified, but without concrete mechanisms, metrics, or mitigation steps. |
| Substantive | Concrete mechanism, tangible action, commitment, metric, or timeline. |

#### Phase 2 Classifier Prompts

The Phase 2 classifier prompts, as well as all other prompts used in the pipeline, are available in the repository at [`/pipeline/prompts/classifiers.yaml`](https://github.com/84rt/ai-risk-observatory/blob/main/pipeline/prompts/classifiers.yaml).

## 4. Quality Assurance

We enforce structured outputs and explicit validation rules to reduce noise and improve reproducibility. We apply the following checks:

- **Structured outputs**: classifiers write to strict JSON response schemas; malformed or labels outside the permitted set are retried or flagged.
- **Conservative prompting**: prompts require explicit AI attribution and discourage over-labelling; the default outcome is *none* or *general_ambiguous*.
- **Temperature zero**: all classifier calls use temperature zero for deterministic, reproducible outputs.
- **Chunk-level traceability**: every annotation maps back to a company, year, and report section via a stable chunk ID.
- **QA scripts**: we run QA tests across each pipeline stage, checking primarily for anomalies and out-of-distribution outputs:
  - Document size, length, duplication, fiscal-year-match, and text anomalies (non-Markdown formatting, unexpected characters).
  - Outlier analysis on the distribution of Phase 1 and Phase 2 labels per company, report, and year; AI mentions extracted per report; and chunk creation keywords.
  - All flagged outputs were manually reviewed.
- **Human review**: the dataset is vast, and while we have made every effort to audit anomalies arising from data processing, some errors and misclassifications may remain. Our data is available for download. If you spot an issue, please file it on the repository.

Dataset and repository links:

- [Download Dataset](https://github.com/84rt/AI-Risk-Observatory/releases/download/dataset-v1.0/airo-dataset-v1.0.zip)
- [GitHub Repository](https://github.com/84rt/ai-risk-observatory)

## Footnotes

[^1]: Some companies have multiple subsidiaries with separate filings, while others were recently listed or spun off and therefore have fewer years of filings available. This means the per-company filing count is not uniform across the dataset.

[^2]: To address the boilerplate problem we apply a substantiveness classifier (see Phase 2 above) that rates each mention on a scale from boilerplate to substantive, allowing users to filter to high-signal disclosures.

[^3]: The ISIC-to-CNI mapping follows two steps: a direct lookup for ISIC codes that clearly correspond to a CNI sector, followed by an LLM classifier for ambiguous cases. Companies that cannot be assigned to any CNI sector are labelled "Other".

[^4]: The following CNI sectors have particularly low public-company representation in our dataset: Space (0), Emergency Services (0), Civil Nuclear (2), Water (18), Defence (20), Government (20), Data Infrastructure (22), Communications (28), Chemicals (34). Conclusions drawn about these sectors should be treated with caution.

[^5]: Main Market issuers are generally subject to FCA disclosure and listing rules, including a four-month reporting deadline, while AIM and AQSE companies typically have up to six months. The auditor's formal opinion covers the financial statements, not the annual report narrative as a whole.
