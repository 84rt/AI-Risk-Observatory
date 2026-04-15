# AI Risk Observatory — Report Outline
**Annotated section-by-section plan. Each entry states: what the section argues, which literature anchors it, and what data it draws on.**

---

## 0. Front matter

**Title:** *Tracking AI Risk and Adoption in UK Public-Company Annual Reports: Evidence from the AI Risk Observatory (2020–2025)*

**Authors / affiliation / sponsor (AISI)**

**Date:** 2026

---

## 1. Abstract / Executive Summary (~1 page)

Two versions of the same content:
- **Abstract** (150–200 words): one paragraph for academic readers. States the object of study, method, key quantitative finding, and contribution.
- **Executive Summary** (400–500 words): for policy readers. Covers the research question, the corpus, the three headline findings (risk disclosure surge, adoption-risk gap, disclosure quality plateau), and three actionable implications.

*Tip: write this last, once all sections are finalized.*

---

## 2. Introduction (~1.5 pages)

### 2.1 Motivation
**Argues:** Annual reports are an underused but high-quality signal for tracking how AI is entering corporate life — audited, legally mandated, and consistent over time. The challenge is extracting structured signal from thousands of long documents without losing reproducibility.

**Anchors:** FCA DTR 4.1 (legal basis for principal risk disclosure); Loughran & McDonald survey (annual reports as established NLP corpus); FRC strategic report guidance.

**Data:** Opening stat — 9,821 reports, 1,362 companies, 2020–2026. Frame the scale.

### 2.2 Research Questions
Three explicit research questions the report answers:
1. How has the prevalence and composition of AI-related disclosure changed across UK listed companies between 2020 and 2025?
2. Is AI risk disclosure keeping pace with adoption disclosure, and is it substantive or boilerplate?
3. Which sectors and market segments show the greatest disclosure gaps relative to plausible AI exposure?

### 2.3 Scope and Contribution
**Argues:** Most prior empirical work is US-centric (SEC 10-K). This is the first large-scale structured analysis of AI disclosure in UK annual reports, covering Main Market and AIM companies, with CNI-sector focus and an explicit substantiveness dimension.

**Anchors:** Uberti-Bona Marin et al. (2025) on US AI risk disclosures (gap to fill); Lang & Stice-Lawrence (2015) on international annual-report text analysis.

---

## 3. Background and Regulatory Context (~2 pages)

### 3.1 UK Annual Reporting Framework
**Argues:** UK annual reports are not optional or strategic communications — they are legally mandated artefacts with defined content requirements. This makes them a reliable monitoring surface.

**Covers:**
- Companies Act / strategic report requirements (narrative reporting for material risks)
- FCA DTR 4.1: annual financial reports within 4 months for Main Market issuers
- AIM Rules: 6-month window, less prescriptive — explains the AIM disclosure gap we observe
- iXBRL mandate: structured data layer that underpins our ingestion pipeline
- Companies House as secondary source for gap-fill (CH PDF route)

**Anchors:** FCA Handbook DTR 4.1; FRC strategic report guidance; AIM Rules for Companies.

### 3.2 Critical National Infrastructure (CNI) Context
**Argues:** The UK designates 13 CNI sectors whose disruption would have catastrophic consequences. AI exposure in these sectors has distinct national security implications beyond corporate risk management — this justifies the observatory's CNI focus.

**Covers:**
- NPSA 13-sector CNI framework
- Mapping LSE-listed companies to CNI sectors (ISIC → CNI crosswalk, with acknowledged ambiguity)
- Why CNI sectors matter: cascading failure risk, supply-chain AI dependencies, OT/IT convergence
- NCSC and DSIT guidance to CNI operators; Cyber Security and Resilience Bill

**Anchors:** NPSA CNI sector definitions; AISI Research Agenda and Frontier AI Trends Report; AISI "Navigating the Uncharted."

**Data:** Companies per CNI sector (Finance: 461, Energy: 141, Health: 111, Transport: 61, Communications: 28, Water: 18, etc.)

### 3.3 Governance Reforms and Disclosure Incentives
**Argues:** The 2024 UK Corporate Governance Code — particularly Provision 29 — is the single most important structural driver of AI disclosure change visible in our dataset. Understanding this reform is essential to correctly interpreting the 2024–2025 surge.

**Covers:**
- Pre-2024 baseline: boards stated that a review had been completed. Low threshold.
- Provision 29 (FY beginning ≥1 Jan 2026): boards must declare effectiveness of *all material internal controls* — financial, operational, reporting, and compliance — including narrative controls.
- Effect visible in data even before the provision's effective date: anticipatory compliance behavior explains part of the 2024–2025 surge in AI risk mentions.
- Comparison to US SOX (auditor attestation) — UK uses "comply or explain" and reputational accountability instead.
- Secondary driver: SEC 2024 State of Disclosure Review — same pressure, different jurisdiction. Confirms global trend.

**Anchors:** FRC UK Corporate Governance Code 2024; FRC guidance on Provision 29; SEC 2024 State of Disclosure Review; Gimini deep-research section on Provision 29.

**Data:** Risk mention trend 2020→2025: 32→674 reports mentioning AI risk (~21× increase). Sharpest inflection point: 2023→2024 (194→582, +200%). Provision 29 anticipation is the leading candidate explanation.

---

## 4. Related Work (~2 pages)

*Framing sentence: The Observatory draws on four established research streams. Each provided a methodological or conceptual foundation; together they position this work as extending prior methods to a new domain (UK AI disclosure) that lacks dedicated empirical coverage.*

### 4.1 Financial Disclosure NLP
**Covers:**
- Why financial language requires domain-specific treatment (Loughran & McDonald 2011 — generic dictionaries misfire on 10-Ks)
- Annual reports as a validated NLP corpus at scale (Loughran & McDonald survey; Lang & Stice-Lawrence 2015 for non-US corpora)
- Section segmentation and targeted extraction as a necessary pre-step before classification (precedents for our chunking approach)
- LLM-era financial text analysis: Kim (2024) on LLMs for financial statement analysis; Park / BIS-IFC (2024) on LLMs for materiality assessment in disclosures — our two-stage pipeline is consistent with this emerging standard

**Anchors:** Loughran & McDonald (2011); Loughran & McDonald survey (SSRN); Lang & Stice-Lawrence (2015); Kim (2024); Park / BIS-IFC (2024).

### 4.2 Substantiveness and Boilerplate
**Covers:**
- The core empirical regularity: regulated reports accumulate length, repetition, and stickiness even when new information does not increase (Dyer, Lang & Stice-Lawrence 2017)
- Year-over-year textual change as a proxy for disclosure informativeness (Brown & Tucker 2011 on MD&A modification)
- The economics of boilerplate: legal safe-harbor incentives (PSLRA equivalent) drive reliance on generic language; our substantiveness classifier is a direct response to this
- Regulatory pressure for specificity: SEC Reg S-K 2020 modernization; SEC 2024 State of Disclosure Review explicitly warning against generic AI buzz
- Readability and obfuscation as related concepts (Li 2008 Fog Index / management obfuscation hypothesis) — useful framing for why substantiveness matters beyond aesthetic preference

**Anchors:** Dyer, Lang & Stice-Lawrence (2017); Brown & Tucker (2011); SEC 2024 State of Disclosure Review; Loughran & McDonald (2011).

### 4.3 AI Disclosure and Governance
**Covers:**
- Rapid empirical growth in AI-risk disclosures: Uberti-Bona Marin et al. (2025) — 30,000+ US 10-Ks, >50% of firms mentioning AI by 2024, but disclosures often generic
- UK-specific: FTSE 100 AI narratives study (2025) — AI disclosure as strategic communication and impression management, not just risk reporting; supports our separation of adoption vs. risk vs. general_ambiguous
- Algorithmic decision-making disclosures in European companies: Bonsón et al. (2023) — uneven disclosure practice, even before AI boom
- Governance perspective: Chiu (2025) on GenAI in narrative reporting — raises the meta-question of AI changing the reporting process itself
- The "AI-washing" problem: disclosure volume rising faster than governance depth

**Anchors:** Uberti-Bona Marin et al. (2025); Bonsón et al. (2023); Chiu (2025); SEC 2024 State of Disclosure Review.

### 4.4 Observatory-Style Monitoring Systems
**Covers:**
- Climate monitoring as the closest methodological analogue: ClimateBERT / ClimaText (2023) — transformer-based monitoring of climate-risk language in financial reports; validates our approach of specialized classifier on a structured corporate corpus
- FTSE 350 ESG annual report monitoring: Ferjančič et al. (2024) — BERTopic over decade-long corpus shows topic prevalence shifts with regulation and events. Near-identical methodology to ours, different thematic domain
- Bank of England NLP on climate disclosures; BIS/IFC on NLP for risk extraction — central-bank precedent for this type of regulatory monitoring infrastructure
- OECD AI Policy Observatory: international precedent for AI-monitoring infrastructure at policy level

**Anchors:** ClimateBERT / ClimaText (arXiv 2303.13373); Ferjančič et al. (2024); FCA TCFD review; Park / BIS-IFC (2024); OECD AI Policy Observatory.

**Key gap statement (end of section):** Prior observatory work covers climate and ESG. Prior AI-disclosure work is US-centric and rarely includes CNI-sector decomposition, vendor dependency signals, or an explicit substantiveness dimension. This Observatory fills that gap for UK listed companies.

---

## 5. Methodology (~2.5 pages)

### 5.1 Corpus and Universe Definition
**Covers:**
- Company universe: 1,469 UK-incorporated LSE-listed companies (FTSE 350, AIM, Other) — how derived (LSE list → UK-incorporated filter → 191 excluded)
- Fiscal-year vs. publication-year primary key decision and why publication year was chosen
- Target: 7,345 company-year slots (1,469 × 5 publication years 2021–2025)
- Achieved: 9,821 reports (includes 2020 and partial 2026); 93.3% coverage of target universe achievable with dual-source strategy

**Data:** Coverage waterfall table (Ideal 7,345 → FR markdown 3,935 → +FR pending → +CH PDFs → achievable 6,854 / 93.3%)

### 5.2 Data Ingestion
**Covers:**
- Primary source: financialreports.eu — iXBRL filings via API, converted to structured markdown. Explain why iXBRL matters (machine-readable structure preserves section metadata)
- Secondary source: Companies House PDF gap-fill — 2,342 additional clean PDFs covering mostly AIM and smaller companies where FR coverage is incomplete
- AIM coverage asymmetry: FR covers only 28% of AIM slots; CH PDF is the primary AIM source. This is a known structural characteristic, not an error
- Known ingestion failures: 178 companies with zero FR coverage despite being in FR's index (e.g., Jet2 misclassification). Documented and partially gap-filled

### 5.3 Preprocessing and Chunking
**Covers:**
- iXBRL → markdown conversion: structure preserved (section headings, report metadata)
- AI-mention extraction: keyword gate (explicit AI/ML/LLM/GenAI or named technique) → context window extraction (2 paragraphs before/after) → deduplication of overlapping spans
- Why keyword gate matters: reduces false positives from "data analytics," "digital tools," "automation" — important given prior work showing these terms are often not AI
- Output: annotated text chunks, each with company, year, report section, and surrounding context

### 5.4 Classification Pipeline
**Covers:**
- Two-stage architecture:
  - Stage 1 (mention-type classifier): assigns adoption / risk / harm / vendor / general_ambiguous / none. Conservative labeling — explicit AI language required; intent/strategy alone is NOT adoption
  - Stage 2 (deep classifiers): for chunks that pass Stage 1 with a signal label, assigns adoption sub-type (non-LLM / LLM / agentic), risk sub-category (10-label taxonomy), vendor tag (OpenAI / Microsoft / Google / Amazon / Meta / Anthropic / internal / other / undisclosed), and substantiveness band (boilerplate / moderate / substantive)
- Schema-constrained outputs: prevents hallucination of non-existent labels
- Conservative prompting rules: explicit AI language required at Stage 1; prevents inflation from adjacent topics (cybersecurity without AI, digital transformation without AI, etc.)
- Model used: Gemini Flash 3 (Google)

**Include:** Classifier taxonomy table (mention types + sub-labels) as a clean visual.

### 5.5 Quality Assurance and Validation
**Covers:**
- Human-annotated golden set: 474 chunks annotated by human reviewers
- Target agreement rate: ~90%
- Reconciliation process: LLM labels vs. human baseline → disagreement review → merged back into golden set
- Traceability: every classified chunk traceable to raw source filing, including model version, classifier version, confidence score, and full prompt/response log
- Low-confidence and multi-model disagreement cases flagged for review

---

## 6. Findings (~3 pages — the longest section)

*Framing: all statistics cover publication years 2020–2025 unless stated. 2026 data is partial and noted where used.*

### 6.1 The Disclosure Surge: Overall Trends
**Argues:** AI-related disclosure has grown dramatically across all signal types, but the pace differs by signal — and the shape of growth tells a story about where corporate attention has moved.

**Key numbers:**
- Reports with any AI mention: 196/1,007 (19.5%) in 2020 → 1,023/1,561 (65.5%) in 2025
- Reports with AI *risk* mention: 32/1,007 (3.2%) in 2020 → 674/1,561 (43.2%) in 2025
- Risk disclosure grew ~21× in absolute terms (32→674), faster than adoption (~5×: 140→721)
- The adoption-to-risk ratio has compressed: in 2020 adoption was 4.4× more common than risk; by 2025 it is 1.1× — risk is nearly catching up

**Inflection point:** The 2023→2024 jump in risk mentions (+200%: 194→582) is the sharpest single-year move in the dataset. Coincides with ChatGPT mass adoption and early Provision 29 anticipation. Note that this is a report-count measure (one per company-year), not a raw mention count.

### 6.2 Adoption Patterns
**Argues:** LLM adoption has grown faster than non-LLM AI, and is now the dominant reported adoption type — a shift that accelerated sharply after 2022.

**Key numbers:**
- LLM adoption reports: 39 (2020) → 557 (2025) — ~14× growth
- Non-LLM: 136 → 672 (~5×)
- Agentic/autonomous: 27 → 172 (~6×)
- LLM's share of all adoption: ~16% in 2020 → ~38% in 2025
- Note: adoption counts are chunk-level (a company can appear in multiple years); interpret as activity level not unique companies

**Sector pattern:** Finance sector dominates (2,301 total adoption signals), followed by Other (1,212) and Health (294). Communications sector shows high LLM-to-non-LLM ratio — suggestive of content/platform businesses embedding generative tools.

### 6.3 Risk Disclosure Patterns
**Argues:** Strategic/competitive risk is the leading risk category, but cybersecurity and information integrity are growing fastest in relative terms, suggesting that the perceived risk landscape is broadening beyond competitive disruption.

**Key numbers (2025 top-5 risk labels):**
- Strategic & competitive: 458
- Cybersecurity: 405
- Operational & technical: 352
- Regulatory & compliance: 322
- Reputational & ethical: 252
- Workforce impacts: 176 (near-zero in 2020, now prominent)
- Information integrity: 184 (emerged in 2023, fast-growing)
- National security: 49 (low but non-zero — notable given CNI focus)

**Sector risk profile:** Finance sector accounts for 3,255 of all risk signals (total), and shows the most diversified risk portfolio (all 10 categories represented). Energy shows high cybersecurity and operational-technical. Health shows high regulatory compliance and third-party supply chain. Communications shows elevated information integrity and national security signals relative to sector size.

### 6.4 Vendor Landscape
**Argues:** Third-party AI dependencies are becoming visible in annual reports, but the picture is more concentrated than it might appear — Microsoft leads, "other" is large and opaque, and undisclosed vendor exposure is growing.

**Key numbers (2025):**
- Microsoft: 87 reports
- "Other" (named but not in top-6): 143 — the largest single category
- OpenAI: 36 (peaked 2024 at 60, slightly declined)
- Google: 42
- Amazon: 31
- Undisclosed: 55 (growing — companies reference AI without naming providers)
- Anthropic: 4 (tiny but first appears in meaningful numbers in 2025–2026)

**Interpretation:** The "other" and "undisclosed" dominance suggests that vendor concentration data from annual reports undercounts real dependency on hyperscalers — companies often reference AI capabilities via cloud provider wrappers without naming the underlying model provider.

### 6.5 Substantiveness: Quality Is Not Keeping Pace with Volume
**Argues:** The most important quality finding — disclosure volume is rising rapidly, but the substantiveness composition has remained flat or worsened slightly. Most disclosure is "moderate," not "substantive."

**Key numbers:**
- 2023: substantive 14% / moderate 77% / boilerplate 10% (of risk-classified chunks)
- 2024: substantive 10% / moderate 77% / boilerplate 12%
- 2025: substantive 9.5% / moderate 81% / boilerplate 9.8%
- Absolute substantive reports grew (27→64), but as a share fell
- The surge in risk mentions is largely driven by "moderate" quality disclosure — companies are mentioning AI risk more often, but not describing it more concretely

**Connect to literature:** Brown & Tucker (2011) on textual change as proxy for informativeness; Dyer et al. (2017) on disclosure inflation; SEC 2024 review warning against generic AI disclosure. The UK data confirms the same pattern at scale.

### 6.6 Sector and CNI Breakdown
**Argues:** Disclosure is not evenly distributed across sectors, and the gap between disclosure prevalence and plausible AI exposure varies significantly — Finance discloses more but is also genuinely AI-intensive; Water and Defence disclose less relative to their operational AI exposure.

**Key patterns:**
- Finance (461 companies): highest absolute disclosure across all signals — expected given sector's AI intensity and regulatory scrutiny
- Health: high regulatory compliance and third-party supply chain risk — consistent with AI in diagnostics and vendor-mediated AI (pathology tools, etc.)
- Energy: cybersecurity and operational-technical dominate — consistent with OT/IT convergence concerns
- Communications: elevated information integrity and national security signals — consistent with content moderation and infrastructure roles
- Water: 82 total risk signals from 18 companies — relatively sparse, but national security mentions present
- Defence: relatively low LLM adoption (15 signals) vs. high agentic mentions (31) — unusual profile suggesting autonomous-systems language in defense reporting
- Government: workforce impacts is a leading risk category — consistent with AI-in-public-services debate

### 6.7 Market Segment Comparison
**Argues:** FTSE 100 companies are by far the most active AI disclosers; AIM companies show a dramatically lower disclosure rate that likely reflects both lower reporting obligations and less AI-intensive business models.

**Key numbers:**
- FTSE 100: 65.6% of reports have AI signal (892/1,359) — near-saturation at the top
- FTSE 350: 56.6% (2,060/3,638)
- Main Market overall: 53.4% (2,225/4,166)
- AIM: 16.9% (239/1,414) — ~3× lower than Main Market
- AIM risk rate is even lower: only 37 risk-signal reports across 1,414 reports (2.6%)

**Interpretation:** AIM's gap is structural — lighter governance requirements, smaller companies, less AI-intensive sectors. But it also represents a disclosure blind spot: AIM includes companies in CNI-adjacent sectors with real operational AI exposure. The gap is a monitoring challenge, not just a data artefact.

---

## 7. Limitations (~1 page)

### 7.1 Coverage Gaps
- 6.7% irreducible gap (shell companies, SPACs, micro-caps that never filed electronically)
- AIM coverage is CH-PDF-dependent (only 28% from FR); PDF quality variable
- 2020 and 2026 are partial years — treated as indicative, not comparable to peak years
- 178 companies with zero FR coverage due to ingestion failures; partially mitigated by CH gap-fill but not fully resolved

### 7.2 Classification Validity
- Golden set size: [N] chunks — larger than many comparable studies but still a constraint on sub-category precision
- Conservative prompt design reduces false positives at the cost of some recall (e.g., indirect AI references may not be captured)
- Substantiveness classification is inherently judgement-dependent — the three-band scale is a simplification; inter-rater agreement on the boundary between "moderate" and "substantive" is lower than for coarser categories
- Vendor tagging: "other" and "undisclosed" categories mask real concentration; not all vendor references are specific enough to classify

### 7.3 Causal and Inferential Limits
- Disclosure ≠ exposure: a company that mentions AI risk is not necessarily more exposed than one that doesn't — it may simply have better governance, or face regulatory pressure to disclose
- Provision 29 / ChatGPT timing overlap: we cannot cleanly decompose how much of the 2024 surge is regulatory (Provision 29 anticipation), technological (ChatGPT diffusion), or market (investor pressure). All three are plausible
- UK-only: findings are not directly generalisable to other jurisdictions, though the methodology is portable
- Annual reports lag real-world events by months — disclosure reflects what companies were willing to commit to at reporting date, not real-time exposure

---

## 8. Discussion (~1.5 pages)

### 8.1 Policy Implications
**Three headline takeaways for AISI / policymakers:**

1. **The disclosure gap is closing but quality is not improving.** Risk mention rates have risen dramatically, but the substantiveness distribution has flatlined. Regulators targeting disclosure quality (not just disclosure presence) should consider metrics beyond mention frequency — our substantiveness dimension provides a tool for this.

2. **CNI sectors show uneven disclosure depth.** Finance and Health are active disclosers. Water, Defence, and Data Infrastructure show thin disclosure relative to their plausible operational AI exposure. These are sectors where regulatory intervention or AISI engagement is most likely to be additive.

3. **Vendor concentration is underreported.** The dominance of "other" and "undisclosed" in vendor tagging suggests that companies are not yet treating AI provider dependencies as material disclosure items, even as hyperscaler concentration at the infrastructure layer grows. This is a systemic risk monitoring gap.

### 8.2 Applications and Use Cases
- **Regulatory benchmarking:** Track year-on-year disclosure improvement as Provision 29 takes full effect (first full disclosure cycle FY2026). The Observatory provides a pre/post baseline.
- **Supervisory prioritisation:** Identify companies and sectors with high plausible AI exposure and low disclosure — a screening tool for AISI or sector regulators.
- **Research replication:** The methodology is fully documented and the pipeline is reproducible. Other jurisdictions can replicate the approach using local filing corpora.
- **Annual report as ground truth:** Confirms or refutes softer signals (earnings calls, press releases, surveys) about AI adoption — the "hard" disclosure baseline.

### 8.3 Future Work
- **Provision 29 follow-through:** Re-run the full pipeline on FY2026 reports (available mid-2027) to measure whether the regulation delivers substantiveness improvement or merely increases disclosure volume.
- **Cross-jurisdictional expansion:** Apply the same pipeline to EU ESEF filings and SEC 10-Ks to enable comparative analysis.
- **Causal modelling:** Correlate disclosure signals with external data (cyber incidents, AI-related regulatory actions, job displacement data) to test whether annual-report language predicts or lags real-world outcomes.
- **Boilerplate tracking over time:** Measure year-over-year textual similarity of AI-risk passages within companies — a firmer operationalisation of Brown & Tucker's MD&A modification measure applied specifically to AI language.
- **Agentic and autonomous systems:** The "agentic" adoption category shows ~6× growth (27→172). This is an early signal worth dedicated monitoring as autonomous AI deployment accelerates.

---

## 9. Conclusion (~0.5 page)

Restates the three research questions and their answers in plain language. Closes with the Observatory's contribution: it establishes a replicable, structured baseline for monitoring AI disclosure in UK annual reports, demonstrates that volume growth is not matched by quality improvement, and provides a tool for AISI and regulators to track the effect of governance interventions over time.

---

## 10. Appendix

**A. Classifier Taxonomy** — Full label definitions for mention types, adoption sub-types, risk categories, vendor tags, and substantiveness bands. (Draw from classifiers.yaml)

**B. Coverage Waterfall** — Full table: Ideal 7,345 → achievable 6,854 → processed 9,821 (including 2020 and partial 2026). With segment and year breakdowns.

**C. Golden Set Summary** — Sample size, annotation process, agreement rates by classifier category.

**D. Sector and Market Segment Full Data Tables** — Complete risk-by-sector, adoption-by-sector, and vendor-by-sector breakdowns for reference.

**E. Selected Example Chunks** — 3–6 illustrative classified excerpts (already in dashboard-data.json exampleChunks), showing range from boilerplate to substantive.

---

## Key statistics reference (for writers)

| Classifier model | Gemini Flash 3 |
| Golden set size | 474 human-annotated chunks |
| Stat | Value |
|------|-------|
| Total reports in corpus | 9,821 |
| Total companies | 1,362 |
| Years covered | 2020–2026 (2026 partial) |
| Reports with AI signal | 4,078 (41.5%) |
| Adoption reports | 3,012 |
| Risk reports | 1,860 |
| Vendor reports | 877 |
| AI risk % in 2020 | 3.2% (32/1,007) |
| AI risk % in 2025 | 43.2% (674/1,561) |
| Risk mention growth 2020→2025 | ~21× |
| LLM adoption growth 2020→2025 | ~14× |
| Substantive % (2025) | ~9.5% of risk-classified chunks |
| FTSE 100 AI signal rate | 65.6% |
| AIM AI signal rate | 16.9% |
| Dominant risk category (2025) | Strategic & competitive (458) |
| Fastest-growing risk (2020→2025) | Workforce impacts (~44×: 4→176) |
| Most vendor-cited company | Microsoft (87 in 2025) |
| Largest "hidden" vendor category | "Other" (143 in 2025) |

---

## Literature reference list (final 12 priority citations)

1. Loughran & McDonald (2011) — finance-specific textual analysis
2. Loughran & McDonald survey (SSRN) — field overview
3. Brown & Tucker (2011) — textual similarity / disclosure modification
4. Dyer, Lang & Stice-Lawrence (2017) — boilerplate growth
5. Lang & Stice-Lawrence (2015) — international annual-report corpus
6. Kim (2024) — LLMs for financial statement analysis
7. Park / BIS-IFC (2024) — LLMs for materiality in risk disclosures
8. Uberti-Bona Marin et al. (2025) — AI risk in US 10-Ks
9. Bonsón et al. (2023) — AI/ADM disclosure in European corporate reports
10. ClimateBERT / ClimaText (2023) — observatory-style disclosure monitoring
11. Ferjančič et al. (2024) — FTSE 350 ESG topic extraction
12. FRC UK Corporate Governance Code 2024; FCA DTR 4.1 (official sources)

Additional:
- Chiu (2025) — GenAI in corporate narrative reporting
- SEC 2024 State of Disclosure Review
- AISI Research Agenda / Frontier AI Trends Report
- NPSA CNI sector definitions
- Li (2008) — Fog Index / management obfuscation hypothesis
