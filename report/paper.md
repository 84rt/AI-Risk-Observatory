# Tracking AI Risk and Adoption in UK Listed Company Annual Reports: Evidence from the AI Risk Observatory (2020–2025)

---

## Abstract

We present the AI Risk Observatory, a large-scale automated system for monitoring AI-related disclosures in UK listed company annual reports. Applying a two-stage LLM classification pipeline — validated against a 474-chunk human-annotated golden set — to 9,821 reports from 1,362 companies (2020–2026), we extract structured signals across adoption type, risk category, vendor dependency, and disclosure substantiveness. We find that AI-related disclosure has grown from 19.5% of reports in 2020 to 65.5% in 2025, with risk-specific disclosure growing approximately 21× over the same period. Despite this volume growth, substantive risk disclosure — language containing named systems, specific operational context, and governance evidence — accounts for fewer than 10% of risk-classified chunks in 2025 with no upward trend since 2023. Critical National Infrastructure sectors including Water, Defence, and Data Infrastructure show thin disclosure relative to plausible operational AI exposure. Vendor concentration is systematically underreported. The Observatory establishes the first large-scale structured baseline for AI disclosure in UK annual reports, directly relevant for regulatory evaluation as the 2024 UK Corporate Governance Code's Provision 29 takes effect.

---

## 1. Introduction

Annual reports are legally mandated, board-accountable artefacts that companies are required to produce under the Companies Act 2006 and the FCA's Disclosure Guidance and Transparency Rules. Their principal risk disclosures must be material and entity-specific, making them a uniquely reliable surface for tracking corporate AI governance at scale — more accountable than earnings calls or voluntary disclosures, and audited in ways that other AI monitoring data sources are not. The challenge is extracting structured, comparable signal from thousands of long documents without sacrificing reproducibility.

The need for such infrastructure has become acute. Regulators on both sides of the Atlantic have begun calling explicitly for AI disclosures to be specific and grounded — not generic statements about industry trends — but lack the tools to assess compliance at scale. The UK's 2024 Corporate Governance Code introduces Provision 29, requiring boards to declare the effectiveness of all material internal controls (including AI governance) from FY2026. The 2024 SEC State of Disclosure Review issued equivalent pressure on US 10-K filers. Whether these interventions shift disclosure *quality*, not just *volume*, is an open empirical question that requires a monitoring infrastructure to answer.

We introduce the AI Risk Observatory, an automated pipeline that applies a validated LLM classifier to the full population of UK listed companies to produce longitudinal, structured AI disclosure signals. We address three research questions: (RQ1) How has the prevalence and composition of AI disclosure changed between 2020 and 2025? (RQ2) Is risk disclosure keeping pace with adoption disclosure, and is it substantive or boilerplate? (RQ3) Which sectors and market segments show the greatest disclosure gaps relative to plausible AI exposure?

Our contributions are: (1) the first large-scale structured analysis of AI disclosure across UK listed companies, spanning both Main Market and AIM; (2) an explicit substantiveness dimension distinguishing boilerplate from operationally grounded disclosure; (3) a CNI-sector decomposition connecting disclosure patterns to national security and infrastructure frameworks; and (4) a reproducible, versioned pipeline enabling annual re-runs to measure regulatory intervention effects.

---

## 2. Related Work

**Financial disclosure NLP.** Loughran and McDonald (2011) established that financial language requires domain-specific treatment and validated annual reports as an NLP corpus. Lang and Stice-Lawrence (2015) extended this to non-US filings. Kim (2024) and Park et al. (BIS/IFC, 2024) demonstrate that LLM-based classification of financial text is now competitive with specialised models, validating our pipeline design.

**Boilerplate and disclosure quality.** Dyer, Lang and Stice-Lawrence (2017) document systematic growth in disclosure length without commensurate informational content. Brown and Tucker (2011) show that year-over-year textual modification in MD&A sections is a proxy for genuine informational updating. These findings motivate our substantiveness classifier: in a domain where boilerplate is the default, measuring quality separately from volume is methodologically necessary.

**AI disclosure.** Uberti-Bona Marin et al. (2025) analyse 30,000+ US 10-K filings and find AI risk mentions exceeded 50% of firms by 2024, but disclosures remain generic — a pattern our UK analysis replicates and extends. Most prior work is US-centric; UK-specific studies are sparse and lack CNI-sector decomposition or substantiveness measurement.

**Observatory-style monitoring.** ClimateBERT (Webersinke et al., 2023) and Ferjančič et al. (2024) — who apply BERTopic to a decade of FTSE 350 ESG reports — establish the genre: specialised classifiers on structured corporate corpora, used to track disclosure evolution in response to regulation and events. Our work extends this approach to AI risk and adoption, adding vendor dependency and substantiveness dimensions not present in prior climate or ESG monitoring work.

---

## 3. Methodology

### 3.1 Corpus and Data Ingestion

The company universe comprises 1,362 UK-incorporated LSE-listed companies across the FTSE 350, AIM, and other market segments, yielding 9,821 annual reports spanning publication years 2020–2026 (2026 partial). Reports are sourced from two pipelines: financialreports.eu provides iXBRL-formatted filings converted to structured markdown (primary source, near-complete Main Market coverage), and Companies House provides PDF gap-fill for AIM and smaller companies where iXBRL coverage is incomplete. financialreports.eu covers only 28% of AIM slots; Companies House PDF is therefore the primary AIM source. A known failure set of 178 companies with zero iXBRL coverage is partially mitigated by the CH route but not fully resolved.

### 3.2 Preprocessing and Chunking

Each report is processed through a keyword gate requiring explicit mention of AI, ML, LLM, GenAI, or a clearly AI-specific technique. Terms such as "data analytics," "automation," and "digital tools" do not pass the gate. Passing passages are expanded to ±400 characters of surrounding context, deduplicated, and annotated with company, year, market segment, CNI sector, and report section metadata.

### 3.3 Two-Stage Classification Pipeline

**Stage 1 (mention-type classifier)** assigns one or more of: *adoption*, *risk*, *harm*, *vendor*, *general\_ambiguous*, or *none*. Labels are non-mutually exclusive except *none*. Adoption requires company-specific deployment language; risk requires AI to be attributed as the risk source; *general\_ambiguous* captures explicit AI mentions that do not meet any other threshold.

**Stage 2 (deep classifiers)** runs conditionally on chunks carrying a signal label. Three parallel classifiers apply:

- *Adoption classifier*: scores three non-mutually-exclusive types — *non-LLM* (traditional ML), *llm* (generative AI), *agentic* (autonomous execution with limited human oversight) — each on a 0–3 signal scale.
- *Risk classifier*: maps to one or more of 10 categories (Table 1) with signal scores (1–3) and a substantiveness band (*boilerplate / moderate / substantive*).
- *Vendor classifier*: assigns named provider tags (Microsoft, Google, OpenAI, Amazon, Meta, Anthropic) or *internal / other / undisclosed*.

All classifiers use schema-constrained outputs. All classification was performed using **Gemini Flash 3**.

**Table 1: Risk category taxonomy**

| Category | Definition |
|---|---|
| `strategic_competitive` | Competitive disadvantage, industry disruption, failure to adopt |
| `operational_technical` | Model failures, reliability/accuracy issues, hallucinations |
| `cybersecurity` | AI-enabled attacks, adversarial abuse, breach exposure |
| `workforce_impacts` | Displacement, skills obsolescence, shadow AI |
| `regulatory_compliance` | AI Act/GDPR/privacy obligations, legal liability |
| `information_integrity` | Misinformation, deepfakes, content authenticity |
| `reputational_ethical` | Trust erosion, bias, fairness, social licence |
| `third_party_supply_chain` | Vendor dependency, concentration risk |
| `environmental_impact` | Energy/carbon footprint of AI |
| `national_security` | Critical infrastructure, geopolitical AI risks |

### 3.4 Validation

The pipeline was validated against a 474-chunk human-annotated golden set drawn from 30 reports (2023–2024 filings for 15 companies across all 13 UK CNI sectors). Six models were compared on a 20-chunk sample; Gemini Flash 3 was selected on accuracy, cost, and structured-output reliability. At temperature 0 (production setting), the classifier produces 100% consistent outputs across repeated runs; at temperature 0.7 consistency is 95%, with divergent outputs predominantly assessed as correct additions rather than errors.

Report-level Jaccard similarity between LLM and human labels is shown in Table 2. The risk taxonomy score of 0.23 reflects LLM *comprehensiveness* rather than error: the LLM assigned 4.77 labels per report versus 3.77 for the human annotator, and the majority of LLM-only labels were confirmed accurate on review. The human baseline is better understood as a conservative lower bound. Agreement improves from 2023 to 2024 (risk Jaccard 0.18 → 0.29), consistent with better classifier performance on more standardised recent AI terminology.

**Table 2: Report-level Jaccard similarity, LLM vs human baseline (n=30 reports)**

| Dimension | All years | 2023 | 2024 |
|---|---|---|---|
| Mention type | 0.75 | 0.74 | 0.76 |
| Adoption type | 0.47 | 0.39 | 0.55 |
| Risk taxonomy | 0.23 | 0.18 | 0.29 |
| Vendor tags | 0.40 | — | — |

Approximately 29% of chunks passing the keyword gate are classified *none* by Stage 1 (false positives from regex); this was reduced 31% through chunking optimisation and is treated as acceptable given the design preference for high recall at the filtering step.

---

## 4. Results

### 4.1 Disclosure Surge and Composition Shift

AI-related disclosure has grown from 19.5% of reports in 2020 (196/1,007) to 65.5% in 2025 (1,023/1,561). Risk-specific disclosure grew substantially faster from a smaller base: 3.2% (32/1,007) in 2020 to 43.2% (674/1,561) in 2025 — a 21× absolute increase. The adoption-to-risk ratio compressed from 4.4:1 in 2020 to 1.07:1 in 2025; in partial 2026 data, risk reports marginally exceed adoption reports for the first time (Table 3).

**Table 3: Annual disclosure prevalence by signal type (reports with ≥1 signal)**

| Year | Reports | Any AI | Adoption | Risk | Gen. ambiguous | Vendor |
|---|---|---|---|---|---|---|
| 2020 | 1,007 | 196 (19.5%) | 140 | 32 (3.2%) | 112 | 28 |
| 2021 | 1,328 | 361 (27.2%) | 285 | 59 (4.4%) | 195 | 41 |
| 2022 | 1,853 | 526 (28.4%) | 434 | 93 (5.0%) | 282 | 83 |
| 2023 | 1,905 | 702 (36.9%) | 511 | 194 (10.2%) | 427 | 119 |
| 2024 | 1,828 | 1,008 (55.1%) | 703 | 582 (31.8%) | 785 | 244 |
| 2025 | 1,561 | 1,023 (65.5%) | 721 | 674 (43.2%) | 853 | 271 |

The sharpest single-year inflection is the 2023→2024 risk surge (+200%: 194→582 reports), coinciding with ChatGPT mass adoption and anticipatory compliance with the 2024 UK Corporate Governance Code (Provision 29). Disentangling these mechanisms is not possible with observational disclosure data alone.

### 4.2 Adoption Type Evolution

LLM adoption (generative AI) grew 14× between 2020 and 2025 (39→557 reports), outpacing non-LLM growth (5×: 136→672). By 2025, LLM adoption appears in 83% as many reports as non-LLM, near-converging. Agentic adoption — autonomous systems executing tasks with limited human oversight — grew approximately 6× (27→172), a smaller absolute number but at a pace that warrants monitoring given its distinct governance implications.

### 4.3 Risk Category Diversification

The risk profile has broadened substantially. In 2020, strategic/competitive and operational/technical risks together dominated. By 2025, all 10 taxonomy categories are materially represented. Strategic/competitive leads at 458 reports (2025), followed by cybersecurity at 405 — a category that registered only 11 reports in 2020. The two fastest-growing categories from near-zero bases are information integrity (1→184 reports, driven by deepfake/misinformation language) and workforce impacts (4→176, 44× growth). National security appears in 49 reports in 2025, concentrated in CNI sectors, up from zero before 2022.

### 4.4 Disclosure Quality: The Substantiveness Gap

The central quality finding is not a plateau — it is a structural decline in substantive share relative to volume. As risk disclosure grew from 3.2% of reports in 2020 to 43.2% in 2025, the fraction of risk-disclosing companies producing *substantive* disclosure fell from 15.6% to 9.5%. The quality gap — risk mention rate minus substantive risk rate — has widened from 2.7 pp in 2020 to **39.1 pp in 2025**. The 2025 substantiveness distribution is 9.5% substantive, 80.7% moderate, 9.8% boilerplate. Risk signal directness is similarly flat: the explicit (signal 3) share was 31.6% in 2024 and 31.2% in 2025. Companies are disclosing more risk categories without attributing them to AI with greater directness. A representative substantive disclosure (Prudential PLC, 2021) identifies a specific causal pathway — particular technology practices creating specific data-handling exposure — rather than asserting AI risk generically. Reports of this kind account for fewer than 1 in 10 risk-disclosing companies.

### 4.5 Vendor Concentration Underreporting

Opaque references — *other* and *undisclosed* — account for **42.7% of all 2025 vendor assignments**. Microsoft is the most cited named provider (87 signals), but OpenAI is the only named vendor declining year-on-year (−1.0 pp), likely reflecting routing of OpenAI access through Azure (which registers as Microsoft). Among explicitly named vendors, the top three account for 75.7% of named-vendor assignments. Structural opacity means foundation-model concentration at the infrastructure layer is larger than named-vendor data suggests.

### 4.6 Sector and Market Segment Gaps

Finance and Communications lead on active disclosure. The largest 2025 blind spots are **Data Infrastructure** (85.0% of reports without an AI risk mention) and **Energy** (79.3%) — neither a low-AI-exposure sector. Communications saw the largest single-year rise (+24.6 pp). Defence is the only sector where agentic adoption signals (31) outnumber LLM signals (15), suggesting autonomous-systems deployment patterns distinct from the commercial sector.

In 2025, FTSE 100 companies reached an 84.8% AI mention rate and **71.3% AI risk rate**. AIM shows a 36.8% mention rate but only a **7.0% risk rate**, against 60.9% for the full Main Market. The risk gap exceeds the mention gap, suggesting AIM companies are aware of AI but not translating it into formal risk governance — a structural effect of lighter governance obligations.

---

## 5. Discussion and Limitations

**Policy implications.** The central finding — that disclosure volume has surged while substantive quality has declined as a share — challenges the regulatory assumption that more disclosure equals better transparency. Provision 29 of the 2024 UK Corporate Governance Code creates an incentive structure that should shift disclosure quality; the first full reporting cycle (FY2026 reports available 2027) will be the primary empirical test of whether it does. The Observatory provides a pre-intervention baseline at company, sector, and market-segment granularity needed to evaluate this. CNI sector decomposition identifies Data Infrastructure (85% blind spot) and Energy (79% blind spot) as the highest-priority targets for AISI engagement: absent disclosure does not imply absent exposure.

**Limitations.** The golden set (474 chunks, 15 large-cap companies, 2023–2024 only) is a calibration instrument, not a production-scale evaluation; agreement rates on simpler filings and pre-2023 reports have not been independently validated. The conservative keyword gate suppresses false positives at the cost of some recall: oblique AI references and proprietary-system language without explicit AI attribution will not be captured. The pipeline produces a lower bound on AI disclosure activity. The 2023–2024 surge cannot be cleanly decomposed into regulatory (Provision 29 anticipation), technological (ChatGPT diffusion), and market (investor pressure) components — all three are plausible and concurrent. Disclosure does not equal exposure: companies with absent AI disclosure may face significant operational AI risk not yet recognised or disclosed. Substantiveness classification has been applied only to risk-classified chunks; adoption and vendor quality assessment remains future work.

---

## 6. Conclusion

The AI Risk Observatory demonstrates that structured, large-scale monitoring of AI disclosure in annual reports is technically feasible and policy-relevant. Across 9,821 reports from 1,362 UK listed companies, we document a 21× growth in AI risk disclosure between 2020 and 2025 — but find that disclosure quality, measured by substantiveness, has not improved since 2023. The adoption-to-risk ratio has compressed to near parity, and all ten risk categories in our taxonomy are now materially represented in the UK listed universe. Vendor concentration and sector-level disclosure gaps remain unresolved monitoring challenges.

The Observatory is designed to be re-run annually. The FY2026 reporting cycle — the first under Provision 29 — will reveal whether the quality plateau reflects a structural feature of voluntary disclosure or a transitional state that governance reform can shift. The methodology is jurisdiction-neutral and portable to ESEF and SEC 10-K corpora, enabling cross-jurisdictional comparison as equivalent monitoring infrastructure develops.

---

## References

Brown, S. V., & Tucker, J. W. (2011). Large-sample evidence on firms' year-over-year MD&A modifications. *Journal of Accounting Research*, 49(2), 309–346.

Dyer, T., Lang, M., & Stice-Lawrence, L. (2017). The evolution of 10-K textual disclosure: Evidence from latent Dirichlet allocation. *Journal of Accounting and Economics*, 64(2–3), 221–245.

Ferjančič, M., et al. (2024). A decade of ESG disclosure in FTSE 350 annual reports: Evidence from BERTopic. *Working paper*.

Kim, A. (2024). Financial statement analysis with large language models. *Working paper*, University of Chicago Booth.

Lang, M., & Stice-Lawrence, L. (2015). Textual analysis and international financial reporting: Large sample evidence. *Journal of Accounting and Economics*, 60(2–3), 110–135.

Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35–65.

Loughran, T., & McDonald, B. (2016). Textual analysis in accounting and finance: A survey. *Journal of Accounting Research*, 54(4), 1187–1230.

Park, H., et al. (2024). Large language models for risk extraction in financial disclosures. *BIS/IFC Working Paper*.

Uberti-Bona Marin, M., et al. (2025). AI risk disclosures in US public company filings: Evidence from 10-K reports (2019–2024). *Working paper*.

Webersinke, N., et al. (2023). ClimateBERT: A pretrained language model for climate-related text. *arXiv:2110.12010*.

Financial Reporting Council. (2024). *UK Corporate Governance Code 2024*. FRC.

Financial Conduct Authority. (2024). *Disclosure Guidance and Transparency Rules (DTR 4.1)*. FCA.

SEC Division of Corporation Finance. (2024). *Staff observations from 2024 disclosure review: Artificial intelligence*. SEC.
