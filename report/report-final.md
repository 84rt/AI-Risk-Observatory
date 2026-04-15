# Tracking AI Risk and Adoption in UK Public-Company Annual Reports: Evidence from the AI Risk Observatory (2020–2025)

*AI Risk Observatory: Research Report*
*riskobservatory.ai*
*Sponsored by the AI Security Institute (AISI), UK*

*2026*

---

## Executive Summary

**Context and motivation.** Artificial intelligence is reshaping the risk and operational landscape of UK listed companies at a pace that has outrun the regulatory frameworks designed to monitor it. Annual reports (legally mandated, board-accountable, and audited artefacts) represent the most reliable systematic evidence base for tracking how companies understand, disclose, and govern their AI exposure. The AI Risk Observatory applies a reproducible, large-language-model-based classification pipeline to the full population of UK listed companies to produce, for the first time, a structured and longitudinal analysis of AI disclosure across the Main Market and AIM.

**What we studied.** The corpus covers 9,821 annual reports from 1,362 companies, spanning publication years 2020 to 2026 (2026 partial). Reports are sourced from financialreports.eu (iXBRL primary) and Companies House (PDF gap-fill). Each report is processed through a two-stage classification pipeline using Gemini Flash 3 as the production model, assigning adoption type, risk category, vendor identity, and substantiveness to individual AI-related text chunks. The pipeline was validated against 474 human-annotated chunks.

**Three headline findings.** First, AI disclosure has grown from a minority feature to a near-majority feature of UK annual reports. In 2020, 19.5% of reports contained any AI signal; by 2025 that figure reached 65.5%. Risk-specific disclosure grew faster than adoption disclosure, from 3.2% of reports in 2020 to 43.2% in 2025, a twenty-one-fold increase. The sharpest single-year inflection occurred between 2023 and 2024 (+200% in risk mentions), coinciding with mass-market generative AI adoption and early anticipatory compliance with the 2024 UK Corporate Governance Code.

Second, disclosure quality has not kept pace with disclosure volume. Fewer than 10% of risk-classified text chunks in 2025 were rated substantive, defined as containing named AI systems, specific operational context, and material governance evidence. The majority of AI risk language remains boilerplate or moderately specific: present in the document but not informative to a reader seeking to understand the real nature of a company's AI exposure. This quality plateau persists despite rising mention rates and represents the most policy-relevant finding in the dataset.

Third, sector and market segment gaps are substantial. Finance (461 companies) and Health dominate the active-disclosure picture. Water, Defence, and Data Infrastructure show thin disclosure relative to their plausible operational AI exposure. FTSE 100 companies are near-saturated (65.6% signal rate) while AIM companies remain dramatically underrepresented (16.9% signal rate, 2.6% risk rate). Vendor concentration is systematically underreported: Microsoft, Google, and Amazon are the most commonly named providers, but the majority of AI provider references in 2025 were categorised as "other" or "undisclosed," masking growing infrastructure dependencies on a small number of hyperscalers.

**Three policy implications.** Regulatory frameworks that treat disclosure *presence* as the primary quality signal will generate rising volumes without rising information content. Sector-specific engagement, particularly for Water, Defence, and Data Infrastructure, is likely to be more additive than generic disclosure requirements. And the underreporting of vendor concentration represents a systemic risk monitoring gap that persists even when companies make good-faith efforts to describe their AI activities.

**What comes next.** The Observatory is designed as a reusable monitoring instrument. The most important near-term milestone is the FY2026 annual report cohort, the first full reporting cycle after Provision 29 of the 2024 UK Corporate Governance Code takes effect. Re-running the pipeline against that cohort will produce a direct, evidence-based assessment of whether the Code amendment delivers improvement in disclosure substantiveness or merely increases disclosure volume. The methodology is fully documented and portable to other jurisdictions; the UK dataset provides a baseline for cross-jurisdictional comparison as equivalent work develops in EU and US markets.

---

## 2. Introduction

### 2.1 Motivation

Annual reports are legally mandated, board-accountable artefacts under the Companies Act and the FCA's Disclosure Guidance and Transparency Rules. Unlike voluntary ESG disclosures or earnings presentations, they are produced under a legal accountability regime that creates strong incentives for accuracy. This makes them a qualitatively different evidence base from surveys, earnings calls, or press releases for tracking how companies understand and govern their AI exposure.

This paper presents the AI Risk Observatory: an automated monitoring system applied to 9,821 annual reports from 1,362 UK-listed companies spanning publication years 2020 to 2026. The Observatory uses a two-stage LLM classification pipeline to identify, categorise, and quality-rate AI-related disclosures, producing structured signals across adoption type, risk category, vendor dependency, and disclosure substantiveness. The goal is to establish a replicable, longitudinal baseline for regulatory policy, supervisory prioritisation, and research on the evolution of corporate AI governance in the United Kingdom.

### 2.2 Research Questions

The Observatory addresses three explicit empirical questions.

First, how has the prevalence and composition of AI-related disclosure changed across UK listed companies between 2020 and 2025? This encompasses both the growth in the share of companies disclosing anything about AI, and the changing mix between adoption signal, risk signal, and vendor disclosure across the period.

Second, is AI risk disclosure keeping pace with adoption disclosure, and is it substantive rather than boilerplate? The concern here is not merely whether companies mention AI risk, they increasingly do, but whether those mentions contain the specific operational detail, named systems, and governance evidence that would make them informative to investors, regulators, and researchers. A disclosure regime in which mention rates grow while quality stagnates is formally responsive but substantively hollow.

Third, which sectors and market segments show the greatest disclosure gaps relative to their plausible AI exposure? This question is motivated by the Observatory's Critical National Infrastructure focus. Sectors whose disruption would have cascading national consequences, Water, Energy, Defence, Communications, may present AI exposure patterns that are underrepresented in the disclosure record, whether because companies in those sectors are less AI-intensive, because they face different disclosure incentives, or because they are genuinely lagging in AI governance maturity.

### 2.3 Scope and Contribution

Most large-scale empirical work on AI disclosure has focused on US markets, specifically the SEC's 10-K mandatory filing. The most comprehensive recent study, Uberti-Bona Marin et al. (2025), analyses over 30,000 US filings and finds a steep rise in AI-risk mentions alongside persistently generic content. Lang and Stice-Lawrence (2015) demonstrated that large-sample textual analysis of annual reports outside the US is both feasible and methodologically sound, but empirical coverage of UK listed companies at scale remains sparse.

This paper makes four contributions. It provides the first large-scale structured analysis of AI disclosure across the full population of UK listed companies, covering both Main Market and AIM issuers. It introduces an explicit substantiveness dimension, distinguishing boilerplate from operationally grounded disclosure, that is absent from most prior work in this domain. It applies a CNI-sector decomposition that connects corporate disclosure patterns to the UK's national security and infrastructure resilience frameworks. And it establishes a reproducible pipeline, using Gemini Flash 3 as the production classifier validated against a 474-chunk human-annotated golden set, that can be re-run annually as the regulatory landscape evolves under the 2024 UK Corporate Governance Code.


---

## 3. Background and Regulatory Context

### 3.1 The UK Annual Reporting Framework

Annual reports produced by UK listed companies are not discretionary communications. They are legally mandated artefacts with defined content requirements, produced on a fixed schedule, subject to audit, and filed with regulators. This combination of legal obligation, audit scrutiny, and structured periodicity makes them a qualitatively different signal source from surveys, earnings calls, or voluntary sustainability disclosures, and it is the foundation of their value as a monitoring surface.

The legal basis for narrative risk reporting sits primarily in the Companies Act 2006 and associated regulations governing the strategic report. For financial years covered by this Observatory, UK-incorporated companies above certain thresholds are required to produce a strategic report that includes a fair review of the business and a description of the principal risks and uncertainties it faces. The Financial Reporting Council's (FRC) guidance on the strategic report makes clear that principal risk disclosures should be entity-specific and material, not generic restatements of sector-wide concerns. Whether AI risk qualifies as a principal risk is a judgement call made at the board level, which is why the prevalence and quality of that judgement across our corpus is itself an informative dataset.

At the market-disclosure layer, the Financial Conduct Authority's Disclosure Guidance and Transparency Rules (DTR 4.1) require issuers subject to those rules to publish an annual financial report within four months of the financial year end. For companies listed on the AIM market, the AIM Rules for Companies impose a less prescriptive six-month window, and AIM companies are not required to comply with the UK Corporate Governance Code. These differences in reporting timelines and governance obligations are directly relevant to the disclosure gap documented in the Findings section: AIM companies face lighter mandatory disclosure pressure, which is reflected in their substantially lower AI signal rates. The difference is regulatory in origin, not solely a function of company size or AI exposure.

The iXBRL filing format (Inline Extensible Business Reporting Language) is mandated for annual financial reports filed with HMRC and, for Main Market companies, with the FCA. iXBRL embeds machine-readable data tags within human-readable HTML, producing documents that are simultaneously readable and structured. This mandate is what makes systematic, large-scale ingestion of annual reports feasible: iXBRL filings can be downloaded programmatically, parsed at the document-structure level, and converted to clean markdown while retaining section metadata. Without this mandate, the Observatory's ingestion pipeline would require substantially more manual effort and would produce less consistent output across companies and years.

### 3.2 Critical National Infrastructure and AI Exposure

The UK's Critical National Infrastructure (CNI) framework designates thirteen sectors whose disruption or destruction would cause catastrophic consequences for public safety, economic stability, or national security. These sectors, including Energy, Finance, Transport, Health, Communications, Water, Food, Government, Defence, Civil Nuclear, Chemicals, Data Infrastructure, and Space, are defined and maintained by the National Protective Security Authority (NPSA) and sit at the intersection of the UK's security and economic governance frameworks.

AI is not a peripheral concern for CNI sectors. In Energy, AI is increasingly embedded in grid management, predictive maintenance, and demand forecasting, and operational technology (OT) systems that have historically been air-gapped are converging with IT infrastructure in ways that expand the attack surface. In Finance, AI underpins trading systems, credit models, fraud detection, and increasingly, customer-facing decision-making at scale. In Health, AI tools are entering clinical pathways, diagnostic support, image analysis, medication management, in deployments where errors have direct patient consequences. In Communications, AI mediates content distribution at infrastructure scale, creating information integrity exposure that is simultaneously a commercial risk and a societal concern.

Two features of AI in CNI sectors make disclosure-based monitoring particularly valuable. The first is interconnection: CNI sectors are not independent, and AI-related failures or dependencies in one sector can cascade across others. A compromised AI system in a financial institution may affect payment infrastructure; an AI-enabled cyber incident in an energy system may affect communications and transport. The second is third-party concentration: most CNI operators do not build their own AI systems. They procure AI capabilities from a small number of technology providers, creating dependency concentration that is not always visible in public disclosure. The vendor landscape documented in the Findings section, particularly the growth of *undisclosed* vendor references, is directly relevant to this systemic risk concern.

This Observatory maps LSE-listed companies to CNI sectors using a crosswalk from the International Standard Industrial Classification (ISIC) to the NPSA sector taxonomy. The mapping involves inherent ambiguity, ISIC codes do not align cleanly with CNI sector boundaries, and some companies operate across multiple CNI sectors simultaneously. Where a company receives multiple CNI sector assignments, the primary classification is used for sector-level analysis. The sector distribution in our corpus (Finance: 461 companies, Energy: 141, Health: 111, Transport: 61, Communications: 28, Water: 18) reflects both the actual composition of the LSE-listed universe and the limitations of the ISIC-to-CNI crosswalk, which is described in Appendix B.

### 3.3 Governance Reforms and Disclosure Incentives

The regulatory environment governing how UK companies disclose risk has changed materially over the period covered by this Observatory, and those changes are visible in the data. The most consequential single development is the Financial Reporting Council's 2024 revision of the UK Corporate Governance Code, which introduced obligations that reshape how boards must account for their risk management and internal control frameworks.

Under the pre-2024 Code, the board's obligation was largely procedural: directors were required to state that they had conducted a review of the effectiveness of the company's risk management and internal control systems. The threshold for compliance was the fact of review, not its depth. The 2024 Code raises this threshold substantially. From financial years beginning on or after 1 January 2025 the revised Code applies broadly; from financial years beginning on or after 1 January 2026, Provision 29 specifically requires boards to make a formal declaration on the effectiveness of all *material controls*, defined to span financial, operational, reporting, and compliance controls. Where material controls have not operated effectively, the board must describe the failure, explain its root cause, and state what corrective action has been taken or is planned. The provision operates within the UK's "comply or explain" regime rather than imposing statutory penalties, relying instead on public transparency and investor accountability to drive compliance, a deliberate contrast with the US Sarbanes-Oxley Act's mandatory auditor attestation model.

The relevance of this reform to AI disclosure is direct. AI governance, covering how boards oversee AI deployment, manage AI-related compliance obligations, and control AI-related operational risks, is a form of material control under the Provision 29 framing. Even before the provision's effective date, companies anticipating its requirements had an incentive to strengthen their disclosure of AI-related risks and controls, to demonstrate that material exposures were being identified and managed. This anticipatory compliance effect is a plausible explanation for a substantial portion of the 2023–2024 risk disclosure surge documented in the Findings. The sharpest single-year increase in the dataset, risk reports growing from 194 in 2023 to 582 in 2024, a 200% increase, coincides precisely with the period in which the FRC published the revised Code and companies began preparing for compliance. Attribution is not straightforward: the ChatGPT diffusion effect, investor pressure, and genuine growth in AI deployment are all concurrent factors. But the governance reform provides a structural mechanism that the diffusion narrative alone cannot.

The UK's reform is not occurring in isolation. In the United States, the Securities and Exchange Commission's 2024 review of disclosure practices observed a significant increase in AI mentions in annual reports and stated explicitly that disclosure teams expected companies' AI disclosures to be tailored to their specific circumstances, tied to actual business use, and grounded in a reasonable factual basis, not generic statements about industry-wide trends. The SEC's framing, that the form of AI disclosure is less important than whether it is material and specific, mirrors the FRC's Provision 29 emphasis on substantive over procedural compliance. The convergence of UK and US regulatory pressure on disclosure quality, rather than just disclosure presence, is directly relevant to the substantiveness gap documented in §6.5: if regulators in both jurisdictions are explicitly calling for specific and material AI disclosure, and the data shows that only approximately 10% of risk-disclosing companies are producing substantive language, the gap between regulatory expectation and disclosure practice remains large.

---

## 4. Related Work

This Observatory draws on four established research streams: financial disclosure NLP, the study of substantiveness and boilerplate in regulated reporting, empirical work on AI disclosure and governance, and observatory-style monitoring systems applied to other risk domains. Together these streams provide the methodological and conceptual foundations for the design choices described in §5, and they position this work within an active research landscape while identifying the gap it addresses.

### 4.1 Financial Disclosure NLP

The application of natural language processing to corporate filings has a well-developed lineage, anchored by the observation that financial language is domain-specific enough to defeat general-purpose text analysis tools. Loughran and McDonald (2011) provided the foundational demonstration of this problem: showing that generic sentiment dictionaries, built on general English corpora, systematically misclassify financial text, treating routine accounting terms such as "liability," "tax," and "impairment" as negative when they carry no such valence in a financial context. Their construction of a finance-specific word list, and their evidence linking filing language to market outcomes, established annual reports as a legitimate and productive NLP corpus. Their subsequent survey of the field (Loughran & McDonald, 2016) maps the methodological terrain, tone, readability, topic extraction, risk language, document length, and provides the framework within which subsequent work, including this Observatory, situates its design choices.

The question of whether textual analysis methods developed on US 10-K filings generalise to other jurisdictions is addressed directly by Lang and Stice-Lawrence (2015), whose study of international annual reports demonstrates that large-sample NLP analysis of non-US filings is both feasible and informative, while noting that disclosure norms, regulatory environments, and reporting cultures vary across markets. This finding supports the treatment of UK annual reports as a valid corpus for structured analysis while reinforcing the need for UK-specific calibration of any taxonomy or classifier.

More recent work has extended this tradition to large language models. Kim (2024) demonstrates that LLMs can extract decision-relevant signals from financial statements at scale, producing outputs that are competitive with or superior to specialised financial models on several evaluation tasks. Park and colleagues (BIS/IFC, 2024) apply LLM-based agents specifically to the problem of materiality assessment in risk disclosures, identifying which disclosed risks are substantive enough to affect investment decisions, a task closely analogous to the substantiveness classification in our pipeline. These studies position LLM-based classification of financial text as an emerging standard rather than a novelty, and they validate the use of Gemini Flash 3 for the classification tasks described in §5.4.

### 4.2 Substantiveness and Boilerplate

A parallel and closely related literature examines a distinctive feature of regulated disclosure: the tendency for mandatory reports to accumulate length and repetition without commensurate growth in informational content. Dyer, Lang and Stice-Lawrence (2017) document this phenomenon at scale across US 10-K filings spanning several decades, showing that reports have grown substantially longer while becoming more redundant and less specific, a pattern they attribute to regulatory accretion, legal caution, and the incentive to copy prior-year language rather than update it. This evidence of systematic disclosure inflation provides the empirical motivation for the Observatory's substantiveness classifier: if the default tendency of regulated disclosure is towards generic and sticky language, then measuring quality separately from volume is not merely useful but necessary.

Brown and Tucker (2011) provide the most influential operationalisation of disclosure quality in this tradition, showing that year-over-year textual modification in the MD&A section of 10-Ks serves as a proxy for informational updating: reports that change more tend to carry more new information. This insight, that similarity to prior-year language is itself an informative signal, underpins the conceptual logic of our substantiveness scale, which assesses individual passages rather than whole-document similarity but shares the same underlying intuition: language that could appear unchanged in any company's report in any year tells the reader very little about that company's specific situation.

The economics of boilerplate are not merely aesthetic. Generic risk factor language has historically provided legal protection under safe-harbour provisions, creating a rational incentive for companies to disclose broadly without disclosing specifically. Regulatory pressure on both sides of the Atlantic has begun to push against this: the SEC's 2020 modernisation of Regulation S-K introduced requirements for risk factors to be organised under relevant headings and summarised when lengthy, reflecting an explicit judgment that long lists of undifferentiated generic risks fail investors. The SEC's 2024 State of Disclosure Review extended this pressure specifically to AI: staff noted that AI disclosure had grown rapidly but remained in many cases untailored and insufficiently tied to actual business use. These regulatory developments contextualise the substantiveness gap documented in §6.5, the gap exists in a regulatory environment that has been actively trying to close it.

### 4.3 AI Disclosure and Governance

Empirical research on AI disclosure in corporate filings has accelerated markedly since 2022, though it remains concentrated on US SEC filings and has not yet produced comparable work on UK annual reports. Uberti-Bona Marin and colleagues (2025) provide the most directly comparable large-scale study: analysing more than 30,000 filings from over 7,000 US firms, they document a steep rise in AI risk mentions between 2020 and 2024, with over 50% of firms mentioning AI in their annual filings by 2024, a pattern consistent in direction with the UK findings reported here, though the US base rate differs given the different disclosure environments. Critically, they also find that many disclosures remain generic and thin on mitigation detail, mirroring the substantiveness gap we document for UK risk disclosure.

On the UK specifically, a 2025 study of AI narratives in FTSE 100 annual reports (2020–2023) frames AI disclosure as a form of strategic communication and impression management rather than straightforward risk reporting, companies shape their AI narratives to signal capability and ambition to investors, not only to discharge disclosure obligations. This framing supports the Observatory's design choice to maintain a separate *general_ambiguous* label for AI mentions that are real but not classifiable as adoption, risk, or vendor: the boundary between strategic positioning and substantive disclosure is a fundamental feature of the corpus, not a noise problem to be resolved away.

Bonsón and colleagues (2023) examine algorithmic decision-making disclosures in major Western European companies, finding that practice is uneven and largely voluntary, a finding that reflects the pre-regulatory baseline and helps explain the heterogeneity we observe. Chiu (2025) raises a related but distinct concern: as companies begin using generative AI in the drafting of narrative disclosures themselves, the authorship and reliability of the text being analysed changes in ways that are difficult to detect. This is a methodological frontier for disclosure-based monitoring systems, including this one.

### 4.4 Observatory-Style Monitoring Systems

The closest methodological precedents for this Observatory are not in AI disclosure research but in climate and ESG monitoring, where comparable systems have been built, validated, and applied to policy questions over the past decade. The genre is well established: define a thematic taxonomy, build or fine-tune a classifier on a structured corporate corpus, apply it at scale, and use the resulting signals to track how disclosure evolves in response to regulation, events, and market pressure.

ClimateBERT (Webersinke et al., 2023) is the most technically proximate precedent: a transformer model fine-tuned on climate-related corporate and policy texts, applied to the classification of climate-risk disclosure in financial reports and related documents. The study makes two findings that are directly relevant here. First, domain-adaptive pre-training substantially improves classification performance over general-purpose models on specialist financial text, consistent with our use of a model carefully prompted with domain-specific taxonomies and examples. Second, monitoring at scale reveals that voluntary climate disclosure frequently lacks the specificity that would make it decision-useful, a finding that parallels the substantiveness gap in AI risk disclosure. Ferjančič and colleagues (2024) analyse a decade of FTSE 350 annual reports using BERTopic to extract ESG themes, demonstrating that topic prevalence in annual reports shifts measurably in response to regulatory changes and major events, validating the temporal monitoring logic underpinning this Observatory. Their corpus and methodology are the closest existing analogue to ours in a UK listed-company context.

Central banks and financial supervisors have also developed this tradition. A Bank of England working paper applies NLP classification to public reports to retrieve climate-related information aligned with TCFD disclosure recommendations, while the FCA's multi-firm review of TCFD-aligned disclosures by premium-listed companies demonstrates that a regulator can systematically read and assess a large corpus of annual reports for thematic coverage, which is precisely the activity this Observatory automates and scales. The BIS/IFC work by Park and colleagues (2024) extends this to LLM-based risk extraction, describing a pipeline, identify, extract, classify, monitor, that closely mirrors the architecture described in §5.

Taken together, these precedents demonstrate that NLP-driven monitoring of structured corporate report corpora is a productive and increasingly standard research methodology. What distinguishes this Observatory from existing work is its specific focus: UK annual reports rather than US 10-Ks, an explicit CNI sector decomposition, a vendor dependency signal not present in prior AI disclosure studies, and a substantiveness dimension that moves beyond mention frequency to assess disclosure quality. Prior observatory work has established the genre and validated the approach; this work extends it to a domain, AI risk and adoption in UK listed companies, where no comparable structured dataset previously existed.

---

---

## 5. Methodology

### 5.1 Corpus and Universe Definition

The starting point for the corpus was the full population of companies listed on the London Stock Exchange. To ensure consistent regulatory coverage and reliable document retrieval, the universe was restricted to UK-incorporated entities, removing 191 companies incorporated outside the UK, primarily in Ireland, the Netherlands, and Australia, that are LSE-listed but hold no Companies House registration. This yielded a target universe of **1,469 companies** across three market segments: the Main Market (including FTSE 350 constituents), AIM, and smaller markets including Aquis.

The temporal scope covers **publication years 2021 to 2025** as the primary analysis window. Publication year, the calendar year in which an annual report was filed or publicly released, is used as the primary temporal key rather than fiscal year. This avoids ambiguity introduced by companies with non-calendar fiscal year ends and aligns with the date-stamped document metadata available from both ingestion sources. Reports published in 2020 and the first months of 2026 are included in the corpus but treated as supplementary given partial coverage.

The full target is **7,345 company-year slots** (1,469 companies × 5 publication years). The final corpus contains **9,821 processed reports** spanning **1,362 companies**, with the difference from the target reflecting the inclusion of 2020 and partial 2026 data alongside the exclusion of a small number of companies for which no processable filing was found across any year.

### 5.2 Data Ingestion

Annual reports were collected from two complementary sources to maximise coverage across market segments and years.

The primary source is **financialreports.eu (FR)**, which aggregates annual filings submitted to the FCA and other European regulators in iXBRL format and converts them to structured markdown. iXBRL, the machine-readable filing format mandated for Main Market issuers, preserves document metadata including section headings and structural tagging, which aids downstream extraction. FR provides strong coverage for Main Market companies, reaching approximately 95% of FTSE 350 company-year slots, but significantly weaker coverage for AIM, where it captures only around 28% of available filings. This asymmetry reflects the lighter electronic filing requirements for AIM companies rather than any gap in actual corporate reporting.

The secondary source is **Companies House (CH)**, which holds the PDF copy of every annual report filed by UK-incorporated companies regardless of market segment. PDF filings were downloaded via the Companies House API and converted to markdown through an OCR pipeline. The CH route serves primarily as a gap-fill for companies absent from or incompletely covered by FR, and is the dominant source for AIM companies. Cross-year deduplication was applied to prevent the same physical document from appearing in adjacent year slots, a known artefact arising when a company files with FR in December and with Companies House the following January.

Despite combining both sources, 178 companies with confirmed Companies House filings could not be matched to any FR annual report, due to ingestion or classification failures on the FR side. The clearest case is Jet2 PLC, where FR entries labelled "Annual Report" were in fact share buyback notices, the actual annual report was absent from FR's index entirely. These gaps are partially mitigated by the CH PDF route but are not fully resolved. Full coverage detail and the source waterfall are provided in Appendix B.

### 5.3 Preprocessing and Chunking

Raw documents, whether iXBRL-sourced markdown from FR or PDF-converted markdown from Companies House, are normalised to a consistent plain-text format with section metadata retained. The normalisation step strips iXBRL tag artefacts and OCR noise while preserving structural signals such as headings and section boundaries, which indicate where in the report a passage appears.

To focus processing on relevant material, a **keyword gate** is applied before any LLM classification. A passage must explicitly mention artificial intelligence, machine learning, large language models, generative AI, or a clearly AI-specific technique, such as neural networks, computer vision, or natural language processing, to pass the gate. Terms that commonly appear in annual reports but do not reliably signal AI, including "data analytics," "digital transformation," "automation," and "advanced analytics", are excluded unless they appear alongside an explicit AI qualifier. This conservative gate reflects a deliberate design choice: reducing false positives is prioritised over maximising recall, given that inflated mention counts would undermine the credibility of trend analysis at scale.

Passages that pass the gate are extracted with a **context window** of two paragraphs before and after the triggering passage, providing the classifier with enough surrounding text to assess intent and tone accurately. Overlapping windows produced by adjacent passages in the same document are deduplicated. Each resulting chunk carries structured metadata: company identity, publication year, market segment, CNI sector assignment, and the report section in which the passage appeared.

### 5.4 Classification Pipeline

Classification proceeds in two sequential stages applied to each chunk.

**Stage 1, Mention-type classification** determines whether the chunk carries a meaningful AI signal and, if so, what kind. The classifier assigns one or more of the following labels:

- *adoption*, AI is being actively deployed or used by the company or for its clients
- *risk*, AI is described as a source of material risk or downside
- *vendor*, a named third-party AI provider is explicitly referenced
- *harm*, AI is described as having caused a past harm (misinformation, fraud, safety incident)
- *general_ambiguous*, AI is explicitly mentioned but does not meet the threshold for any of the above; typically high-level strategy, vague plans, or non-specific AI references
- *none*, the passage is a false positive: a place name, unrelated abbreviation, or automation language with no AI specificity

Labels are not mutually exclusive except for *none*, which must appear alone. Stage 1 enforces a strict **explicit AI language requirement**: adoption requires company-specific deployment language, "we deployed," "our system uses", not strategic intent or aspiration. Risk requires AI to be attributed as the source of a downside, not merely mentioned in proximity to risk language. A passage describing a board committee that "monitors risks including AI" does not qualify as a risk disclosure unless AI is named as the risk source. This conservative approach suppresses false positives at the cost of some recall, a deliberate trade-off for a corpus of nearly ten thousand documents.

**Stage 2, Deep classification** is applied conditionally to chunks carrying a Stage 1 signal label. Three parallel classifiers run depending on which signal types were assigned:

The *adoption classifier* characterises the type of AI being reported. Each chunk is scored across three non-mutually-exclusive categories: *non-LLM* (traditional machine learning, computer vision, predictive analytics, fraud detection, recommendation systems), *llm* (large language models and generative AI tools, including named products such as ChatGPT, Gemini, and Microsoft Copilot), and *agentic* (autonomous AI systems that execute tasks without continuous human oversight, the key distinguishing characteristic is autonomous action, not AI-assistance). Each category receives a signal score from 0 to 3 reflecting the directness of evidence: 0 indicates the type is not present, 1 a weak or implicit signal, 2 a strong implicit signal, and 3 an explicit and unambiguous mention.

The *risk classifier* maps the AI-related risk to one or more of ten categories: strategic and competitive, cybersecurity, operational and technical, regulatory and compliance, reputational and ethical, third-party and supply chain, information integrity, workforce impacts, environmental impact, and national security. Each assigned category receives a signal strength score on the same 1–3 scale. In addition, each risk chunk receives a **substantiveness** rating assessing how concrete and decision-useful the disclosure is: *boilerplate* denotes generic language that could appear unchanged in any company's report ("AI poses risks to our business"); *moderate* identifies a specific risk area without concrete mitigation detail ("AI regulation may affect our compliance obligations"); and *substantive* describes specific risk mechanisms with named consequences or operational detail. The following 2021 disclosure from Prudential PLC illustrates the substantive band: *"The risk to the Group of not meeting these requirements and expectations may be increased by the development and usage of digital distribution and service channels, which can collect a broader range of personal and health-related data from individuals at increased scale, and the use of complex tools, machine learning and artificial intelligence technologies to process, analyse and interpret this data."* This passage earns its label because it identifies a specific causal pathway, a particular technology practice creating a specific data-handling exposure, rather than asserting AI risk in the abstract. The full definitions for each substantiveness band and risk category are reproduced in Appendix A.

The *vendor classifier* identifies the AI provider referenced in the chunk. Named tags cover Microsoft, Google, OpenAI, Amazon, Meta, and Anthropic; an *internal* tag captures companies describing in-house AI development; *other* covers explicitly named providers not in the main list; and *undisclosed* is used when a company references an external AI capability without naming the provider.

**Table 1: Classifier Taxonomy**

*Stage 1, Mention Types*

| Label | Definition |
|---|---|
| `adoption` | AI actively deployed, used, piloted, or implemented by the company or for clients. Requires company-specific language ("we/our"). Intent, strategy, or roadmaps alone do not qualify. |
| `risk` | AI described as a source of material risk or downside. AI must be attributed as the risk source, proximate co-occurrence with generic risk language does not qualify. |
| `harm` | AI described as having caused a past harm (misinformation, fraud, safety incident, discrimination). |
| `vendor` | Explicit named reference to a third-party AI provider or platform. |
| `general_ambiguous` | AI explicitly mentioned but not meeting any of the above thresholds, high-level plans, strategic positioning, or non-specific AI references. |
| `none` | No AI mention, false positive (place name, unrelated abbreviation), or automation language without explicit AI specificity. |

*Stage 2a, Adoption Types*

| Label | Definition |
|---|---|
| `non_llm` | Traditional AI/ML: computer vision, predictive analytics, fraud detection, recommendation engines, anomaly detection, RPA with ML components. |
| `llm` | Large language models and generative AI: GPT, ChatGPT, Gemini, Claude, Copilot, text generation, NLP chatbots, document summarisation, code generation. Copilots and AI assistants default to this category unless explicitly described as autonomous. |
| `agentic` | AI systems that autonomously execute tasks with limited human oversight. The defining characteristic is autonomous action, the system acts and decides independently rather than assisting a human who decides. Standard automation and copilots do not qualify. |

*Stage 2b, Risk Categories*

| Category | Definition |
|---|---|
| `strategic_competitive` | AI-driven competitive disadvantage, industry disruption, failure to adapt or adopt. |
| `operational_technical` | AI model failures, reliability/accuracy issues, hallucinations, system errors, decision-quality degradation. |
| `cybersecurity` | AI-enabled cyberattacks, fraud, impersonation, adversarial attacks on AI systems, breach exposure linked to AI. |
| `workforce_impacts` | AI-driven displacement, skills obsolescence, shadow AI usage by employees. |
| `regulatory_compliance` | AI Act/GDPR/privacy obligations, IP/copyright risk, legal liability from AI decisions, regulatory uncertainty. |
| `information_integrity` | AI-generated misinformation, deepfakes, content authenticity erosion, manipulation risk. |
| `reputational_ethical` | Public trust erosion, algorithmic bias, ethical concerns, fairness, social licence risk. |
| `third_party_supply_chain` | Vendor dependency/concentration risk, downstream misuse of third-party AI, over-reliance on AI providers. |
| `environmental_impact` | Energy consumption and carbon footprint from AI training and inference. |
| `national_security` | AI in critical infrastructure, geopolitical AI risks, security-of-state concerns, export controls. |

*Stage 2c, Substantiveness Bands*

| Band | Definition | Example |
|---|---|---|
| `boilerplate` | Generic language with no information content; could appear unchanged in any company's report. | *"AI poses risks to our business."* |
| `moderate` | Identifies a specific risk area or use case but lacks concrete mechanism, mitigation detail, or quantification. | *"AI regulation may affect our compliance obligations."* |
| `substantive` | Describes specific risk mechanisms with named systems, quantified impacts, concrete mitigation actions, or causal pathways. | *"We deployed GPT-4 for document review, cutting processing time by 40%."* |

Note: the substantiveness definitions for the adoption classifier and the risk classifier are operationally distinct. For risk, *substantive* requires a specific risk mechanism **and** concrete mitigation actions or commitments; for adoption, it requires named systems, quantified impact, or technical specificity. Both share the same three-band scale for comparability, but the criteria differ because the underlying disclosure tasks differ.

*Stage 2d, Vendor Tags*

| Tag | Scope |
|---|---|
| `microsoft` | Microsoft AI products: Azure AI, Copilot, Power Platform AI, Bing AI |
| `google` | Google AI products: Gemini, Vertex AI, Google Cloud AI |
| `openai` | OpenAI products: GPT series, ChatGPT, DALL·E |
| `amazon` | Amazon/AWS AI services: Bedrock, SageMaker, Rekognition |
| `meta` | Meta AI products: Llama series, Meta AI |
| `anthropic` | Anthropic products: Claude series |
| `internal` | Company describes in-house or proprietary AI development |
| `other` | Explicitly named provider not in the named-vendor list |
| `undisclosed` | External AI capability referenced without naming the provider |

All classifiers use **schema-constrained outputs**, preventing the model from returning labels outside the defined taxonomy and eliminating free-text hallucination of novel categories. All classification was performed using **Gemini Flash 3** (Google). Prompt definitions for all classifiers are reproduced in full in Appendix A.

### 5.5 Quality Assurance and Validation

**Golden set construction.** Classifier accuracy was evaluated against a human-annotated golden set of 474 chunks drawn from 30 reports (two consecutive annual reports, 2023 and 2024, for each of 15 companies) selected to provide coverage across all 13 CNI sectors. The full list of golden set companies and their sector assignments is reproduced in Appendix C. Human annotation was performed independently across all classifier dimensions (mention type, adoption type, risk category, and vendor tag) by the lead researcher, who reviewed all reports manually and cross-checked the generated chunks against the original filings to confirm recall. This process took approximately four hours and confirmed that all genuine AI mentions were captured, with a small number of false positives (subsequently classified as `none` by Stage 1). The golden set is intentionally small relative to the full corpus; it is designed as a targeted calibration instrument for design decisions rather than a production-scale evaluation.

**Model selection.** Before committing to a production classifier, six models from three families were compared on the same 20-chunk sample drawn from the golden set: Gemini Flash 3, Gemini 2 Flash, Claude Sonnet 4.5, Claude Haiku 4.5, GPT-4o mini, and GPT-5 nano. Given the small test sample, results should be read as indicative rather than statistically decisive. Gemini Flash 3 and Claude Sonnet 4.5 produced the highest exact-match rates (50% each), with Gemini Flash 3 selected as the production model on grounds of combined accuracy, cost-efficiency, and structured-output reliability. GPT-5 nano was disqualified from consideration due to consistent structured output failures on this test. The comparison cost approximately $0.60 in API credits across all six models.

**Consistency testing.** A single model was run repeatedly on the same chunks to verify output stability (the equivalent of a blank-taxonomy sanity check). At temperature 0 (the setting used in all production runs) the model produced identical outputs across all runs: 100% consistency. At temperature 0.7 (a more creative setting not used in production), average consistency across ten chunks tested ten times each was 95%. The 5% of divergent outputs were predominantly cases where the model added `general_ambiguous` alongside `adoption` on a second pass; on review these additions were assessed to be correct, indicating that even the inconsistent outputs were improvements rather than errors.

**Agreement with the human baseline.** LLM outputs were compared against the human annotation using report-level Jaccard similarity (the intersection divided by the union of assigned labels). The results by category, measured on the 30-report golden set, are shown in Table 2.

**Table 2: Report-level Jaccard similarity, LLM vs human baseline (30-report golden set)**

| Classifier dimension | Jaccard (all years) | Jaccard (2023) | Jaccard (2024) |
|---|---|---|---|
| Mention type | 0.75 | 0.74 | 0.76 |
| Adoption type | 0.47 | 0.39 | 0.55 |
| Risk taxonomy | 0.23 | 0.18 | 0.29 |
| Vendor tags | 0.40 | — | — |

The risk taxonomy score of 0.23 warrants explanation. The primary driver is not classifier error but systematic differences in labelling comprehensiveness: the LLM assigned an average of 4.77 mention-type labels per report against the human annotator's 3.77, a pattern that repeats across all dimensions. Detailed review of disagreements confirmed that the majority of LLM-only labels, where the model assigned a category the human did not, were assessed on inspection to be accurate. The LLM was identifying signals the human annotator missed rather than hallucinating signals that were not present. The human baseline is accordingly better understood as a conservative lower bound than as a gold standard, and the low Jaccard on risk taxonomy reflects human under-labelling at least as much as LLM over-labelling. The year-on-year improvement (risk Jaccard 0.18 in 2023 → 0.29 in 2024) is consistent with the classifier performing better on more recent, more explicit AI-risk language, a pattern expected given that 2023 and 2024 reports use more standardised AI terminology than earlier years.

**False positive rate.** Approximately 29% of chunks passing the keyword gate are classified as `none` by Stage 1: genuine false positives from the regex search (place names, abbreviations, and adjacent technology language). This rate was reduced substantially during development through chunking algorithm optimisation, including filtering of false-positive substrings such as "ml" in "html" or "ai" in "repair," which alone reduced the false positive rate by 31%. The current false positive rate is treated as acceptable given the design preference for high recall at the keyword stage: it is preferable to pass a false positive to the classifier for rejection than to miss a genuine AI mention at the filtering step.

**Traceability.** Every annotation in the dataset is fully traceable from raw source filing to final label. Each record stores the model name, classifier version, prompt key, confidence scores for each assigned label, and a reasoning trace where available. Chunks where the classifier returned confidence below threshold or where multiple plausible labels were in close contention are flagged in the dataset. Where low-confidence outputs would materially affect a headline statistic, they are excluded from that calculation and noted.

---

## 6. Findings

*All figures cover publication years 2020 to 2025 unless otherwise stated. 2026 data is partial (339 reports) and is noted where referenced but not used for trend conclusions. Report-level counts reflect unique company-year filings containing at least one signal of the relevant type; a single report may contribute to multiple signal categories. Risk-category and adoption-type counts are label assignments and may exceed unique report counts.*

### 6.1 The Disclosure Surge: Overall Trends

The most striking finding in the corpus is the scale and speed of growth in AI-related disclosure. In 2020, 196 of 1,007 processed reports (19.5%) contained any AI mention at all. By 2025, that figure had risen to 1,023 of 1,561 reports, or 65.5%. Risk disclosure grew even faster from a much smaller base: in 2020, just 32 reports (3.2%) named AI as a material risk. By 2025, 674 reports did (43.2% of the annual cohort), a 21-fold increase in absolute terms. Table 1 provides the full year-by-year breakdown.

**Table 1: Annual disclosure prevalence by signal type (% of all reports that year)**

| Year | Reports | Any AI | Adoption | Risk | Gen. ambiguous | Vendor | Adoption–risk gap |
|---|---|---|---|---|---|---|---|
| 2020 | 1,007 | 19.5% | 13.9% | 3.2% | 11.1% | 2.8% | 10.7 pp |
| 2021 | 1,328 | 27.2% | 21.5% | 4.4% | 14.7% | 3.1% | 17.0 pp |
| 2022 | 1,853 | 28.4% | 23.4% | 5.0% | 15.2% | 4.5% | 18.4 pp |
| 2023 | 1,905 | 36.9% | 26.8% | 10.2% | 22.4% | 6.2% | 16.6 pp |
| 2024 | 1,828 | 55.1% | 38.5% | 31.8% | 42.9% | 13.3% | 6.6 pp |
| 2025 | 1,561 | 65.5% | 46.2% | 43.2% | 54.6% | 17.4% | 3.0 pp |

[Figure 1: Line chart, annual disclosure rates for Adoption, Risk, General/Ambiguous, and Vendor, 2020–2025, with adoption-risk gap shaded]

Growth across the full period falls into three broadly distinct phases. Between 2020 and 2022, adoption disclosure grew steadily, from 13.9% to 23.4% of reports, while risk disclosure lagged, reaching only 5.0% by 2022. The adoption-to-risk gap actually widened over this phase, peaking at 18.4 percentage points in 2022, as companies began describing their AI use without yet framing it as a governance concern. The second phase opens in 2023: risk mentions more than doubled in a single year (5.0% → 10.2%), and general strategy language surged, as ChatGPT reached mass awareness and AI moved rapidly from a technology consideration to a board-level agenda item. The third phase, 2024 onwards, is defined by the risk disclosure surge: risk reports grew 200% in a single year (10.2% → 31.8%), compressing the adoption-risk gap from 16.6 to 6.6 percentage points. By 2025, the gap had narrowed to 3.0 pp; risk disclosure is now nearly at parity with adoption disclosure. In the partial 2026 data, risk reports (66.7%) marginally exceed adoption reports (64.3%) for the first time, suggesting the two signals may have crossed.

Despite this growth, the disclosure gap remains substantial in absolute terms. In 2025, 887 of 1,561 annual reports (56.8%) contained no AI risk mention. The surge in headline statistics is concentrated among larger, better-resourced, and more closely scrutinised companies.

### 6.2 Adoption Patterns

Adoption disclosure appeared in 3,012 unique company-year filings across the full corpus. Its composition has shifted significantly over time as the balance between AI types has reordered.

**Table 2: Adoption type disclosure rates (% of all reports), 2024–2025**

| Adoption type | 2024 | 2025 | YoY change |
|---|---|---|---|
| Traditional AI (non-LLM) | 36.3% | 43.0% | +6.8 pp |
| LLM / Generative AI | 24.6% | 35.7% | +11.1 pp |
| Agentic AI | 6.9% | 11.0% | +4.1 pp |

[Figure 2: Stacked area chart, non-LLM, LLM, and agentic adoption report counts by year, 2020–2025]

Non-LLM adoption (traditional machine learning, computer vision, predictive analytics, fraud detection) remains the numerically dominant category, growing from 136 reports in 2020 to 672 in 2025. But LLM and generative AI adoption has grown far faster, 14-fold versus five-fold for non-LLM, and by 2025 LLM adoption appears in 35.7% of all reports, narrowing fast on non-LLM at 43.0%. LLM adoption was essentially invisible before 2022; its acceleration is the clearest signal in the corpus of the ChatGPT diffusion effect translating into formal corporate disclosure.

Agentic AI (autonomous systems that execute tasks without continuous human oversight) appears in 172 reports by 2025, up from 27 in 2020. The category is growing in significance beyond its absolute count: average adoption labels per adoption-reporting company rose from 1.76 in 2024 to 1.94 in 2025, partly driven by companies reporting multiple adoption types simultaneously as AI deployment portfolios diversify. Agentic signals are concentrated in Finance, Transport, and Defence: sectors where autonomous process execution carries distinct governance implications that extend beyond decision-support tools.

The *general_ambiguous* category grew from 11.1% of reports in 2020 to 54.6% in 2025, overtaking adoption as the largest single-year category. It represents companies acknowledging AI without providing operational specificity, and its sustained growth reflects a reporting environment where mentioning AI has become normative but concrete characterisation has not.

### 6.3 Risk Disclosure Patterns

AI risk disclosure not only grew in volume but broadened substantially in scope. In 2020, the risk profile was narrow: strategic/competitive and operational/technical together accounted for most signals. By 2025, all ten taxonomy categories are materially represented, the risk profile has diversified, and companies reporting risk are reporting more of it, average risk labels per risk-reporting company rose from 3.28 in 2024 to 3.63 in 2025.

**Table 3: Risk category disclosure rates (label assignments as % of all 2025 reports), 2024–2025**

| Risk category | 2025 assignments | 2025 % of reports | 2024 % of reports | YoY change |
|---|---|---|---|---|
| Strategic / Competitive | 458 | 29.3% | 20.6% | +8.8 pp |
| Cybersecurity | 405 | 25.9% | 17.9% | +8.1 pp |
| Operational / Technical | 352 | 22.5% | 15.9% | +6.6 pp |
| Regulatory / Compliance | 322 | 20.6% | 15.3% | +5.4 pp |
| Reputational / Ethical | 252 | 16.1% | 10.9% | +5.2 pp |
| Third-Party / Supply Chain | 197 | 12.6% | 7.4% | +5.2 pp |
| Information Integrity | 184 | 11.8% | 7.9% | +3.9 pp |
| Workforce Impacts | 176 | 11.3% | 5.8% | +5.5 pp |
| Environmental Impact | 52 | 3.3% | 1.1% | +2.2 pp |
| National Security | 49 | 3.1% | 1.7% | +1.4 pp |

[Figure 3: Horizontal bar chart, risk category YoY percentage-point changes, 2024→2025, sorted by magnitude]

Strategic/competitive risk remains the leading category (29.3%), but the most consequential compositional shift is the rise of cybersecurity: from 11 reports in 2020 to 405 in 2025, it has moved from a peripheral concern to the second-largest category. This represents a reframing of AI in corporate risk language, from a strategic threat (competitive obsolescence) to an operational one (AI-enabled attacks, adversarial exploitation, AI-accelerated fraud), consistent with NCSC guidance on AI as a force multiplier for hostile actors.

Two categories show exceptional growth from near-zero bases. Information integrity (AI-generated misinformation, deepfakes, content authenticity erosion) grew from 1 report in 2020 to 184 in 2025. Workforce impacts grew from 4 to 176 reports, a 44-fold increase. Their emergence as mainstream disclosure categories reflects how the perceived scope of AI risk has broadened from competitive and technical concerns to societal and human-capital questions. National security, while small (49 reports in 2025), is non-trivially concentrated in CNI sectors and growing in the 2026 partial data.

Critically, the *directness* of risk attribution has not improved. The distribution of signal strength (the classifier's rating of how explicitly AI is attributed as the risk source) was essentially flat year-on-year: explicit (signal 3) signals were 31.6% in 2024 and 31.2% in 2025; strong implicit (signal 2) was 30.7% and 30.3%; weak implicit (signal 1) rose from 37.7% to 38.5%. Companies are disclosing more risk categories, but not attributing them to AI with greater directness or specificity.

### 6.4 The Vendor Landscape

Vendor references appeared in 877 unique company-year filings. The distribution is heavily skewed, and its most important feature is not which vendors are named but how many are not.

**Table 4: Vendor reference rates (label assignments as % of all 2025 reports), 2024–2025**

| Vendor | 2025 assignments | 2025 % of reports | 2024 % of reports | YoY change |
|---|---|---|---|---|
| Other (named, unlisted) | 143 | 9.2% | 5.8% | +3.4 pp |
| Microsoft | 87 | 5.6% | 5.1% | +0.5 pp |
| Internal / proprietary | 48 | 3.1% | 2.2% | +0.8 pp |
| Undisclosed | 55 | 3.5% | 1.6% | +1.9 pp |
| Google | 42 | 2.7% | 2.0% | +0.7 pp |
| OpenAI | 36 | 2.3% | 3.3% | **−1.0 pp** |
| Amazon / AWS | 31 | 2.0% | 1.8% | +0.2 pp |
| Meta | 18 | 1.2% | 0.7% | +0.5 pp |
| Anthropic | 4 | 0.3% | 0.1% | +0.1 pp |

[Figure 4: Treemap, 2025 vendor reference distribution, sized by assignment count, with named vs opaque grouping highlighted]

Two findings stand out. First, *other* and *undisclosed* together (references to external AI capabilities without identifying the provider) account for 198 of 464 total 2025 vendor assignments, or **42.7% of all vendor references**. Among only the explicitly named vendors, the three largest (Microsoft, Google, OpenAI) account for 75.7% of named-vendor assignments, revealing extreme concentration in the named portion of the landscape. Second, **OpenAI is the only named vendor declining** year-on-year (−1.0 pp), likely reflecting a combination of reduced direct API usage and the routing of OpenAI model access through Azure, which registers as Microsoft. This structural opacity means that concentration at the foundation-model layer is larger than the named-vendor data suggests.

Dependency is growing faster than disclosure: companies are acquiring AI capabilities from a small number of providers but not yet treating those dependencies as material disclosure items.

### 6.5 Disclosure Quality: The Substantiveness Gap

*Note: Substantiveness scoring applies to risk-classified chunks only. The following analysis covers 674 risk-signal reports in 2025. Adoption and vendor quality remain unassessed and represent a priority for future work.*

The central quality finding is not a plateau, it is a **structural decline in substantive share relative to disclosure volume**. As risk disclosure has grown from 3.2% of reports in 2020 to 43.2% in 2025, the fraction of risk-disclosing companies producing substantive disclosure has fallen from 15.6% to 9.5%. The quality gap, the difference between the risk mention rate and the substantive risk rate, has widened from 2.7 percentage points in 2020 to **39.1 percentage points in 2025**.

**Table 5: Quality-gap analysis, risk mention rate vs substantive risk rate**

| Year | Risk reports | Risk rate | Substantive reports | Substantive rate | Substantive share of risk reports | Quality gap |
|---|---|---|---|---|---|---|
| 2020 | 32 | 3.2% | 5 | 0.5% | 15.6% | 2.7 pp |
| 2021 | 59 | 4.4% | 9 | 0.7% | 15.3% | 3.8 pp |
| 2022 | 93 | 5.0% | 17 | 0.9% | 18.3% | 4.1 pp |
| 2023 | 194 | 10.2% | 27 | 1.4% | 13.9% | 8.8 pp |
| 2024 | 582 | 31.8% | 60 | 3.3% | 10.3% | 28.6 pp |
| 2025 | 674 | 43.2% | 64 | 4.1% | 9.5% | 39.1 pp |

[Figure 5: Dual-axis chart, risk mention rate (left axis) and substantive risk rate (left axis) as lines, quality gap (right axis, shaded area) by year, 2020–2025]

The substantiveness distribution in 2025 breaks down as follows: 9.5% substantive (64 reports), 80.7% moderate (544 reports), and 9.8% boilerplate (66 reports). The moderate category has grown most, from 78.0% in 2024. This is the dominant disclosure pattern: companies have moved beyond pure boilerplate, generic one-line AI risk statements, but have not reached the specificity needed to be decision-useful. A typical moderate disclosure identifies a risk category (regulatory compliance, cybersecurity) and attributes it to AI, but describes a monitoring or watching-brief posture rather than a concrete governance response.

Signal strength data corroborates this independently: the share of explicit risk signals (signal 3) was 31.6% in 2024 and 31.2% in 2025, while weak-implicit signals rose from 37.7% to 38.5%. Companies are acknowledging more risk categories without attributing them to AI with greater directness. The absolute number of substantive disclosures grew from 5 in 2020 to 64 in 2025, but the substantive share has fallen every year since 2022. Whether the FY2026 cohort reverses this trend is the single most important empirical question the Observatory is positioned to answer.

### 6.6 Sector and CNI Patterns

Sector-level findings reveal sharp variation in both disclosure intensity and disclosure depth across the UK's Critical National Infrastructure framework. Table 6 provides the full 2025 breakdown.

**Table 6: CNI sector disclosure summary (2025)**

| Sector | Companies | 2025 reports | AI mention % | AI risk % | No AI-risk % | Risk YoY change |
|---|---|---|---|---|---|---|
| Government | 20 | 23 | 87.0% | 60.9% | 39.1% | +6.7 pp |
| Communications | 28 | 29 | 86.2% | 55.2% | 44.8% | **+24.6 pp** |
| Finance | 461 | 657 | 74.7% | 49.5% | 50.5% | +10.4 pp |
| Water | 18 | 17 | 64.7% | 52.9% | 47.1% | +19.6 pp |
| Chemicals | 34 | 24 | 62.5% | 50.0% | 50.0% | +21.1 pp |
| Food | 52 | 70 | 62.9% | 44.3% | 55.7% | +12.6 pp |
| Transport | 61 | 67 | 73.1% | 41.8% | 58.2% | +15.3 pp |
| Defence | 20 | 22 | 77.3% | 40.9% | 59.1% | +14.0 pp |
| Health | 111 | 100 | 50.0% | 37.0% | 63.0% | +7.3 pp |
| Energy | 141 | 140 | 30.7% | 20.7% | **79.3%** | +5.3 pp |
| Data Infrastructure | 22 | 20 | 60.0% | 15.0% | **85.0%** | +5.0 pp |
| Civil Nuclear | 2 | 3 | 0.0% | 0.0% | 100.0% | — |

[Figure 6: Horizontal bar chart, CNI sector AI-risk disclosure rate and no-AI-risk rate (blind spot) side by side, 2025, sorted by risk rate]

[Figure 7: Heatmap, AI-risk YoY percentage-point change by CNI sector, 2021–2025, sectors as rows, years as columns]

Finance dominates in absolute volume (461 companies, all ten risk categories materially represented), and its disclosure depth reflects genuine AI embeddedness across trading, credit, fraud detection, and compliance. Communications had the largest single-year rise in risk disclosure in 2025 (+24.6 pp), consistent with the sector's exposure to both AI-enabled content moderation and information integrity risks. Government leads on AI mention rate (87.0%), with workforce impacts its most prominent risk category, consistent with the public policy debate around AI in public services.

Two sectors warrant particular attention. **Data Infrastructure** shows the largest blind spot: 85% of reports in 2025 contain no AI risk mention, despite this sector's role as the layer on which most AI deployment depends. **Energy** is close behind at 79.3%, even though AI is increasingly embedded in grid management, predictive maintenance, and operational technology, and OT/IT convergence in energy infrastructure is among the most documented systemic risk concerns in the national security literature. These are not sectors with low AI exposure; they are sectors with low AI disclosure.

Water, by contrast, reached 52.9% AI risk disclosure in 2025 (up from 33.3% in 2024), a sharp rise that partly reflects the small sample size (17 reports from 18 companies) amplifying individual company changes. The absolute numbers warrant caution, but the direction is notable.

Defence presents the most distinctive adoption profile. Of its 97 lifetime adoption signals, agentic signals (31) outnumber LLM signals (15), making it the only sector where this is true. This is consistent with defence sector language around autonomous systems and unmanned platforms, and it suggests AI deployment patterns that differ structurally from the commercial sector.

### 6.7 Market Segment Patterns

[Figure 8: Side-by-side bar chart, 2025 AI mention rate, adoption rate, and risk rate by market segment (FTSE 100, FTSE 350, Main Market, AIM)]

The gap between Main Market and AIM disclosure is one of the most structurally significant findings in the corpus. Table 7 presents 2025-specific rates, which are materially higher than lifetime rates and provide the clearest picture of the current state.

**Table 7: Market segment disclosure summary (2025)**

| Market segment | Lifetime reports | 2025 reports | 2025 AI mention % | 2025 Adoption % | 2025 Risk % | 2025 Vendor % |
|---|---|---|---|---|---|---|
| FTSE 100 | 1,359 | 223 | 84.8% | 79.8% | 71.3% | 28.7% |
| FTSE 350 | 3,638 | 606 | 82.8% | 63.2% | 66.5% | 26.9% |
| Main Market | 4,166 | 703 | 77.5% | 57.0% | 60.9% | 23.8% |
| AIM | 1,414 | 171 | 36.8% | 31.0% | 7.0% | 9.4% |

Among FTSE 100 companies, AI risk disclosure has reached 71.3% in 2025, approaching saturation at the mention level. The question for the largest companies is no longer whether they disclose AI risk but whether they disclose it substantively. For the FTSE 350 and broader Main Market, risk rates of 66.5% and 60.9% respectively suggest that governance reform pressure has propagated well down the Main Market.

AIM is in a different category. A 7.0% AI risk rate in 2025, against 60.9% for the Main Market, reflects a structural gap driven by lighter governance obligations (no UK Corporate Governance Code compliance, six-month reporting window, lighter institutional investor scrutiny) and a different company size and sector mix. The risk gap is wider than the AI mention gap (36.8% AI mention for AIM, suggesting companies are aware of AI; 7.0% risk, suggesting they are not translating that awareness into formal risk governance). This is not a data artefact, the Observatory has reasonable AIM document coverage via Companies House. The gap is in what is being said, not what can be seen.

For sectors where AIM-listed companies operate in CNI-adjacent industries, smaller energy producers, specialist communications firms, health technology businesses, the absence of AI risk disclosure should not be interpreted as an absence of AI exposure.
---

## 7. Limitations

Every large-scale empirical study of corporate disclosure carries inherited constraints that qualify the strength of its conclusions. We organise ours into three categories: coverage gaps that affect the completeness of the corpus, classification validity issues that bear on the accuracy of the signals we extract, and causal and inferential limits that govern how findings should be interpreted.

### 7.1 Coverage Gaps

The corpus covers 9,821 reports across 1,362 companies and six publication years, but it does not cover the full universe of UK-listed companies. The irreducible gap is approximately 6.7% of target company-year slots, attributable primarily to shell companies, special-purpose acquisition vehicles (SPACs), and micro-caps that have never filed electronically in a machine-parseable format. This segment is structurally unlikely to contain material AI disclosure, so its exclusion is not expected to bias aggregate trend findings, but the gap cannot be reduced without manual document retrieval.

AIM-listed companies present a more specific coverage concern. Because financialreports.eu covers only 28% of AIM slots (compared with near-complete Main Market coverage), the majority of AIM reports in the corpus are sourced from Companies House as PDF documents. PDF quality is variable: converted PDFs sometimes lose table structure, misparse section boundaries, or contain OCR artefacts that introduce noise at the chunking stage. AIM findings should therefore be interpreted with a wider confidence interval than Main Market results.

Two years at the edges of the time window are structurally partial. Publication year 2020 reflects a mid-year build-up of company filings, not a full annual cycle; 2026 contains only reports filed in the first months of the year. Both years are retained as indicative series anchors but are not directly comparable to peak-coverage years (2022–2025). Year-over-year comparisons that include 2020 or 2026 should be treated as directional rather than precise.

Finally, 178 companies have zero coverage from financialreports.eu despite appearing in that source's filing index, in some cases because the source misclassifies non-annual-report documents (for example, Jet2 PLC's prospectus and share-buyback notices were returned as annual-report candidates). The Companies House gap-fill reduces but does not fully resolve this exposure, and any systematic pattern in which companies are missed, if, for example, certain industry registrants are more likely to file in non-standard formats, could introduce a modest sector-level bias.

### 7.2 Classification Validity

The primary validation instrument is a human-annotated golden set of 474 chunks reviewed against the full two-stage classifier pipeline. This is a larger evaluation corpus than many comparable studies in financial-NLP research, but it remains a constraint on our ability to estimate sub-category precision independently. In particular, rare signal types, agentic adoption, national security risk, and sector-specific vendor mentions, appear infrequently enough in the golden set that sub-category confidence intervals remain wide.

The pipeline's conservative prompting strategy is a deliberate design choice that creates a known recall trade-off. By requiring explicit AI language at Stage 1 and refusing to infer adoption from intent or strategy statements alone, the classifier excludes some passages that a human annotator might reasonably flag. Companies that describe AI capabilities obliquely, through references to "intelligent systems," "predictive tools," or proprietary platform features without explicit AI attribution, will not be captured. The pipeline therefore produces a conservative lower bound on AI disclosure activity rather than an upper bound.

Substantiveness classification is inherently judgement-dependent in a way that the mention-type and risk-category classifiers are not. The three-band scale (boilerplate / moderate / substantive) requires the classifier to distinguish passages that are highly similar in surface form but differ in specificity, concreteness, and operational grounding. Inter-rater agreement at the boundary between *moderate* and *substantive* is lower than agreement on coarser distinctions such as adoption versus risk. The substantiveness scores reported here, and particularly the finding that the substantive share of risk-classified chunks remains below 10% in 2025, should be read as an approximate signal of disclosure quality rather than a precise measurement. A further methodological limitation is that substantiveness classification has only been applied to risk-classified chunks (5,178 chunks in total); adoption and vendor chunks do not yet carry substantiveness scores. This means the finding of a quality plateau cannot currently be extended to the full disclosure picture, and the characterisation of adoption disclosure quality in particular remains an open empirical question for future work.

Vendor tagging carries its own opacity. The "other" category captured 143 signals in 2025 alone, and "undisclosed" represents an additional substantial group of references where vendor identity cannot be inferred from the disclosure text. These categories mask real concentration patterns in AI supply chains. Until reporting practice improves, or until external supplementary data sources are integrated, vendor concentration analysis at the firm level will remain limited by the deliberate or inadvertent opacity of the source documents themselves.

### 7.3 Causal and Inferential Limits

The most fundamental inferential limit is that disclosure does not equal exposure. A company that discusses AI risk at length in its annual report is not necessarily more exposed to AI-related harm than a company that says nothing. It may simply have more sophisticated governance processes, be operating in a more heavily scrutinised sector, or be responding to investor or regulatory pressure. Conversely, a company with thin or absent AI disclosure may face significant operational AI risk that it has not yet recognised, has chosen not to disclose, or has disclosed under language our keyword gate does not capture. The corpus measures what companies say, not what they experience.

The 2023–2024 surge in AI risk mentions, a 200% increase in a single year, is the most striking finding in the dataset and the most difficult to attribute. Three mechanisms are simultaneously operative and individually plausible: the mass-market adoption of generative AI tools (ChatGPT launched November 2022; enterprise GenAI adoption accelerated through 2023), early anticipatory compliance with the 2024 UK Corporate Governance Code as companies revised their risk frameworks ahead of Provision 29, and growing investor and analyst pressure to address AI risk specifically. We cannot cleanly decompose these effects with observational disclosure data alone. The absence of a clean control group, a set of companies demonstrably unaffected by the ChatGPT shock or by Provision 29 anticipation, means that the causal weight on any single driver must remain a hypothesis rather than an empirical result.

The UK-only scope of the corpus is a strength for depth but a limitation for generalisability. The regulatory architecture underpinning UK annual reports, the FCA DTR 4.1 requirement, the FRC's strategic report guidance, and the comply-or-explain structure of the UK Corporate Governance Code, differs from the SEC mandatory-disclosure regime for US 10-Ks, ESEF for European-listed companies, and the very different reporting cultures of other major jurisdictions. Findings about disclosure patterns, sector composition, and quality should not be assumed to transfer directly to other markets. The pipeline methodology, however, is jurisdiction-neutral and can be applied to any structured machine-readable filing corpus.

Finally, annual reports are a lagging instrument. They reflect decisions, risks, and governance arrangements as they existed at the close of a fiscal year, typically published three to four months later and sometimes consulted many months after that. Material changes in AI exposure, a significant vendor failure, a new agentic deployment, a regulatory enforcement action, will not appear in disclosed language until the following annual cycle. The Observatory captures where companies were, not where they are.


---

## 8. Discussion

### 8.1 Policy Implications

The Observatory's findings point to three actionable policy concerns.

First, regulatory frameworks that treat disclosure *presence* as the primary compliance signal will generate rising volumes without rising information content. The risk mention rate has grown 21-fold since 2020, yet the substantive share of risk disclosures has fallen from 15.6% in 2020 to 9.5% in 2025. The boilerplate problem documented in the financial NLP literature is replicating in AI disclosure before that domain has had time to mature. Provision 29's requirement for boards to assess the *effectiveness* of material controls is the right intervention design, shifting the obligation from mentioning risk to evidencing governance, but the FY2026 data will be the first real test.

Second, the CNI sector decomposition reveals disclosure gaps that aggregate statistics obscure. Data Infrastructure (85% of reports without an AI risk mention in 2025) and Energy (79.3%) are the largest blind spots, despite being high-AI-exposure sectors. The absence of disclosure more plausibly reflects the absence of regulatory and investor pressure than an absence of risk. These are where targeted AISI engagement or sectoral regulator guidance would be most additive. By contrast, Communications showed the largest single-year increase in AI risk disclosure in 2025 (+24.6 pp), suggesting that sector-specific scrutiny accelerates disclosure practice when it arrives.

Third, vendor concentration is systematically underreported. Opaque references (the *other* and *undisclosed* categories) accounted for 42.7% of all vendor assignments in 2025. Among named providers, the top three (Microsoft, Google, OpenAI) account for 75.7% of named-vendor references. OpenAI is the only named vendor declining year-on-year (−1.0 pp), likely due to routing through Azure rather than a genuine reduction in dependency. Foundation-model concentration at the infrastructure layer is larger than the named-vendor data suggests.

### 8.2 Applications and Use Cases

The Observatory was designed to be a reusable monitoring instrument rather than a one-time study. Several concrete applications follow directly from the current pipeline and corpus.

**Regulatory benchmarking.** The Provision 29 obligation, requiring boards to declare the effectiveness of all material internal controls, enters its first full annual reporting cycle for fiscal years beginning on or after 1 January 2026. The Observatory provides a pre-intervention baseline across the full universe of UK listed companies at a granularity (company, sector, market segment, and disclosure type) that enables clean before-and-after comparison. Rerunning the pipeline on FY2026 reports (which will become available across 2027) will produce a direct measure of whether the Code amendment delivers substantiveness improvement or merely increases disclosure volume. This is precisely the type of regulatory evaluation that is difficult to conduct without a structured, reproducible monitoring infrastructure.

**Supervisory prioritisation.** Regulators and sector bodies can use the corpus to identify companies with high plausible AI exposure and low disclosure, a screening tool for follow-up supervisory inquiry. A company in the Water or Communications CNI sector with zero AI signal across multiple years is not necessarily unexposed; it may be a candidate for direct supervisory engagement. The Observatory does not replace supervisory judgment, but it provides a structured starting point that narrows the search space.

**Research replication and jurisdictional comparison.** The pipeline is fully documented, the classifier taxonomy is explicit, and the prompts are versioned. Any jurisdiction with access to machine-readable annual report filings, ESEF in the EU, 10-K EDGAR filings in the US, or equivalent national repositories, can run the same pipeline against its own corpus and produce directly comparable signals. The UK dataset provides a baseline against which cross-jurisdictional comparisons can be anchored.

**Annual report as a hard disclosure baseline.** Earnings calls, investor presentations, and corporate surveys are softer forms of AI disclosure: they are not audited, they are not legally mandated to discuss principal risks, and they are produced for audiences with very different information needs. Annual reports occupy a distinct position in the information ecosystem as the closest thing to a legally accountable, audited, structured statement of what a company believes its material risks and capabilities to be. The Observatory's signals can serve as ground truth against which softer disclosure channels are calibrated, confirming, for instance, whether what companies say in earnings calls about AI adoption appears in the harder disclosure record.

### 8.3 Future Work

Several extensions to the current pipeline are either technically straightforward or clearly motivated by the findings.

**Provision 29 follow-through** is the highest-priority empirical next step. The 2024 surge in risk mentions may represent genuine governance improvement that will be consolidated in the first post-Provision-29 reporting cycle, or it may represent anticipatory disclosure volume that was not matched by underlying governance depth. Only the 2026 and 2027 report cohorts will resolve this. Annual re-running of the pipeline with a fixed methodology will produce the longitudinal series needed to answer the question.

**Substantiveness extension to adoption and vendor chunks** is a methodological gap the current study acknowledges but does not close. The adoption-type and vendor classifiers do not yet produce substantiveness scores; only risk-classified chunks carry the three-band quality rating. Extending the substantiveness schema to adoption chunks in particular would allow a more complete assessment of whether companies that claim AI adoption are providing material operational detail or producing impressionistic statements. The classifier prompt architecture already defines the relevant output fields; the extension requires running the Stage 2 adoption classifier against the substantiveness scoring rule set and validating against the golden set.

**Causal modelling** would connect disclosure signals to external data sources, cyber incident databases, AI-related regulatory enforcement actions, sector-level employment data, or company-level stock volatility around AI announcements, to test whether annual-report language predicts, follows, or is orthogonal to real-world AI-related outcomes. This requires merging the Observatory's structured disclosure signals with third-party event data, which is methodologically tractable but outside the current scope.

**Boilerplate tracking over time within companies** is a direct extension of Brown and Tucker's (2011) MD&A modification measure applied to AI-specific language. By computing year-over-year textual similarity of AI-risk passages within the same company, it becomes possible to operationalise "disclosure staleness" at the firm level, identifying companies that have copied forward risk language for multiple years without material revision. This is a more precise and more defensible substantiveness measure than the current three-band classifier, and it would complement rather than replace the existing architecture.

**Agentic and autonomous system monitoring** warrants dedicated attention. The agentic adoption category recorded approximately six-fold growth between 2020 and 2025 (27 to 172 signals), a pace considerably faster than non-LLM or general LLM adoption. Annual reports are beginning to describe autonomous AI agents in operational contexts, systems that act, not merely systems that analyse. The risk taxonomy, the disclosure patterns, and the governance language around agentic systems are likely to differ substantially from what has developed around predictive analytics and language models. Establishing a monitoring baseline for this category before it matures into a common disclosure topic will be considerably easier than retrospectively reconstructing it.


---

## 9. Conclusion

The AI Risk Observatory documents a disclosure landscape defined by volume growth without equivalent quality growth. AI mention rates have risen from 19.5% of UK listed company reports in 2020 to 65.5% in 2025. Risk disclosure grew twenty-one-fold over the same period. Yet the substantive share of risk disclosures has fallen every year since 2022, sitting at 9.5% in 2025. More companies are mentioning AI risk; fewer, proportionally, are saying anything decision-useful about it.

The sector and market segment findings reinforce this picture. Finance and Health are active disclosers; Data Infrastructure and Energy are not, despite high plausible AI exposure. FTSE 100 companies are near-saturated on disclosure volume. AIM companies remain structurally underrepresented, reflecting lighter governance obligations rather than lower AI risk.

The Observatory's most important near-term function is as a pre-intervention baseline for Provision 29 of the 2024 UK Corporate Governance Code, which takes effect for FY2026. If the Code is working as intended, the next annual cohort should show improvement in disclosure specificity and governance depth, not just mention rates. Annual reports are a lagging instrument, but they are the most systematic and legally accountable evidence base available for monitoring corporate AI governance at scale.


---
---

## Appendix A: Classifier Definitions

This appendix reproduces the canonical label definitions used in production classification. These definitions are the operative specifications against which the golden set was annotated and against which classifier output was evaluated. Minor formatting differences from the production prompt YAML have been made for readability; no substantive content has been altered.

---

### A.1 Keyword Gate (Stage 1 Pre-filter)

Before any classification is attempted, a chunk must pass the following hard gate:

> The excerpt must explicitly mention **AI, ML, LLM, GenAI**, or a clearly AI-specific technique such as machine learning, neural networks, or computer vision. Terms such as "data analytics," "automation," "digital tools," "advanced analytics," or "predictive tools" do **not** qualify as AI under this definition unless AI is explicitly named or a specific AI technique is unambiguously described.

Chunks that do not pass this gate are assigned `none` and do not enter Stage 2. This rule is the single most consequential boundary decision in the pipeline: it excludes a large class of adjacent technology language that prior studies have sometimes treated as proxies for AI activity. The rationale is that a disclosure monitor that conflates "data analytics" with AI will systematically overstate adoption prevalence and may attribute to AI both language and risks that belong to a different (if adjacent) technology domain.

---

### A.2 Stage 1: Mention Type Labels

**`adoption`**
Describes real, current deployment, implementation, rollout, pilot, or use of AI by the company or for its clients. Requires company-specific language ("we," "our," "our clients"). Generic intent, strategy, roadmaps, or future plans do not qualify. Phrases such as "exploring," "piloting," or "investigating" qualify only when they refer to a specific initiative currently underway (e.g., "our current trial resulted in..."). Delivering AI systems to clients counts as adoption; pure consulting or advisory work without deployment does not.

**`risk`**
AI is described as a risk or material concern to the company: legal, cybersecurity, operational, reputational, or regulatory risk directly caused by or attributed to AI. The classifier must verify that AI is the named source of the risk. A passage that mentions a risk category in one sentence and AI in a separate, unconnected sentence does not qualify. Downstream or indirect risks from AI technologies do count (e.g., "rapid AI adoption creates skills gaps" → `workforce_impacts`).

**`harm`**
AI is described as having caused a past, specific harm: misinformation spread, fraud enabled by AI, safety incident, or discriminatory outcome. Harm is distinguished from risk (prospective) by its past tense or completed framing.

**`vendor`**
Explicit mention of a named third-party AI vendor or platform that provides AI technology to the company, or a named product that is clearly AI (e.g., GPT, Google Gemini, Microsoft Copilot). The emphasis is on *what AI models or systems the company uses*, not merely that a technology partnership exists.

**`general_ambiguous`**
AI is explicitly mentioned (satisfying the keyword gate) but the passage does not meet the threshold for any of the above labels. Typical examples: high-level strategy statements, board-level AI acknowledgements, industry trend commentary, or AI opportunity language without operational specificity. `general_ambiguous` must not co-occur with other non-`none` labels; if any other label applies, that label is preferred.

**`none`**
No AI mention, a false positive (place name containing "ai," unrelated abbreviation, foreign language fragment), or automation/digital language that does not pass the keyword gate.

---

### A.3 Stage 2a: Adoption Type Definitions

All three adoption types are non-mutually exclusive and each receives a signal score of 0–3.

**`non_llm`, Traditional AI/ML**
Everything that is AI but not LLM-based or agentic: computer vision, predictive analytics, fraud detection models, recommendation engines, anomaly detection, robotic process automation with ML components, natural language processing systems that predate the LLM era. This is the baseline AI adoption category.

**`llm`, Large Language Models and Generative AI**
GPT, ChatGPT, Gemini, Claude, Copilot, text generation systems, NLP chatbots, document summarisation, code generation, and any system in the GenAI category. AI copilots and AI assistants default to `llm` unless explicitly described as operating autonomously. When a system is both LLM-based and agentic, both labels apply.

**`agentic`, Autonomous AI Systems**
AI systems or agents that autonomously execute tasks and take actions with limited human oversight. The key characteristic is **autonomous execution**: the AI acts, decides, and operates on its own rather than assisting a human who decides. The word "agentic" need not appear, but there must be clear evidence of autonomous AI action, not just AI-assisted decision support. Copilots, AI assistants, and standard automation without autonomous decision-making do not qualify.

*Signal guidance (0–3):* 0 = type not present; 1 = weak or implicit signal, type is plausible but not stated; 2 = strong implicit signal, type is clearly implied but not explicit; 3 = explicit and unambiguous mention.

---

### A.4 Stage 2b: Risk Category Definitions

All ten categories are non-mutually exclusive. Each assigned category receives a signal score of 1–3. Assignment requires that AI is attributed as the source of the risk; generic risk language not linked to AI yields `none`.

| Category | Operational Definition |
|---|---|
| `strategic_competitive` | AI changes market structure, customer behaviour, pricing power, or competitive position. Failure to adopt AI, competitive obsolescence, or industry disruption attributable to AI. |
| `operational_technical` | AI quality, reliability, and model-risk issues. Model failures, accuracy problems, hallucinations, system instability, unsafe outputs, or decision-quality degradation caused by AI systems. |
| `cybersecurity` | AI-linked attack or defence exposure. AI-enabled phishing/fraud/impersonation, adversarial attacks on AI systems, AI-accelerated data breaches, AI-enhanced social engineering. |
| `workforce_impacts` | AI-driven workforce transition risk. Skills gaps caused by AI adoption, displacement pressure, retraining obligations, unsafe or unauthorised employee AI use (shadow AI). |
| `regulatory_compliance` | AI-specific legal, regulatory, privacy, or IP exposure. EU AI Act compliance costs, GDPR/privacy implications of AI, IP and copyright risks from AI-generated content, legal liability from AI decisions, regulatory uncertainty. |
| `information_integrity` | AI-enabled misinformation, deepfakes, content authenticity erosion, manipulation risk. The AI must be attributed as the mechanism producing or enabling false or manipulated content. |
| `reputational_ethical` | AI-linked trust erosion, algorithmic bias, fairness concerns, ethical objections, or social licence risk. Covers both internal (employee) and external (public/investor) trust. |
| `third_party_supply_chain` | Over-reliance on AI vendors, concentration risk from a small number of AI providers, downstream misuse of AI in the supply chain, or risks arising from third-party AI embedded in the company's products or services. |
| `environmental_impact` | AI energy consumption, carbon footprint from training or inference, hardware resource intensity, and sustainability concerns attributable to AI use. |
| `national_security` | AI-linked geopolitical or security instability, exposure of critical infrastructure to AI-enabled threats, export-control and security-of-state concerns, and AI in defence/intelligence contexts where national security implications are explicit. |

*Signal guidance (1–3):* 1 = weak implicit attribution, plausible but lightly supported; 2 = strong implicit attribution, clear with some interpretation; 3 = explicit attribution, AI is directly stated as the cause of that category risk.

---

### A.5 Stage 2c: Substantiveness Definitions

The substantiveness scale applies separately to risk chunks and (where implemented) adoption chunks. The definitions are functionally similar but operationally distinct because the disclosure tasks differ.

**Risk substantiveness:**

- `boilerplate`: Generic risk or governance language with little concrete mechanism. Could appear unchanged in any company's report (e.g., *"AI poses risks to our business"*; *"We monitor AI-related developments"*).
- `moderate`: Identifies a specific AI-risk area but provides limited mechanism or mitigation detail. The reader learns *what* risk area is relevant but not *how* the risk operates or *what* is being done about it (e.g., *"AI regulation may affect our compliance obligations"*; *"Cyberattacks using AI are increasing"*).
- `substantive`: Describes a specific AI-risk mechanism **and** provides concrete mitigation actions, operational commitments, named systems, or causal pathways. The reader learns something specific about this company's situation that would not apply equally to every other company in the sector (e.g., *"We allocated £5M to reclassify three high-risk AI systems under the EU AI Act by Q3 2025"*; the Prudential PLC example in §5.4).

**Adoption substantiveness:**

- `boilerplate`: Pure jargon with no information content. Could appear in any company's report unchanged (e.g., *"We leverage AI to drive innovation and improve operations"*).
- `moderate`: Identifies a specific use case or domain but lacks concrete detail (e.g., *"We use AI in our underwriting process"*; *"We deployed AI in risk management"*).
- `substantive`: Names specific systems, quantifies impact, or explains what/how/why with technical or operational detail (e.g., *"We deployed GPT-4 for document review, reducing processing time by 40%"*).

The key difference: for risk, `substantive` requires both a specific mechanism **and** mitigation evidence; for adoption, it requires specificity in the description of *what* was deployed and *to what effect*.

---

### A.6 Stage 2d: Vendor Tag Definitions

The vendor classifier identifies the AI provider referenced in the chunk. Named tags in priority order:

| Tag | Coverage |
|---|---|
| `microsoft` | Microsoft Azure AI, Microsoft Copilot (all variants), Power Platform AI features, Bing/Search AI, and any Microsoft-branded AI product |
| `google` | Google Gemini, Vertex AI, Google Cloud AI/ML APIs, DeepMind products deployed commercially, Google Workspace AI features |
| `openai` | GPT series (GPT-3, GPT-4, GPT-4o), ChatGPT, DALL·E, Whisper, OpenAI API references |
| `amazon` | AWS Bedrock, Amazon SageMaker, Amazon Rekognition, Amazon Comprehend, Amazon Lex, and other AWS AI/ML services |
| `meta` | Llama series (Llama 2, Llama 3, Code Llama), Meta AI assistant, PyTorch where deployed as a production AI system |
| `anthropic` | Claude series (Claude 1, 2, 3, Haiku, Sonnet, Opus) |
| `internal` | Company explicitly describes building, training, or maintaining its own AI systems in-house or through a proprietary development programme |
| `other` | A named AI provider or product that does not match any of the above tags (e.g., Palantir, Scale AI, Snowflake Cortex, IBM watsonx) |
| `undisclosed` | An external AI capability or provider is referenced but not named (e.g., *"our AI vendor"*, *"third-party AI tools"*, *"external AI platform"*) |

Multiple vendor tags may be assigned to a single chunk if more than one provider is named. The `internal` and `undisclosed` tags may co-occur with named-vendor tags only if the passage clearly refers to both.


---

## Appendix C: Golden Set Composition

The golden set comprises 474 AI-mention chunks drawn from 30 annual reports, two consecutive years (2023 and 2024) for each of 15 companies. Companies were selected to provide coverage across all 13 UK CNI sectors. All 30 reports were manually reviewed by the lead researcher both to produce the human annotation baseline and to verify that the chunking pipeline had achieved full recall of genuine AI mentions.

| Company | CNI Sector | Reports included |
|---|---|---|
| Croda International plc | Chemicals | 2023, 2024 |
| Rolls-Royce Holdings plc | Civil Nuclear / Space | 2023, 2024 |
| BT Group plc | Communications | 2023, 2024 |
| BAE Systems plc | Defence | 2023, 2024 |
| Serco Group plc | Government Services | 2023, 2024 |
| Shell plc | Energy (Extraction) | 2023, 2024 |
| Lloyds Banking Group plc | Finance (Banking) | 2023, 2024 |
| Tesco plc | Food Retail | 2023, 2024 |
| AstraZeneca plc | Health (Pharma) | 2023, 2024 |
| National Grid plc | Energy (Transmission) | 2023, 2024 |
| Severn Trent plc | Water | 2023, 2024 |
| Aviva plc | Finance (Insurance) | 2023, 2024 |
| Schroders plc | Finance (Asset Management) | 2023, 2024 |
| FirstGroup plc | Transport | 2023, 2024 |
| Clarkson plc | Shipping | 2023, 2024 |

The 15 companies are all large-cap or mid-cap listed firms with well-resourced investor relations and governance functions. This was deliberate: the golden set was designed to contain substantive, nuanced AI language rather than thin or absent disclosure, so that the classifier could be evaluated on the full range of signal types. The trade-off is that golden set agreement rates, measured against these more complex, multi-signal reports, may be slightly pessimistic relative to what the classifier achieves on simpler filings with single-label AI mentions. The golden set does not include AIM companies or reports from 2020–2022; agreement performance on these cohorts has not been independently validated.

