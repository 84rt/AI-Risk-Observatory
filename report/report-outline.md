# Report Outline: Tracking AI Risk and Adoption in UK Public-Company Annual Reports

*AI Risk Observatory, 2026*

---

## Executive Summary
Three headline findings: AI disclosure has grown from 19.5% to 65.5% of UK listed company reports (2020–2025); disclosure quality has not kept pace, with substantive risk disclosure below 10% in 2025; and sector gaps are large, with Data Infrastructure and Energy showing the highest blind spots relative to AI exposure. Three policy implications follow: volume-based regulatory frameworks are insufficient, sector-specific engagement is more effective than generic requirements, and vendor concentration is systematically underreported.

---

## 1. Introduction
**Motivation**: Annual reports are legally mandated, board-accountable artefacts, making them the most reliable systematic evidence base for tracking corporate AI governance. **Research questions**: (1) How has AI disclosure prevalence changed 2020–2025? (2) Is risk disclosure substantive or boilerplate? (3) Which sectors show the largest disclosure gaps relative to AI exposure? **Scope**: First large-scale analysis of AI disclosure across the full UK listed universe (Main Market and AIM), with an explicit CNI-sector decomposition and substantiveness dimension.

---

## 2. Background and Regulatory Context
**UK annual reporting framework**: The Companies Act, FCA DTR 4.1, and iXBRL mandate create a structured, machine-parseable filing corpus. AIM companies face lighter requirements, explaining much of the disclosure gap observed. **Critical National Infrastructure**: AI is embedded in all 13 CNI sectors; third-party AI concentration and cross-sector interdependencies make disclosure-based monitoring particularly valuable. **Governance reforms**: The 2024 UK Corporate Governance Code's Provision 29 shifts board obligations from mentioning risk to evidencing the effectiveness of material controls, taking full effect for FY2026.

---

## 3. Related Work
Four research streams inform the Observatory: financial disclosure NLP (Loughran & McDonald; Lang & Stice-Lawrence; Kim; Park et al.); the substantiveness and boilerplate literature (Dyer, Lang & Stice-Lawrence; Brown & Tucker); empirical AI disclosure research (Uberti-Bona Marin et al.; Bonsón et al.; Chiu); and observatory-style NLP monitoring systems applied to ESG and climate domains (ClimateBERT; Ferjančič et al.; Bank of England; FCA).

---

## 4. Methodology
**Corpus**: 9,821 annual reports from 1,362 UK-listed companies, 2020–2026, sourced from financialreports.eu (iXBRL) and Companies House (PDF gap-fill). **Preprocessing**: Keyword gate requiring explicit AI language; chunking with two-paragraph context windows. **Classification pipeline**: Two-stage LLM pipeline using Gemini Flash 3. Stage 1 assigns mention type (adoption, risk, harm, vendor, general_ambiguous, none). Stage 2 assigns adoption type (non-LLM, LLM, agentic), risk category (10 categories), substantiveness (boilerplate/moderate/substantive), and vendor tags. **Validation**: 474-chunk human-annotated golden set; Jaccard similarity 0.75 on mention type, 0.23 on risk taxonomy (LLM over-labels rather than mislabels).

---

## 5. Findings

**5.1 Overall trends**: AI mention rate grew from 19.5% (2020) to 65.5% (2025). Risk disclosure grew 21-fold, from 3.2% to 43.2%. The sharpest inflection was 2023–2024 (+200% in risk mentions). By 2025 the adoption–risk gap had narrowed to 3 percentage points.

**5.2 Adoption patterns**: Non-LLM adoption remains dominant but LLM adoption grew 14-fold (essentially zero pre-2022 to 35.7% in 2025). Agentic signals grew to 172 reports. General/ambiguous AI mentions surpassed adoption as the largest single-year category (54.6% in 2025).

**5.3 Risk disclosure patterns**: All ten risk categories are now materially represented. Cybersecurity moved from peripheral to second-largest category. Information integrity and workforce impacts grew from near-zero to mainstream. Risk attribution directness has not improved year-on-year.

**5.4 Vendor landscape**: 42.7% of 2025 vendor assignments are "other" or "undisclosed." Among named vendors, Microsoft, Google, and OpenAI account for 75.7% of references. OpenAI is the only named vendor declining year-on-year (−1.0 pp), likely due to routing through Azure.

**5.5 Substantiveness gap**: The substantive share of risk-disclosing companies has fallen from 15.6% (2020) to 9.5% (2025), even as the risk mention rate grew 21-fold. The quality gap widened from 2.7 pp to 39.1 pp. 80.7% of risk disclosure is "moderate": acknowledges a risk category but provides no concrete governance evidence.

**5.6 Sector patterns**: Government (87%), Communications (86%), and Finance (75%) lead on AI mention rates. Data Infrastructure (85% no AI-risk mention) and Energy (79%) are the largest blind spots. Defence is the only sector where agentic signals outnumber LLM signals.

**5.7 Market segment patterns**: FTSE 100 is near-saturated at 71.3% AI risk disclosure. AIM sits at 7.0% risk rate, reflecting lighter governance obligations rather than lower AI exposure.

---

## 6. Limitations
Coverage gaps: ~6.7% of target slots missing; AIM PDF quality variable; 2020 and 2026 partial years. Classification validity: conservative prompting creates recall trade-off; substantiveness scoring is judgement-dependent; risk taxonomy Jaccard of 0.23 reflects human under-labelling as much as LLM over-labelling. Causal limits: disclosure does not equal exposure; the 2023–2024 surge cannot be cleanly attributed to ChatGPT diffusion vs. Provision 29 anticipation; annual reports are a lagging instrument.

---

## 7. Discussion
**Policy implications**: Volume-based frameworks incentivise mention without specificity; Provision 29 is the right design but FY2026 will be the first test. Data Infrastructure and Energy warrant targeted AISI engagement. Vendor concentration is a systemic monitoring gap. **Applications**: Regulatory benchmarking for Provision 29; supervisory screening for high-exposure/low-disclosure companies; jurisdictional replication. **Future work**: Provision 29 follow-through (FY2026 cohort); substantiveness extension to adoption and vendor chunks; causal modelling; firm-level boilerplate tracking; dedicated agentic AI monitoring.

---

## 8. Conclusion
AI disclosure volume is up; quality is not. The substantive share of risk disclosures has declined every year since 2022 while mention rates have surged. Sector and market segment gaps reflect regulatory obligations more than AI exposure. The Observatory provides the pre-intervention baseline for assessing whether Provision 29 delivers substantive improvement in FY2026.

---

## Appendices
**Appendix A**: Full classifier definitions (keyword gate; Stage 1 mention types; Stage 2 adoption, risk, substantiveness, and vendor labels with operational definitions and signal guidance). **Appendix C**: Golden set composition (15 companies, 30 reports, two annual cycles each, covering all 13 CNI sectors).
