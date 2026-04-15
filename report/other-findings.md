Yes — this is a very workable literature review topic, and the source base is strong.

For **riskobservatory.ai**, I’d frame the review around a core claim: your project sits at the intersection of **financial-disclosure NLP**, **AI-governance disclosure**, **boilerplate-vs-substantive reporting**, and **regulatory risk monitoring**. That framing is well supported by existing work. The Observatory itself already presents the project as tracking AI-related risk, adoption, vendor dependence, and disclosure gaps across UK public-company annual reports, especially in CNI sectors, which gives you a clear applied policy use case. ([Risk Observatory][1])

## 1) Core methodology precedents

These are the best anchors for justifying your pipeline design.

**Loughran & McDonald (2011)** is the classic starting point for any financial-text pipeline. Their core contribution is showing that generic sentiment dictionaries misfire in financial documents and that finance-specific lexicons do better on 10-Ks. This is the citation that justifies domain-specific vocabulary, careful prompt design, and why you should not treat annual reports as ordinary prose. ([Wiley Online Library][2])

**Loughran & McDonald, “Textual Analysis in Accounting and Finance: A Survey”** is the best review article to cite near the front of the report. It gives you the map of the field: tone, readability, topic extraction, risk language, and market consequences. Use it to justify why annual reports are a legitimate and established NLP corpus rather than an ad hoc data source. ([SSRN][3])

**Brown & Tucker (2011)** is central for your substantiveness logic. Their year-over-year MD&A modification measure shows that textual change itself can proxy for disclosure informativeness. That is directly relevant to any classifier that tries to separate meaningful AI-risk discussion from recycled wording. ([bear.warrington.ufl.edu][4])

**Dyer, Lang & Stice-Lawrence (2017)** is the strongest paper for the “annual reports have become longer, more redundant, more boilerplate” story. It gives you an academic basis for expecting disclosure inflation and for building methods that discount repetition, stickiness, and low-specificity language. ([ScienceDirect][5])

**Lang & Stice-Lawrence (2015), “Textual Analysis and International Financial Reporting”** matters because your corpus is UK annual reports, not only US 10-Ks. This paper is useful for arguing that large-sample textual analysis of annual reports outside the US is both feasible and informative. ([ScienceDirect][6])

For more recent AI-era methods, two sources stand out. **Kim (2024), “Financial Statement Analysis with Large Language Models”** gives you evidence that LLMs can extract decision-relevant signals from financial statements at scale. **Park (2024), BIS/IFC, “Quantifying Material Risks from Textual Disclosures in Financial Statements using LLM Agents”** is especially close to your use case because it applies LLMs to disclosure materiality assessment rather than just generic summarization. ([Bayes Business School][7])

A good way to write this section is:

* lexicon and classical text-analysis foundations: Loughran & McDonald
* disclosure-change / similarity foundations: Brown & Tucker; Dyer et al.
* international annual-report corpus precedent: Lang & Stice-Lawrence
* LLM-era extraction and materiality: Kim; Park

## 2) AI disclosure and governance

This is the clearest “research gap” section.

The most directly relevant recent paper I found is **“Are Companies Taking AI Risks Seriously? A Systematic Analysis of Companies’ AI Risk Disclosures in SEC 10-K forms”**. It analyzes more than 30,000 filings from over 7,000 firms and finds a steep rise in AI-risk mentions from 2020 to 2024, while also concluding that many disclosures remain generic and thin on mitigation detail. That is almost a one-to-one precedent for your own empirical angle. ([arXiv][8])

For broader disclosure practice in Europe, **Bonsón et al. (2023), “Disclosures about algorithmic decision making in the annual or sustainability reports of major Western European companies”** is highly relevant. It is not identical to AI-risk disclosure, but it shows that automated-decision-making disclosure can be studied systematically from corporate reports and that disclosure practice is still uneven. ([ScienceDirect][9])

On the governance side, **Chiu (2025), “Using generative artificial intelligence in corporate narrative reporting”** is useful for the report’s discussion section. It focuses on firms using GenAI in mandatory narrative reporting and the attendant governance, legal, and control risks. That helps you position the Observatory not just as observing AI discussed *by* firms, but also as part of a wider concern about AI changing the reporting process itself. ([Cambridge University Press & Assessment][10])

For policy framing, the **SEC’s 2024 “State of Disclosure Review”** is a very strong source. It says staff observed a significant increase in AI mentions in annual reports and explicitly expects disclosures to be tailored rather than boilerplate, material to the company, tied to actual business use, and grounded in a reasonable basis. Even though your corpus is UK, this is powerful evidence that AI disclosure quality is already a live regulatory issue. ([SEC][11])

For corporate-governance framing beyond securities law, the **OECD AI Principles** and the **OECD Due Diligence Guidance for Responsible AI** are useful normative references because they emphasize transparency, responsible disclosure, and enterprise-level governance processes around AI. ([OECD][12])

## 3) Substantiveness and boilerplate

This is where your classifier has the clearest conceptual justification.

Your anchor trio should be:

**Loughran & McDonald (2011)** for finance-specific language measurement. ([Wiley Online Library][2])

**Brown & Tucker (2011)** for year-over-year textual change as evidence of informative updating. ([bear.warrington.ufl.edu][4])

**Dyer, Lang & Stice-Lawrence (2017)** for rising boilerplate, redundancy, and declining specificity in 10-K disclosure over time. ([ScienceDirect][5])

You can then connect these to your own classifier design:

* **similarity to prior-year disclosures** as a proxy for boilerplate
* **specificity / concreteness / named systems / operational detail** as a proxy for substantive disclosure
* **mitigation language** as a proxy for governance maturity
* **firm-tailored vs generic sectoral language** as a proxy for materiality

If you want one supporting contemporary citation, the SEC’s own guidance language is unusually aligned with your classifier objective because it explicitly warns against generic AI buzz and asks for tailored, material discussion. ([SEC][11])

## 4) Monitoring / observatory precedents

This is the section that will make your project feel like part of an established research genre rather than a one-off dashboard.

The strongest regulatory-monitoring parallel I found is the **FCA’s review of TCFD-aligned disclosures by premium-listed companies**. It shows a regulator systematically reading annual financial reports for principal and emerging risks, then reporting how often climate appears as such a risk. Methodologically, that is very close to what your Observatory does for AI. ([FCA][13])

For climate-risk text monitoring specifically, **ClimateBERT / ClimaText** is a strong methodological analogue. The paper explicitly argues that transformer models can monitor climate-risk disclosure in financial reports and related policy texts. That is a useful precedent for “specialized transformer + structured monitoring of corporate disclosure.” ([arXiv][14])

**Ferjančič et al. (2024)** on ESG reporting in FTSE 350 annual reports is another good UK-adjacent precedent. It uses BERTopic over a decade-long annual-report corpus and shows how topic prevalence moves with regulation and major events. That is very close to your observatory logic: build a corpus, extract structured signals, relate shifts to external policy events. ([ScienceDirect][15])

For risk-language measurement more generally, **Davis et al. (2020)** use 10-K risk-factor text to characterize firm-level exposures, showing that filing language can proxy for economically meaningful risk channels. That supports your claim that narrative disclosures can be treated as a measurement instrument for real-world exposure. ([NBER][16])

## 5) UK / CNI regulatory context

This is the policy section you should treat as non-optional.

For listed-company annual reports, the FCA’s **DTR 4.1** requires the management report to include a description of the issuer’s principal risks and uncertainties. The FCA’s own TCFD review also reiterates that annual financial reports must describe principal risks and uncertainties, and shows that regulators already assess whether particular themes, such as climate, are being recognized as principal or emerging risks. That gives you a direct doctrinal basis for tracking whether AI is entering the formal risk vocabulary of UK companies. ([FCA Handbook][17])

The **2024 UK Corporate Governance Code** is directly relevant. The FRC states that the revised Code applies from financial years beginning on or after 1 January 2025, while **Provision 29** applies to financial years beginning on or after **1 January 2026**. The FRC also says the key change is that boards now make a declaration on the effectiveness of their material internal controls. In the guidance, the FRC clarifies that this extends beyond traditional financial controls to include controls over reporting, including **narrative and ESG reporting controls**. That is highly relevant to your finding that AI mentions shift after Provision 29 enters the scene: even before the first full post-2026 reporting cycle, firms have an incentive to strengthen how they identify and narrate emerging AI-related risks and controls. ([FRC (Financial Reporting Council)][18])

On AISI, the official material you will want is:

* **AISI Research Agenda**, which emphasizes evaluation, risk scenarios, and capability-risk mapping
* **Frontier AI Trends Report**, which presents accessible, data-driven analysis of frontier AI capabilities and risks
* **Navigating the uncharted: Building societal resilience to frontier AI**, which is especially useful if your report is framed around resilience and critical infrastructure. ([AI Security Institute][19])

Those sources help connect your observatory from firm-level disclosures to the UK’s broader public-interest framing around resilience, security, and critical-risk monitoring. ([AI Security Institute][20])

## Recommended reading list by section

If you want a manageable first-pass bibliography, I’d start with these 12:

1. **Loughran & McDonald (2011)** — finance-specific textual analysis foundation. ([Wiley Online Library][2])
2. **Loughran & McDonald survey** — field overview. ([SSRN][3])
3. **Brown & Tucker (2011)** — textual similarity / disclosure modification. ([bear.warrington.ufl.edu][4])
4. **Dyer, Lang & Stice-Lawrence (2017)** — boilerplate growth and declining specificity. ([ScienceDirect][5])
5. **Lang & Stice-Lawrence (2015)** — international annual-report text analysis. ([ScienceDirect][6])
6. **Kim (2024), Financial Statement Analysis with LLMs** — LLM-era precedent. ([Bayes Business School][7])
7. **Park (2024), BIS/IFC** — LLMs for materiality in risk disclosures. ([Bank for International Settlements][21])
8. **Uberti-Bona Marin et al. (2025)** — large-scale AI-risk disclosure study in 10-Ks. ([arXiv][8])
9. **Bonsón et al. (2023)** — AI/ADM disclosure in European corporate reports. ([ScienceDirect][9])
10. **ClimateBERT / ClimaText (2023)** — disclosure monitoring precedent. ([arXiv][14])
11. **Ferjančič et al. (2024)** — FTSE 350 ESG topic extraction from annual reports. ([ScienceDirect][15])
12. **FRC + FCA official sources on Provision 29 / DTR 4 / TCFD review** — UK doctrinal anchor. ([FRC (Financial Reporting Council)][18])

## The core argument your report can make

A concise thesis for the literature review could be:

> Prior research shows that corporate filings are an established and informative NLP corpus; that disclosure quality is shaped by boilerplate, repetition, and firm-specific updating; that AI-related corporate disclosure is rapidly increasing but often remains generic; and that regulators already use disclosure-monitoring frameworks for other emerging risks such as climate. Risk Observatory extends this lineage to AI exposure and AI-risk monitoring in UK public-company annual reports, with special relevance for Critical National Infrastructure and the UK’s evolving governance regime under the FCA, the FRC’s 2024 Code, and AISI’s resilience-oriented risk agenda. ([Risk Observatory][1])

The biggest gap I see is that there is now emerging work on **AI-risk disclosure in US 10-Ks**, but much less on **UK annual reports**, **CNI sectors**, **vendor dependence**, and **substantiveness of AI-risk language** specifically. That makes your project feel timely rather than derivative. ([arXiv][8])

I can turn this into a **formal annotated bibliography** or a **draft literature review section** next.

[1]: https://riskobservatory.ai/?utm_source=chatgpt.com "AI Risk Observatory"
[2]: https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2010.01625.x?utm_source=chatgpt.com "When Is a Liability Not a Liability? Textual Analysis ..."
[3]: https://ssrn.com/abstract%3D2504147?utm_source=chatgpt.com "Textual Analysis in Accounting and Finance: A Survey"
[4]: https://bear.warrington.ufl.edu/tucker/2011_Brown_and_Tucker_MDA.pdf?utm_source=chatgpt.com "Large sample evidence on firms' year-over-year MD&A ..."
[5]: https://www.sciencedirect.com/science/article/abs/pii/S0165410117300484?utm_source=chatgpt.com "The evolution of 10-K textual disclosure: Evidence from ..."
[6]: https://www.sciencedirect.com/science/article/abs/pii/S0165410115000658?utm_source=chatgpt.com "Textual analysis and international financial reporting"
[7]: https://www.bayes.citystgeorges.ac.uk/__data/assets/pdf_file/0009/799794/Alex-Kim_Financial_Statement_Analysis_with_Large_Language_Models__2024_-6.pdf?utm_source=chatgpt.com "Financial Statement Analysis with Large Language Models"
[8]: https://arxiv.org/abs/2508.19313?utm_source=chatgpt.com "Are Companies Taking AI Risks Seriously? A Systematic Analysis of Companies' AI Risk Disclosures in SEC 10-K forms"
[9]: https://www.sciencedirect.com/science/article/pii/S1467089522000483?utm_source=chatgpt.com "Disclosures about algorithmic decision making in the ..."
[10]: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/00E8C459675F433EBBF1277A3162B947/S3033373325100380a.pdf/div-class-title-using-generative-artificial-intelligence-in-corporate-narrative-reporting-understanding-risks-within-the-lens-of-a-reporting-chain-approach-div.pdf?utm_source=chatgpt.com "Using generative artificial intelligence in corporate ..."
[11]: https://www.sec.gov/newsroom/speeches-statements/gerding-statement-state-disclosure-review-062424 "SEC.gov | The State of Disclosure Review"
[12]: https://www.oecd.org/en/topics/ai-principles.html?utm_source=chatgpt.com "AI principles"
[13]: https://www.fca.org.uk/publications/multi-firm-reviews/tcfd-aligned-disclosures-premium-listed-commercial-companies "Review of TCFD-aligned disclosures by premium listed commercial companies | FCA"
[14]: https://arxiv.org/pdf/2303.13373?utm_source=chatgpt.com "arXiv:2303.13373v1 [cs.CL] 21 Mar 2023"
[15]: https://www.sciencedirect.com/science/article/pii/S105752192400601X?utm_source=chatgpt.com "Textual analysis of corporate sustainability reporting and ..."
[16]: https://www.nber.org/system/files/working_papers/w27867/w27867.pdf?utm_source=chatgpt.com "Firm-Level Risk Exposures and Stock Returns in the Wake ..."
[17]: https://handbook.fca.org.uk/handbook/DTR/4/1.html?utm_source=chatgpt.com "DTR 4.1 Annual financial report"
[18]: https://www.frc.org.uk/library/standards-codes-policy/corporate-governance/uk-corporate-governance-code/ "UK Corporate Governance Code 2024"
[19]: https://www.aisi.gov.uk/research-agenda "AISI Research Agenda | The AI Security Institute"
[20]: https://www.aisi.gov.uk/about?utm_source=chatgpt.com "About | The AI Security Institute (AISI ..."
[21]: https://www.bis.org/ifc/publ/ifcb65_09_rh.pdf?utm_source=chatgpt.com "Quantifying material risks from textual disclosures in ..."

