Progress Report
Disclaimer: all of the claims and findings in this report are from a limited sample that has been only preliminarily tested and thus all of the findings and conclusions might end up being at least partly wrong. 
Summary
We have built a working prototype that extracts and classifies AI risk disclosures from UK company annual reports. The system processes iXBRL/XHTML filings, applies five classifiers (harms, adoption types, substantiveness, vendors, risk categories), and validates results across multiple AI model families. Testing on 20 FTSE 350 companies suggests AI risk disclosure is growing rapidly but remains largely superficial.
Why This Is Difficult
Corporate AI disclosure sits at the intersection of inconsistent regulatory frameworks. Listed firms follow IFRS, private entities use UK GAAP, and sector regulators add varying requirements. The 2024 UK Corporate Governance Code now requires boards to declare effectiveness of "material controls" including AI systems, but specifics remain unclear.
More fundamentally, we face a temporal discontinuity. Mentions of AI as a principal risk jumped from 11 companies in 2023 to 68 in 2024-a post-ChatGPT phenomenon that makes historical comparisons difficult. We are tracking something new, not measuring change in established practice.
Technical Approach
We focused on iXBRL/XHTML filings (mandatory since 2022) rather than PDFs, which suffer from persistent extraction errors when tables and structure are flattened. Our pipeline achieves roughly 95% extraction quality. Classifications are validated across three AI model families (Gemini, Claude, GPT) with 83% agreement, following research showing multi-model consensus reduces hallucinations significantly.
Early Indicators
From our limited sample:
90% of companies mentioned AI-related harms in 2024 (up from 70% in 2023)
~75% of disclosure is contextual boilerplate; only ~15% contains substantive detail
Cybersecurity is the dominant concern (70% of companies), specifically AI-enabled attacks
~60% of companies appear to rely on internal AI development rather than external vendors
Open Questions
15-20% of risk mentions resist clean categorisation (e.g., "AI-enabled cyberattack" could be cybersecurity or operational risk). Some sectors like energy show minimal disclosure, unclear whether this reflects lower adoption or less transparency. Extending analysis before 2022 requires PDF processing we have deliberately deferred.
Proposed Next Steps
Pending AISI input: expand to broader FTSE 350 coverage, extend timeline to 2020-2024, refine classifiers based on human annotation, and build a visualisation dashboard.



