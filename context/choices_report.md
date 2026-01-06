Purpose of this Document
We are currently in the initial testing phase of the AIRO project. To ensure the final dashboard provides actionable insights within the allocated resources, we must make specific trade-offs regarding scope and technical complexity.

This document outlines our recommended methodology based on preliminary feasibility research. Its purpose is to finalize these design choices before we scale up data processing.

How to Use This Document: This report is divided into five decision sections (A–E). Each section presents a specific design question, the context, and our recommendation.
Basis of Recommendations: We incorporated the initial "steer" provided by AISI during preliminary discussions.
Color Coding:
Green: The steer from AISI (as understood) form the kick off call.
Yellow: Our recommended approach based on preliminary testing and expected technical feasibility or value.
Bright yellow: Sections where we need an answer from AISI. While some questions can be answered directly, more elaboration on AISI’s preferences is greatly helpful to us in navigating technical tradeoffs moving forward. 
Required Action: We need AISI to review the recommendations in the following five areas and confirm approval or provide specific adjustments:
A. Sampling: Which annual reports (sectors/companies) do we use for the initial test?
B. Classification: Which taxonomies (Adoption, Risk, Harms) do we apply?
C. Reliability: What is the acceptable trade-off between volume and accuracy?
D. Validation: How do we test correspondence (Ground Truth vs. Report Content)?
E. Visualization: What are the priority views for the dashboard? 
A. Sampling: Which annual reports (sectors/companies) do we use for the initial test?

Context:
We want to start with a small set of annual reports (~20–30) to calibrate the pipeline, then scale up depending on time/budget. After calibration, we will prioritize breadth across the whole economy where feasible, using CNI sectors as the organizing lens.

Decision question:
Which reports should we use for the initial calibration sample (sectors/companies/years), before scaling up?

Initial AISI steer:
Sector: “Ideally, focus on Critical National Infrastructure (CNI) sectors. The general goal is to get a rough sense of trends in the economy rather than specific analysis of individual firms’ reports.”
Years: “We are mainly interested in ‘post-ChatGPT’ type AI”

Recommendation:
We recommend doing initial testing on thirteen companies over two years each (i.e., a total of 26 annual reports). Each company will be from a different CNI sector and will be selected manually from among FTSE 350 firms in that sector. Reports will be the two most recent available years (e.g., 2023 and 2024) to test whether there is an indication of a trend in AI mentions. The list of suggested companies can be found in Appendix 1.


Rationale and drawbacks:
Number of companies: One firm per CNI sector gives breadth across sectors while keeping manual quality testing feasible.
Years: Two years gives us technical insight into comparing years (plus some very minimal substantive insight into trends). Post-2021 is ‘post-ChatGPT’, plus reports are available in XML.
Epistemic drawback: This sample is not representative of the whole economy.
Important epistemic note: AISI ultimately prefers coverage (“less robust, on more companies”). This small sample is for the purpose of initial testing (to refine methodology and set confidence bounds). We expect to scale to a bigger sample later in the process.

Alternatives:
Choose companies from before 2021 or from outside FTSE 350: FAIRLY DIFFICULT – this involves reading scanned PDFs (rather than XMLs). Little value in doing pre-2021 (unless AISI wished specifically to test ‘before and after’ frontier AI became prominent in discourse). Selecting non-FTSE 350 firms (i.e. smaller listed firms) would give some additional insight into this population, but also increases processing cost and probably decrease the accuracy of findings. 
Add more companies or annual reports to the initial sample: FAIRLY EASY - the work scales less-than-linearly. However, we believe 26 reports is sufficient to test the main technical questions (including accuracy of classifiers). It is insufficient to draw robust conclusions on most substantive questions, but this can be done when we scale up.
Use different sectors or companies: EASY - if AISI has a different list it prefers, we can use that. We proposed to use one from each CNI sector in response to AISI’s steer. This does not give us robust insight into any one sector, but does give us a chance to detect any idiosyncratic features related to any of the 13 CNI sectors. We could instead analyse more companies from fewer sectors, or more years for fewer companies, or more companies if we test only one year per company. Ideally the number of reports tested will be below 30 to facilitate manual quality control.

Key considerations: 
FTSE 350 companies from 2022 onward are required to file in XML format. Non-XML file types are costly to work with: the filing type is a PDF photo scan of a physical document, which requires us to convert it to text. This introduces additional processing cost and can introduce errors. We recommend avoiding processing PDFs if possible, as errors can decrease the reliability of the downstream findings.

Question for AISI:
1) Do you agree with the proposed calibration sample (13 firms × 2 years)? If not, what changes (sectors/years/firms)?






2) Do you want to approve the specific list of 13 firms (Appendix 1), or should we use different firms in the initial test sample?






3) Do you agree with “calibrate first, then scale breadth‑first”? If not, should we move to broader coverage immediately (accepting higher uncertainty per classification)?





B. Classification: Which taxonomies (Adoption, Risk, Harms) do we apply? 

Context:
We need to choose (a) adoption taxonomy, (b) risk taxonomy, and (c) whether to add other classifiers (e.g. ones to detect ‘boilerplate’  vs substantive comment). In general, fewer, well-defined categories yield higher reliability; broad/overlapping taxonomies reduce accuracy.

Initial AISI steer:
Interested in three areas: Type of AI used, Depth of use (Pilot vs Deployment), and Criticality (Core vs Peripheral).
Cares about distinguishing non-LLM, LLM, Agentic AI adoption.
Expressed a strong interest in distinguishing 'Boilerplate' from 'Substantive' disclosures.
AISI is more interested in systemic misuse / operational risks for this dataset than in very long‑term “gradual disempowerment” scenarios.
AISI may have a preferred risk taxonomy (potentially an Excel/government register). We anticipate some mismatch between policy-oriented categories and how companies phrase risks in annual reports, so we may need to map/merge categories for reliability.
Preference for investigating the concentration of tools/vendors (single points of failure analysis).

Recommendation:
We are proposing to attempt the following classifications in our initial sample:
Level of AI adoption: We recommend strictly classifying Type of AI (non-LLM, LLM, Agentic AI) for the initial test.
Note: We acknowledge your interest in "Depth" (Pilot vs. Deployment) and "Criticality" (Peripheral vs. Core). We recommend excluding these from the initial test to ensure we first establish high accuracy on the base taxonomy. Adding them now risks compounding errors.
Risk from AI: We propose to use the taxonomy in Appendix 2, unless provided with an alternative one.
Harms from AI: We propose looking for and analyzing any mentions form harms from AI within the reports. We expect these mentions to be rare and highly irregular in both frequency and form, making them hard to identify and classify reliably. (In addition, establishing any credible “source of truth” for harm(s) would likely require manual external research, increasing time/cost significantly. See later section). At this stage, we recommend focusing on whether the core classifier works and is worth scaling up.
Boilerplate detection: We recommend adding a "Substantiveness" classifier. We will tag risk mentions as "Substantive" if they contain specific evidence (e.g., specific tool names, quantitative impact, or mitigation steps) and "Boilerplate" if they use generic legal phrasing. We expect this to be one of the harder classifiers to implement.
We propose to defer testing of the following classifications (and any others):
Single points of failure / concentration signals: AISI expressed interest in concentration of tools/vendors (single points of failure analysis). We propose deferring this for the initial calibration pass to reduce complexity, but we will implement it immediately after calibration if AISI prioritizes it.
Classifying the severity/likelihood of risk: We expect the descriptions of the risks mentioned to be highly lawyered and hard to reliably classify. We propose testing the classifier of severity or likelihood or risk mentioned once the core classifiers work reliably. 
Internal deployment vs. customer-facing AI products: Classification based on the primary subject of each section containing the mention of AI. This classification dimension can be added to the analysis after our classifiers are robust and well working. (We did manage to get the initial version of this classifier working already, however we recommend focusing on the core classifiers for the initial sample)
A note on Measurement Methodology (Mention Density/Propensity): We propose to use segment-level tagging and aggregate statistically, (rather than one label per report) to reduce long-context errors and quantify intensity. In other words, tag and classify the report by analysing it segment-by-segment and applying all the classifiers that appear to be merited. This could mean that multiple classifiers are applied (e.g. “non-LLM” and “Agentic AI” might both get applied). Each will be given a propensity score, and we can then establish rules later to determine which tags are accepted as valid. (This entails setting rules about which tags are mutually exclusive and in what circumstances, and how conflicts are to be resolved). The alternative approach is to require the model to do this in one step, so that we get the final decision but never ‘see the working’. However, this one-step approach is likely to be more error prone so at least in the testing phase we prefer a multi-step process.


Potential Adjustments:
Adoption taxonomy / categories (GENERALLY EASY): Changing the taxonomy is straightforward, but once a taxonomy is set, re-running the classifier at scale can be expensive. Reliability is highest when categories are few, mutually exclusive, and easy for an LLM to apply; overlapping or ambiguous categories can significantly reduce accuracy.
Measuring adoption intensity (mention density/propensity) vs. exclusive labels (MODERATE DIFFICULTY): We can switch to a more exclusive, report-level classification system instead of measuring propensity, but deciding which mention is “dominant” is unlikely to be done accurately by an LLM alone. Segment-level tagging + statistical aggregation is typically more robust and gives a better measure of intensity.
Concentration / vendor risk (MODERATE DIFFICULTY): We can look for “vendor concentration” signals (e.g., “How many CNI sectors rely on Microsoft?”) by extracting named entities (OpenAI, AWS, Microsoft, etc.) during processing. Classification is straightforward, but vendor normalization (aliases, subsidiaries, product vs. parent) can be challenging; this likely requires a separate vendor extractor/classifier and a normalization layer.
Severity/likelihood (HIGHLY UNCERTAIN): If severity/likelihood is explicitly stated, we can extract it. However, we expect it is often omitted or only implicit, which makes it difficult to classify. If needed, we can test the frequency and nature of explicit severity/likelihood statements in the initial test sample.
Internal use vs. customer-facing products (MODERATE DIFFICULTY): We might be able to find explicit mentions containing enough detail for the LLM to correctly classify internal vs customer-facing AI deployment, but there is a lot of room for error as the mentions of AI deployment are non-standarized. Likely, a heavy manual quality assurance process would have to be done on the annotated sections by the AI to guarantee the reliability of findings. 

Key consideration:
Although this differs from AISI’s current preference, we recommend a limited-time, small-sample test of highly difficult classification types to validate the methodology before scaling. AISI ultimately sets testing priorities, and while difficulty is relevant, the value of the findings will determine next steps. Testing these difficult, potentially low-value cases allows us to confirm or falsify our preliminary results and could ultimately deliver the high value AISI seeks.

Questions for AISI
Question
Answer
1. Risk Taxonomy: Do you accept the default risk taxonomy in Appendix 2 for the initial test?
If "No", please provide the preferred Excel taxonomy immediately.
2. Adoption dimensions: Should we limit the initial test to “Type of AI” only? 
If "No", should we prioritize (A) "Depth" (Pilot vs Deployment) or (B) "Criticality" (Core vs Peripheral) or both?
3. Should we deprioritize the concentration/vendor risk classification in the initial sample?


4. Should we deprioritize the severity/likelihood classification in the initial sample?


5. Should we deprioritize the internal use vs. customer-facing classification in the initial sample? 


5. Any other comments regarding the general scope of the initial testing? 




C. Reliability: What is the acceptable trade-off between volume and accuracy? 
Context:
We need to decide how to trade off volume vs robustness of classifications done. Classifier difficulty varies by label (some are straightforward; others are nuanced and low-signal).

Initial AISI steer:
“Robust enough that I can tell, for example, Ofgem, and they won’t laugh at me.”

Recommendation:
Set a ‘minimum robustness’ threshold (based on initial experimentation) and thereafter prioritise coverage over robustness : We propose using the initial sample to calibrate expected error rates and establish minimum acceptable performance (e.g., for each classifier, a human‑checked precision/recall threshold and a clear “unknown/uncertain” bucket). After that, we propose prioritizing broader coverage (“less robust on more”), while clearly communicating uncertainty and known failure modes.

Potential Adjustments:
Instead we could select either extreme: (A) optimize for very high accuracy on a small set of items to classify, or (B) accept more uncertainty per classification to maximize coverage. We won’t know the exact trade-off between these two until we have done our initial testing.

Questions for AISI
Do you agree with “set a minimum robustness threshold, then prioritise for coverage”? If not, should we bias toward (A) higher confidence on fewer reports or (B) broader coverage with uncertainty bounds?






D. Validation: Should we test correspondence to Ground Truth??
Context:
In principle we can test two different things: (a) whether classifiers correctly extract what is in reports, and (b) whether reports track real-world reality (“ground truth”). However, testing (b) is expensive and often subjective, so we must make a decision on whether, and to what extent, to attempt it.

Initial AISI steer:
“Testing ground truth is less important, given that it is likely to be very difficult to do robustly. It is not the best use of this technical project’s time and resources.”
“Testing the correspondence between what is in the report and what an LLM can detect and classify seems the higher priority.”


Recommendation:
Focus primarily on report-classifier correspondence, plus a light “vibe check” against known incidents: Prioritize robust testing of whether LLMs can reliably detect, extract, and classify the risks and signals that are explicitly present in the reports. Use a small, carefully constructed set of human-annotated reports as a reference benchmark to evaluate classifier precision, recall, and error modes. In addition, we propose a light-touch validation pass using a small number of known incidents / third‑party sources to sanity‑check whether report signals feel directionally plausible. We do not propose a large, expensive “ground truth” validation exercise.
We will conduct a light test of performance across multiple model families (e.g., Gemini vs. Claude). If there is no difference in output quality, we’ll use the more cost-effective option.

Rationale:
Ground-truth risk is hard to determine (e.g. if a report says that the main AI-related risk the company faces is cybercrime on a one-year horizon and shifting customer behaviour on a three-year horizon, how do we assess whether that is true?); robust validation is expensive even where possible (e.g. if the media reports an AI-related harm to a given company, we could manually check whether that is disclosed in the report, but doing this systematically would be very costly, and remains reliant on the completeness and correctness of media reporting).
The core technical deliverable is reliable extraction/classification of what firms report.
A small “vibe check” provides a sanity check without becoming a major research project.

Potential Adjustments:
Focus on correspondence to ground truth (EXTREMELY DIFFICULT).  We could make this a major focus of our work, though we believe the results will be of very limited use unless we are able to identify a more targeted and robust methodology (e.g. if a specific sector has a mandatory regulatory-disclosure process, and we can access the incident list, we could test that against annual report disclosure).
Make not even a cursory attempt to look at ground truth correspondence (EASY): i.e. conversely, we could remove the “vibe check” / directional checking from the work plan.

Questions for AISI:
Does AISI agree with the recommended approach to correspondence testing? If not, how should we adjust?





E. Visualization: What are the priority views for the dashboard?
Context:
We need to determine the visualization priorities to ensure the dashboard provides actionable insights for AISI. The focus is on usefulness to the end user. 

Initial AISI steer:
“Prefers simple, legible dashboards over a single very complex all‑in‑one interface.” High-value views include risk trends over time and by sector. Heatmaps (sectors × risk types) and adoption intensity views are requested.
Preference for including a visualization of “blind spots” - risks not mentioned at all in sectors.

Recommendation:
Develop a simple dashboard focusing on three core views:
Risk Trends: Visualizing risk over time and by sector (e.g., stacked charts).
Sector Heatmaps: Grids displaying sectors vs. risk types to identify "hot spots" of intensity.
Adoption Intensity: A view analogous to risk views, showing adoption depth by sector.
Additionally, we will explicitly visualize “Blind Spots” (empty heatmap cells) as requested to highlight sectors reporting no risks.

We should prioritize the ability to select a risk type and see its distribution across sectors or over time. Comparison of two risks on the same chart should be treated as a secondary "nice-to-have".

Potential Adjustments:
Advanced Multi-Variable Comparisons: MODERATELY COSTLY - Focusing on complex comparisons (e.g., two risks simultaneously) adds development overhead for features explicitly marked as only "nice-to-have". While not technically difficult, it detracts development time.
LLM chatbot interfaces: EXPERIMENTAL. We suggest only attempting this at the end of the project, if all else has been achieved and there is no other priority. While perhaps convenient for the user, without robust testing the chatbot interface could itself introduce errors that misrepresent the research.

Questions for AISI
Question
Answer
1. Do you agree with the priority views (Trends, Heatmaps, Adoption)? Any other comments on what visualisations would be helpful.


2. How important is it to have a dedicated “Blind Spots” view? Would lack of it be highly undesirable?


3. For the "Blind Spots” (highlighting which sectors report no expected risk), how should we know what risk is expected to be reported in a given sector? How do we know if there is a blind spot in reporting? 


4. How should we prioritise the LLM chatbot interface? 





Appendix 1
List of suggested companies for the initial testing sample (some sectors don’t have a major public company with core value proposition in that sector, hence we are forced to use a best proxy we can find):


Company
CNI Sector
Index
Fit to sector
1
Johnson Matthey plc
Chemicals
FTSE 100
Best proxy
2
Rolls-Royce Holdings plc
Civil Nuclear
FTSE 100
Best proxy
3
BT Group plc
Communications
FTSE 100
Direct
4
BAE Systems plc
Defence
FTSE 100
Direct
5
Babcock International Group plc
Emergency Services
FTSE 250
Best proxy
6
Shell plc
Energy
FTSE 100
Direct
7
HSBC Holdings plc
Finance
FTSE 100
Direct
8
Tesco plc
Food
FTSE 100
Direct
9
Capita plc
Government
FTSE 250
Best proxy
10
AstraZeneca plc
Health
FTSE 100
Direct
11
Rolls-Royce Holdings plc
Space
FTSE 100
Best proxy
12
National Grid plc
Transport
FTSE 100
Direct
13
Severn Trent plc
Water
FTSE 100
Direct



Appendix 2
The default risk taxonomy that we created in aims of capturing all potential risk mentions into insightful categories. The definition of these categories is our best guess of what kind of risk taxonomy would be the most useful, while keeping in mind the limitations of LLMs (primarily regarding having too many, or overlapping, categories to choose from). The proposed default risk taxonomy:
Name
Description
Operational & Technical Risk
Model failures, bias, reliability, system errors
Cybersecurity Risk
AI-enabled attacks, data breaches, system vulnerabilities
Workforce Impacts
Job displacement, skill requirements, automation
Regulatory & Compliance Risk
Legal liability, compliance costs, AI regulations
Information Integrity
Misinformation, content authenticity, deepfakes
Reputational & Ethical Risk
Public trust, ethical concerns, human rights, bias
Third-Party & Supply Chain Risk
Vendor reliance, downstream misuse, LLM provider dependence
Environmental Impact
Energy use, carbon footprint, sustainability
National Security Risk
Geopolitical, export controls, adversarial use



