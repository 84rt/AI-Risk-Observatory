# Golden Set Implementation Plan (Minimal)

## Phase 1: Scope & Ground Truth
~~1. Get the initial sample (one for each CNI sector (list in appendix), two recent years only).~~
~~2. Preprocess filings to clean, human-readable text.~~
~~3. Manually annotate all samples to create the golden set.~~
	~~1. create tools to make this easier for human annotator~~
	2. save the golden set data as classified by the human (in the mata data the same way we save classifier version)
## Phase 2: Core Classifiers
6. Implement base classifiers:
   - AI harms
   - AI adoption
   - AI risk disclosure
7. Benchmark against the golden set.
8. Run reliability checks:
   - 3 model families
   - 10-run consistency
   - low-confidence (<80%) and disagreement analysis
9. Iterate or halt based on predefined quality thresholds.

## Phase 3: Expansion
10. Extend classifiers:
    - vendor extraction (adoption)
    - substantiveness / boilerplate scoring (risk)
11. Re-benchmark expanded dimensions.
12. Add second-order analyses:
    - cowboy risk (severity × mitigation)
    - workforce impact test

## Phase 4: Scaling
13. Expand temporal scope (>5 years).
	1. Add PDF ingestion pipeline with explicit quality checks.
	2. Run comprehensive test on the error rate of PDF processing 

### Appendix: list of companies to process

|     | Company                         | CNI Sector         | Index    | Fit to sector |
| --- | ------------------------------- | ------------------ | -------- | ------------- |
| 1   | Johnson Matthey plc             | Chemicals          | FTSE 100 | Best proxy    |
| 2   | Rolls-Royce Holdings plc        | Civil Nuclear      | FTSE 100 | Best proxy    |
| 3   | BT Group plc                    | Communications     | FTSE 100 | Direct        |
| 4   | BAE Systems plc                 | Defence            | FTSE 100 | Direct        |
| 5   | Babcock International Group plc | Emergency Services | FTSE 250 | Best proxy    |
| 6   | Shell plc                       | Energy             | FTSE 100 | Direct        |
| 7   | HSBC Holdings plc               | Finance            | FTSE 100 | Direct        |
| 8   | Tesco plc                       | Food               | FTSE 100 | Direct        |
| 9   | Capita plc                      | Government         | FTSE 250 | Best proxy    |
| 10  | AstraZeneca plc                 | Health             | FTSE 100 | Direct        |
| 11  | Rolls-Royce Holdings plc        | Space              | FTSE 100 | Best proxy    |
| 12  | National Grid plc               | Transport          | FTSE 100 | Direct        |
| 13  | Severn Trent plc                | Water              | FTSE 100 | Direct        |


***Important note: Rolls-Royce applies twice to both Civil Nuclear and Space sectors as best proxy, it appears in the database only once for annotation but it should be used for both sectors***


### Apendix 2: the proposed taxonomy
- Adoption (initial focus):
  - Classify Type of AI: non-LLM, LLM, agentic AI
  - Depth/Criticality are deferred
- Risk (Appendix 2 default taxonomy):
  - Operational & Technical Risk (model failures, bias, reliability)
  - Cybersecurity Risk
  - Workforce Impacts
  - Regulatory & Compliance Risk
  - Information Integrity (misinformation/deepfakes)
  - Reputational & Ethical Risk
  - Third-Party & Supply Chain Risk (vendor dependence)
  - Environmental Impact
  - National Security Risk
- Harms:
  - Look for harms mentions but expect rarity; keep simple
- Substantiveness:
  - Classify disclosures as substantive vs boilerplate
- Other classifiers (vendor concentration, severity/likelihood, internal vs customer-facing, depth/criticality):
  - Proposed to defer until after the core set is stable

