# Dashboard Findings

Generated on 2026-05-01 from `data/dashboard-data.json`.

## Scope and caveats

- This report uses the existing precomputed dashboard artifact only. It does not add any new metrics to the dashboard.
- 2026 is present in the artifact, but because the current date is 2026-05-01, 2026 should be treated as a partial year. The main YoY comparisons below therefore use 2024 -> 2025.
- Tables based on `riskTrend`, `adoptionTrend`, and `vendorTrend` are label-assignment counts, not unique-report counts. A single report can contribute to multiple labels.
- Sector-level unique adoption and vendor rates are not derivable from the current artifact because the available sector arrays for adoption and vendor are label-level counts rather than unique report counts.

## Recommended comparison windows

- Main research window: `2021 -> 2025`. This matches the paper draft's primary analysis period and avoids over-weighting the partial 2020 and partial 2026 edges of the series.
- Supporting long-run window: `2020 -> 2025`. This is useful as a start-of-series anchor and for communicating scale of change since the earliest observable baseline.
- Directional snapshot only: `2026`. Because the current date is 2026-05-01, 2026 should be treated as a partial-year directional signal, not as a directly comparable full-year endpoint.

## Priority metrics for the paper

- Any AI mention rate by year
- AI risk mention rate by year
- AI adoption mention rate by year
- Adoption-to-risk ratio by year
- General / ambiguous rate by year
- Substantive risk rate by year, both as a share of all reports and as a share of risk-reporting reports
- Quality gap: AI risk mention rate minus substantive risk rate
- Sector AI-risk rate and sector AI-risk blind-spot rate
- Market-segment gap: FTSE 100, FTSE 350, Main Market, and AIM
- Risk-category composition shift over time
- Vendor opacity rate: other + undisclosed as a share of all vendor references
- Named-vendor concentration among explicitly named provider references

## Headline findings

- The current artifact covers 9,821 reports across 1,362 companies from 2020-2026. 41.6% of all reports contain at least one non-`none` AI signal.
- In 2025, 65.5% of reports mentioned AI at all, up +10.3 pp from 2024; AI-risk disclosure rose even faster to 41.2% (+10.8 pp).
- The adoption-versus-risk disclosure gap narrowed from 6.4 pp in 2024 to 4.0 pp in 2025, suggesting risk disclosure is catching up with general AI adoption language.
- Cybersecurity was the fastest-rising risk category in 2025, increasing +7.9 pp to 25.7% of all reports. Cybersecurity and Operational / Technical risk were close behind.
- LLM disclosure was the fastest-rising adoption category, reaching 20.0% of all reports in 2025 (+8.6 pp YoY). Agentic references also rose to 10.6%.
- Main Market (FTSE 100 only) had the highest AI-risk rate in 2025 at 68.6%, while AIM remained far lower at 6.4%.
- Among CNI sectors with at least 20 reports in 2025, Communications saw the biggest rise in AI-risk disclosure (+26.7 pp), while Data Infrastructure had the largest remaining AI-risk blind spot (90.0% of reports still without an AI-risk mention).
- Vendor references remain fragmented. The largest vendor bucket in 2025 was `other` at 25.3% of vendor assignments; the leading named vendor was Microsoft at 16.1%.
- In 2025, opaque vendor references (`other` + `undisclosed`) accounted for 36.0% of all vendor assignments. Among explicitly named vendors, the top three accounted for 56.0% of named-vendor assignments.
- Risk disclosures became denser over time: average risk labels per risk-reporting company rose from 3.42 in 2024 to 3.65 in 2025.
- At ISIC level, the strongest large-sample AI-risk disclosure rate in 2025 was in Other monetary intermediation (77.2%; n=57 reports, using a minimum-sample filter of 20).

## Coverage summary

| Reports | Companies | AI signal reports | AI signal rate | Avg phase-1 labels / signal report |
| ---: | ---: | ---: | ---: | ---: |
| 9,821 | 1,362 | 4,084 | 41.6% | 2.14 |

### Company distribution by CNI sector

| Sector | Companies | Share of all companies |
| :--- | ---: | ---: |
| Finance | 461 | 33.8% |
| Other | 392 | 28.8% |
| Energy | 141 | 10.4% |
| Health | 111 | 8.1% |
| Transport | 61 | 4.5% |
| Food | 52 | 3.8% |
| Chemicals | 34 | 2.5% |
| Communications | 28 | 2.1% |
| Data Infrastructure | 22 | 1.6% |
| Government | 20 | 1.5% |
| Defence | 20 | 1.5% |
| Water | 18 | 1.3% |
| Civil Nuclear | 2 | 0.1% |

## Annual report-level trend summary

| Year | Reports | AI mention % | Adoption % | Risk % | General / ambiguous % | Vendor % | Adoption-risk gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020 | 1,007 | 19.8% | 13.8% | 2.8% | 0.0% | 3.1% | 11.0 pp |
| 2021 | 1,328 | 27.4% | 20.9% | 4.2% | 0.0% | 4.2% | 16.6 pp |
| 2022 | 1,853 | 28.4% | 22.5% | 4.8% | 0.0% | 5.7% | 17.6 pp |
| 2023 | 1,905 | 36.8% | 25.8% | 9.8% | 0.0% | 7.5% | 16.1 pp |
| 2024 | 1,828 | 55.2% | 36.8% | 30.4% | 0.0% | 14.6% | 6.4 pp |
| 2025 | 1,561 | 65.5% | 45.2% | 41.2% | 0.0% | 19.2% | 4.0 pp |
| 2026 | 339 | 77.3% | 63.1% | 66.4% | 0.0% | 29.5% | -3.2 pp |

### Core 2021 -> 2025 research metrics

| Metric | 2021 | 2025 | Change |
| :--- | ---: | ---: | ---: |
| Any AI mention rate | 27.4% | 65.5% | +38.1 pp |
| AI adoption rate | 20.9% | 45.2% | +24.3 pp |
| AI risk rate | 4.2% | 41.2% | +37.0 pp |
| General / ambiguous rate | 0.0% | 0.0% | +0.0 pp |
| AI vendor rate | 4.2% | 19.2% | +14.9 pp |
| Substantive risk rate (of all reports) | 0.6% | 4.3% | +3.7 pp |
| Substantive share of risk reports | 14.3% | 10.4% | -3.9 pp |
| Quality gap: risk minus substantive risk | 3.6 pp | 36.9 pp | +33.3 pp |
| Adoption-to-risk ratio | 4.95 | 1.10 | -3.85 |

### Supporting 2020 -> 2025 start-of-series comparison

| Metric | 2020 | 2025 | Change |
| :--- | ---: | ---: | ---: |
| Any AI mention rate | 19.8% | 65.5% | +45.8 pp |
| AI adoption rate | 13.8% | 45.2% | +31.4 pp |
| AI risk rate | 2.8% | 41.2% | +38.4 pp |
| General / ambiguous rate | 0.0% | 0.0% | +0.0 pp |
| Substantive risk rate (of all reports) | 0.3% | 4.3% | +4.0 pp |
| Adoption-to-risk ratio | 4.96 | 1.10 | -3.87 |

### Partial 2026 directional snapshot

2026 is partial and should not be compared to full-year 2025 as if both were complete annual cohorts. This table is included only to show direction of travel.

| Metric | 2025 | 2026 partial |
| :--- | ---: | ---: |
| Reports in sample | 1,561 | 339 |
| Any AI mention rate | 65.5% | 77.3% |
| AI adoption rate | 45.2% | 63.1% |
| AI risk rate | 41.2% | 66.4% |
| Substantive risk rate (of all reports) | 4.3% | 7.1% |
| Adoption-to-risk ratio | 1.10 | 0.95 |

## Quality-gap analysis

This is the most directly policy-relevant metric family in the current dataset: it separates the growth in AI-risk mentions from the much smaller share of reports that contain genuinely substantive AI-risk disclosure.

| Year | Risk reports | Risk rate | Substantive risk reports | Substantive rate | Substantive share of risk reports | Quality gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020 | 28 | 2.8% | 3 | 0.3% | 10.7% | 2.5 pp |
| 2021 | 56 | 4.2% | 8 | 0.6% | 14.3% | 3.6 pp |
| 2022 | 89 | 4.8% | 16 | 0.9% | 18.0% | 3.9 pp |
| 2023 | 186 | 9.8% | 23 | 1.2% | 12.4% | 8.6 pp |
| 2024 | 556 | 30.4% | 61 | 3.3% | 11.0% | 27.1 pp |
| 2025 | 643 | 41.2% | 67 | 4.3% | 10.4% | 36.9 pp |
| 2026 | 225 | 66.4% | 24 | 7.1% | 10.7% | 59.3 pp |

### 2024 -> 2025 headline rate changes

| Metric | 2024 | 2025 | Change | Count change |
| :--- | ---: | ---: | ---: | ---: |
| Any AI mention | 55.2% | 65.5% | +10.3 pp | +14 |
| AI adoption mention | 36.8% | 45.2% | +8.3 pp | +32 |
| AI risk mention | 30.4% | 41.2% | +10.8 pp | +87 |
| AI vendor mention | 14.6% | 19.2% | +4.6 pp | +33 |
| Adoption-risk gap | 6.4 pp | 4.0 pp | -2.4 pp | n/a |

## Risk taxonomy findings

- Risk-category counts are label assignments, so totals exceed the number of unique risk-reporting companies.
- In 2025, the top three risk categories accounted for 51.4% of all risk-label assignments. The HHI for risk-category concentration was 1,333.
- Average risk labels per risk-reporting report rose from 3.42 in 2024 to 3.65 in 2025.

| Risk category | 2025 count | 2025 % of reports | 2024 % of reports | YoY change |
| :--- | ---: | ---: | ---: | ---: |
| Cybersecurity | 401 | 25.7% | 17.8% | +7.9 pp |
| Strategic / Competitive | 427 | 27.4% | 19.6% | +7.7 pp |
| Operational / Technical | 377 | 24.2% | 18.3% | +5.8 pp |
| Regulatory / Compliance | 320 | 20.5% | 15.2% | +5.3 pp |
| Third-Party Supply Chain | 191 | 12.2% | 7.5% | +4.7 pp |
| Reputational / Ethical | 248 | 15.9% | 11.2% | +4.7 pp |
| Workforce Impacts | 132 | 8.5% | 4.5% | +3.9 pp |
| Information Integrity | 162 | 10.4% | 7.3% | +3.1 pp |
| Environmental Impact | 45 | 2.9% | 1.1% | +1.8 pp |
| National Security | 42 | 2.7% | 1.5% | +1.2 pp |

## Adoption findings

- Adoption-category counts are label assignments, not unique reports.
- In 2025, the top three adoption categories accounted for 88.4% of all adoption-label assignments.
- Average adoption labels per adoption-reporting report rose from 1.75 in 2024 to 2.01 in 2025.

| Adoption category | 2025 count | 2025 % of reports | 2024 % of reports | YoY change |
| :--- | ---: | ---: | ---: | ---: |
| Ambiguous | 312 | 20.0% | 11.4% | +8.6 pp |
| LLM | 387 | 24.8% | 16.7% | +8.1 pp |
| Traditional AI (non-LLM) | 555 | 35.6% | 30.4% | +5.1 pp |
| Agentic | 165 | 10.6% | 6.0% | +4.6 pp |

## Vendor findings

- Vendor-category counts are label assignments, not unique reports.
- In 2025, the top three vendor buckets accounted for 52.1% of all vendor-label assignments. The HHI for vendor concentration was 1,306.
- Average vendor labels per vendor-reporting report rose from 2.12 in 2024 to 2.04 in 2025.

| Vendor bucket | 2025 count | 2025 % of reports | 2024 % of reports | YoY change |
| :--- | ---: | ---: | ---: | ---: |
| Other | 154 | 9.9% | 5.9% | +4.0 pp |
| Undisclosed | 65 | 4.2% | 1.8% | +2.4 pp |
| Internal | 56 | 3.6% | 2.5% | +1.1 pp |
| Amazon / AWS | 42 | 2.7% | 2.1% | +0.6 pp |
| Salesforce | 16 | 1.0% | 0.5% | +0.5 pp |
| Meta | 22 | 1.4% | 1.0% | +0.4 pp |
| Google | 47 | 3.0% | 2.7% | +0.3 pp |
| Palantir | 4 | 0.3% | 0.0% | +0.3 pp |
| Snowflake | 5 | 0.3% | 0.1% | +0.2 pp |
| Anthropic | 5 | 0.3% | 0.1% | +0.2 pp |
| Microsoft | 98 | 6.3% | 6.1% | +0.2 pp |
| Open Source Model | 3 | 0.2% | 0.1% | +0.1 pp |
| Xai | 2 | 0.1% | 0.0% | +0.1 pp |
| Ibm | 7 | 0.4% | 0.4% | +0.1 pp |
| Cohere | 1 | 0.1% | 0.0% | +0.1 pp |
| Databricks | 3 | 0.2% | 0.2% | +0.0 pp |
| Mistral | 0 | 0.0% | 0.0% | +0.0 pp |
| Huggingface | 0 | 0.0% | 0.0% | +0.0 pp |
| Pinecone | 0 | 0.0% | 0.0% | +0.0 pp |
| Arm | 3 | 0.2% | 0.3% | -0.1 pp |
| Uk Ai | 0 | 0.0% | 0.3% | -0.3 pp |
| Nvidia | 38 | 2.4% | 3.1% | -0.6 pp |
| OpenAI | 38 | 2.4% | 3.8% | -1.3 pp |

## Market segment comparison

| Market segment | Lifetime reports | 2025 reports | 2025 AI mention % | 2025 adoption % | 2025 risk % | 2025 vendor % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Main Market (FTSE 100 only) | 1,359 | 223 | 84.8% | 79.8% | 68.6% | 31.8% |
| Main Market (FTSE 350 only) | 3,638 | 606 | 82.8% | 61.2% | 63.0% | 28.9% |
| Main Market | 4,166 | 703 | 77.5% | 55.6% | 57.8% | 25.6% |
| AIM | 1,414 | 171 | 36.8% | 30.4% | 6.4% | 9.4% |

## CNI-only versus all companies

| Scope | Lifetime reports | Lifetime companies | 2025 reports | 2025 AI mention % | 2025 adoption % | 2025 risk % | 2025 vendor % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 9,821 | 1,362 | 1,561 | 65.5% | 45.2% | 41.2% | 19.2% |
| cniOnly | 7,340 | 970 | 1,172 | 66.3% | 44.0% | 41.6% | 19.4% |

## CNI sector summary (2025)

| Sector | Companies | 2025 reports | AI mention % | AI risk % | No AI-risk % | AI mention YoY | AI risk YoY |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Government | 20 | 23 | 87.0% | 60.9% | 39.1% | +3.6 pp | +2.5 pp |
| Communications | 28 | 29 | 86.2% | 51.7% | 48.3% | +14.0 pp | +26.7 pp |
| Chemicals | 34 | 24 | 62.5% | 50.0% | 50.0% | +25.7 pp | +21.1 pp |
| Water | 18 | 17 | 64.7% | 47.1% | 52.9% | +2.8 pp | +13.7 pp |
| Finance | 461 | 657 | 74.7% | 47.0% | 53.0% | +9.7 pp | +10.7 pp |
| Defence | 20 | 22 | 77.3% | 40.9% | 59.1% | +27.3 pp | +14.0 pp |
| Food | 52 | 70 | 62.9% | 40.0% | 60.0% | +17.3 pp | +8.4 pp |
| Other | 392 | 389 | 63.2% | 39.8% | 60.2% | +10.6 pp | +12.9 pp |
| Transport | 61 | 67 | 73.1% | 38.8% | 61.2% | +22.5 pp | +12.3 pp |
| Health | 111 | 100 | 50.0% | 37.0% | 63.0% | +2.3 pp | +7.3 pp |
| Energy | 141 | 140 | 30.7% | 20.0% | 80.0% | +8.3 pp | +5.9 pp |
| Data Infrastructure | 22 | 20 | 60.0% | 10.0% | 90.0% | +0.0 pp | +5.0 pp |
| Civil Nuclear | 2 | 3 | 0.0% | 0.0% | 100.0% | +0.0 pp | +0.0 pp |

### Largest CNI sector rises in AI-risk disclosure (2024 -> 2025)

| Sector | 2024 reports | 2025 reports | 2025 AI risk % | YoY change |
| :--- | ---: | ---: | ---: | ---: |
| Communications | 36 | 29 | 51.7% | +26.7 pp |
| Chemicals | 38 | 24 | 50.0% | +21.1 pp |
| Defence | 26 | 22 | 40.9% | +14.0 pp |
| Other | 467 | 389 | 39.8% | +12.9 pp |
| Transport | 83 | 67 | 38.8% | +12.3 pp |
| Finance | 767 | 657 | 47.0% | +10.7 pp |
| Food | 79 | 70 | 40.0% | +8.4 pp |
| Health | 111 | 100 | 37.0% | +7.3 pp |
| Energy | 156 | 140 | 20.0% | +5.9 pp |
| Data Infrastructure | 20 | 20 | 10.0% | +5.0 pp |
| Government | 24 | 23 | 60.9% | +2.5 pp |

### Largest CNI-sector AI-risk blind spots (2025)

| Sector | 2025 reports | No AI-risk reports | No AI-risk % |
| :--- | ---: | ---: | ---: |
| Data Infrastructure | 20 | 18 | 90.0% |
| Energy | 140 | 112 | 80.0% |
| Health | 100 | 63 | 63.0% |
| Transport | 67 | 41 | 61.2% |
| Other | 389 | 234 | 60.2% |
| Food | 70 | 42 | 60.0% |
| Defence | 22 | 13 | 59.1% |
| Finance | 657 | 348 | 53.0% |
| Chemicals | 24 | 12 | 50.0% |
| Communications | 29 | 14 | 48.3% |
| Government | 23 | 9 | 39.1% |

## ISIC industries with strongest AI-risk disclosure

Minimum sample filter: 20 reports in 2025.

### Highest AI-risk disclosure rates

| ISIC industry | 2025 reports | 2025 risk reports | 2025 AI risk % | YoY change |
| :--- | ---: | ---: | ---: | ---: |
| Other monetary intermediation | 57 | 44 | 77.2% | +15.9 pp |
| Web search portals activities and other information service activities | 23 | 14 | 60.9% | +12.6 pp |
| Computer consultancy and computer facilities management activities | 21 | 12 | 57.1% | +11.0 pp |
| Other financial service activities n.e.c., except insurance and pension funding activities | 25 | 12 | 48.0% | +3.6 pp |
| Construction of residential and non-residential buildings | 20 | 9 | 45.0% | +8.0 pp |
| Activities of non-money market investments funds | 441 | 198 | 44.9% | +11.2 pp |
| Real estate activities with own or leased property | 89 | 39 | 43.8% | +25.1 pp |
| Manufacture of pharmaceuticals, medicinal chemical and botanical products | 42 | 15 | 35.7% | +9.2 pp |
| Other computer programming activities | 28 | 10 | 35.7% | +18.1 pp |
| Other credit granting activities | 20 | 6 | 30.0% | -6.4 pp |
| Electric power generation activities from renewable sources | 21 | 6 | 28.6% | +14.9 pp |
| Activities of holding companies | 25 | 5 | 20.0% | -1.4 pp |
| Extraction of crude petroleum | 31 | 5 | 16.1% | +0.3 pp |
| Research and experimental development on natural sciences and engineering | 22 | 3 | 13.6% | -2.2 pp |
| Mining of other non-ferrous metal ores | 66 | 5 | 7.6% | +1.6 pp |

### Fastest-rising AI-risk disclosure rates

| ISIC industry | 2025 reports | 2025 AI risk % | YoY change |
| :--- | ---: | ---: | ---: |
| Real estate activities with own or leased property | 89 | 43.8% | +25.1 pp |
| Other computer programming activities | 28 | 35.7% | +18.1 pp |
| Other monetary intermediation | 57 | 77.2% | +15.9 pp |
| Electric power generation activities from renewable sources | 21 | 28.6% | +14.9 pp |
| Web search portals activities and other information service activities | 23 | 60.9% | +12.6 pp |
| Activities of non-money market investments funds | 441 | 44.9% | +11.2 pp |
| Computer consultancy and computer facilities management activities | 21 | 57.1% | +11.0 pp |
| Manufacture of pharmaceuticals, medicinal chemical and botanical products | 42 | 35.7% | +9.2 pp |
| Construction of residential and non-residential buildings | 20 | 45.0% | +8.0 pp |
| Other financial service activities n.e.c., except insurance and pension funding activities | 25 | 48.0% | +3.6 pp |
| Mining of other non-ferrous metal ores | 66 | 7.6% | +1.6 pp |
| Extraction of crude petroleum | 31 | 16.1% | +0.3 pp |
| Activities of holding companies | 25 | 20.0% | -1.4 pp |
| Research and experimental development on natural sciences and engineering | 22 | 13.6% | -2.2 pp |
| Other credit granting activities | 20 | 30.0% | -6.4 pp |

## Signal quality

Risk signal strength is based on label-level assignments, not unique reports. Risk substantiveness is reported at report level.

### Risk signal mix (2024 vs 2025)

| Signal level | 2024 share | 2025 share | YoY change | 2025 assignments |
| :--- | ---: | ---: | ---: | ---: |
| 3-explicit | 30.6% | 33.2% | +2.5 pp | 1,516 |
| 2-strong_implicit | 29.8% | 28.8% | -1.0 pp | 1,317 |
| 1-weak_implicit | 39.6% | 38.0% | -1.5 pp | 1,739 |

### Risk substantiveness mix (2024 vs 2025)

| Band | 2024 share | 2025 share | YoY change | 2025 reports |
| :--- | ---: | ---: | ---: | ---: |
| substantive | 11.0% | 10.4% | -0.6 pp | 67 |
| moderate | 79.5% | 78.5% | -1.0 pp | 505 |
| boilerplate | 9.5% | 11.0% | +1.5 pp | 71 |

## Notes for follow-up analysis

- The artifact already supports strong report-level findings for annual trends, CNI sectors, market segments, and ISIC risk rates.
- The cleanest candidate metrics for later dashboard work are the ones that are already robust here: report-level rates, YoY percentage-point changes, blind-spot rates, segment gaps, and quality-adjusted risk rates.
- If we later want sector-level unique adoption or vendor rates, the artifact will need unique report counts by sector for those dimensions rather than label-assignment counts.

## Additional analyses to run next

- Company transition analysis: How many firms move from no AI disclosure to adoption, then from adoption to risk, and how many remain stuck in general / ambiguous language? Requires company-year panel data or regeneration from raw report rows; not recoverable from the current dashboard artifact alone.
- Persistence analysis: Once a company starts mentioning AI risk or reaches substantive disclosure, does it keep doing so in later years? Requires company-year panel data.
- Quality-adjusted sector analysis: Which sectors produce substantive risk disclosure rather than merely mentioning AI risk? Partly supported now at aggregate level; best done with sector-level substantive report counts.
- Over-index / under-index analysis: Which sectors and market segments disclose AI risk above or below the overall baseline once normalized? Supported now from the current artifact.
- Pre/post inflection analysis: Does the 2023 -> 2024 break look like a slope change or a level shift, consistent with a ChatGPT / anticipatory Provision 29 shock? Supported now from annual series.
- Boilerplate / staleness tracking: Are firms repeating the same AI-risk language year after year, or materially updating it? Requires company-level text history; this is one of the highest-value next analyses for the paper.
- Adoption-quality analysis: Are adoption disclosures becoming more operationally specific, or merely more common? Requires extending substantiveness scoring to adoption chunks.

## Recommended headline outputs for the paper

- A 2021-2025 core metrics table with percentage-point changes.
- A quality-gap table showing risk mention rate versus substantive risk rate.
- A CNI sector blind-spot table.
- A market-segment comparison centered on FTSE 100 versus AIM.
- A company transition analysis showing movement from no disclosure to adoption to risk.
