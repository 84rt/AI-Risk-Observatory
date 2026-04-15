# Dashboard Findings

Generated on 2026-04-13 from `data/dashboard-data.json`.

## Scope and caveats

- This report uses the existing precomputed dashboard artifact only. It does not add any new metrics to the dashboard.
- 2026 is present in the artifact, but because the current date is 2026-04-13, 2026 should be treated as a partial year. The main YoY comparisons below therefore use 2024 -> 2025.
- Tables based on `riskTrend`, `adoptionTrend`, and `vendorTrend` are label-assignment counts, not unique-report counts. A single report can contribute to multiple labels.
- Sector-level unique adoption and vendor rates are not derivable from the current artifact because the available sector arrays for adoption and vendor are label-level counts rather than unique report counts.

## Recommended comparison windows

- Main research window: `2021 -> 2025`. This matches the paper draft's primary analysis period and avoids over-weighting the partial 2020 and partial 2026 edges of the series.
- Supporting long-run window: `2020 -> 2025`. This is useful as a start-of-series anchor and for communicating scale of change since the earliest observable baseline.
- Directional snapshot only: `2026`. Because the current date is 2026-04-13, 2026 should be treated as a partial-year directional signal, not as a directly comparable full-year endpoint.

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

- The current artifact covers 9,821 reports across 1,362 companies from 2020-2026. 41.5% of all reports contain at least one non-`none` AI signal.
- In 2025, 65.5% of reports mentioned AI at all, up +10.4 pp from 2024; AI-risk disclosure rose even faster to 43.2% (+11.3 pp).
- The adoption-versus-risk disclosure gap narrowed from 6.6 pp in 2024 to 3.0 pp in 2025, suggesting risk disclosure is catching up with general AI adoption language.
- Strategic / Competitive was the fastest-rising risk category in 2025, increasing +8.8 pp to 29.3% of all reports. Cybersecurity and Operational / Technical risk were close behind.
- LLM disclosure was the fastest-rising adoption category, reaching 35.7% of all reports in 2025 (+11.1 pp YoY). Agentic references also rose to 11.0%.
- Main Market (FTSE 100 only) had the highest AI-risk rate in 2025 at 71.3%, while AIM remained far lower at 7.0%.
- Among CNI sectors with at least 20 reports in 2025, Communications saw the biggest rise in AI-risk disclosure (+24.6 pp), while Data Infrastructure had the largest remaining AI-risk blind spot (85.0% of reports still without an AI-risk mention).
- Vendor references remain fragmented. The largest vendor bucket in 2025 was `other` at 30.8% of vendor assignments; the leading named vendor was Microsoft at 18.8%.
- In 2025, opaque vendor references (`other` + `undisclosed`) accounted for 42.7% of all vendor assignments. Among explicitly named vendors, the top three accounted for 75.7% of named-vendor assignments.
- Risk disclosures became denser over time: average risk labels per risk-reporting company rose from 3.28 in 2024 to 3.63 in 2025.
- At ISIC level, the strongest large-sample AI-risk disclosure rate in 2025 was in Other monetary intermediation (78.9%; n=57 reports, using a minimum-sample filter of 20).

## Coverage summary

| Reports | Companies | AI signal reports | AI signal rate | Avg phase-1 labels / signal report |
| ---: | ---: | ---: | ---: | ---: |
| 9,821 | 1,362 | 4,078 | 41.5% | 2.12 |

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
| 2020 | 1,007 | 19.5% | 13.9% | 3.2% | 11.1% | 2.8% | 10.7 pp |
| 2021 | 1,328 | 27.2% | 21.5% | 4.4% | 14.7% | 3.1% | 17.0 pp |
| 2022 | 1,853 | 28.4% | 23.4% | 5.0% | 15.2% | 4.5% | 18.4 pp |
| 2023 | 1,905 | 36.9% | 26.8% | 10.2% | 22.4% | 6.2% | 16.6 pp |
| 2024 | 1,828 | 55.1% | 38.5% | 31.8% | 42.9% | 13.3% | 6.6 pp |
| 2025 | 1,561 | 65.5% | 46.2% | 43.2% | 54.6% | 17.4% | 3.0 pp |
| 2026 | 339 | 77.3% | 64.3% | 66.7% | 67.8% | 26.8% | -2.4 pp |

### Core 2021 -> 2025 research metrics

| Metric | 2021 | 2025 | Change |
| :--- | ---: | ---: | ---: |
| Any AI mention rate | 27.2% | 65.5% | +38.4 pp |
| AI adoption rate | 21.5% | 46.2% | +24.7 pp |
| AI risk rate | 4.4% | 43.2% | +38.7 pp |
| General / ambiguous rate | 14.7% | 54.6% | +40.0 pp |
| AI vendor rate | 3.1% | 17.4% | +14.3 pp |
| Substantive risk rate (of all reports) | 0.7% | 4.1% | +3.4 pp |
| Substantive share of risk reports | 15.3% | 9.5% | -5.8 pp |
| Quality gap: risk minus substantive risk | 3.8 pp | 39.1 pp | +35.3 pp |
| Adoption-to-risk ratio | 4.83 | 1.07 | -3.76 |

### Supporting 2020 -> 2025 start-of-series comparison

| Metric | 2020 | 2025 | Change |
| :--- | ---: | ---: | ---: |
| Any AI mention rate | 19.5% | 65.5% | +46.1 pp |
| AI adoption rate | 13.9% | 46.2% | +32.3 pp |
| AI risk rate | 3.2% | 43.2% | +40.0 pp |
| General / ambiguous rate | 11.1% | 54.6% | +43.5 pp |
| Substantive risk rate (of all reports) | 0.5% | 4.1% | +3.6 pp |
| Adoption-to-risk ratio | 4.38 | 1.07 | -3.31 |

### Partial 2026 directional snapshot

2026 is partial and should not be compared to full-year 2025 as if both were complete annual cohorts. This table is included only to show direction of travel.

| Metric | 2025 | 2026 partial |
| :--- | ---: | ---: |
| Reports in sample | 1,561 | 339 |
| Any AI mention rate | 65.5% | 77.3% |
| AI adoption rate | 46.2% | 64.3% |
| AI risk rate | 43.2% | 66.7% |
| Substantive risk rate (of all reports) | 4.1% | 9.4% |
| Adoption-to-risk ratio | 1.07 | 0.96 |

## Quality-gap analysis

This is the most directly policy-relevant metric family in the current dataset: it separates the growth in AI-risk mentions from the much smaller share of reports that contain genuinely substantive AI-risk disclosure.

| Year | Risk reports | Risk rate | Substantive risk reports | Substantive rate | Substantive share of risk reports | Quality gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020 | 32 | 3.2% | 5 | 0.5% | 15.6% | 2.7 pp |
| 2021 | 59 | 4.4% | 9 | 0.7% | 15.3% | 3.8 pp |
| 2022 | 93 | 5.0% | 17 | 0.9% | 18.3% | 4.1 pp |
| 2023 | 194 | 10.2% | 27 | 1.4% | 13.9% | 8.8 pp |
| 2024 | 582 | 31.8% | 60 | 3.3% | 10.3% | 28.6 pp |
| 2025 | 674 | 43.2% | 64 | 4.1% | 9.5% | 39.1 pp |
| 2026 | 226 | 66.7% | 32 | 9.4% | 14.2% | 57.2 pp |

### 2024 -> 2025 headline rate changes

| Metric | 2024 | 2025 | Change | Count change |
| :--- | ---: | ---: | ---: | ---: |
| Any AI mention | 55.1% | 65.5% | +10.4 pp | +15 |
| AI adoption mention | 38.5% | 46.2% | +7.7 pp | +18 |
| AI risk mention | 31.8% | 43.2% | +11.3 pp | +92 |
| AI vendor mention | 13.3% | 17.4% | +4.0 pp | +27 |
| Adoption-risk gap | 6.6 pp | 3.0 pp | -3.6 pp | n/a |

## Risk taxonomy findings

- Risk-category counts are label assignments, so totals exceed the number of unique risk-reporting companies.
- In 2025, the top three risk categories accounted for 49.7% of all risk-label assignments. The HHI for risk-category concentration was 1,292.
- Average risk labels per risk-reporting report rose from 3.28 in 2024 to 3.63 in 2025.

| Risk category | 2025 count | 2025 % of reports | 2024 % of reports | YoY change |
| :--- | ---: | ---: | ---: | ---: |
| Strategic / Competitive | 458 | 29.3% | 20.6% | +8.8 pp |
| Cybersecurity | 405 | 25.9% | 17.9% | +8.1 pp |
| Operational / Technical | 352 | 22.5% | 15.9% | +6.6 pp |
| Workforce Impacts | 176 | 11.3% | 5.8% | +5.5 pp |
| Regulatory / Compliance | 322 | 20.6% | 15.3% | +5.4 pp |
| Reputational / Ethical | 252 | 16.1% | 10.9% | +5.2 pp |
| Third-Party Supply Chain | 197 | 12.6% | 7.4% | +5.2 pp |
| Information Integrity | 184 | 11.8% | 7.9% | +3.9 pp |
| Environmental Impact | 52 | 3.3% | 1.1% | +2.2 pp |
| National Security | 49 | 3.1% | 1.7% | +1.4 pp |

## Adoption findings

- Adoption-category counts are label assignments, not unique reports.
- In 2025, the top three adoption categories accounted for 100.0% of all adoption-label assignments.
- Average adoption labels per adoption-reporting report rose from 1.76 in 2024 to 1.94 in 2025.

| Adoption category | 2025 count | 2025 % of reports | 2024 % of reports | YoY change |
| :--- | ---: | ---: | ---: | ---: |
| LLM | 557 | 35.7% | 24.6% | +11.1 pp |
| Traditional AI (non-LLM) | 672 | 43.0% | 36.3% | +6.8 pp |
| Agentic | 172 | 11.0% | 6.9% | +4.1 pp |

## Vendor findings

- Vendor-category counts are label assignments, not unique reports.
- In 2025, the top three vendor buckets accounted for 61.4% of all vendor-label assignments. The HHI for vendor concentration was 1,751.
- Average vendor labels per vendor-reporting report rose from 1.69 in 2024 to 1.71 in 2025.

| Vendor bucket | 2025 count | 2025 % of reports | 2024 % of reports | YoY change |
| :--- | ---: | ---: | ---: | ---: |
| Other | 143 | 9.2% | 5.8% | +3.4 pp |
| Undisclosed | 55 | 3.5% | 1.6% | +1.9 pp |
| Internal | 48 | 3.1% | 2.2% | +0.8 pp |
| Google | 42 | 2.7% | 2.0% | +0.7 pp |
| Meta | 18 | 1.2% | 0.7% | +0.5 pp |
| Microsoft | 87 | 5.6% | 5.1% | +0.5 pp |
| Amazon / AWS | 31 | 2.0% | 1.8% | +0.2 pp |
| Anthropic | 4 | 0.3% | 0.1% | +0.1 pp |
| OpenAI | 36 | 2.3% | 3.3% | -1.0 pp |

## Market segment comparison

| Market segment | Lifetime reports | 2025 reports | 2025 AI mention % | 2025 adoption % | 2025 risk % | 2025 vendor % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Main Market (FTSE 100 only) | 1,359 | 223 | 84.8% | 79.8% | 71.3% | 28.7% |
| Main Market (FTSE 350 only) | 3,638 | 606 | 82.8% | 63.2% | 66.5% | 26.9% |
| Main Market | 4,166 | 703 | 77.5% | 57.0% | 60.9% | 23.8% |
| AIM | 1,414 | 171 | 36.8% | 31.0% | 7.0% | 9.4% |

## CNI-only versus all companies

| Scope | Lifetime reports | Lifetime companies | 2025 reports | 2025 AI mention % | 2025 adoption % | 2025 risk % | 2025 vendor % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 9,821 | 1,362 | 1,561 | 65.5% | 46.2% | 43.2% | 17.4% |
| cniOnly | 7,340 | 970 | 1,172 | 66.3% | 45.6% | 43.8% | 17.5% |

## CNI sector summary (2025)

| Sector | Companies | 2025 reports | AI mention % | AI risk % | No AI-risk % | AI mention YoY | AI risk YoY |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Government | 20 | 23 | 87.0% | 60.9% | 39.1% | +3.6 pp | +6.7 pp |
| Communications | 28 | 29 | 86.2% | 55.2% | 44.8% | +14.0 pp | +24.6 pp |
| Water | 18 | 17 | 64.7% | 52.9% | 47.1% | +2.8 pp | +19.6 pp |
| Chemicals | 34 | 24 | 62.5% | 50.0% | 50.0% | +25.7 pp | +21.1 pp |
| Finance | 461 | 657 | 74.7% | 49.5% | 50.5% | +9.9 pp | +10.4 pp |
| Food | 52 | 70 | 62.9% | 44.3% | 55.7% | +17.3 pp | +12.6 pp |
| Transport | 61 | 67 | 73.1% | 41.8% | 58.2% | +22.5 pp | +15.3 pp |
| Other | 392 | 389 | 63.2% | 41.4% | 58.6% | +10.6 pp | +14.2 pp |
| Defence | 20 | 22 | 77.3% | 40.9% | 59.1% | +27.3 pp | +14.0 pp |
| Health | 111 | 100 | 50.0% | 37.0% | 63.0% | +2.3 pp | +7.3 pp |
| Energy | 141 | 140 | 30.7% | 20.7% | 79.3% | +7.6 pp | +5.3 pp |
| Data Infrastructure | 22 | 20 | 60.0% | 15.0% | 85.0% | +0.0 pp | +5.0 pp |
| Civil Nuclear | 2 | 3 | 0.0% | 0.0% | 100.0% | +0.0 pp | +0.0 pp |

### Largest CNI sector rises in AI-risk disclosure (2024 -> 2025)

| Sector | 2024 reports | 2025 reports | 2025 AI risk % | YoY change |
| :--- | ---: | ---: | ---: | ---: |
| Communications | 36 | 29 | 55.2% | +24.6 pp |
| Chemicals | 38 | 24 | 50.0% | +21.1 pp |
| Transport | 83 | 67 | 41.8% | +15.3 pp |
| Other | 467 | 389 | 41.4% | +14.2 pp |
| Defence | 26 | 22 | 40.9% | +14.0 pp |
| Food | 79 | 70 | 44.3% | +12.6 pp |
| Finance | 767 | 657 | 49.5% | +10.4 pp |
| Health | 111 | 100 | 37.0% | +7.3 pp |
| Government | 24 | 23 | 60.9% | +6.7 pp |
| Energy | 156 | 140 | 20.7% | +5.3 pp |
| Data Infrastructure | 20 | 20 | 15.0% | +5.0 pp |

### Largest CNI-sector AI-risk blind spots (2025)

| Sector | 2025 reports | No AI-risk reports | No AI-risk % |
| :--- | ---: | ---: | ---: |
| Data Infrastructure | 20 | 17 | 85.0% |
| Energy | 140 | 111 | 79.3% |
| Health | 100 | 63 | 63.0% |
| Defence | 22 | 13 | 59.1% |
| Other | 389 | 228 | 58.6% |
| Transport | 67 | 39 | 58.2% |
| Food | 70 | 39 | 55.7% |
| Finance | 657 | 332 | 50.5% |
| Chemicals | 24 | 12 | 50.0% |
| Communications | 29 | 13 | 44.8% |
| Government | 23 | 9 | 39.1% |

## ISIC industries with strongest AI-risk disclosure

Minimum sample filter: 20 reports in 2025.

### Highest AI-risk disclosure rates

| ISIC industry | 2025 reports | 2025 risk reports | 2025 AI risk % | YoY change |
| :--- | ---: | ---: | ---: | ---: |
| Other monetary intermediation | 57 | 45 | 78.9% | +17.6 pp |
| Computer consultancy and computer facilities management activities | 21 | 13 | 61.9% | +15.8 pp |
| Web search portals activities and other information service activities | 23 | 14 | 60.9% | +12.6 pp |
| Other financial service activities n.e.c., except insurance and pension funding activities | 25 | 12 | 48.0% | +3.6 pp |
| Activities of non-money market investments funds | 441 | 211 | 47.8% | +10.2 pp |
| Real estate activities with own or leased property | 89 | 41 | 46.1% | +27.4 pp |
| Construction of residential and non-residential buildings | 20 | 9 | 45.0% | +8.0 pp |
| Manufacture of pharmaceuticals, medicinal chemical and botanical products | 42 | 15 | 35.7% | +9.2 pp |
| Other computer programming activities | 28 | 10 | 35.7% | +18.1 pp |
| Other credit granting activities | 20 | 7 | 35.0% | -1.4 pp |
| Electric power generation activities from renewable sources | 21 | 7 | 33.3% | +15.2 pp |
| Activities of holding companies | 25 | 5 | 20.0% | -1.4 pp |
| Extraction of crude petroleum | 31 | 5 | 16.1% | -2.3 pp |
| Research and experimental development on natural sciences and engineering | 22 | 3 | 13.6% | -2.2 pp |
| Mining of other non-ferrous metal ores | 66 | 6 | 9.1% | +3.1 pp |

### Fastest-rising AI-risk disclosure rates

| ISIC industry | 2025 reports | 2025 AI risk % | YoY change |
| :--- | ---: | ---: | ---: |
| Real estate activities with own or leased property | 89 | 46.1% | +27.4 pp |
| Other computer programming activities | 28 | 35.7% | +18.1 pp |
| Other monetary intermediation | 57 | 78.9% | +17.6 pp |
| Computer consultancy and computer facilities management activities | 21 | 61.9% | +15.8 pp |
| Electric power generation activities from renewable sources | 21 | 33.3% | +15.2 pp |
| Web search portals activities and other information service activities | 23 | 60.9% | +12.6 pp |
| Activities of non-money market investments funds | 441 | 47.8% | +10.2 pp |
| Manufacture of pharmaceuticals, medicinal chemical and botanical products | 42 | 35.7% | +9.2 pp |
| Construction of residential and non-residential buildings | 20 | 45.0% | +8.0 pp |
| Other financial service activities n.e.c., except insurance and pension funding activities | 25 | 48.0% | +3.6 pp |
| Mining of other non-ferrous metal ores | 66 | 9.1% | +3.1 pp |
| Other credit granting activities | 20 | 35.0% | -1.4 pp |
| Activities of holding companies | 25 | 20.0% | -1.4 pp |
| Research and experimental development on natural sciences and engineering | 22 | 13.6% | -2.2 pp |
| Extraction of crude petroleum | 31 | 16.1% | -2.3 pp |

## Signal quality

Risk signal strength is based on label-level assignments, not unique reports. Risk substantiveness is reported at report level.

### Risk signal mix (2024 vs 2025)

| Signal level | 2024 share | 2025 share | YoY change | 2025 assignments |
| :--- | ---: | ---: | ---: | ---: |
| 3-explicit | 31.6% | 31.2% | -0.4 pp | 1,516 |
| 2-strong_implicit | 30.7% | 30.3% | -0.4 pp | 1,473 |
| 1-weak_implicit | 37.7% | 38.5% | +0.8 pp | 1,870 |

### Risk substantiveness mix (2024 vs 2025)

| Band | 2024 share | 2025 share | YoY change | 2025 reports |
| :--- | ---: | ---: | ---: | ---: |
| substantive | 10.3% | 9.5% | -0.8 pp | 64 |
| moderate | 78.0% | 80.7% | +2.7 pp | 544 |
| boilerplate | 11.7% | 9.8% | -1.9 pp | 66 |

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
