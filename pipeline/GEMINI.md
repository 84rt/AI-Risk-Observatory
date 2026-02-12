# Gemini CLI Notes

## Maintenance & Audit Scripts
Scripts created by Gemini for data auditing, chunking comparisons, and database mapping are located in:
`pipeline/scripts/`

### Key Audit Scripts:
- `run_audit_comparison.py`: General stats and side-by-side comparison of FR vs iXBRL.
- `generate_aviva_dumps.py`: Generates full chunk text dumps for manual inspection.
- `map_filings.py`: Maps new company targets to the FinancialReports database.
- `diagnostic_chunk_count.py`: Validates chunking algorithm consistency across sources.

## Data Artifacts
Comparison outputs and audit logs are stored in:
`data/results/comparison_audit/`
