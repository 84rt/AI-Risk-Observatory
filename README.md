# AI Risk Observatory (AIRO)

The AI Risk Observatory tracks how UK public companies are adopting, governing, and disclosing risks related to Artificial Intelligence.

**Components:**
- **Data Pipeline:** Ingests annual reports, extracts AI-relevant text, and uses LLMs to tag and classify mentions using a risk taxonomy.
- **Dashboard:** Next.js web application to visualize risk disclosure trends by sector, risk category, time, and governance maturity.

**Goals:**
- Identify emerging AI threats and disclosure patterns.
- Monitor gaps between AI adoption and governance.
- Enable regulators and stakeholders to focus on high-risk sectors.

**Key Questions:**
- How reliably can LLMs identify/classify AI risk disclosures in financial filings?
- What are the trends in AI risk severity, nature, and mitigation across sectors?

**Architecture:**
- **Pipeline:** Processes annual reports, extracts content, applies a multi-stage LLM filter for relevance and classification, outputs structured data to SQL (SQLite/PostgreSQL).
- **Dashboard:** Built with Next.js and Recharts/Tremor, offers visualizations (e.g., risk composition, governance maturity, sector trends).

**Repo Structure:**
- `/airo-dashboard`: Next.js frontend
- `/pipeline`: (Planned) Python ETL/classification scripts
- `/data`: Mock data and schemas
- `AIRO_DATA_PIPELINE_SPECIFICATION.md`: Data model specification

