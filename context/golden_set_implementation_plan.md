# Golden Set Implementation Plan (AISI-Updated)

## Goals
- Build a minimal, reliable pipeline for the initial sample.
- Prioritize preprocessing quality and AI-mention chunking.
- Ensure outputs are saved consistently in the `/data` structure.
- Design the database schema to support later classification work.

## Starting Sample (Updated)
We have a new starting set of companies (about 13). We may add a helper to pull filings by identifier (LEI / Companies House), but manual ingestion is acceptable for the initial sample.

If manual:
1. Download iXBRL filings for each company and year.
2. Place raw files in `/data/raw/` using a consistent naming scheme.
3. Create a small metadata file (company, year, identifier, source).

### New Company List
| ID | Company | Sector | Index | Type |
| -- | ------- | ------ | ----- | ---- |
| 1 | Croda International plc | Chemicals | FTSE 100 | Best proxy |
| 2 | Rolls-Royce Holdings plc | Civil Nuclear/Space | FTSE 100 | Best proxy |
| 3 | BT Group plc | Communications | FTSE 100 | Direct |
| 4 | BAE Systems plc | Defence | FTSE 100 | Direct |
| 5 | Serco Group plc | Government Services | FTSE 250 | Best proxy |
| 6 | Shell plc | Energy (Extraction) | FTSE 100 | Direct |
| 7 | Lloyds Banking Group plc | Finance (Banking) | FTSE 100 | Direct |
| 8 | Tesco plc | Food (Retail) | FTSE 100 | Direct |
| 9 | AstraZeneca plc | Health (Pharma) | FTSE 100 | Direct |
| 10 | National Grid plc | Energy (Transmission) | FTSE 100 | Direct |
| 11 | Severn Trent plc | Water | FTSE 100 | Direct |
| 12 | Aviva plc | Insurance | FTSE 100 | Direct |
| 13 | Schroders plc | Asset Management | FTSE 100 | Direct |
| 14 | FirstGroup plc | Transport | FTSE 250 | Direct |
| 15 | Clarkson plc | Shipping | FTSE 250 | Direct |

## Data Layout
- `/data/raw/` for original iXBRL files and download metadata.
- `/data/processed/` for cleaned, readable markdown outputs.
- `/data/processed/chunks/` for AI-mention chunks (initial processing output).
- `/data/results/` reserved for classifier outputs later.

## Phase 1: Preprocessing (Primary Focus)
Goal: Convert raw iXBRL filings into readable markdown.

### Requirements
- Preserve headings and section boundaries as much as possible.
- Normalize whitespace and remove boilerplate where it hurts readability.
- Capture basic structure metadata (section titles, page numbers if available).

### Outputs
- One markdown file per report in `/data/processed/`.
- A lightweight JSON sidecar with:
  - source file path
  - company identifier
  - year
  - extraction method/version
  - section and page mapping (if available)

## Phase 2: Processing (Chunking AI Mentions)
Goal: Convert full markdown into a set of AI-mention chunks with enough context for later classifiers.

### Chunking Rules
- Detect AI mentions (AI, artificial intelligence, ML, machine learning, LLM, large language model, etc.).
- Create a chunk from:
  - the paragraph containing the mention
  - plus one paragraph before and after (configurable)
- Save multiple mentions in the same sentence as a single chunk.
- Store matched terms as a list (do not split into separate chunks yet).

### Metadata to Save
- Document identifier and file path
- Section title and page number(s) if available
- Chunk text
- Matched terms (list)
- Character offsets in the markdown (start/end)
- Context window size (paragraphs before/after)

### Output
- One chunk record per AI-mention chunk in `/data/processed/chunks/`.
- Each chunk is also stored in the database for later classification.

## Phase 3: Classification (Deferred)
Classification comes after we trust preprocessing and chunking. Later stages will:
- Group AI types (non-LLM, LLM, agentic)
- Apply risk taxonomy and substantiveness scoring
- Add harms tagging if present

## Database Schema Considerations
We need a schema that cleanly separates:

1. Documents (raw filings)
   - company identifier, year, source
   - raw file path, checksum, download metadata
2. Processed Documents (markdown)
   - processed file path, extraction version
   - section/page map
3. Chunks (AI-mention contexts)
   - document_id (FK)
   - section title, page range
   - chunk text
   - matched terms (array or JSON)
   - offsets and context window metadata
4. Future Classifications (later)
   - chunk_id (FK)
   - classifier type, version, outputs, confidence

This structure keeps preprocessing outputs stable and allows later classifiers to evolve without re-ingesting raw data.