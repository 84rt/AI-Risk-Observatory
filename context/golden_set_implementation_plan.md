# Golden Set Implementation Plan (AISI-Updated)

## Goals
- Build a minimal, reliable pipeline for the initial sample.
- Prioritize preprocessing quality and AI-mention chunking.
- Ensure outputs are saved consistently in the `/data` structure.
- Design the database schema to support later classification work.

## Current Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Ingestion | Complete | Raw iXBRL files in `data/raw/ixbrl/{year}/` |
| Phase 2: Preprocessing | Complete | Markdown + JSON sidecars in `data/processed/<run_id>/documents/` |
| Phase 3: Chunking | In Progress | AI-mention chunks with 2+2 context window, JSONL output |
| Phase 4: Human Annotation | Next | Guidelines below |
| Phase 5: Classification | Deferred | LLM classification after annotation |

## Starting Sample (Updated)
We have a new starting set of companies. We may add a helper to pull filings by identifier (LEI / Companies House), but manual ingestion is acceptable for the initial sample.

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
- `/data/processed/<run_id>/documents/` for cleaned, readable markdown outputs.
- `/data/processed/<run_id>/metadata/` for JSON sidecars per document.
- `/data/processed/<run_id>/documents_manifest.json` for processed document index.
- `/data/processed/<run_id>/chunks/` for AI-mention chunks (JSONL).
- `/data/results/` reserved for classifier outputs later.

## Phase 1: Preprocessing (Primary Focus)
Goal: Convert raw iXBRL filings into readable markdown.

### Requirements
- Preserve headings and section boundaries as much as possible.
- Normalize whitespace and remove boilerplate where it hurts readability.
- Capture basic structure metadata (section titles, page numbers if available).

### Outputs
- One markdown file per report in `/data/processed/<run_id>/documents/`.
- A lightweight JSON sidecar with:
  - source file path
  - company identifier
  - year
  - extraction method/version
  - section and page mapping (if available)
  - preprocessing stats (retention, spans, sections)
- A `documents_manifest.json` index for all processed outputs in the run.

### Quality Assurance
- Use the QA manager in `pipeline/tests/qa_manager.py` to run preprocessing checks (warn-only).

## Phase 2: Processing (Chunking AI Mentions)
Goal: Convert full markdown into a set of AI-mention chunks with enough context for later classifiers.

### Chunking Rules
- Detect AI mentions (AI, artificial intelligence, ML, machine learning, LLM, large language model, etc.).
- Create chunks from processed markdown (source of truth for later classifiers).
- Use a context window of 2 paragraphs before and after by default (configurable).
- Deduplicate overlapping mention windows so each chunk is unique, even with multiple mentions.
- Store matched terms as a list per chunk (do not split into separate chunks yet).

### Metadata to Save
- Document identifier and file path
- Section title and page number(s) if available
- Chunk text
- Matched terms (list)
- Character offsets in the markdown (start/end)
- Context window size (paragraphs before/after)

### Output
- One chunk record per AI-mention chunk in `/data/processed/<run_id>/chunks/chunks.jsonl`.
- Each chunk can be stored in the database later for classification.

## Phase 3: Post-processing (Classification & Enrichment) (Deferred)
Post-processing comes after we trust preprocessing and chunking. Later stages will:
- LLM classification (AI type, risk taxonomy, substantiveness, confidence levels on classifications etc.)
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


# To make a new run

  ./pipeline/venv/bin/python pipeline/scripts/golden_set_phase1.py --all

  That auto-generates a new run_id and writes data/runs/<run_id>/ingestion.json plus processed outputs in data/processed/<run_id>/....

  If you only want a new run id without download/preprocess:

  ./pipeline/venv/bin/python pipeline/scripts/golden_set_phase1.py --run-id <your-id>