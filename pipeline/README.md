# Pipeline

This directory houses the pipeline source, scripts, and tests. Full docs live in `context/pipeline_docs/` (architecture, guides, and status notes).

## Layout
- `src/` — core pipeline code
- `scripts/` — helper/maintenance CLIs (diagnostics, downloads, setup)
- `tests/` — test entrypoints, fixtures, and QA suite

## Data locations
Pipeline reads/writes under the repo-level `data/` tree:
- Inputs/reference: `data/reference/`
- Raw files: `data/raw/ixbrl/{year}/`, `data/raw/pdfs/{year}/`
- Processed markdown: `data/processed/<run_id>/documents/*.md`
- Metadata sidecars: `data/processed/<run_id>/metadata/*.json`
- Processed manifest: `data/processed/<run_id>/documents_manifest.json`
- AI-mention chunks: `data/processed/<run_id>/chunks/chunks.jsonl`
- Results and annotations: `data/results/`, `data/annotations/`
- Database: `data/db/airo.db`
Note: classifier API calls log prompt/response char counts and token estimates at DEBUG in `data/logs/pipeline/classifier_runs/*.log`.

## Golden Set Pipeline

The golden set pipeline has three phases:

### Phase 1: Ingestion (Complete)
Raw iXBRL files are downloaded and placed in `data/raw/ixbrl/{year}/`.

```bash
python scripts/golden_set_phase1.py --download --years 2024 2023
```

### Phase 2: Preprocessing (Complete)
Converts iXBRL to readable markdown with metadata sidecars.

```bash
python scripts/golden_set_phase1.py --preprocess --years 2024 2023
```

Outputs:
- `data/processed/<run_id>/documents/*.md` — full markdown
- `data/processed/<run_id>/metadata/*.json` — metadata sidecars
- `data/processed/<run_id>/documents_manifest.json` — manifest of processed docs

The script prints the `run_id` it used; reuse it for chunking and QA.

### Phase 3: Chunking (In Progress)
Extracts AI-mention chunks with context window (2 paragraphs before/after by default). Deduplicates overlapping windows.

```bash
python scripts/chunk_markdown.py --run-id <run_id>
```

Outputs:
- `data/processed/<run_id>/chunks/chunks.jsonl` — one chunk per line

### Phase 4: Human Annotation (Next)
After chunking, chunks are ready for human annotation. See annotation guidelines in `context/golden_set_implementation_plan.md`.

### QA Checks
Run QA after each pipeline step:

```bash
python tests/qa_manager.py --run-id <run_id> --stage all
```
