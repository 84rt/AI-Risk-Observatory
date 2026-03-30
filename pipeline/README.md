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
- Database: `pipeline/pipeline/data/airo.db`
Note: classifier API calls log prompt/response char counts and token estimates at DEBUG in `data/logs/pipeline/classifier_runs/*.log`.

## Metadata surfaces

The main metadata file for the whole company-year universe is:
- `data/reference/target_manifest.csv`

This is the closest thing to a master manifest for the database:
- one row per `(company, fiscal_year)`
- includes companies/years even when no consolidated markdown file exists yet

At a high level, `target_manifest.csv` contains:
- company identifiers and names
- legacy and refined market-segment fields
- CNI and ISIC classification fields
- fiscal-year fields and fiscal-year provenance
- FR filing linkage/status fields
- Companies House date/jurisdiction fields

The metadata file for the actual consolidated markdown corpus is:
- `data/FR_consolidated/metadata.csv`

This is a subset of the broader manifest:
- one row per retained filing/document in `data/FR_consolidated/`
- only includes filings with a consolidated markdown file on disk

At a high level, `data/FR_consolidated/metadata.csv` contains:
- filing/document identifiers
- company identifiers and names
- refined market-segment fields
- CNI and ISIC classification fields
- release-year / fiscal-year / filing-type fields
- title, source, and source-path fields

Reference mapping tables used to populate those metadata files:
- `data/reference/market_segments_refined.csv` — canonical refined market mapping (`Main Market`, `AIM`, `AQSE`, `Other`)
- `data/reference/company_cni_sectors.csv` — canonical company-level CNI/ISIC mapping
- `data/reference/cni_distribution_summary.csv` — derived sector distribution summary, not a row-level metadata source

Notes:
- `market_segment_refined` is the canonical market field for downstream work.
- `cni_sector_primary` is the canonical single-label CNI field for grouping and charts.
- `cni_sectors` preserves multi-label CNI assignments when present.
- `cni_isic_ambiguous=true` means the same ISIC code maps to multiple primary CNI sectors in the current universe, so the CNI assignment should be treated as company-level rather than a universal ISIC rule.

## Reconcile LLM vs Human Annotations (from a testbed run)

Use this when you’ve run `classifier_testbed.py` and want to compare/reconcile
LLM labels against the current human baseline **without** re-running the LLM.

1) Export the testbed run into LLM-annotations format:
   This export is required for each new testbed run (`<run_id>`).

```bash
python3 scripts/export_testbed_run_for_reconcile.py \
  --testbed-run data/testbed_runs/<run_id>.jsonl \
  --human data/golden_set/human_reconciled/annotations.jsonl \
  --output-dir data/golden_set/llm/<run_id> \
  --confidence-mode uniform
```

2) Reconcile (interactive). Tip: use `--only-disagreements` and `--resume`:

```bash
python3 scripts/reconcile_annotations.py \
  --human data/golden_set/human_reconciled/annotations.jsonl \
  --llm data/golden_set/llm/<run_id>/annotations.jsonl \
  --output-dir data/golden_set/reconciled/<run_id> \
  --only-disagreements \
  --max-chunks 50 \
  --resume
```

If you only want to reconcile one field (and preserve the rest of the human baseline),
add `--focus-field` with one of: `mention_types`, `adoption`, `risk`, or `vendor`.

```bash
python3 scripts/reconcile_annotations.py \
  --human data/golden_set/human_reconciled/annotations.jsonl \
  --llm data/golden_set/llm/<run_id>/annotations.jsonl \
  --output-dir data/golden_set/reconciled/<run_id> \
  --only-disagreements \
  --resume \
  --focus-field vendor
```

3) Merge reconciled updates into the human baseline:

```bash
python3 scripts/merge_reconciled_golden_set.py \
  --human data/golden_set/human_reconciled/annotations.jsonl \
  --reconciled data/golden_set/reconciled/<run_id>/annotations.jsonl \
  --output data/golden_set/human_reconciled/annotations.jsonl
```

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

---

## Known Issues / To Investigate

### CH gap-fill: 87 filings with no CH annual accounts found (2025-03-29)

When running `scripts/patch_ch_gaps.py --phase download_ch --include-pending`, 87 out of 129
`needs_ch` items failed with "no CH filing found". These are **not code bugs** — they fall into
two categories:

**1. Foreign-incorporated companies** (GLEIF returns a non-UK registration number):
`CNH Industrial N.V.` (NL), `Pepco Group N.V.` (NL), `BenevolentAI` (BE),
`Prairie Mining Ltd.` (AU), `Linde PLC` (IE), `Glencore PLC` (JE),
`Flex LNG` (BM), `Panther Metals PLC` (non-standard `009753V`).
Their `registeredAs` from GLEIF is a home-country number that CH doesn't recognise.

**2. UK-incorporated vehicles that don't file standard AA accounts on CH**:
Investment trusts (`Schroder Real Estate`, `Schroder Oriental Income`, `Henderson Far East Income`,
`Picton Property Income`), holding/management companies (`Worsley Investors`, `DCI Advisors`,
`Raymond James Wealth Management`), and smaller vehicles (`Fragrant Prosperity Holdings`,
`AdvancedAdvT`, `Vinanz`).

Affected CH numbers: `008 677 852`, `009753V`, `00039117`, `00041959`, `00043007`, `00043298`,
`00043673`, `00045582`, `00052644`, `00060527`, `00064865`, `00074891`, `00081792`, `00095064`,
`00098465`, `00107710`, `00111714`, `00126344`, `00268512`, `00444750`, `00606357`, `00669758`,
`01905051`, `02040954`, `02073995`, `56532474`, `81928491`, `B255412`.

Possible future actions:
- For foreign-incorporated companies: find PDF directly from company IR page or stock exchange filing.
- For investment trusts: check if they file accounts under a different CH category.
