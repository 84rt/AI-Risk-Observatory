# Classifier Testbed

Interactive notebook-style script for evaluating LLM classifiers against human-annotated baselines.

**Location:** `pipeline/scripts/classifier_testbed.py`

## Quick Start

```bash
cd pipeline
source venv/bin/activate
# Open in VS Code with Python extension or Jupyter
```

Run cells in order:
1. **SETUP** - Imports and paths
2. **FUNCTIONS** - Helper functions (collapse after running)
3. **LOAD DATA** - Load golden set chunks
4. **CONFIG** - Set model parameters
5. **RUN** - Execute classifier (sync or batch)
6. **ANALYSIS** - View results

## Two Modes

### Sync Mode (immediate results)
```python
results = run_classifier(RUN_ID, chunks, MODEL_NAME, TEMPERATURE, THINKING_BUDGET)
```

### Batch Mode (50% cost, async)
```python
from src.utils.batch_api import BatchClient
batch = BatchClient(runs_dir=RUNS_DIR)

# Submit
requests = batch.prepare_requests(chunks, temperature=0.0)
job_name = batch.submit("my-run", requests, model_name="gemini-2.0-flash")

# Check later
batch.check_status(job_name)

# Get results when done
results = batch.get_results(job_name, chunks)
```

## Key Functions

| Function | Description |
|----------|-------------|
| `load_chunks(limit)` | Load N chunks from golden set |
| `run_classifier(...)` | Run sync classification |
| `show_summary(results)` | Print EXACT/PARTIAL/DIFF counts |
| `show_table(results, diff_only)` | Comparison table |
| `show_details(results, indices)` | Detailed view with reasoning |
| `show_diff_summary(results)` | Label transition analysis |

## Data Paths

- **Golden set:** `data/golden_set/human_reconciled/annotations.jsonl`
- **Run cache:** `data/testbed_runs/<run_id>.jsonl`
- **Run metadata:** `data/testbed_runs/<run_id>.meta.json`

## Config Parameters

```python
RUN_ID = "my-experiment"        # Unique run identifier
MODEL_NAME = "gemini-2.0-flash" # or gemini-3-flash-preview
TEMPERATURE = 0.0               # 0.0 for deterministic
THINKING_BUDGET = 1024          # 0 to disable, or token budget
```

## Output Format

Each result contains:
```python
{
    "chunk_id": "...",
    "company_name": "...",
    "report_year": 2024,
    "human_mention_types": ["adoption", "risk"],
    "llm_mention_types": ["adoption"],
    "confidence": 0.85,
    "reasoning": "...",
    "chunk_text": "..."
}
```

## Match Types

- **EXACT** - LLM labels == Human labels
- **PARTIAL** - Some overlap
- **DIFF** - No overlap
