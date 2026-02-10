# Context Dump: Risk Classifier + Reconciliation Workstream

## Goal
Manually QA and reconcile risk-classifier outputs into the golden set, with strict precision for AI-risk tagging.

## Main user intent
- Reduce false positives where text is about AI opportunity/adoption/governance but not concrete AI downside risk.
- Ensure export + reconcile tooling correctly surfaces and preserves new risk schema fields.
- Validate batch run processing for `batch-p2-risk-gemini-3-flash-preview-schema-v2.1-full`.

## What was implemented

### 1) `scripts/phase2_testbed.py`
- Added support to persist:
  - `risk_signals`
  - `risk_substantiveness`
- Added normalization/parsing helpers for these fields.
- Updated sync and batch paths to save these fields.
- Added fallback parsing/salvage for malformed batch JSON responses.
- Added sane defaults for error rows.
- Removed duplicate `risk_signal_map` definition.
- Fixed syntax/indentation issue in sync execution block.

### 2) `pipeline/scripts/export_testbed_run_for_reconcile.py`
- For risk exports, now preserves/sets:
  - top-level `risk_substantiveness`
  - `llm_details.risk_substantiveness`
  - `risk_signals` and derived `risk_confidences`
- Prevented `--confidence-mode uniform` from overwriting the structured `risk_signals` list:
  - writes `risk_signals_uniform` instead.

### 3) `pipeline/scripts/reconcile_annotations.py`
- Added normalization for legacy human risk labels before compare/save:
  - `strategic_market -> strategic_competitive`
  - `regulatory -> regulatory_compliance`
  - `workforce -> workforce_impacts`
  - `environmental -> environmental_impact`
- Added display fields requested for QA:
  - HUMAN `risk_substantiveness`
  - LLM raw `risk_signals` (`type:signal`)
  - LLM `risk_substantiveness`

### 4) `pipeline/prompts/classifiers.yaml`
- Fixed prompt formatting brace-escaping issues that caused `.format(...)` KeyErrors.
- Added new prompt variant: `risk_v4`.

### 5) `pipeline/src/classifiers/risk_classifier.py`
- Current classifier prompt key is hardcoded in code (not notebook `PROMPT_KEY` preview var).
- Comment updated to include `risk_v4` as available option.

## Prompt issues diagnosed
- Root issue in current risk prompt versions: scope ambiguity lets model convert AI opportunity/governance text into risk tags.
- `risk_v3` precision phrase alone was insufficiently operationalized.

## `risk_v4` design intent
`risk_v4` was created to enforce high precision with:
- Hard gate requiring all three:
  1. AI anchor
  2. downside/exposure anchor
  3. AI->downside causal link
- Strong disqualifiers for:
  - megatrend/opportunity-only language
  - governance/roadmap/process-only language
  - AI mention + unrelated generic risk language
- Conservative category assignment and sparse labeling behavior.

## User example interpretation captured
For the Rolls-Royce chunk:
- The prior output was considered overcalled.
- Under stricter policy it is at best borderline and likely `none` unless explicit downside mechanism is stated.

## Operational notes for running
- Reconciliation command pattern used:

```bash
python3 pipeline/scripts/reconcile_annotations.py \
  --human data/golden_set/human_reconciled/annotations.jsonl \
  --llm data/golden_set/llm/batch-p2-risk-gemini-3-flash-preview-schema-v2.1-full/annotations.jsonl \
  --output-dir data/golden_set/reconciled/batch-p2-risk-gemini-3-flash-preview-schema-v2.1-full \
  --only-disagreements \
  --include-subtype-disagreements \
  --focus-field risk \
  --resume
```

- To use new prompt version, set key in:
  - `pipeline/src/classifiers/risk_classifier.py`
- If prompts were edited, restart kernel (or clear prompt cache) before batch prep.

## Known batch issue seen
- One batch produced many malformed JSON responses (e.g., unterminated strings / invalid JSON tokens).
- Parser hardening and salvage logic was added in testbed processing to improve recovery.

## Current status
- Tooling now surfaces risk signal tags + risk substantiveness for manual QA.
- Prompt `risk_v4` exists and is ready to test.
- Next likely step: run a focused eval slice comparing `risk_v3` vs `risk_v4` on known false-positive chunks.
