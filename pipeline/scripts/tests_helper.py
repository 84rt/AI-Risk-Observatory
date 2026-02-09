"""Test utilities for classifier consistency and model comparison."""

from collections import Counter
from functools import lru_cache
from typing import Any

import requests

from src.config import settings


def _looks_like_schema(classification: Any) -> bool:
    """Heuristic check for schema-echo responses."""
    if not isinstance(classification, dict):
        return False

    keys = set(classification.keys())
    if {"title", "description", "type", "properties"}.issubset(keys):
        return True
    if "properties" in keys and "type" in keys and isinstance(classification.get("properties"), dict):
        return True
    for v in classification.values():
        if isinstance(v, str) and "schema" in v.lower():
            return True
    return False


@lru_cache(maxsize=1)
def _fetch_openrouter_models() -> set[str]:
    """Fetch available model IDs from OpenRouter API (cached)."""
    api_key = settings.openrouter_api_key
    if not api_key:
        return set()

    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {m["id"] for m in data.get("data", [])}
    except Exception:
        pass
    return set()


def validate_models(models: list[str], verbose: bool = True) -> tuple[list[str], list[str]]:
    """Validate model slugs against OpenRouter's available models.

    Args:
        models: List of model names to validate
        verbose: Print validation results

    Returns:
        Tuple of (valid_models, invalid_models)
    """
    available = _fetch_openrouter_models()

    valid = []
    invalid = []

    for model in models:
        # Models without "/" are assumed to be Gemini API models
        if "/" not in model:
            valid.append(model)
            if verbose:
                print(f"  ✓ {model} (Gemini API - not validated)")
        elif model in available:
            valid.append(model)
            if verbose:
                print(f"  ✓ {model}")
        else:
            invalid.append(model)
            if verbose:
                print(f"  ✗ {model} (not found on OpenRouter)")

    return valid, invalid


def _chunk_to_metadata(chunk: dict) -> dict[str, Any]:
    """Extract classifier metadata from a chunk dict."""
    return {
        "firm_id": chunk.get("company_id", "Unknown"),
        "firm_name": chunk.get("company_name", "Unknown"),
        "report_year": chunk.get("report_year", 0),
        "sector": "Unknown",
        "report_section": chunk.get("report_sections", ["Unknown"])[0]
        if chunk.get("report_sections")
        else "Unknown",
    }


def run_best_of_n(
    classifier_cls,
    chunk: dict,
    n: int = 5,
    temperature: float = 0.0,
    model_name: str | None = None,
    tqdm_func=None,
) -> dict[str, Any]:
    """Run best-of-n consistency test on a single chunk.

    Args:
        classifier_cls: Classifier class (e.g., MentionTypeClassifier)
        chunk: Chunk dict with 'chunk_text' and metadata fields
        n: Number of runs
        temperature: Model temperature (0.0 = deterministic)
        model_name: Optional model override
        tqdm_func: Optional progress bar function

    Returns:
        Result dict with consistency metrics
    """
    return best_of_n_test(
        classifier_cls,
        chunk["chunk_text"],
        _chunk_to_metadata(chunk),
        n=n,
        temperature=temperature,
        model_name=model_name,
        tqdm_func=tqdm_func,
    )


def run_model_family(
    classifier_cls,
    chunk: dict,
    models: list[str],
    temperature: float = 0.0,
    tqdm_func=None,
) -> dict[str, Any]:
    """Run model family comparison test on a single chunk.

    Args:
        classifier_cls: Classifier class (e.g., MentionTypeClassifier)
        chunk: Chunk dict with 'chunk_text' and metadata fields
        models: List of model names to test
        temperature: Model temperature
        tqdm_func: Optional progress bar function

    Returns:
        Result dict with agreement metrics
    """
    return model_family_test(
        classifier_cls,
        chunk["chunk_text"],
        _chunk_to_metadata(chunk),
        models=models,
        temperature=temperature,
        tqdm_func=tqdm_func,
    )


def best_of_n_test(
    classifier_cls,
    text: str,
    metadata: dict[str, Any],
    n: int = 5,
    temperature: float = 0.0,
    run_id: str = "best-of-n",
    model_name: str | None = None,
    tqdm_func=None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the same classifier n times on identical input to test consistency.

    Args:
        classifier_cls: Classifier class (e.g., MentionTypeClassifier)
        text: The text to classify
        metadata: Metadata dict for classification
        n: Number of runs (default 5)
        temperature: Model temperature (0.0 = deterministic, higher = more random)
        run_id: Run identifier for logging
        model_name: Optional model override
        tqdm_func: Optional progress bar function
        verbose: Print results as they come in

    Returns:
        Dict with:
            - runs: List of individual run results
            - classifications: List of primary classification sets
            - consistency_score: Fraction of runs matching the most common result
            - most_common: The most frequent classification
            - unique_count: Number of distinct classifications seen
            - temperature: Temperature used
    """
    progress = tqdm_func or (lambda x, **_: x)
    runs = []
    classifications = []

    if verbose:
        print(f"\n{'='*80}")
        print(f"BEST-OF-{n} TEST")
        print(f"{'='*80}")
        print(f"Classifier: {classifier_cls.__name__}")
        print(f"Model: {model_name or 'default'}")
        print(f"Temperature: {temperature}")
        print(f"Text preview: {text[:100].strip()}...")
        print()

    for i in progress(range(n), desc=f"Best-of-{n} runs"):
        # Create fresh classifier for each run
        clf_kwargs = {"run_id": f"{run_id}-{i}", "temperature": temperature}
        if model_name:
            clf_kwargs["model_name"] = model_name

        clf = classifier_cls(**clf_kwargs)
        result = clf.classify(text, metadata)

        # Extract the primary classification for comparison
        classification = result.classification
        schema_echo = _looks_like_schema(classification)
        if schema_echo:
            if verbose:
                print(
                    "    WARNING: Model returned a schema-like response; "
                    f"treating as empty classification. Keys={list(classification.keys())}"
                )
            classification = None
        primary_set = _extract_primary_set(classification)

        runs.append({
            "run": i + 1,
            "primary_label": result.primary_label,
            "confidence": result.confidence_score,
            "reasoning": result.reasoning,
            "classification": classification,
            "primary_set": primary_set,
            "schema_echo": schema_echo,
        })
        classifications.append(tuple(sorted(primary_set)))

        if verbose:
            print(f"  Run {i+1}: {list(primary_set)} (conf={result.confidence_score:.2f})")

    # Compute consistency metrics
    counter = Counter(classifications)
    most_common_tuple, most_common_count = counter.most_common(1)[0]
    consistency_score = most_common_count / n
    unique_count = len(counter)

    if verbose:
        print(f"\n{'='*80}")
        print("CONSISTENCY RESULTS")
        print(f"{'='*80}")
        print(f"Unique classifications: {unique_count}")
        print(f"Most common: {list(most_common_tuple)} ({most_common_count}/{n} = {consistency_score:.0%})")
        print(f"Consistency score: {consistency_score:.2%}")
        if unique_count > 1:
            print("\nAll variants seen:")
            for variant, count in counter.most_common():
                print(f"  {list(variant)}: {count}x ({count/n:.0%})")

    return {
        "runs": runs,
        "classifications": [list(c) for c in classifications],
        "consistency_score": consistency_score,
        "most_common": list(most_common_tuple),
        "most_common_count": most_common_count,
        "unique_count": unique_count,
        "temperature": temperature,
        "n": n,
    }


def model_family_test(
    classifier_cls,
    text: str,
    metadata: dict[str, Any],
    models: list[str],
    temperature: float = 0.0,
    run_id: str = "model-family",
    tqdm_func=None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Test the same text across multiple models to compare classifications.

    Args:
        classifier_cls: Classifier class (e.g., MentionTypeClassifier)
        text: The text to classify
        metadata: Metadata dict for classification
        models: List of model names to test
        temperature: Model temperature
        run_id: Run identifier for logging
        tqdm_func: Optional progress bar function
        verbose: Print results as they come in

    Returns:
        Dict with:
            - results: Dict mapping model name to result
            - classifications: Dict mapping model name to classification set
            - agreement_score: Fraction of models agreeing with majority
            - majority: The most common classification across models
    """
    progress = tqdm_func or (lambda x, **_: x)
    results = {}
    classifications = {}

    if verbose:
        print(f"\n{'='*80}")
        print("MODEL FAMILY TEST")
        print(f"{'='*80}")
        print(f"Classifier: {classifier_cls.__name__}")
        print(f"Temperature: {temperature}")
        print(f"Text preview: {text[:100].strip()}...")
        print()
        print("Validating models...")

    valid_models, invalid_models = validate_models(models, verbose=verbose)

    if invalid_models and verbose:
        print(f"\n  Skipping {len(invalid_models)} invalid model(s)")

    if not valid_models:
        raise ValueError(f"No valid models to test. Invalid: {invalid_models}")

    if verbose:
        print()

    for model_name in progress(valid_models, desc="Testing models"):
        if verbose:
            print(f"\n  Testing: {model_name}")

        clf = classifier_cls(
            run_id=f"{run_id}-{model_name.replace('/', '-')}",
            model_name=model_name,
            temperature=temperature,
        )

        if verbose:
            api_type = "OpenRouter" if clf.use_openrouter else "Gemini API"
            print(f"    Using: {api_type}")

        try:
            result = clf.classify(text, metadata)
            classification = result.classification
            schema_echo = _looks_like_schema(classification)
            if schema_echo:
                if verbose:
                    print(
                        "    WARNING: Model returned a schema-like response; "
                        f"treating as empty classification. Keys={list(classification.keys())}"
                    )
                classification = None
            primary_set = _extract_primary_set(classification)

            results[model_name] = {
                "primary_label": result.primary_label,
                "confidence": result.confidence_score,
                "reasoning": result.reasoning,
                "classification": classification,
                "primary_set": primary_set,
                "error": "Schema echo detected" if schema_echo else result.error_message,
                "success": False if schema_echo else result.success,
                "response_raw": getattr(result, "response_raw", None),
            }
            classifications[model_name] = tuple(sorted(primary_set))

            if verbose:
                print(f"    Result: {list(primary_set)} (conf={result.confidence_score:.2f})")
                if not result.success:
                    print(f"    ERROR: {result.error_message}")
                if not primary_set and classification:
                    print(f"    Raw classification: {classification}")
                if hasattr(result, "response_raw") and result.response_raw:
                    print(f"    Raw response: {result.response_raw[:200]}...")

        except Exception as e:
            results[model_name] = {
                "primary_label": None,
                "confidence": None,
                "reasoning": None,
                "classification": None,
                "primary_set": set(),
                "error": str(e),
            }
            classifications[model_name] = ()

            if verbose:
                print(f"    ERROR: {e}")

    # Compute agreement metrics
    valid_classifications = [c for c in classifications.values() if c]
    if valid_classifications:
        counter = Counter(valid_classifications)
        majority_tuple, majority_count = counter.most_common(1)[0]
        agreement_score = majority_count / len(valid_classifications)
    else:
        majority_tuple = ()
        majority_count = 0
        agreement_score = 0.0

    if verbose:
        print(f"\n{'='*80}")
        print("MODEL AGREEMENT RESULTS")
        print(f"{'='*80}")
        print(f"Models validated: {len(valid_models)}/{len(models)}")
        if invalid_models:
            print(f"Skipped (invalid): {invalid_models}")
        print(f"Successful API calls: {len(valid_classifications)}")
        if valid_classifications:
            print(f"Majority classification: {list(majority_tuple)} ({majority_count}/{len(valid_classifications)})")
            print(f"Agreement score: {agreement_score:.2%}")
        if valid_classifications and len(counter) > 1:
            print("\nAll variants seen:")
            for variant, count in counter.most_common():
                matching_models = [m for m, c in classifications.items() if c == variant]
                print(f"  {list(variant)}: {matching_models}")

    return {
        "results": results,
        "classifications": {m: list(c) for m, c in classifications.items()},
        "agreement_score": agreement_score,
        "majority": list(majority_tuple),
        "majority_count": majority_count,
        "models_tested": len(valid_models),
        "models_successful": len(valid_classifications),
        "models_invalid": invalid_models,
    }


def best_of_n_batch(
    classifier_cls,
    chunks: list[dict],
    n: int = 3,
    temperature: float = 0.0,
    run_id: str = "best-of-n-batch",
    model_name: str | None = None,
    tqdm_func=None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run best-of-n test across multiple chunks.

    Args:
        classifier_cls: Classifier class
        chunks: List of chunk dicts with 'chunk_text' and metadata fields
        n: Number of runs per chunk
        temperature: Model temperature
        run_id: Run identifier
        model_name: Optional model override
        tqdm_func: Optional progress bar function
        verbose: Print results

    Returns:
        List of best_of_n_test results, one per chunk
    """
    progress = tqdm_func or (lambda x, **_: x)
    results = []

    if verbose:
        print(f"\n{'='*80}")
        print(f"BEST-OF-{n} BATCH TEST ({len(chunks)} chunks)")
        print(f"{'='*80}")

    for i, chunk in enumerate(progress(chunks, desc="Processing chunks")):
        metadata = {
            "firm_id": chunk.get("company_id", "Unknown"),
            "firm_name": chunk.get("company_name", "Unknown"),
            "report_year": chunk.get("report_year", 0),
            "sector": "Unknown",
            "report_section": chunk.get("report_sections", ["Unknown"])[0]
            if chunk.get("report_sections")
            else "Unknown",
        }

        if verbose:
            print(f"\n[{i+1}/{len(chunks)}] {chunk.get('company_name', 'Unknown')} ({chunk.get('report_year', '?')})")

        result = best_of_n_test(
            classifier_cls,
            chunk["chunk_text"],
            metadata,
            n=n,
            temperature=temperature,
            run_id=f"{run_id}-chunk{i}",
            model_name=model_name,
            tqdm_func=None,  # Don't nest progress bars
            verbose=False,  # Summarize at the end
        )

        result["chunk_id"] = chunk.get("chunk_id", f"chunk-{i}")
        result["company_name"] = chunk.get("company_name", "Unknown")
        result["human_labels"] = chunk.get("mention_types", [])
        results.append(result)

        if verbose:
            human = set(chunk.get("mention_types", []))
            llm_majority = set(result["most_common"])
            match = "✓" if human == llm_majority else ("~" if human & llm_majority else "✗")
            print(f"  Consistency: {result['consistency_score']:.0%} | "
                  f"Human: {list(human)} | LLM: {list(llm_majority)} {match}")

    # Summary
    if verbose and results:
        avg_consistency = sum(r["consistency_score"] for r in results) / len(results)
        perfect = sum(1 for r in results if r["consistency_score"] == 1.0)
        print(f"\n{'='*80}")
        print("BATCH SUMMARY")
        print(f"{'='*80}")
        print(f"Average consistency: {avg_consistency:.2%}")
        print(f"Perfectly consistent: {perfect}/{len(results)} ({perfect/len(results):.0%})")

    return results


def best_of_n_batch_detailed(
    classifier_cls,
    chunks: list[dict],
    n: int = 5,
    temperature: float = 0.0,
    run_id: str = "best-of-n-batch",
    model_name: str | None = None,
    tqdm_func=None,
    verbose: bool = True,
    show_text: bool = False,
    max_text_len: int = 140,
) -> list[dict[str, Any]]:
    """Run best-of-n test across multiple chunks with detailed per-chunk breakdowns."""
    progress = tqdm_func or (lambda x, **_: x)
    results = []

    if verbose:
        print(f"\n{'='*80}")
        print(f"BEST-OF-{n} BATCH (detailed)")
        print(f"{'='*80}")
        print(f"Chunks: {len(chunks)} | Temperature: {temperature}")
        print()

    for i, chunk in enumerate(progress(chunks, desc=f"Best-of-{n} batch")):
        result = best_of_n_test(
            classifier_cls,
            chunk["chunk_text"],
            _chunk_to_metadata(chunk),
            n=n,
            temperature=temperature,
            run_id=f"{run_id}-{i}",
            model_name=model_name,
            tqdm_func=None,
            verbose=False,
        )

        classifications = [tuple(sorted(c)) for c in result["classifications"]]
        counter = Counter(classifications)
        most_common_tuple, most_common_count = counter.most_common(1)[0]
        unique_count = len(counter)
        schema_echo_count = sum(1 for r in result.get("runs", []) if r.get("schema_echo"))

        if verbose:
            company = chunk.get("company_name", "Unknown")
            year = chunk.get("report_year", "?")
            human = list(set(chunk.get("mention_types", [])))
            print(f"\n[{i+1}/{len(chunks)}] {company} ({year}) | Human: {human}")
            print(
                f"  Agreement: {most_common_count}/{n} | "
                f"Most common: {list(most_common_tuple)} | Variants: {unique_count}"
            )
            if schema_echo_count:
                print(f"  WARNING: schema-echo runs: {schema_echo_count}/{n}")
            if unique_count > 1:
                print("  Variants:")
                for variant, count in counter.most_common():
                    print(f"    {list(variant)}: {count}")
            if show_text:
                preview = chunk.get("chunk_text", "").strip().replace("\n", " ")
                print(f"  Text: {preview[:max_text_len]}...")

        result["chunk_id"] = chunk.get("chunk_id", f"chunk-{i}")
        result["company_name"] = chunk.get("company_name", "Unknown")
        result["report_year"] = chunk.get("report_year", "?")
        result["human_labels"] = list(set(chunk.get("mention_types", [])))
        result["agreement_count"] = most_common_count
        result["agreement_labels"] = list(most_common_tuple)
        result["variant_counts"] = [{"labels": list(k), "count": v} for k, v in counter.items()]
        result["schema_echo_count"] = schema_echo_count
        results.append(result)

    if verbose and results:
        avg_consistency = sum(r["consistency_score"] for r in results) / len(results)
        perfect = sum(1 for r in results if r["consistency_score"] == 1.0)
        any_schema = sum(1 for r in results if r.get("schema_echo_count", 0) > 0)
        print(f"\n{'='*80}")
        print("BATCH SUMMARY")
        print(f"{'='*80}")
        print(f"Average consistency: {avg_consistency:.2%}")
        print(f"Perfectly consistent: {perfect}/{len(results)} ({perfect/len(results):.0%})")
        if any_schema:
            print(f"Chunks with schema-echo: {any_schema}/{len(results)}")

    return results


def model_family_batch(
    classifier_cls,
    chunks: list[dict],
    models: list[str],
    temperature: float = 0.0,
    run_id: str = "model-family-batch",
    tqdm_func=None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run model family comparison across multiple chunks.

    Args:
        classifier_cls: Classifier class
        chunks: List of chunk dicts with 'chunk_text' and metadata fields
        models: List of model names to test
        temperature: Model temperature
        run_id: Run identifier
        tqdm_func: Optional progress bar function
        verbose: Print results

    Returns:
        Dict with:
            - chunk_results: List of per-chunk results
            - model_stats: Per-model accuracy vs human labels
            - agreement_matrix: How often each pair of models agrees
            - overall_agreement: Average agreement across chunks
    """
    progress = tqdm_func or (lambda x, **_: x)

    if verbose:
        print(f"\n{'='*80}")
        print(f"MODEL FAMILY BATCH TEST ({len(chunks)} chunks, {len(models)} models)")
        print(f"{'='*80}")
        print("Validating models...")

    valid_models, invalid_models = validate_models(models, verbose=verbose)

    if not valid_models:
        raise ValueError(f"No valid models to test. Invalid: {invalid_models}")

    if verbose:
        print(f"\nRunning {len(chunks)} chunks × {len(valid_models)} models = {len(chunks) * len(valid_models)} API calls\n")

    chunk_results = []
    model_correct = {m: 0 for m in valid_models}
    model_total = {m: 0 for m in valid_models}
    agreement_counts = {m1: {m2: 0 for m2 in valid_models} for m1 in valid_models}

    for i, chunk in enumerate(progress(chunks, desc="Processing chunks")):
        human_labels = set(chunk.get("mention_types", []))

        if verbose:
            company = chunk.get("company_name", "Unknown")
            year = chunk.get("report_year", "?")
            print(f"\n[{i+1}/{len(chunks)}] {company} ({year}) | Human: {list(human_labels)}")

        result = run_model_family(
            classifier_cls,
            chunk,
            models=valid_models,
            temperature=temperature,
            tqdm_func=None,
        )

        result["chunk_id"] = chunk.get("chunk_id", f"chunk-{i}")
        result["company_name"] = chunk.get("company_name", "Unknown")
        result["human_labels"] = list(human_labels)
        chunk_results.append(result)

        # Track per-model accuracy
        for model_name, clf_set in result["classifications"].items():
            clf_set = set(clf_set)
            model_total[model_name] += 1
            if clf_set == human_labels:
                model_correct[model_name] += 1
                match_symbol = "✓"
            elif clf_set & human_labels:
                match_symbol = "~"
            else:
                match_symbol = "✗"

            if verbose:
                print(f"  {model_name}: {list(clf_set)} {match_symbol}")

        # Track pairwise agreement
        for m1 in valid_models:
            for m2 in valid_models:
                if result["classifications"].get(m1) == result["classifications"].get(m2):
                    agreement_counts[m1][m2] += 1

    # Compute stats
    model_accuracy = {
        m: model_correct[m] / model_total[m] if model_total[m] > 0 else 0.0
        for m in valid_models
    }

    total_chunks = len(chunk_results)
    agreement_matrix = {
        m1: {m2: agreement_counts[m1][m2] / total_chunks if total_chunks > 0 else 0.0
             for m2 in valid_models}
        for m1 in valid_models
    }

    # Overall agreement = average of non-diagonal pairwise agreements
    pairwise = []
    for i, m1 in enumerate(valid_models):
        for m2 in valid_models[i+1:]:
            pairwise.append(agreement_matrix[m1][m2])
    overall_agreement = sum(pairwise) / len(pairwise) if pairwise else 1.0

    avg_chunk_agreement = sum(r["agreement_score"] for r in chunk_results) / len(chunk_results) if chunk_results else 0.0

    if verbose:
        print(f"\n{'='*80}")
        print("BATCH SUMMARY")
        print(f"{'='*80}")
        print(f"Chunks processed: {total_chunks}")
        print(f"Models tested: {len(valid_models)}")
        if invalid_models:
            print(f"Models skipped: {invalid_models}")

        print(f"\nModel accuracy vs human labels:")
        for m in sorted(valid_models, key=lambda x: model_accuracy[x], reverse=True):
            exact = model_correct[m]
            total = model_total[m]
            print(f"  {m}: {exact}/{total} ({model_accuracy[m]:.0%})")

        print(f"\nAverage within-chunk agreement: {avg_chunk_agreement:.1%}")
        print(f"Overall pairwise agreement: {overall_agreement:.1%}")

    return {
        "chunk_results": chunk_results,
        "model_accuracy": model_accuracy,
        "model_correct": model_correct,
        "model_total": model_total,
        "agreement_matrix": agreement_matrix,
        "overall_agreement": overall_agreement,
        "avg_chunk_agreement": avg_chunk_agreement,
        "models_tested": valid_models,
        "models_invalid": invalid_models,
    }


def run_thinking_levels(
    classifier_cls,
    chunk: dict,
    budgets: list[int] | None = None,
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    tqdm_func=None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Test different thinking budgets on a single chunk.

    Args:
        classifier_cls: Classifier class (e.g., MentionTypeClassifier)
        chunk: Chunk dict with 'chunk_text' and metadata fields
        budgets: List of thinking budgets to test (default: [0, 1024, 4096, 8192])
        model_name: Model to use (must support native thinking)
        temperature: Model temperature
        tqdm_func: Optional progress bar function
        verbose: Print results

    Returns:
        Dict with per-budget results and comparison metrics
    """
    if budgets is None:
        budgets = [0, 1024, 4096, 8192]

    progress = tqdm_func or (lambda x, **_: x)
    results = {}
    classifications = {}
    human_labels = set(chunk.get("mention_types", []))

    if verbose:
        print(f"\n{'='*80}")
        print("THINKING LEVELS TEST")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Budgets: {budgets}")
        print(f"Human labels: {list(human_labels)}")
        print(f"Text preview: {chunk.get('chunk_text', '')[:100].strip()}...")
        print()

    metadata = _chunk_to_metadata(chunk)

    for budget in progress(budgets, desc="Testing thinking levels"):
        if verbose:
            budget_label = "disabled" if budget == 0 else f"{budget} tokens"
            print(f"\n  Testing thinking_budget={budget} ({budget_label})")

        clf = classifier_cls(
            run_id=f"thinking-{budget}",
            model_name=model_name,
            temperature=temperature,
            thinking_budget=budget,
        )

        try:
            result = clf.classify(chunk["chunk_text"], metadata)
            classification = result.classification
            schema_echo = _looks_like_schema(classification)
            if schema_echo:
                if verbose:
                    print(f"    WARNING: Schema echo detected")
                classification = None
            primary_set = _extract_primary_set(classification)

            # Compare to human labels
            if primary_set == human_labels:
                match = "✅ EXACT"
            elif primary_set & human_labels:
                match = "🟡 PARTIAL"
            else:
                match = "❌ DIFF"

            results[budget] = {
                "primary_label": result.primary_label,
                "confidence": result.confidence_score,
                "reasoning": result.reasoning,
                "classification": classification,
                "primary_set": primary_set,
                "latency_ms": result.api_latency_ms,
                "tokens_used": result.tokens_used,
                "success": result.success,
                "error": result.error_message,
                "match_status": match,
            }
            classifications[budget] = tuple(sorted(primary_set))

            if verbose:
                print(f"    Result: {list(primary_set)} {match}")
                print(f"    Confidence: {result.confidence_score:.2f}")
                print(f"    Latency: {result.api_latency_ms}ms")
                if result.reasoning:
                    reasoning_preview = result.reasoning[:150]
                    if len(result.reasoning) > 150:
                        reasoning_preview += "..."
                    print(f"    Reasoning: {reasoning_preview}")

        except Exception as e:
            results[budget] = {
                "primary_set": set(),
                "error": str(e),
                "success": False,
                "match_status": "❌ ERROR",
            }
            classifications[budget] = ()
            if verbose:
                print(f"    ERROR: {e}")

    # Summary
    if verbose:
        print(f"\n{'='*80}")
        print("THINKING LEVELS SUMMARY")
        print(f"{'='*80}")
        print(f"{'Budget':<12} {'Result':<30} {'Match':<12} {'Latency':<10}")
        print("-" * 70)
        for budget in budgets:
            r = results.get(budget, {})
            labels = list(r.get("primary_set", []))
            match = r.get("match_status", "N/A")
            latency = r.get("latency_ms", 0)
            print(f"{budget:<12} {str(labels):<30} {match:<12} {latency}ms")

    return {
        "results": results,
        "classifications": {b: list(c) for b, c in classifications.items()},
        "human_labels": list(human_labels),
        "budgets_tested": budgets,
        "model": model_name,
    }


def thinking_levels_batch(
    classifier_cls,
    chunks: list[dict],
    budgets: list[int] | None = None,
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    tqdm_func=None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Test thinking levels across multiple chunks.

    Args:
        classifier_cls: Classifier class
        chunks: List of chunk dicts
        budgets: List of thinking budgets (default: [0, 1024, 4096, 8192])
        model_name: Model to use
        temperature: Model temperature
        tqdm_func: Optional progress bar function
        verbose: Print results

    Returns:
        Dict with per-chunk results and aggregate statistics
    """
    if budgets is None:
        budgets = [0, 1024, 4096, 8192]

    progress = tqdm_func or (lambda x, **_: x)

    if verbose:
        print(f"\n{'='*80}")
        print(f"THINKING LEVELS BATCH TEST ({len(chunks)} chunks, {len(budgets)} budgets)")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Budgets: {budgets}")
        print(f"Total API calls: {len(chunks) * len(budgets)}")
        print()

    chunk_results = []
    budget_stats = {
        b: {"exact": 0, "partial": 0, "diff": 0, "total_latency": 0, "total_tokens": 0}
        for b in budgets
    }

    for i, chunk in enumerate(progress(chunks, desc="Processing chunks")):
        human_labels = set(chunk.get("mention_types", []))

        if verbose:
            company = chunk.get("company_name", "Unknown")
            year = chunk.get("report_year", "?")
            print(f"\n[{i+1}/{len(chunks)}] {company} ({year}) | Human: {list(human_labels)}")

        result = run_thinking_levels(
            classifier_cls,
            chunk,
            budgets=budgets,
            model_name=model_name,
            temperature=temperature,
            tqdm_func=None,
            verbose=False,
        )

        result["chunk_id"] = chunk.get("chunk_id", f"chunk-{i}")
        result["company_name"] = chunk.get("company_name", "Unknown")
        chunk_results.append(result)

        # Update stats and print row
        if verbose:
            row = f"  "
            for budget in budgets:
                r = result["results"].get(budget, {})
                match = r.get("match_status", "N/A")
                latency = r.get("latency_ms", 0) or 0
                tokens = r.get("tokens_used", 0) or 0

                if "EXACT" in match:
                    budget_stats[budget]["exact"] += 1
                elif "PARTIAL" in match:
                    budget_stats[budget]["partial"] += 1
                else:
                    budget_stats[budget]["diff"] += 1
                budget_stats[budget]["total_latency"] += latency
                budget_stats[budget]["total_tokens"] += tokens

                symbol = "✅" if "EXACT" in match else ("🟡" if "PARTIAL" in match else "❌")
                row += f"  {budget}: {symbol}"
            print(row)

    # Compute accuracy per budget
    total = len(chunks)
    budget_accuracy = {}
    for budget in budgets:
        exact = budget_stats[budget]["exact"]
        partial = budget_stats[budget]["partial"]
        budget_accuracy[budget] = {
            "exact_match": exact / total if total > 0 else 0,
            "partial_or_better": (exact + partial) / total if total > 0 else 0,
            "avg_latency_ms": budget_stats[budget]["total_latency"] / total if total > 0 else 0,
            "avg_tokens": budget_stats[budget]["total_tokens"] / total if total > 0 else 0,
        }

    if verbose:
        print(f"\n{'='*80}")
        print("BATCH SUMMARY")
        print(f"{'='*80}")
        print(f"{'Budget':<12} {'Exact':<10} {'Partial+':<10} {'Avg Latency':<15} {'Avg Tokens':<12}")
        print("-" * 60)
        for budget in budgets:
            acc = budget_accuracy[budget]
            exact_pct = f"{acc['exact_match']:.0%}"
            partial_pct = f"{acc['partial_or_better']:.0%}"
            latency = f"{acc['avg_latency_ms']:.0f}ms"
            tokens = f"{acc['avg_tokens']:.0f}"
            print(f"{budget:<12} {exact_pct:<10} {partial_pct:<10} {latency:<15} {tokens:<12}")

        # Highlight best
        best_budget = max(budgets, key=lambda b: budget_accuracy[b]["exact_match"])
        print(f"\nBest accuracy: thinking_budget={best_budget} ({budget_accuracy[best_budget]['exact_match']:.0%} exact)")

    return {
        "chunk_results": chunk_results,
        "budget_accuracy": budget_accuracy,
        "budget_stats": budget_stats,
        "budgets_tested": budgets,
        "model": model_name,
        "total_chunks": total,
    }


def _extract_primary_set(classification: Any) -> set[str]:
    """Extract the primary classification as a set of strings for comparison."""
    if classification is None:
        return set()

    if isinstance(classification, dict):
        # Handle mention_types style
        if "mention_types" in classification:
            types = classification["mention_types"]
            if isinstance(types, list):
                return {str(t.value) if hasattr(t, "value") else str(t) for t in types}

        # Handle adoption_types style (with signals)
        if "adoption_signals" in classification or "adoption_confidences" in classification:
            confs = classification.get("adoption_signals") or classification.get("adoption_confidences")
            if isinstance(confs, list):
                return {
                    str(e.get("type"))
                    for e in confs
                    if isinstance(e, dict) and isinstance(e.get("signal"), (int, float)) and e.get("signal") > 0
                }
            if isinstance(confs, dict):
                return {k for k, v in confs.items() if isinstance(v, (int, float)) and v > 0}

        # Handle risk_types style
        if "risk_types" in classification:
            types = classification["risk_types"]
            if isinstance(types, list):
                return {str(t.value) if hasattr(t, "value") else str(t)
                        for t in types if str(t) != "none"}

        # Fallback: use all non-None, non-empty values
        result = set()
        for k, v in classification.items():
            if v and k not in ("reasoning", "confidence_scores"):
                if isinstance(v, list):
                    result.update(str(x.value) if hasattr(x, "value") else str(x) for x in v)
                elif isinstance(v, str):
                    result.add(v)
        return result

    if isinstance(classification, list):
        return {str(c.value) if hasattr(c, "value") else str(c) for c in classification}

    return {str(classification)}
