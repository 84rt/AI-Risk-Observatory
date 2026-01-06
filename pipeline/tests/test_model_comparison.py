#!/usr/bin/env python3
"""
Model Comparison Tests for AIRO Classifiers.

Compares classification agreement across different model families and versions:
- Gemini 2.0 Flash (baseline)
- Gemini 2.0 Flash Thinking (reasoning mode)
- Gemini 2.5 Pro (highest quality)

Usage:
    python tests/test_model_comparison.py [--samples 5]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

# Add pipeline root to path for proper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifiers.harms_classifier import HarmsClassifier
from src.classifiers.substantiveness_classifier import SubstantivenessClassifier
from src.utils.logging_config import setup_logging
from src.utils.data_export import DataExporter

# Paths
PIPELINE_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PIPELINE_ROOT / "output"
PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed" / "keyword"
LOGS_DIR = PIPELINE_ROOT / "logs" / "model_comparison"

# Models to compare
# Note: gemini-2.0-flash-exp has higher rate limits (2000 RPM)
# Other models may require different API access or billing
MODELS_TO_TEST = [
    {
        "name": "gemini-2.0-flash",
        "description": "Fast, good quality (baseline)",
    },
    {
        "name": "gemini-2.0-flash-exp",
        "description": "Experimental version with higher rate limits",
    },
]

# Sample files for comparison
SAMPLE_FILES = [
    "00617987_HSBA_2024.md",  # HSBC
    "00048839_BARC_2024.md",  # Barclays
    "02723534_AZN_2024.md",   # AstraZeneca
    "00077536_REL_2024.md",   # RELX
    "03888792_GSK_2024.md",   # GSK
]


def get_sample_files(num_samples: int = 5) -> List[Path]:
    """Get sample files for comparison."""
    files = []
    for filename in SAMPLE_FILES[:num_samples]:
        file_path = PREPROCESSED_DIR / filename
        if file_path.exists():
            files.append(file_path)
    return files


def extract_company_info(file_path: Path) -> Dict[str, Any]:
    """Extract company info from filename."""
    # Format: COMPANY_NUMBER_TICKER_YEAR.md
    parts = file_path.stem.split("_")
    return {
        "company_number": parts[0] if len(parts) > 0 else "",
        "ticker": parts[1] if len(parts) > 1 else "",
        "year": int(parts[2]) if len(parts) > 2 else 2024,
        "firm_name": parts[1] if len(parts) > 1 else "Unknown",
    }


def compare_models(
    num_samples: int = 5,
    classifiers_to_test: List[str] = ["harms", "substantiveness"],
    rate_limit: float = 2.0,
) -> Dict[str, Any]:
    """
    Compare classification results across models.

    Args:
        num_samples: Number of sample files to test
        classifiers_to_test: Which classifiers to compare
        rate_limit: Delay between API calls

    Returns:
        Comparison results dict
    """
    run_id = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logging(log_level="INFO", run_id=run_id)
    logger.info(f"Starting model comparison: {run_id}")

    # Get sample files
    sample_files = get_sample_files(num_samples)
    logger.info(f"Testing with {len(sample_files)} sample files")

    if not sample_files:
        logger.error("No sample files found!")
        return {"error": "No sample files found"}

    results = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "samples_tested": len(sample_files),
        "models_tested": [m["name"] for m in MODELS_TO_TEST],
        "classifiers_tested": classifiers_to_test,
        "comparisons": [],
        "agreement_summary": {},
    }

    # For each sample file
    for file_idx, file_path in enumerate(sample_files, 1):
        company_info = extract_company_info(file_path)
        logger.info(f"\n[{file_idx}/{len(sample_files)}] Testing: {file_path.name}")

        # Read file content
        text = file_path.read_text(encoding="utf-8")

        # For each classifier
        for clf_name in classifiers_to_test:
            comparison = {
                "sample": file_path.name,
                "company": company_info["ticker"],
                "year": company_info["year"],
                "classifier": clf_name,
                "model_results": {},
            }

            # Get the classifier class
            if clf_name == "harms":
                classifier_class = HarmsClassifier
            elif clf_name == "substantiveness":
                classifier_class = SubstantivenessClassifier
            else:
                logger.warning(f"Unknown classifier: {clf_name}")
                continue

            # Test each model
            for model_info in MODELS_TO_TEST:
                model_name = model_info["name"]
                logger.info(f"  Testing {clf_name} with {model_name}...")

                try:
                    classifier = classifier_class(
                        run_id=run_id,
                        model_name=model_name,
                    )

                    start_time = time.time()
                    result = classifier.classify(
                        text=text,
                        metadata={
                            "firm_id": company_info["ticker"],
                            "firm_name": company_info["firm_name"],
                            "report_year": company_info["year"],
                            "sector": "Unknown",
                        },
                        source_file=str(file_path),
                    )
                    latency = time.time() - start_time

                    comparison["model_results"][model_name] = {
                        "primary_label": result.primary_label,
                        "confidence": result.confidence_score,
                        "evidence_count": len(result.evidence),
                        "latency_seconds": round(latency, 2),
                        "success": result.success,
                    }

                    logger.info(
                        f"    {model_name}: {result.primary_label} "
                        f"(conf={result.confidence_score:.2f}, latency={latency:.1f}s)"
                    )

                except Exception as e:
                    logger.error(f"    {model_name}: Error - {e}")
                    comparison["model_results"][model_name] = {
                        "error": str(e),
                        "success": False,
                    }

                # Rate limiting
                time.sleep(rate_limit)

            # Check agreement
            labels = [
                r.get("primary_label")
                for r in comparison["model_results"].values()
                if r.get("success", False)
            ]
            comparison["all_agree"] = len(set(labels)) == 1 if labels else False
            comparison["labels"] = labels

            results["comparisons"].append(comparison)

    # Calculate agreement summary
    by_classifier = {}
    for clf_name in classifiers_to_test:
        clf_comparisons = [c for c in results["comparisons"] if c["classifier"] == clf_name]
        agree_count = sum(1 for c in clf_comparisons if c.get("all_agree", False))
        total = len(clf_comparisons)

        by_classifier[clf_name] = {
            "total": total,
            "full_agreement": agree_count,
            "agreement_rate": agree_count / total if total > 0 else 0,
        }

    results["agreement_summary"] = by_classifier

    # Calculate overall
    total_comparisons = len(results["comparisons"])
    total_agreements = sum(1 for c in results["comparisons"] if c.get("all_agree", False))
    results["overall_agreement_rate"] = total_agreements / total_comparisons if total_comparisons > 0 else 0

    # Save results
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOGS_DIR / f"{run_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 60)
    print(f"Samples tested: {len(sample_files)}")
    print(f"Models compared: {', '.join(m['name'] for m in MODELS_TO_TEST)}")
    print(f"\nAgreement by Classifier:")
    for clf_name, stats in by_classifier.items():
        print(f"  {clf_name}: {stats['full_agreement']}/{stats['total']} ({stats['agreement_rate']:.1%})")
    print(f"\nOverall Agreement: {results['overall_agreement_rate']:.1%}")
    print("=" * 60)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare classification across model families"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples to test (default: 5)",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["harms", "substantiveness"],
        help="Classifiers to compare (default: harms substantiveness)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=2.0,
        help="Delay between API calls in seconds (default: 2.0)",
    )

    args = parser.parse_args()

    results = compare_models(
        num_samples=args.samples,
        classifiers_to_test=args.classifiers,
        rate_limit=args.rate_limit,
    )

    return 0 if results.get("overall_agreement_rate", 0) >= 0.6 else 1


if __name__ == "__main__":
    sys.exit(main())

