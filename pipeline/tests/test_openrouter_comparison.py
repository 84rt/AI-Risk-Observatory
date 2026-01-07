#!/usr/bin/env python3
"""
OpenRouter Model Family Comparison Test for AIRO Classifiers.

Compares classification results across different model families using OpenRouter:
- Claude (Anthropic)
- Gemini (Google)
- GPT (OpenAI)

Usage:
    python tests/test_openrouter_comparison.py [--samples 3] [--classifiers harms]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Add pipeline root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.utils.logging_config import setup_logging

# Paths
PIPELINE_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PIPELINE_ROOT / "output"
PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed" / "keyword"
LOGS_DIR = PIPELINE_ROOT / "logs" / "model_comparison"

# Models to compare via OpenRouter
MODELS_TO_COMPARE = [
    {
        "id": "google/gemini-2.0-flash-001",
        "name": "Gemini 2.0 Flash",
        "family": "google",
    },
    {
        "id": "anthropic/claude-3.5-sonnet",
        "name": "Claude 3.5 Sonnet",
        "family": "anthropic",
    },
    {
        "id": "openai/gpt-4o-mini",
        "name": "GPT-4o Mini",
        "family": "openai",
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


class OpenRouterClient:
    """Client for OpenRouter API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/airo-pipeline",
            "X-Title": "AIRO Classifier Test",
            "Content-Type": "application/json",
        }

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Send a chat completion request to OpenRouter."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

        return response.json()


def get_sample_files(num_samples: int = 3) -> List[Path]:
    """Get sample files for comparison."""
    files = []
    for filename in SAMPLE_FILES[:num_samples]:
        file_path = PREPROCESSED_DIR / filename
        if file_path.exists():
            files.append(file_path)
    return files


def extract_company_info(file_path: Path) -> Dict[str, Any]:
    """Extract company info from filename."""
    parts = file_path.stem.split("_")
    return {
        "company_number": parts[0] if len(parts) > 0 else "",
        "ticker": parts[1] if len(parts) > 1 else "",
        "year": int(parts[2]) if len(parts) > 2 else 2024,
        "firm_name": parts[1] if len(parts) > 1 else "Unknown",
    }


def build_harms_prompt(text: str, company_info: Dict[str, Any]) -> str:
    """Build the harms classification prompt."""
    max_chars = 25000
    if len(text) > max_chars:
        text = text[:12500] + "\n\n[...content truncated...]\n\n" + text[-12500:]

    return f"""You are an expert analyst for the UK AI Safety Institute, analyzing company annual reports for mentions of AI-related HARMS.

## CONTEXT
Company: {company_info['firm_name']}
Report Year: {company_info['year']}

## TASK
Analyze this annual report excerpt and determine if it mentions any AI-related HARMS.

AI-related harms include:
- Risks to safety, security, or privacy from AI systems
- Bias, discrimination, or unfair outcomes from AI
- Job displacement or workforce impacts from AI
- Misinformation or content authenticity issues
- Cybersecurity threats enabled by AI
- Operational failures or errors from AI systems
- Regulatory or legal risks from AI use
- Reputational damage from AI incidents
- Environmental impacts from AI infrastructure

## REPORT EXCERPT
\"\"\"
{text}
\"\"\"

## OUTPUT FORMAT
Return a JSON object with these fields:
{{
    "harms_mentioned": true/false,
    "confidence": 0.0-1.0,
    "evidence": ["quote1", "quote2", ...],
    "reasoning": "Brief explanation"
}}

Return ONLY valid JSON."""


def build_adoption_prompt(text: str, company_info: Dict[str, Any]) -> str:
    """Build the adoption type classification prompt."""
    max_chars = 25000
    if len(text) > max_chars:
        text = text[:12500] + "\n\n[...content truncated...]\n\n" + text[-12500:]

    return f"""You are an expert analyst classifying AI adoption types in company annual reports.

## CONTEXT
Company: {company_info['firm_name']}
Report Year: {company_info['year']}

## TASK
Classify the types of AI adoption mentioned in this report.

## AI ADOPTION CATEGORIES
- **non_llm**: Traditional AI/ML (computer vision, predictive analytics, fraud detection)
- **llm**: Large Language Models (GPT, ChatGPT, chatbots, text generation)
- **agentic**: Autonomous AI agents (self-directed systems)

## REPORT EXCERPT
\"\"\"
{text}
\"\"\"

## OUTPUT FORMAT
Return a JSON object:
{{
    "adoption_types": ["non_llm", "llm", "agentic"],
    "confidence": 0.0-1.0,
    "evidence": {{"non_llm": ["quote..."], "llm": ["quote..."]}},
    "reasoning": "Brief explanation"
}}

Return ONLY valid JSON."""


def classify_with_openrouter(
    client: OpenRouterClient,
    model_id: str,
    prompt: str,
    logger,
) -> Dict[str, Any]:
    """Run classification using OpenRouter."""
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    try:
        response = client.chat_completion(
            model=model_id,
            messages=messages,
            temperature=0.0,
        )

        latency = time.time() - start_time
        content = response["choices"][0]["message"]["content"]

        # Parse JSON response
        try:
            result = json.loads(content)
            result["success"] = True
            result["latency_seconds"] = round(latency, 2)
        except json.JSONDecodeError:
            result = {
                "success": False,
                "error": "JSON parse error",
                "raw_response": content[:500],
                "latency_seconds": round(latency, 2),
            }

        return result

    except Exception as e:
        latency = time.time() - start_time
        logger.error(f"OpenRouter error: {e}")
        return {
            "success": False,
            "error": str(e),
            "latency_seconds": round(latency, 2),
        }


def compare_models(
    num_samples: int = 3,
    classifiers_to_test: List[str] = ["harms"],
    rate_limit: float = 2.0,
) -> Dict[str, Any]:
    """
    Compare classification results across model families via OpenRouter.

    Args:
        num_samples: Number of sample files to test
        classifiers_to_test: Which classifiers to compare
        rate_limit: Delay between API calls

    Returns:
        Comparison results dict
    """
    settings = get_settings()

    if not settings.openrouter_api_key:
        print("ERROR: OPENROUTER_API_KEY not set in environment")
        return {"error": "OPENROUTER_API_KEY not configured"}

    run_id = f"openrouter_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logging(log_level="INFO", run_id=run_id)
    logger.info(f"Starting OpenRouter model comparison: {run_id}")

    client = OpenRouterClient(settings.openrouter_api_key)

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
        "models_tested": [m["name"] for m in MODELS_TO_COMPARE],
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

            # Build prompt
            if clf_name == "harms":
                prompt = build_harms_prompt(text, company_info)
            elif clf_name == "adoption":
                prompt = build_adoption_prompt(text, company_info)
            else:
                logger.warning(f"Unknown classifier: {clf_name}")
                continue

            # Test each model
            for model_info in MODELS_TO_COMPARE:
                model_id = model_info["id"]
                model_name = model_info["name"]
                logger.info(f"  Testing {clf_name} with {model_name}...")

                result = classify_with_openrouter(client, model_id, prompt, logger)

                # Extract primary label
                if result.get("success"):
                    if clf_name == "harms":
                        primary_label = "true" if result.get("harms_mentioned") else "false"
                        confidence = result.get("confidence", 0.5)
                    elif clf_name == "adoption":
                        types = result.get("adoption_types", [])
                        primary_label = ",".join(sorted(types)) if types else "none"
                        confidence = result.get("confidence", 0.5)
                    else:
                        primary_label = "unknown"
                        confidence = 0.5
                else:
                    primary_label = "error"
                    confidence = 0.0

                comparison["model_results"][model_name] = {
                    "primary_label": primary_label,
                    "confidence": confidence,
                    "latency_seconds": result.get("latency_seconds", 0),
                    "success": result.get("success", False),
                    "error": result.get("error"),
                }

                logger.info(
                    f"    {model_name}: {primary_label} "
                    f"(conf={confidence:.2f}, latency={result.get('latency_seconds', 0):.1f}s)"
                )

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
    print("\n" + "=" * 70)
    print("OPENROUTER MODEL FAMILY COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Samples tested: {len(sample_files)}")
    print(f"Models compared:")
    for m in MODELS_TO_COMPARE:
        print(f"  - {m['name']} ({m['family']})")
    print(f"\nAgreement by Classifier:")
    for clf_name, stats in by_classifier.items():
        print(f"  {clf_name}: {stats['full_agreement']}/{stats['total']} ({stats['agreement_rate']:.1%})")
    print(f"\nOverall Agreement: {results['overall_agreement_rate']:.1%}")
    print(f"\nResults saved to: {output_path}")
    print("=" * 70)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare classification across model families using OpenRouter"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of samples to test (default: 3)",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["harms"],
        choices=["harms", "adoption"],
        help="Classifiers to compare (default: harms)",
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

    return 0 if results.get("overall_agreement_rate", 0) >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())





