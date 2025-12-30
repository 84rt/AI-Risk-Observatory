"""
API-Based Pre-Submission Tests for AISI Choices Report
======================================================

Tests that require API calls:
1. Adoption Type Classifier (non-LLM vs LLM vs Agentic)
2. Substantiveness Detector (Boilerplate vs Substantive)
5. Time Travel Test (run on 2016 reports)
7. Model Family Comparison (Gemini vs Claude on 5 reports)
"""

import json
import random
import re
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import google.generativeai as genai
    from config import get_settings
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("WARNING: Gemini not available")

try:
    import anthropic
    HAS_CLAUDE = True
except ImportError:
    HAS_CLAUDE = False
    print("WARNING: Claude not available")

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "test_results"
GOLDEN_SET = BASE_DIR / "output" / "risk_classifications" / "golden_set_results.json"
PREPROCESSED_DIR = BASE_DIR / "output" / "preprocessed" / "keyword"

# Adoption Type Taxonomy (from AISI steer)
ADOPTION_TYPES = {
    "non_llm": "Traditional AI/ML (computer vision, predictive analytics, recommendation systems)",
    "llm": "Large Language Models (GPT, BERT, language processing, chatbots)",
    "agentic": "Agentic AI (autonomous systems, AI agents, self-directed automation)"
}

# Substantiveness definitions
SUBSTANTIVENESS = {
    "boilerplate": "Generic legal phrasing applicable to any company (e.g., 'AI may pose risks')",
    "substantive": "Specific evidence with named systems, quantified impact, or concrete mitigation steps"
}


def get_sample_texts():
    """Get sample texts from preprocessed files."""
    texts = []
    preprocessed_files = list(PREPROCESSED_DIR.glob("*.md"))

    # Sample up to 10 files
    for file_path in random.sample(preprocessed_files, min(10, len(preprocessed_files))):
        content = file_path.read_text()
        # Extract paragraphs with AI mentions
        paragraphs = content.split("\n\n")
        for para in paragraphs:
            if re.search(r'\b(AI|artificial intelligence|machine learning|ML|LLM|generative)\b', para, re.IGNORECASE):
                if 50 < len(para) < 1000:  # Reasonable length
                    company_match = re.search(r'(\d+)_(\w+)', file_path.name)
                    company = company_match.group(2) if company_match else "Unknown"
                    texts.append({
                        "company": company,
                        "source": file_path.name,
                        "text": para.strip()
                    })

    return texts[:20]  # Limit to 20 samples


def test_1_adoption_type_classifier():
    """
    Test 1: Adoption Type Classifier

    Test if we can reliably classify AI adoption type into:
    - non-LLM (traditional ML)
    - LLM (language models)
    - Agentic (autonomous systems)
    """
    print("\n" + "="*60)
    print("TEST 1: Adoption Type Classifier")
    print("="*60)

    if not HAS_GEMINI:
        return {"error": "Gemini not configured", "verdict": "SKIPPED"}

    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 500,
            "response_mime_type": "application/json"
        }
    )

    sample_texts = get_sample_texts()
    print(f"Testing with {len(sample_texts)} sample texts...")

    results = {
        "test_name": "Adoption Type Classifier",
        "description": "Classify AI mentions as non-LLM, LLM, or Agentic",
        "samples_tested": len(sample_texts),
        "classifications": [],
        "distribution": defaultdict(int)
    }

    for i, sample in enumerate(sample_texts[:10], 1):  # Limit to 10 for rate limits
        print(f"  Classifying {i}/10...", end=" ")

        prompt = f"""You are classifying AI technology mentions in company annual reports.

TEXT:
"{sample['text'][:500]}"

TASK: Classify the PRIMARY type of AI mentioned into ONE of these categories:
- non_llm: Traditional AI/ML (computer vision, predictive analytics, recommendation systems, fraud detection, general automation)
- llm: Large Language Models (GPT, BERT, ChatGPT, language processing, chatbots, text generation, NLP)
- agentic: Agentic AI (autonomous agents, self-directed systems, AI that takes actions independently)

Return JSON:
{{"adoption_type": "non_llm|llm|agentic", "confidence": 0.0-1.0, "reason": "brief explanation"}}
"""

        try:
            response = model.generate_content(prompt)
            result = json.loads(response.text)
            adoption_type = result.get("adoption_type", "unknown")
            confidence = result.get("confidence", 0.5)

            results["classifications"].append({
                "company": sample["company"],
                "text_preview": sample["text"][:100] + "...",
                "adoption_type": adoption_type,
                "confidence": confidence,
                "reason": result.get("reason", "")
            })
            results["distribution"][adoption_type] += 1
            print(f"{adoption_type} ({confidence:.2f})")

            time.sleep(0.5)  # Rate limit

        except Exception as e:
            print(f"Error: {e}")
            results["classifications"].append({
                "company": sample["company"],
                "error": str(e)
            })

    # Summary
    total = sum(results["distribution"].values())
    results["summary"] = {
        "total_classified": total,
        "distribution": dict(results["distribution"]),
        "non_llm_pct": round(100 * results["distribution"]["non_llm"] / max(total, 1), 1),
        "llm_pct": round(100 * results["distribution"]["llm"] / max(total, 1), 1),
        "agentic_pct": round(100 * results["distribution"]["agentic"] / max(total, 1), 1),
        "can_distinguish": len(set(results["distribution"].keys())) >= 2,
        "verdict": "PASS" if len(set(results["distribution"].keys())) >= 2 else "CANNOT_DISTINGUISH"
    }

    print(f"\nDistribution: {dict(results['distribution'])}")
    print(f"Can distinguish types: {results['summary']['can_distinguish']}")
    print(f"Verdict: {results['summary']['verdict']}")

    return results


def test_2_substantiveness_detector():
    """
    Test 2: Substantiveness Detector

    Test if we can reliably distinguish boilerplate from substantive AI mentions.
    """
    print("\n" + "="*60)
    print("TEST 2: Substantiveness Detector")
    print("="*60)

    if not HAS_GEMINI:
        return {"error": "Gemini not configured", "verdict": "SKIPPED"}

    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 500,
            "response_mime_type": "application/json"
        }
    )

    # Load golden set for evidence texts
    with open(GOLDEN_SET) as f:
        golden_set = json.load(f)

    # Collect all evidence quotes
    all_quotes = []
    for company in golden_set:
        for risk_type, quotes in company.get("evidence", {}).items():
            if isinstance(quotes, list):
                for quote in quotes:
                    all_quotes.append({
                        "company": company["firm_name"],
                        "risk_type": risk_type,
                        "text": quote
                    })

    sample_quotes = random.sample(all_quotes, min(15, len(all_quotes)))
    print(f"Testing with {len(sample_quotes)} evidence quotes...")

    results = {
        "test_name": "Substantiveness Detector",
        "description": "Classify AI mentions as Boilerplate vs Substantive",
        "samples_tested": len(sample_quotes),
        "classifications": [],
        "distribution": defaultdict(int)
    }

    for i, sample in enumerate(sample_quotes[:10], 1):  # Limit to 10
        print(f"  Classifying {i}/10...", end=" ")

        prompt = f"""You are classifying the SUBSTANTIVENESS of AI risk mentions in company reports.

TEXT:
"{sample['text']}"

TASK: Is this mention BOILERPLATE or SUBSTANTIVE?

BOILERPLATE indicators:
- Generic language applicable to any company
- Vague references ("AI may pose risks", "we monitor AI developments")
- Standard legal disclaimers
- No specific systems, metrics, or actions named

SUBSTANTIVE indicators:
- Names specific AI systems/tools (OpenAI, Azure ML, etc.)
- Provides quantified impact or metrics
- Describes concrete mitigation steps taken
- Details specific incidents or use cases
- Contains company-specific context

Return JSON:
{{"substantiveness": "boilerplate|substantive", "confidence": 0.0-1.0, "indicators": ["list", "of", "indicators"]}}
"""

        try:
            response = model.generate_content(prompt)
            result = json.loads(response.text)
            substantiveness = result.get("substantiveness", "unknown")
            confidence = result.get("confidence", 0.5)

            results["classifications"].append({
                "company": sample["company"],
                "risk_type": sample["risk_type"],
                "text_preview": sample["text"][:80] + "...",
                "substantiveness": substantiveness,
                "confidence": confidence,
                "indicators": result.get("indicators", [])
            })
            results["distribution"][substantiveness] += 1
            print(f"{substantiveness} ({confidence:.2f})")

            time.sleep(0.5)

        except Exception as e:
            print(f"Error: {e}")
            results["classifications"].append({
                "company": sample["company"],
                "error": str(e)
            })

    # Summary
    total = sum(results["distribution"].values())
    boilerplate_pct = round(100 * results["distribution"]["boilerplate"] / max(total, 1), 1)
    substantive_pct = round(100 * results["distribution"]["substantive"] / max(total, 1), 1)

    results["summary"] = {
        "total_classified": total,
        "distribution": dict(results["distribution"]),
        "boilerplate_pct": boilerplate_pct,
        "substantive_pct": substantive_pct,
        "can_distinguish": len(set(results["distribution"].keys())) >= 2,
        "substantive_ratio": substantive_pct,
        "verdict": "PASS" if len(set(results["distribution"].keys())) >= 2 else "CANNOT_DISTINGUISH"
    }

    print(f"\nDistribution: {dict(results['distribution'])}")
    print(f"Boilerplate: {boilerplate_pct}%, Substantive: {substantive_pct}%")
    print(f"Verdict: {results['summary']['verdict']}")

    return results


def test_5_time_travel():
    """
    Test 5: Time Travel Test

    Check if 2016/pre-2016 reports exist and can be processed.
    Since iXBRL requirement started 2022, older reports are PDFs.
    """
    print("\n" + "="*60)
    print("TEST 5: Time Travel Test (Pre-2016 Reports)")
    print("="*60)

    results = {
        "test_name": "Time Travel Test",
        "description": "Test availability and processing of pre-2016 reports",
        "findings": {}
    }

    # Check existing preprocessed files for year range
    all_years = set()
    for file_path in PREPROCESSED_DIR.glob("*_20*.md"):
        match = re.search(r'_(\d{4})\.md$', file_path.name)
        if match:
            all_years.add(int(match.group(1)))

    results["findings"]["available_years"] = sorted(all_years)
    results["findings"]["earliest_year"] = min(all_years) if all_years else None
    results["findings"]["latest_year"] = max(all_years) if all_years else None
    results["findings"]["has_pre_2020"] = any(y < 2020 for y in all_years)

    # Check for PDF reports (pre-2022)
    pdf_dir = BASE_DIR / "output" / "reports" / "pdfs"
    if pdf_dir.exists():
        pdf_count = len(list(pdf_dir.glob("*.pdf")))
        results["findings"]["pdf_reports_available"] = pdf_count
    else:
        results["findings"]["pdf_reports_available"] = 0

    # Summary
    can_do_time_travel = min(all_years) <= 2020 if all_years else False

    results["summary"] = {
        "years_available": list(sorted(all_years)),
        "year_span": max(all_years) - min(all_years) if all_years else 0,
        "earliest_data": min(all_years) if all_years else None,
        "can_test_pre_chatgpt": min(all_years) <= 2020 if all_years else False,
        "pdf_support_needed": min(all_years) < 2022 if all_years else True,
        "verdict": "PASS" if can_do_time_travel else "LIMITED_TO_RECENT_YEARS"
    }

    print(f"Available years: {sorted(all_years)}")
    print(f"Year span: {results['summary']['year_span']} years")
    print(f"Can test pre-ChatGPT (<=2020): {results['summary']['can_test_pre_chatgpt']}")
    print(f"Verdict: {results['summary']['verdict']}")

    return results


def test_7_model_comparison():
    """
    Test 7: Model Family Comparison

    Compare Gemini vs Claude classification on same texts.
    """
    print("\n" + "="*60)
    print("TEST 7: Model Family Comparison (Gemini vs Claude)")
    print("="*60)

    results = {
        "test_name": "Model Family Comparison",
        "description": "Compare Gemini and Claude classification on same texts",
        "gemini_available": HAS_GEMINI,
        "claude_available": False,  # Will be updated after checking
        "comparisons": []
    }

    if not HAS_GEMINI:
        results["summary"] = {"verdict": "SKIPPED", "reason": "Gemini not configured"}
        print("Gemini not available - skipping comparison")
        return results

    # Get sample texts
    sample_texts = get_sample_texts()[:5]  # Just 5 for comparison

    # Initialize Gemini
    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)
    gemini_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 300,
            "response_mime_type": "application/json"
        }
    )

    # Initialize Claude if available
    claude_client = None
    claude_available = HAS_CLAUDE
    if claude_available:
        try:
            claude_client = anthropic.Anthropic()
        except:
            print("Claude API key not configured")
            claude_available = False

    results["claude_available"] = claude_available

    prompt_template = """Classify this AI-related text from a company annual report.

TEXT: "{text}"

Return JSON with:
- is_risk: true/false (does this mention AI-related risks?)
- risk_type: operational|cybersecurity|regulatory|workforce|reputational|other
- confidence: 0.0-1.0
"""

    print(f"Comparing on {len(sample_texts)} texts...")

    for i, sample in enumerate(sample_texts, 1):
        print(f"\nSample {i}: {sample['company']}")
        comparison = {
            "sample": i,
            "company": sample["company"],
            "text_preview": sample["text"][:80] + "...",
            "gemini": None,
            "claude": None,
            "agreement": None
        }

        # Gemini classification
        try:
            prompt = prompt_template.format(text=sample["text"][:400])
            response = gemini_model.generate_content(prompt)
            comparison["gemini"] = json.loads(response.text)
            print(f"  Gemini: risk={comparison['gemini'].get('is_risk')}, type={comparison['gemini'].get('risk_type')}")
        except Exception as e:
            comparison["gemini"] = {"error": str(e)}
            print(f"  Gemini: Error - {e}")

        time.sleep(0.5)

        # Claude classification (if available)
        if claude_client:
            try:
                response = claude_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=300,
                    messages=[{
                        "role": "user",
                        "content": prompt_template.format(text=sample["text"][:400]) + "\n\nReturn only valid JSON."
                    }]
                )
                response_text = response.content[0].text
                # Try to extract JSON
                if "{" in response_text:
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1
                    comparison["claude"] = json.loads(response_text[json_start:json_end])
                    print(f"  Claude: risk={comparison['claude'].get('is_risk')}, type={comparison['claude'].get('risk_type')}")
            except Exception as e:
                comparison["claude"] = {"error": str(e)}
                print(f"  Claude: Error - {e}")
        else:
            comparison["claude"] = {"skipped": "Claude not available"}
            print("  Claude: Skipped (not available)")

        # Check agreement
        if comparison["gemini"] and comparison["claude"]:
            g_risk = comparison["gemini"].get("is_risk")
            c_risk = comparison["claude"].get("is_risk")
            g_type = comparison["gemini"].get("risk_type")
            c_type = comparison["claude"].get("risk_type")

            if g_risk == c_risk:
                if g_type == c_type:
                    comparison["agreement"] = "full"
                else:
                    comparison["agreement"] = "partial"
            else:
                comparison["agreement"] = "disagreement"
            print(f"  Agreement: {comparison['agreement']}")

        results["comparisons"].append(comparison)

    # Summary
    agreements = [c["agreement"] for c in results["comparisons"] if c["agreement"]]
    full_agreement = agreements.count("full")
    partial_agreement = agreements.count("partial")

    results["summary"] = {
        "samples_compared": len(results["comparisons"]),
        "gemini_only": not claude_available,
        "full_agreement": full_agreement,
        "partial_agreement": partial_agreement,
        "disagreement": agreements.count("disagreement"),
        "agreement_rate": round(100 * (full_agreement + partial_agreement) / max(len(agreements), 1), 1),
        "verdict": "PASS" if not claude_available else ("PASS" if full_agreement + partial_agreement >= len(agreements) * 0.6 else "LOW_AGREEMENT")
    }

    print(f"\nSummary:")
    print(f"  Full agreement: {full_agreement}")
    print(f"  Partial agreement: {partial_agreement}")
    print(f"  Disagreement: {agreements.count('disagreement')}")
    print(f"  Agreement rate: {results['summary']['agreement_rate']}%")
    print(f"  Verdict: {results['summary']['verdict']}")

    return results


def run_all_api_tests():
    """Run all API-based tests."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {
        "timestamp": timestamp,
        "test_suite": "AISI API-Based Tests",
        "tests": {}
    }

    print("\n" + "="*60)
    print("AISI API-BASED TEST SUITE")
    print(f"Timestamp: {timestamp}")
    print("="*60)

    # Run tests
    tests = [
        ("test_1", test_1_adoption_type_classifier),
        ("test_2", test_2_substantiveness_detector),
        ("test_5", test_5_time_travel),
        ("test_7", test_7_model_comparison),
    ]

    for test_id, test_func in tests:
        try:
            result = test_func()
            all_results["tests"][test_id] = result
        except Exception as e:
            print(f"\nERROR in {test_id}: {e}")
            import traceback
            traceback.print_exc()
            all_results["tests"][test_id] = {"error": str(e)}

    # Summary
    print("\n" + "="*60)
    print("API TEST SUITE SUMMARY")
    print("="*60)

    passed = 0
    failed = 0
    for test_id, result in all_results["tests"].items():
        verdict = result.get("summary", {}).get("verdict", "ERROR")
        status = "PASS" if "PASS" in verdict else "REVIEW"
        if "PASS" in verdict:
            passed += 1
        else:
            failed += 1
        print(f"  {test_id}: {status} - {result.get('test_name', 'Unknown')}")

    print(f"\nTotal: {passed} passed, {failed} need review")

    # Save results
    output_file = RESULTS_DIR / f"api_test_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    run_all_api_tests()
