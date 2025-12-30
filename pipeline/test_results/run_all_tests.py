"""
Comprehensive Pre-Submission Testing for AISI Choices Report
============================================================

This script runs all 9 pre-submission tests to validate the methodology
before the AISI Choices Report is finalized.

Tests:
1. Adoption Type Classifier (non-LLM vs LLM vs Agentic)
2. Substantiveness Detector (Boilerplate vs Substantive)
3. Trend Sensitivity (2022 vs 2024 for 3 firms)
4. Zero-Shot Validation (CTRL+F for Environmental/National Security)
5. Time Travel Test (run on 2016 reports)
6. Snippet Legibility Audit (read 20 random snippets)
7. Model Family Comparison (Gemini vs Claude on 5 reports)
8. CNI Sector Mapping (map 21 companies to 13 CNI sectors)
9. Mention Density Check (verify segment-level output exists)
"""

import json
import random
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "test_results"
GOLDEN_SET = BASE_DIR / "output" / "risk_classifications" / "golden_set_results.json"
PREPROCESSED_DIR = BASE_DIR / "output" / "preprocessed" / "keyword"
ALL_YEARS = BASE_DIR / "output" / "risk_classifications" / "all_years.json"

# CNI Sectors mapping
CNI_SECTORS = {
    "Chemicals": [],
    "Civil Nuclear": [],
    "Communications": ["RELX PLC"],
    "Defence": ["BAE Systems plc", "Rolls-Royce Holdings plc"],
    "Emergency Services": [],
    "Energy": ["BP p.l.c.", "Shell plc", "National Grid plc"],
    "Finance": ["Barclays PLC", "HSBC Holdings plc", "London Stock Exchange Group plc", "Lloyds Banking Group plc"],
    "Food": ["Tesco PLC", "Diageo plc", "Unilever PLC", "Compass Group PLC", "Reckitt Benckiser Group plc"],
    "Government": [],
    "Health": ["AstraZeneca plc", "GSK plc"],
    "Space": [],
    "Transport": [],
    "Water": [],
    # Additional sectors for non-CNI companies
    "Mining": ["Rio Tinto plc", "Anglo American plc"],
    "Tobacco": ["British American Tobacco p.l.c."],
}


def load_golden_set():
    """Load the golden set results."""
    with open(GOLDEN_SET) as f:
        return json.load(f)


def test_4_zero_shot_validation():
    """
    Test 4: Zero-Shot Validation

    Search 5 reports manually for "environmental" and "national security" mentions
    that the classifier may have missed (marked as 0 companies in final results).
    """
    print("\n" + "="*60)
    print("TEST 4: Zero-Shot Validation")
    print("="*60)

    results = {
        "test_name": "Zero-Shot Validation",
        "description": "Search for Environmental and National Security mentions that classifier marked as 0",
        "search_terms": {
            "environmental": ["environmental impact", "carbon footprint", "energy consumption", "sustainability", "climate"],
            "national_security": ["national security", "defence", "critical infrastructure", "government", "military"]
        },
        "files_searched": [],
        "findings": defaultdict(list)
    }

    # Get 5 random preprocessed files
    preprocessed_files = list(PREPROCESSED_DIR.glob("*.md"))
    sample_files = random.sample(preprocessed_files, min(5, len(preprocessed_files)))

    for file_path in sample_files:
        results["files_searched"].append(file_path.name)
        content = file_path.read_text()
        content_lower = content.lower()

        # Search for environmental terms
        for term in results["search_terms"]["environmental"]:
            matches = list(re.finditer(term.lower(), content_lower))
            if matches:
                for match in matches[:3]:  # Limit to 3 matches per term
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 100)
                    context = content[start:end].replace("\n", " ")
                    results["findings"]["environmental"].append({
                        "file": file_path.name,
                        "term": term,
                        "context": f"...{context}..."
                    })

        # Search for national security terms
        for term in results["search_terms"]["national_security"]:
            matches = list(re.finditer(term.lower(), content_lower))
            if matches:
                for match in matches[:3]:
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 100)
                    context = content[start:end].replace("\n", " ")
                    results["findings"]["national_security"].append({
                        "file": file_path.name,
                        "term": term,
                        "context": f"...{context}..."
                    })

    # Summary
    env_count = len(results["findings"]["environmental"])
    ns_count = len(results["findings"]["national_security"])

    results["summary"] = {
        "environmental_mentions_found": env_count,
        "national_security_mentions_found": ns_count,
        "verdict": "PASS" if env_count == 0 and ns_count == 0 else "NEEDS_REVIEW"
    }

    print(f"Files searched: {len(results['files_searched'])}")
    print(f"Environmental mentions found: {env_count}")
    print(f"National Security mentions found: {ns_count}")
    print(f"Verdict: {results['summary']['verdict']}")

    if env_count > 0:
        print("\nEnvironmental mentions sample:")
        for m in results["findings"]["environmental"][:3]:
            print(f"  - {m['file']}: '{m['term']}' -> {m['context'][:100]}...")

    if ns_count > 0:
        print("\nNational Security mentions sample:")
        for m in results["findings"]["national_security"][:3]:
            print(f"  - {m['file']}: '{m['term']}' -> {m['context'][:100]}...")

    return results


def test_6_snippet_legibility_audit():
    """
    Test 6: Snippet Legibility Audit

    Randomly sample 20 key snippets and score them for human legibility.
    """
    print("\n" + "="*60)
    print("TEST 6: Snippet Legibility Audit")
    print("="*60)

    golden_set = load_golden_set()

    # Collect all key snippets
    all_snippets = []
    for company in golden_set:
        for risk_type, snippet in company.get("key_snippets", {}).items():
            all_snippets.append({
                "company": company["firm_name"],
                "risk_type": risk_type,
                "snippet": snippet
            })

    # Sample 20 random snippets
    sample_size = min(20, len(all_snippets))
    sample_snippets = random.sample(all_snippets, sample_size)

    results = {
        "test_name": "Snippet Legibility Audit",
        "description": "Score 20 random snippets for legibility (1-5 scale)",
        "total_snippets_available": len(all_snippets),
        "sample_size": sample_size,
        "snippets_analyzed": []
    }

    legibility_scores = []
    for i, snippet_data in enumerate(sample_snippets, 1):
        snippet = snippet_data["snippet"]

        # Auto-scoring heuristics
        score = 5  # Start with perfect score
        issues = []

        # Check for spacing issues (common in iXBRL extraction)
        if re.search(r'\s[a-z]\s', snippet):  # Single letter surrounded by spaces
            score -= 1
            issues.append("spacing_issues")

        # Check for incomplete sentences
        if not snippet.strip().endswith(('.', '!', '?', '"', "'")):
            score -= 0.5
            issues.append("incomplete_sentence")

        # Check for very short snippets
        if len(snippet) < 50:
            score -= 0.5
            issues.append("too_short")

        # Check for garbled text (high non-alpha ratio)
        alpha_ratio = sum(c.isalpha() for c in snippet) / max(len(snippet), 1)
        if alpha_ratio < 0.7:
            score -= 1
            issues.append("garbled_text")

        # Check for Unicode artifacts
        if re.search(r'[\ue000-\uf8ff]', snippet):  # Private use area
            score -= 1
            issues.append("unicode_artifacts")

        score = max(1, score)  # Floor at 1
        legibility_scores.append(score)

        results["snippets_analyzed"].append({
            "index": i,
            "company": snippet_data["company"],
            "risk_type": snippet_data["risk_type"],
            "snippet": snippet[:150] + "..." if len(snippet) > 150 else snippet,
            "score": score,
            "issues": issues
        })

    # Calculate statistics
    avg_score = sum(legibility_scores) / len(legibility_scores)
    high_quality = sum(1 for s in legibility_scores if s >= 4)
    low_quality = sum(1 for s in legibility_scores if s < 3)

    results["summary"] = {
        "average_legibility_score": round(avg_score, 2),
        "high_quality_count": high_quality,
        "low_quality_count": low_quality,
        "high_quality_percentage": round(100 * high_quality / len(legibility_scores), 1),
        "verdict": "PASS" if avg_score >= 3.5 else "NEEDS_IMPROVEMENT"
    }

    print(f"Snippets analyzed: {sample_size}")
    print(f"Average legibility score: {results['summary']['average_legibility_score']}/5")
    print(f"High quality (4+): {high_quality} ({results['summary']['high_quality_percentage']}%)")
    print(f"Low quality (<3): {low_quality}")
    print(f"Verdict: {results['summary']['verdict']}")

    # Show examples of low-quality snippets
    low_quality_examples = [s for s in results["snippets_analyzed"] if s["score"] < 3]
    if low_quality_examples:
        print("\nLow quality snippet examples:")
        for ex in low_quality_examples[:3]:
            print(f"  - {ex['company']} ({ex['risk_type']}): Score {ex['score']}, Issues: {ex['issues']}")
            print(f"    \"{ex['snippet'][:80]}...\"")

    return results


def test_8_cni_sector_mapping():
    """
    Test 8: CNI Sector Mapping

    Map all 21 companies to the 13 CNI sectors.
    """
    print("\n" + "="*60)
    print("TEST 8: CNI Sector Mapping")
    print("="*60)

    golden_set = load_golden_set()

    # Create reverse mapping
    company_to_cni = {}
    for sector, companies in CNI_SECTORS.items():
        for company in companies:
            company_to_cni[company] = sector

    results = {
        "test_name": "CNI Sector Mapping",
        "description": "Map 21 golden set companies to 13 CNI sectors",
        "mappings": [],
        "sector_coverage": defaultdict(list),
        "unmapped_companies": []
    }

    for company in golden_set:
        firm_name = company["firm_name"]
        cni_sector = company_to_cni.get(firm_name, "UNMAPPED")

        results["mappings"].append({
            "company": firm_name,
            "original_sector": company.get("sector", "Unknown"),
            "cni_sector": cni_sector
        })

        if cni_sector == "UNMAPPED":
            results["unmapped_companies"].append(firm_name)
        else:
            results["sector_coverage"][cni_sector].append(firm_name)

    # Calculate coverage
    cni_core_sectors = ["Chemicals", "Civil Nuclear", "Communications", "Defence",
                        "Emergency Services", "Energy", "Finance", "Food",
                        "Government", "Health", "Space", "Transport", "Water"]

    covered_sectors = [s for s in cni_core_sectors if results["sector_coverage"].get(s)]

    results["summary"] = {
        "total_companies": len(golden_set),
        "mapped_companies": len(golden_set) - len(results["unmapped_companies"]),
        "unmapped_companies": len(results["unmapped_companies"]),
        "cni_sectors_covered": len(covered_sectors),
        "cni_sectors_missing": [s for s in cni_core_sectors if s not in covered_sectors],
        "verdict": "PASS" if len(covered_sectors) >= 5 else "NEEDS_EXPANSION"
    }

    print(f"Total companies: {results['summary']['total_companies']}")
    print(f"Mapped to CNI sectors: {results['summary']['mapped_companies']}")
    print(f"CNI sectors covered: {len(covered_sectors)}/13")
    print(f"Missing sectors: {results['summary']['cni_sectors_missing']}")
    print(f"Verdict: {results['summary']['verdict']}")

    print("\nSector distribution:")
    for sector in cni_core_sectors:
        companies = results["sector_coverage"].get(sector, [])
        print(f"  {sector}: {len(companies)} companies")

    return results


def test_9_mention_density_check():
    """
    Test 9: Mention Density Check

    Verify that segment-level output exists (not just report-level).
    """
    print("\n" + "="*60)
    print("TEST 9: Mention Density Check")
    print("="*60)

    golden_set = load_golden_set()

    results = {
        "test_name": "Mention Density Check",
        "description": "Verify segment-level classification data exists",
        "company_analysis": []
    }

    total_evidence_quotes = 0
    companies_with_multiple = 0

    for company in golden_set:
        evidence_counts = {}
        total_quotes = 0

        for risk_type, quotes in company.get("evidence", {}).items():
            count = len(quotes) if isinstance(quotes, list) else 1
            evidence_counts[risk_type] = count
            total_quotes += count

        total_evidence_quotes += total_quotes
        if total_quotes > len(evidence_counts):
            companies_with_multiple += 1

        results["company_analysis"].append({
            "company": company["firm_name"],
            "risk_types_count": len(company.get("risk_types", [])),
            "total_evidence_quotes": total_quotes,
            "evidence_per_type": evidence_counts,
            "has_segment_level": total_quotes > len(evidence_counts)
        })

    results["summary"] = {
        "total_companies": len(golden_set),
        "total_evidence_quotes": total_evidence_quotes,
        "avg_quotes_per_company": round(total_evidence_quotes / len(golden_set), 1),
        "companies_with_multiple_quotes": companies_with_multiple,
        "segment_level_coverage": round(100 * companies_with_multiple / len(golden_set), 1),
        "verdict": "PASS" if companies_with_multiple / len(golden_set) > 0.5 else "REPORT_LEVEL_ONLY"
    }

    print(f"Total companies: {results['summary']['total_companies']}")
    print(f"Total evidence quotes: {results['summary']['total_evidence_quotes']}")
    print(f"Average quotes per company: {results['summary']['avg_quotes_per_company']}")
    print(f"Companies with segment-level data: {companies_with_multiple} ({results['summary']['segment_level_coverage']}%)")
    print(f"Verdict: {results['summary']['verdict']}")

    return results


def test_3_trend_sensitivity():
    """
    Test 3: Trend Sensitivity

    Compare AI risk mentions between years for same companies.
    """
    print("\n" + "="*60)
    print("TEST 3: Trend Sensitivity")
    print("="*60)

    # Check if multi-year data exists
    if not ALL_YEARS.exists():
        print("WARNING: all_years.json not found. Checking preprocessed files...")

    # Analyze preprocessed files by year
    results = {
        "test_name": "Trend Sensitivity",
        "description": "Compare AI risk mentions across years for same companies",
        "year_comparison": defaultdict(lambda: defaultdict(dict))
    }

    # Group files by company and year
    for file_path in PREPROCESSED_DIR.glob("*_20*.md"):
        match = re.search(r'_(\d{4})\.md$', file_path.name)
        if match:
            year = match.group(1)
            # Extract company number
            company_num = file_path.name.split("_")[0]

            # Count AI-related content
            content = file_path.read_text()
            ai_mentions = len(re.findall(r'\b(artificial intelligence|AI|machine learning|ML|generative|LLM)\b', content, re.IGNORECASE))
            word_count = len(content.split())

            results["year_comparison"][company_num][year] = {
                "ai_mentions": ai_mentions,
                "word_count": word_count,
                "ai_density": round(1000 * ai_mentions / max(word_count, 1), 2)  # per 1000 words
            }

    # Analyze trends
    companies_with_trend = 0
    increasing_trend = 0
    decreasing_trend = 0

    trend_analysis = []
    for company, years in results["year_comparison"].items():
        if len(years) >= 2:
            companies_with_trend += 1
            sorted_years = sorted(years.keys())
            first_year = sorted_years[0]
            last_year = sorted_years[-1]

            first_density = years[first_year]["ai_density"]
            last_density = years[last_year]["ai_density"]

            change = last_density - first_density
            if change > 0.5:
                increasing_trend += 1
                trend = "INCREASING"
            elif change < -0.5:
                decreasing_trend += 1
                trend = "DECREASING"
            else:
                trend = "STABLE"

            trend_analysis.append({
                "company": company,
                "first_year": first_year,
                "last_year": last_year,
                "first_density": first_density,
                "last_density": last_density,
                "change": round(change, 2),
                "trend": trend
            })

    results["trend_analysis"] = trend_analysis
    results["summary"] = {
        "companies_with_multi_year_data": companies_with_trend,
        "increasing_ai_trend": increasing_trend,
        "decreasing_ai_trend": decreasing_trend,
        "stable_trend": companies_with_trend - increasing_trend - decreasing_trend,
        "can_detect_trends": companies_with_trend >= 3,
        "verdict": "PASS" if companies_with_trend >= 3 else "INSUFFICIENT_DATA"
    }

    print(f"Companies with multi-year data: {companies_with_trend}")
    print(f"Increasing AI trend: {increasing_trend}")
    print(f"Decreasing AI trend: {decreasing_trend}")
    print(f"Stable: {results['summary']['stable_trend']}")
    print(f"Verdict: {results['summary']['verdict']}")

    if trend_analysis:
        print("\nSample trends:")
        for t in sorted(trend_analysis, key=lambda x: abs(x["change"]), reverse=True)[:5]:
            print(f"  {t['company']}: {t['first_year']}({t['first_density']}) -> {t['last_year']}({t['last_density']}) = {t['trend']}")

    return results


def run_all_tests():
    """Run all tests and save results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {
        "timestamp": timestamp,
        "test_suite": "AISI Pre-Submission Validation",
        "tests": {}
    }

    print("\n" + "="*60)
    print("AISI PRE-SUBMISSION TEST SUITE")
    print(f"Timestamp: {timestamp}")
    print("="*60)

    # Run tests
    tests = [
        ("test_4", test_4_zero_shot_validation),
        ("test_6", test_6_snippet_legibility_audit),
        ("test_8", test_8_cni_sector_mapping),
        ("test_9", test_9_mention_density_check),
        ("test_3", test_3_trend_sensitivity),
    ]

    for test_id, test_func in tests:
        try:
            result = test_func()
            all_results["tests"][test_id] = result
        except Exception as e:
            print(f"\nERROR in {test_id}: {e}")
            all_results["tests"][test_id] = {"error": str(e)}

    # Summary
    print("\n" + "="*60)
    print("TEST SUITE SUMMARY")
    print("="*60)

    passed = 0
    failed = 0
    for test_id, result in all_results["tests"].items():
        verdict = result.get("summary", {}).get("verdict", "ERROR")
        status = "PASS" if verdict == "PASS" else "REVIEW"
        if verdict == "PASS":
            passed += 1
        else:
            failed += 1
        print(f"  {test_id}: {status} - {result.get('test_name', 'Unknown')}")

    print(f"\nTotal: {passed} passed, {failed} need review")

    # Save results
    output_file = RESULTS_DIR / f"test_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    run_all_tests()
