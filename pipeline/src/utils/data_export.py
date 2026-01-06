"""Data export utilities for AIRO classifier results.

Provides export functions for:
- JSON/CSV exports for human annotation review
- Full results archival
- Summary report generation
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Base paths
PIPELINE_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PIPELINE_ROOT / "output"
ANNOTATION_EXPORT_DIR = OUTPUT_DIR / "annotation_export"
CLASSIFICATION_RESULTS_DIR = OUTPUT_DIR / "classification_results"

# Ensure directories exist
ANNOTATION_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
CLASSIFICATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class DataExporter:
    """Export classification results for human review and archival."""

    def __init__(self, db=None):
        """Initialize the exporter.

        Args:
            db: Optional database connection for querying results
        """
        self.db = db

    def export_for_annotation(
        self,
        run_id: str,
        results: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Export results for human annotation review.

        Args:
            run_id: Run ID for this export
            results: List of classification results to export
            output_dir: Optional custom output directory

        Returns:
            Dict mapping export type to file path
        """
        if output_dir is None:
            output_dir = ANNOTATION_EXPORT_DIR / run_id

        output_dir.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # 1. Export all snippets as JSON
        snippets_json_path = output_dir / "snippets_for_review.json"
        self._export_snippets_json(results, snippets_json_path)
        exported_files["snippets_json"] = snippets_json_path

        # 2. Export all snippets as CSV
        snippets_csv_path = output_dir / "snippets_for_review.csv"
        self._export_snippets_csv(results, snippets_csv_path)
        exported_files["snippets_csv"] = snippets_csv_path

        # 3. Export low-confidence items for priority review
        low_conf_path = output_dir / "low_confidence_flags.csv"
        low_conf_count = self._export_low_confidence(results, low_conf_path)
        exported_files["low_confidence_csv"] = low_conf_path

        # 4. Generate summary report
        summary_path = output_dir / "summary_report.md"
        self._generate_summary_report(run_id, results, low_conf_count, summary_path)
        exported_files["summary_report"] = summary_path

        return exported_files

    def export_full_results(
        self,
        run_id: str,
        results: List[Dict[str, Any]],
        config: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Export complete results for archival.

        Args:
            run_id: Run ID for this export
            results: List of classification results
            config: Run configuration
            output_dir: Optional custom output directory

        Returns:
            Dict mapping export type to file path
        """
        if output_dir is None:
            output_dir = CLASSIFICATION_RESULTS_DIR / run_id

        output_dir.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # 1. Full JSON with all metadata
        full_json_path = output_dir / "full_results.json"
        full_data = {
            "run_id": run_id,
            "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "config": config,
            "results": results,
        }
        with open(full_json_path, "w", encoding="utf-8") as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False, default=str)
        exported_files["full_json"] = full_json_path

        # 2. Config snapshot
        config_path = output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)
        exported_files["config"] = config_path

        # 3. Per-classifier results
        by_classifier = self._group_by_classifier(results)
        for classifier, clf_results in by_classifier.items():
            clf_path = output_dir / f"{classifier}_results.json"
            with open(clf_path, "w", encoding="utf-8") as f:
                json.dump(clf_results, f, indent=2, ensure_ascii=False, default=str)
            exported_files[f"{classifier}_results"] = clf_path

        return exported_files

    def _export_snippets_json(
        self,
        results: List[Dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Export snippets grouped by company."""
        by_company = {}

        for result in results:
            firm_id = result.get("firm_id", "unknown")
            firm_name = result.get("firm_name", firm_id)

            if firm_name not in by_company:
                by_company[firm_name] = {
                    "firm_id": firm_id,
                    "firm_name": firm_name,
                    "snippets": [],
                }

            # Extract snippets from evidence
            evidence = result.get("evidence", [])
            if isinstance(evidence, dict):
                # Evidence is categorized by type
                for category, quotes in evidence.items():
                    if isinstance(quotes, list):
                        for quote in quotes:
                            by_company[firm_name]["snippets"].append({
                                "year": result.get("report_year"),
                                "classifier": result.get("classifier_type"),
                                "category": category,
                                "label": result.get("primary_label"),
                                "confidence": result.get("confidence_score"),
                                "text": quote,
                                "source_file": result.get("source_file"),
                                "review_priority": self._get_review_priority(
                                    result.get("confidence_score", 0.5)
                                ),
                            })
            elif isinstance(evidence, list):
                for quote in evidence:
                    by_company[firm_name]["snippets"].append({
                        "year": result.get("report_year"),
                        "classifier": result.get("classifier_type"),
                        "label": result.get("primary_label"),
                        "confidence": result.get("confidence_score"),
                        "text": quote,
                        "source_file": result.get("source_file"),
                        "review_priority": self._get_review_priority(
                            result.get("confidence_score", 0.5)
                        ),
                    })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(by_company, f, indent=2, ensure_ascii=False)

    def _export_snippets_csv(
        self,
        results: List[Dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Export snippets as flat CSV for spreadsheet review."""
        rows = []

        for result in results:
            firm_id = result.get("firm_id", "unknown")
            firm_name = result.get("firm_name", firm_id)

            evidence = result.get("evidence", [])
            if isinstance(evidence, dict):
                for category, quotes in evidence.items():
                    if isinstance(quotes, list):
                        for quote in quotes:
                            rows.append({
                                "company": firm_name,
                                "firm_id": firm_id,
                                "year": result.get("report_year"),
                                "classifier": result.get("classifier_type"),
                                "category": category,
                                "label": result.get("primary_label"),
                                "confidence": result.get("confidence_score"),
                                "snippet": quote[:500] if quote else "",
                                "source_file": result.get("source_file"),
                                "review_priority": self._get_review_priority(
                                    result.get("confidence_score", 0.5)
                                ),
                                "reviewer_decision": "",
                                "reviewer_notes": "",
                            })
            elif isinstance(evidence, list):
                for quote in evidence:
                    rows.append({
                        "company": firm_name,
                        "firm_id": firm_id,
                        "year": result.get("report_year"),
                        "classifier": result.get("classifier_type"),
                        "category": "",
                        "label": result.get("primary_label"),
                        "confidence": result.get("confidence_score"),
                        "snippet": quote[:500] if quote else "",
                        "source_file": result.get("source_file"),
                        "review_priority": self._get_review_priority(
                            result.get("confidence_score", 0.5)
                        ),
                        "reviewer_decision": "",
                        "reviewer_notes": "",
                    })

        if rows:
            fieldnames = list(rows[0].keys())
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    def _export_low_confidence(
        self,
        results: List[Dict[str, Any]],
        output_path: Path,
        threshold: float = 0.7,
    ) -> int:
        """Export low-confidence results for priority review."""
        low_conf_results = [
            r for r in results if r.get("confidence_score", 1.0) < threshold
        ]

        rows = []
        for result in low_conf_results:
            rows.append({
                "company": result.get("firm_name", result.get("firm_id")),
                "firm_id": result.get("firm_id"),
                "year": result.get("report_year"),
                "classifier": result.get("classifier_type"),
                "label": result.get("primary_label"),
                "confidence": result.get("confidence_score"),
                "reason": result.get("reasoning", ""),
                "source_file": result.get("source_file"),
                "needs_review": True,
            })

        if rows:
            fieldnames = list(rows[0].keys())
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        return len(rows)

    def _generate_summary_report(
        self,
        run_id: str,
        results: List[Dict[str, Any]],
        low_conf_count: int,
        output_path: Path,
    ) -> None:
        """Generate a human-readable summary report."""
        total = len(results)
        if total == 0:
            avg_confidence = 0.0
        else:
            confidences = [r.get("confidence_score", 0) for r in results if r.get("confidence_score")]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        by_classifier = self._group_by_classifier(results)
        by_company = self._group_by_company(results)

        report = f"""# Classification Results Summary

**Run ID:** {run_id}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview

| Metric | Value |
|--------|-------|
| Total Classifications | {total} |
| Unique Companies | {len(by_company)} |
| Classifiers Run | {len(by_classifier)} |
| Average Confidence | {avg_confidence:.2f} |
| Low Confidence (< 0.7) | {low_conf_count} |
| Review Priority High | {low_conf_count} |

## By Classifier

"""
        for classifier, clf_results in sorted(by_classifier.items()):
            clf_confidences = [r.get("confidence_score", 0) for r in clf_results if r.get("confidence_score")]
            clf_avg = sum(clf_confidences) / len(clf_confidences) if clf_confidences else 0.0
            report += f"### {classifier.title()}\n\n"
            report += f"- Total: {len(clf_results)}\n"
            report += f"- Avg Confidence: {clf_avg:.2f}\n\n"

        report += "## By Company\n\n"
        report += "| Company | Classifications | Avg Confidence |\n"
        report += "|---------|-----------------|----------------|\n"

        for company, comp_results in sorted(by_company.items()):
            comp_confidences = [r.get("confidence_score", 0) for r in comp_results if r.get("confidence_score")]
            comp_avg = sum(comp_confidences) / len(comp_confidences) if comp_confidences else 0.0
            report += f"| {company} | {len(comp_results)} | {comp_avg:.2f} |\n"

        report += """
## Files Generated

- `snippets_for_review.json` - All evidence snippets grouped by company
- `snippets_for_review.csv` - Flat format for spreadsheet review
- `low_confidence_flags.csv` - Items needing priority review
- `summary_report.md` - This summary

## Review Priority

- **1 = High** (confidence < 0.7) - Needs human verification
- **2 = Medium** (confidence 0.7-0.85) - Spot check recommended
- **3 = Low** (confidence > 0.85) - Auto-accept with sampling
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    def _get_review_priority(self, confidence: float) -> int:
        """Determine review priority based on confidence score."""
        if confidence < 0.7:
            return 1  # High priority
        elif confidence < 0.85:
            return 2  # Medium priority
        else:
            return 3  # Low priority

    def _group_by_classifier(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict]]:
        """Group results by classifier type."""
        by_classifier = {}
        for result in results:
            clf = result.get("classifier_type", "unknown")
            if clf not in by_classifier:
                by_classifier[clf] = []
            by_classifier[clf].append(result)
        return by_classifier

    def _group_by_company(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict]]:
        """Group results by company."""
        by_company = {}
        for result in results:
            company = result.get("firm_name", result.get("firm_id", "unknown"))
            if company not in by_company:
                by_company[company] = []
            by_company[company].append(result)
        return by_company


def export_run_metadata(
    run_id: str,
    config: Dict[str, Any],
    results_summary: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Path:
    """Export run metadata as JSON.

    Args:
        run_id: Run ID
        config: Run configuration
        results_summary: Summary statistics
        output_dir: Output directory (defaults to data/classification_runs/)

    Returns:
        Path to the metadata file
    """
    if output_dir is None:
        output_dir = PIPELINE_ROOT / "data" / "classification_runs"

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "started_at": config.get("started_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": config,
        "results_summary": results_summary,
    }

    output_path = output_dir / f"{run_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return output_path

