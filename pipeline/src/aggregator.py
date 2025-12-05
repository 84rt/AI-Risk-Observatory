"""Aggregate mention-level data to firm-year metrics."""

import json
import logging
from collections import Counter
from datetime import datetime
from typing import Optional

from .database import Database, Firm, Mention

logger = logging.getLogger(__name__)


# Governance maturity scoring
GOVERNANCE_SCORES = {
    "none": 0.0,
    "basic": 0.33,
    "intermediate": 0.66,
    "advanced": 1.0
}

# Materiality scoring
MATERIALITY_SCORES = {
    "unspecified": 0,
    "low": 1,
    "medium": 2,
    "high": 3
}

# Specificity scoring
SPECIFICITY_SCORES = {
    "boilerplate": 1,
    "contextual": 2,
    "concrete": 3
}


class FirmAggregator:
    """Aggregate mentions to firm-year level."""

    def __init__(self, database: Optional[Database] = None):
        """Initialize aggregator.

        Args:
            database: Database instance. If not provided, creates new one.
        """
        self.db = database or Database()

    def aggregate_firm_year(
        self,
        firm_id: str,
        report_year: int
    ) -> Optional[Firm]:
        """Aggregate all mentions for a firm-year into firm-level metrics.

        Args:
            firm_id: Firm identifier
            report_year: Report year

        Returns:
            Firm object with aggregated metrics, or None if no mentions
        """
        # Get all mentions for this firm-year
        mentions = self.db.get_mentions_for_firm(firm_id, report_year)

        if not mentions:
            logger.warning(
                f"No mentions found for {firm_id} in {report_year}"
            )
            return None

        logger.info(
            f"Aggregating {len(mentions)} mentions for "
            f"{mentions[0].firm_name} ({report_year})"
        )

        # Extract firm metadata from first mention
        firm_name = mentions[0].firm_name
        sector = mentions[0].sector
        sector_code = mentions[0].sector_code

        # Compute aggregates
        ai_mentioned = len(mentions) > 0
        total_ai_mentions = len(mentions)

        # Count risk mentions
        risk_mention_types = {
            "risk_statement", "incident_event", "regulatory_environment"
        }
        risk_mentions = [
            m for m in mentions
            if m.mention_type in risk_mention_types
        ]
        ai_risk_mentioned = len(risk_mentions) > 0
        total_ai_risk_mentions = len(risk_mentions)

        # Check for frontier AI
        frontier_mentions = [m for m in mentions if m.frontier_tech_flag]
        frontier_ai_mentioned = len(frontier_mentions) > 0

        # Tier 1 distribution
        tier_1_counts = Counter(
            m.tier_1_category for m in mentions
            if m.tier_1_category is not None
        )
        dominant_tier_1 = tier_1_counts.most_common(1)[0][0] if tier_1_counts else None
        tier_1_distribution = json.dumps(dict(tier_1_counts))

        # Max specificity
        specificity_levels = [m.specificity_level for m in mentions if m.specificity_level]
        max_specificity = self._get_max_by_score(
            specificity_levels,
            SPECIFICITY_SCORES
        )

        # Max materiality
        materiality_signals = [
            m.materiality_signal for m in mentions
            if m.materiality_signal
        ]
        max_materiality = self._get_max_by_score(
            materiality_signals,
            MATERIALITY_SCORES
        )

        # Governance
        governance_mentions = [
            m for m in mentions
            if m.governance_maturity and m.governance_maturity != "none"
        ]
        has_ai_governance = len(governance_mentions) > 0

        governance_levels = [m.governance_maturity for m in governance_mentions]
        max_governance_maturity = self._get_max_by_score(
            governance_levels,
            GOVERNANCE_SCORES
        )

        # Check if AI in Principal Risks section
        principal_risks_mentions = [
            m for m in mentions
            if m.report_section and "principal" in m.report_section.lower()
        ]
        ai_in_principal_risks = len(principal_risks_mentions) > 0

        # Derived metrics
        specificity_ratio = self._compute_specificity_ratio(mentions)
        mitigation_gap_score = self._compute_mitigation_gap(risk_mentions)

        # Create or update firm record
        session = self.db.get_session()
        try:
            # Check if firm exists
            firm = session.query(Firm).filter(
                Firm.firm_id == firm_id,
                Firm.report_year == report_year
            ).first()

            if firm:
                # Update existing
                firm.ai_mentioned = ai_mentioned
                firm.ai_risk_mentioned = ai_risk_mentioned
                firm.frontier_ai_mentioned = frontier_ai_mentioned
                firm.total_ai_mentions = total_ai_mentions
                firm.total_ai_risk_mentions = total_ai_risk_mentions
                firm.dominant_tier_1_category = dominant_tier_1
                firm.tier_1_distribution = tier_1_distribution
                firm.max_specificity_level = max_specificity
                firm.max_materiality_signal = max_materiality
                firm.has_ai_governance = has_ai_governance
                firm.max_governance_maturity = max_governance_maturity
                firm.ai_in_principal_risks = ai_in_principal_risks
                firm.specificity_ratio = specificity_ratio
                firm.mitigation_gap_score = mitigation_gap_score
                firm.last_updated = datetime.now().date()
            else:
                # Create new
                firm = Firm(
                    firm_id=firm_id,
                    firm_name=firm_name,
                    sector=sector,
                    sector_code=sector_code,
                    report_year=report_year,
                    ai_mentioned=ai_mentioned,
                    ai_risk_mentioned=ai_risk_mentioned,
                    frontier_ai_mentioned=frontier_ai_mentioned,
                    total_ai_mentions=total_ai_mentions,
                    total_ai_risk_mentions=total_ai_risk_mentions,
                    dominant_tier_1_category=dominant_tier_1,
                    tier_1_distribution=tier_1_distribution,
                    max_specificity_level=max_specificity,
                    max_materiality_signal=max_materiality,
                    has_ai_governance=has_ai_governance,
                    max_governance_maturity=max_governance_maturity,
                    ai_in_principal_risks=ai_in_principal_risks,
                    specificity_ratio=specificity_ratio,
                    mitigation_gap_score=mitigation_gap_score,
                    last_updated=datetime.now().date()
                )
                session.add(firm)

            session.commit()
            logger.info(f"Saved aggregated metrics for {firm_name} ({report_year})")

            return firm

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving firm aggregates: {e}")
            raise
        finally:
            session.close()

    @staticmethod
    def _get_max_by_score(values: list, score_map: dict) -> Optional[str]:
        """Get the value with maximum score.

        Args:
            values: List of values
            score_map: Dict mapping values to scores

        Returns:
            Value with max score, or None
        """
        if not values:
            return None

        return max(values, key=lambda x: score_map.get(x, 0))

    @staticmethod
    def _compute_specificity_ratio(mentions: list[Mention]) -> float:
        """Compute ratio of concrete mentions.

        Args:
            mentions: List of Mention objects

        Returns:
            Ratio of concrete mentions (0.0 - 1.0)
        """
        if not mentions:
            return 0.0

        concrete_count = sum(
            1 for m in mentions
            if m.specificity_level == "concrete"
        )

        return concrete_count / len(mentions)

    @staticmethod
    def _compute_mitigation_gap(risk_mentions: list[Mention]) -> float:
        """Compute mitigation gap score.

        Higher score = high severity risks with low governance.

        Args:
            risk_mentions: List of risk-related Mention objects

        Returns:
            Gap score (0.0 - 1.0)
        """
        if not risk_mentions:
            return 0.0

        # Compute average severity
        severity_scores = [
            MATERIALITY_SCORES.get(m.materiality_signal, 0)
            for m in risk_mentions
        ]
        avg_severity = sum(severity_scores) / len(severity_scores) if severity_scores else 0
        normalized_severity = avg_severity / 3.0  # Max score is 3

        # Get max governance maturity
        governance_levels = [
            m.governance_maturity for m in risk_mentions
            if m.governance_maturity
        ]

        if governance_levels:
            max_gov_level = max(
                governance_levels,
                key=lambda x: GOVERNANCE_SCORES.get(x, 0)
            )
            governance_score = GOVERNANCE_SCORES.get(max_gov_level, 0)
        else:
            governance_score = 0.0

        # Gap = high severity * low governance
        gap_score = normalized_severity * (1.0 - governance_score)

        return gap_score


def aggregate_firm(
    firm_id: str,
    report_year: int,
    database: Optional[Database] = None
) -> Optional[Firm]:
    """Convenience function to aggregate a firm-year.

    Args:
        firm_id: Firm identifier
        report_year: Report year
        database: Optional database instance

    Returns:
        Firm object with aggregated metrics
    """
    aggregator = FirmAggregator(database)
    return aggregator.aggregate_firm_year(firm_id, report_year)
