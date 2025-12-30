"""Database models and operations for AIRO pipeline."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import (
    Boolean, Column, Date, Float, ForeignKeyConstraint, Integer, String, Text,
    create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

Base = declarative_base()


class RiskClassification(Base):
    """Table for AI risk type classifications at the report level.

    This is the primary table for storing classification results.
    Each row represents one company's annual report for a specific year.
    """

    __tablename__ = "risk_classifications"

    # Primary key: composite of firm_id and report_year
    firm_id = Column(String, nullable=False, primary_key=True)
    report_year = Column(Integer, nullable=False, primary_key=True)

    # Company metadata
    firm_name = Column(String, nullable=False)
    company_number = Column(String, nullable=False)
    sector = Column(String, default="Unknown")

    # Classification results
    ai_mentioned = Column(Boolean, default=False)

    # Risk types as JSON array: ["cybersecurity", "regulatory_compliance", ...]
    risk_types = Column(Text, default="[]")

    # Evidence as JSON: {"cybersecurity": ["quote1", "quote2"], ...}
    evidence = Column(Text, default="{}")

    # Key snippets as JSON: {"cybersecurity": "key quote", ...}
    key_snippets = Column(Text, default="{}")

    # Confidence scores as JSON: {"cybersecurity": 0.9, ...}
    confidence_scores = Column(Text, default="{}")

    # LLM reasoning
    reasoning = Column(Text)

    # Metadata
    model_version = Column(String)
    classification_date = Column(Date)
    source_file = Column(String)  # Path to source markdown/report

    def __repr__(self):
        return f"<RiskClassification {self.firm_name} ({self.firm_id}) - {self.report_year}>"

    def get_risk_types(self) -> List[str]:
        """Get risk types as a Python list."""
        return json.loads(self.risk_types) if self.risk_types else []

    def get_evidence(self) -> dict:
        """Get evidence as a Python dict."""
        return json.loads(self.evidence) if self.evidence else {}

    def get_key_snippets(self) -> dict:
        """Get key snippets as a Python dict."""
        return json.loads(self.key_snippets) if self.key_snippets else {}

    def get_confidence_scores(self) -> dict:
        """Get confidence scores as a Python dict."""
        return json.loads(self.confidence_scores) if self.confidence_scores else {}

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "firm_id": self.firm_id,
            "firm_name": self.firm_name,
            "company_number": self.company_number,
            "sector": self.sector,
            "report_year": self.report_year,
            "ai_mentioned": self.ai_mentioned,
            "risk_types": self.get_risk_types(),
            "evidence": self.get_evidence(),
            "key_snippets": self.get_key_snippets(),
            "confidence_scores": self.get_confidence_scores(),
            "reasoning": self.reasoning,
            "model_version": self.model_version,
            "classification_date": str(self.classification_date) if self.classification_date else None,
            "source_file": self.source_file
        }


class Mention(Base):
    """Table for AI-relevant mentions from annual reports."""

    __tablename__ = "mentions"

    # Identifiers & Context
    mention_id = Column(String, primary_key=True)
    firm_id = Column(String, nullable=False)
    firm_name = Column(String, nullable=False)
    sector = Column(String, nullable=False)
    sector_code = Column(String)
    report_year = Column(Integer, nullable=False)
    report_section = Column(String)

    # Source Text & Traceability
    text_excerpt = Column(Text, nullable=False)
    page_number = Column(Integer)

    # Mention Type & AI Context
    mention_type = Column(String, nullable=False)
    ai_specificity = Column(String, nullable=False)
    frontier_tech_flag = Column(Boolean, default=False)

    # Risk Classification
    tier_1_category = Column(String)
    tier_2_driver = Column(String)

    # Severity & Substance
    specificity_level = Column(String, nullable=False)
    materiality_signal = Column(String)

    # Governance & Mitigation
    mitigation_mentioned = Column(Boolean, default=False)
    governance_maturity = Column(String)

    # LLM Metadata & Quality Control
    confidence_score = Column(Float)
    reasoning_summary = Column(Text)
    model_version = Column(String)
    extraction_date = Column(Date)
    review_status = Column(String, default="unreviewed")
    reviewer_notes = Column(Text)

    def __repr__(self):
        return f"<Mention {self.mention_id} - {self.firm_name} {self.report_year}>"


class Firm(Base):
    """Table for firm-year aggregated metrics."""

    __tablename__ = "firms"

    # Identifiers
    firm_id = Column(String, nullable=False, primary_key=True)
    firm_name = Column(String, nullable=False)
    sector = Column(String, nullable=False)
    sector_code = Column(String)
    report_year = Column(Integer, nullable=False, primary_key=True)

    # AI Adoption Indicators
    ai_mentioned = Column(Boolean, default=False)
    ai_risk_mentioned = Column(Boolean, default=False)
    frontier_ai_mentioned = Column(Boolean, default=False)
    total_ai_mentions = Column(Integer, default=0)
    total_ai_risk_mentions = Column(Integer, default=0)

    # Risk Profile (Aggregated)
    dominant_tier_1_category = Column(String)
    tier_1_distribution = Column(Text)  # JSON string
    max_specificity_level = Column(String)
    max_materiality_signal = Column(String)

    # Governance Profile (Aggregated)
    has_ai_governance = Column(Boolean, default=False)
    max_governance_maturity = Column(String)
    ai_in_principal_risks = Column(Boolean, default=False)

    # Derived Metrics
    specificity_ratio = Column(Float)
    mitigation_gap_score = Column(Float)

    # Metadata
    last_updated = Column(Date)

    def __repr__(self):
        return f"<Firm {self.firm_name} ({self.firm_id}) - {self.report_year}>"


class Database:
    """Database manager for AIRO pipeline."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database. If not provided, uses settings.
        """
        if db_path is None:
            db_path = settings.database_path

        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine
        db_url = f"sqlite:///{db_path}"
        self.engine = create_engine(db_url, echo=False)

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

        logger.info(f"Database initialized at {db_path}")

    def get_session(self) -> Session:
        """Get a new database session.

        Returns:
            SQLAlchemy session
        """
        return self.SessionLocal()

    def save_mention(
        self,
        session: Session,
        candidate,
        classification,
        model_version: str
    ) -> Mention:
        """Save a classified mention to the database.

        Args:
            session: Database session
            candidate: CandidateSpan object
            classification: Classification object
            model_version: LLM model version used

        Returns:
            Saved Mention object
        """
        mention = Mention(
            mention_id=candidate.span_id,
            firm_id=candidate.firm_id,
            firm_name=candidate.firm_name,
            sector=candidate.sector,
            report_year=candidate.report_year,
            report_section=candidate.report_section,
            text_excerpt=candidate.text,
            page_number=candidate.page_number,
            mention_type=classification.mention_type,
            ai_specificity=classification.ai_specificity,
            frontier_tech_flag=classification.frontier_tech_flag or False,
            tier_1_category=classification.tier_1_category,
            tier_2_driver=classification.tier_2_driver,
            specificity_level=classification.specificity_level,
            materiality_signal=classification.materiality_signal,
            mitigation_mentioned=classification.mitigation_mentioned or False,
            governance_maturity=classification.governance_maturity,
            confidence_score=classification.confidence_score,
            reasoning_summary=classification.reasoning_summary,
            model_version=model_version,
            extraction_date=datetime.now().date(),
            review_status="unreviewed"
        )

        session.add(mention)
        return mention

    def save_mentions_batch(
        self,
        results: list,
        model_version: str
    ) -> int:
        """Save a batch of classified mentions.

        Args:
            results: List of (candidate, classification) tuples
            model_version: LLM model version used

        Returns:
            Number of mentions saved
        """
        session = self.get_session()
        count = 0

        try:
            for candidate, classification in results:
                # Only save if relevant
                if classification.is_relevant:
                    self.save_mention(
                        session=session,
                        candidate=candidate,
                        classification=classification,
                        model_version=model_version
                    )
                    count += 1

            session.commit()
            logger.info(f"Saved {count} mentions to database")

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving mentions: {e}")
            raise
        finally:
            session.close()

        return count

    def get_mentions_for_firm(
        self,
        firm_id: str,
        report_year: int
    ) -> list[Mention]:
        """Get all mentions for a firm-year.

        Args:
            firm_id: Firm identifier
            report_year: Report year

        Returns:
            List of Mention objects
        """
        session = self.get_session()
        try:
            mentions = session.query(Mention).filter(
                Mention.firm_id == firm_id,
                Mention.report_year == report_year
            ).all()
            return mentions
        finally:
            session.close()

    def firm_exists(self, firm_id: str, report_year: int) -> bool:
        """Check if a firm-year already exists.

        Args:
            firm_id: Firm identifier
            report_year: Report year

        Returns:
            True if exists
        """
        session = self.get_session()
        try:
            exists = session.query(Firm).filter(
                Firm.firm_id == firm_id,
                Firm.report_year == report_year
            ).first() is not None
            return exists
        finally:
            session.close()

    # ========================================================================
    # Risk Classification Methods (Report-Level)
    # ========================================================================

    def save_risk_classification(
        self,
        firm_id: str,
        firm_name: str,
        company_number: str,
        report_year: int,
        classification_data: dict,
        model_version: str,
        sector: str = "Unknown",
        source_file: Optional[str] = None
    ) -> RiskClassification:
        """Save a risk classification to the database.

        Args:
            firm_id: Firm identifier (ticker)
            firm_name: Company name
            company_number: Companies House number
            report_year: Report year
            classification_data: Dict with risk_types, evidence, etc.
            model_version: LLM model used
            sector: Company sector
            source_file: Path to source file

        Returns:
            Saved RiskClassification object
        """
        session = self.get_session()

        try:
            # Check if exists and delete if so (upsert behavior)
            existing = session.query(RiskClassification).filter(
                RiskClassification.firm_id == firm_id,
                RiskClassification.report_year == report_year
            ).first()

            if existing:
                session.delete(existing)
                session.flush()

            # Create new record
            record = RiskClassification(
                firm_id=firm_id,
                report_year=report_year,
                firm_name=firm_name,
                company_number=company_number,
                sector=sector,
                ai_mentioned=classification_data.get("ai_mentioned", False),
                risk_types=json.dumps(classification_data.get("risk_types", [])),
                evidence=json.dumps(classification_data.get("evidence", {})),
                key_snippets=json.dumps(classification_data.get("key_snippets", {})),
                confidence_scores=json.dumps(classification_data.get("confidence_scores", {})),
                reasoning=classification_data.get("reasoning", ""),
                model_version=model_version,
                classification_date=datetime.now().date(),
                source_file=source_file
            )

            session.add(record)
            session.commit()

            logger.info(f"Saved risk classification: {firm_name} ({report_year})")
            return record

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving risk classification: {e}")
            raise
        finally:
            session.close()

    def get_risk_classification(
        self,
        firm_id: str,
        report_year: int
    ) -> Optional[RiskClassification]:
        """Get a risk classification for a firm-year.

        Args:
            firm_id: Firm identifier
            report_year: Report year

        Returns:
            RiskClassification object or None
        """
        session = self.get_session()
        try:
            return session.query(RiskClassification).filter(
                RiskClassification.firm_id == firm_id,
                RiskClassification.report_year == report_year
            ).first()
        finally:
            session.close()

    def get_all_risk_classifications(
        self,
        year: Optional[int] = None,
        firm_id: Optional[str] = None
    ) -> List[RiskClassification]:
        """Get all risk classifications, optionally filtered.

        Args:
            year: Filter by report year
            firm_id: Filter by firm

        Returns:
            List of RiskClassification objects
        """
        session = self.get_session()
        try:
            query = session.query(RiskClassification)

            if year:
                query = query.filter(RiskClassification.report_year == year)
            if firm_id:
                query = query.filter(RiskClassification.firm_id == firm_id)

            return query.order_by(
                RiskClassification.firm_id,
                RiskClassification.report_year
            ).all()
        finally:
            session.close()

    def risk_classification_exists(self, firm_id: str, report_year: int) -> bool:
        """Check if a risk classification exists.

        Args:
            firm_id: Firm identifier
            report_year: Report year

        Returns:
            True if exists
        """
        session = self.get_session()
        try:
            exists = session.query(RiskClassification).filter(
                RiskClassification.firm_id == firm_id,
                RiskClassification.report_year == report_year
            ).first() is not None
            return exists
        finally:
            session.close()

    def export_risk_classifications_to_json(
        self,
        output_path: Path,
        year: Optional[int] = None
    ) -> int:
        """Export risk classifications to JSON file.

        Args:
            output_path: Path to output JSON file
            year: Optional year filter

        Returns:
            Number of records exported
        """
        classifications = self.get_all_risk_classifications(year=year)

        data = [c.to_dict() for c in classifications]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(data)} risk classifications to {output_path}")
        return len(data)

    def get_risk_summary_stats(self) -> dict:
        """Get summary statistics for risk classifications.

        Returns:
            Dict with summary statistics
        """
        session = self.get_session()
        try:
            all_records = session.query(RiskClassification).all()

            if not all_records:
                return {"total_records": 0}

            # Count by year
            years = {}
            for r in all_records:
                years[r.report_year] = years.get(r.report_year, 0) + 1

            # Count AI mentions
            ai_mentioned_count = sum(1 for r in all_records if r.ai_mentioned)

            # Count risk types
            risk_type_counts = {}
            for r in all_records:
                for rt in r.get_risk_types():
                    risk_type_counts[rt] = risk_type_counts.get(rt, 0) + 1

            return {
                "total_records": len(all_records),
                "by_year": years,
                "ai_mentioned_count": ai_mentioned_count,
                "ai_mentioned_pct": ai_mentioned_count / len(all_records) * 100,
                "risk_type_counts": risk_type_counts,
                "unique_firms": len(set(r.firm_id for r in all_records))
            }
        finally:
            session.close()


def get_database() -> Database:
    """Get database instance.

    Returns:
        Database instance
    """
    return Database()
