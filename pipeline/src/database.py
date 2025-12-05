"""Database models and operations for AIRO pipeline."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

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


def get_database() -> Database:
    """Get database instance.

    Returns:
        Database instance
    """
    return Database()
