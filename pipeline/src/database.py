"""Database models and operations for AIRO pipeline."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, ForeignKey, ForeignKeyConstraint,
    Integer, String, Text, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

Base = declarative_base()


def _max_confidence(confidences: dict) -> float:
    """Return the max confidence score in a dict."""
    if not isinstance(confidences, dict):
        return 0.0
    values = [
        score for score in confidences.values()
        if isinstance(score, (int, float))
    ]
    return max(values) if values else 0.0


def _normalize_adoption_signals(adoption_result: dict) -> dict:
    """Normalize adoption signals to a dict for storage."""
    if not isinstance(adoption_result, dict):
        return {}
    signals = adoption_result.get("adoption_signals") or adoption_result.get("adoption_confidences") or {}
    if isinstance(signals, list):
        out = {}
        for entry in signals:
            if isinstance(entry, dict):
                k = entry.get("type")
                v = entry.get("signal")
                if k is not None and isinstance(v, (int, float)):
                    out[str(k)] = float(v)
        return out
    if isinstance(signals, dict):
        return {
            str(k): float(v)
            for k, v in signals.items()
            if isinstance(v, (int, float))
        }
    return {}


def _normalize_risk_signals(risk_result: dict) -> dict:
    """Normalize risk signals to a dict for storage."""
    if not isinstance(risk_result, dict):
        return {}

    signals = risk_result.get("risk_signals")
    if isinstance(signals, list):
        out = {}
        for entry in signals:
            if not isinstance(entry, dict):
                continue
            k = entry.get("type")
            v = entry.get("signal")
            if k is not None and isinstance(v, (int, float)):
                out[str(k)] = float(v)
        return out

    legacy = risk_result.get("confidence_scores")
    if isinstance(legacy, dict):
        return {
            str(k): float(v)
            for k, v in legacy.items()
            if isinstance(v, (int, float))
        }

    return {}


class Company(Base):
    """Table for canonical company records."""

    __tablename__ = "companies"

    company_id = Column(String, primary_key=True)
    company_name = Column(String, nullable=False)
    company_number = Column(String)
    lei = Column(String)
    ticker = Column(String)
    sector = Column(String)
    index_name = Column(String)
    company_type = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    documents = relationship("Document", back_populates="company")

    def __repr__(self):
        return f"<Company {self.company_name} ({self.company_id})>"


class Document(Base):
    """Table for raw filing documents."""

    __tablename__ = "documents"

    document_id = Column(String, primary_key=True)
    company_id = Column(String, ForeignKey("companies.company_id"), nullable=False)
    company_name = Column(String, nullable=False)
    company_number = Column(String)
    lei = Column(String)
    ticker = Column(String)
    sector = Column(String)
    report_year = Column(Integer, nullable=False)
    source_format = Column(String)
    raw_path = Column(String)
    checksum_sha256 = Column(String)
    source = Column(String)
    status = Column(String)
    error = Column(Text)
    run_id = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    company = relationship("Company", back_populates="documents")

    def __repr__(self):
        return f"<Document {self.document_id} {self.report_year}>"


class ProcessedDocument(Base):
    """Table for preprocessed markdown documents."""

    __tablename__ = "processed_documents"

    processed_id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.document_id"), nullable=False)
    company_id = Column(String, ForeignKey("companies.company_id"))
    company_name = Column(String, nullable=False)
    report_year = Column(Integer, nullable=False)
    source_format = Column(String)
    preprocess_strategy = Column(String)
    markdown_text = Column(Text, nullable=False)
    run_id = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    document = relationship("Document")
    company = relationship("Company")
    chunks = relationship("DocumentChunk", back_populates="processed_document")

    def __repr__(self):
        return f"<ProcessedDocument {self.processed_id} {self.report_year}>"


class DocumentChunk(Base):
    """Table for AI keyword chunks derived from markdown."""

    __tablename__ = "document_chunks"

    chunk_id = Column(String, primary_key=True)
    processed_id = Column(String, ForeignKey("processed_documents.processed_id"), nullable=False)
    document_id = Column(String, ForeignKey("documents.document_id"))
    company_id = Column(String, ForeignKey("companies.company_id"))
    company_name = Column(String, nullable=False)
    report_year = Column(Integer, nullable=False)
    report_section = Column(String)
    paragraph_index = Column(Integer)
    context_before = Column(Integer)
    context_after = Column(Integer)
    chunk_text = Column(Text, nullable=False)
    keyword_matches = Column(Text, default="[]")  # JSON list
    created_at = Column(DateTime, default=datetime.now)

    processed_document = relationship("ProcessedDocument", back_populates="chunks")

    def __repr__(self):
        return f"<DocumentChunk {self.chunk_id}>"


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
    keyword = Column(String)
    keyword_text = Column(String)
    match_start = Column(Integer)
    match_end = Column(Integer)

    # Mention Type & AI Context
    mention_type = Column(String)
    ai_specificity = Column(String)
    frontier_tech_flag = Column(Boolean, default=False)
    mention_types = Column(Text, default="[]")  # JSON list
    mention_type_confidences = Column(Text, default="{}")  # JSON dict
    mention_reasoning = Column(Text)

    # Risk Classification
    tier_1_category = Column(String)
    tier_2_driver = Column(String)
    risk_types = Column(Text, default="[]")  # JSON list
    risk_confidences = Column(Text, default="{}")  # JSON dict
    risk_evidence = Column(Text, default="{}")  # JSON dict
    risk_key_snippets = Column(Text, default="{}")  # JSON dict
    risk_substantiveness = Column(Float)
    risk_reasoning = Column(Text)

    # Adoption Classification
    adoption_confidences = Column(Text, default="{}")  # JSON dict
    adoption_evidence = Column(Text, default="{}")  # JSON dict
    adoption_reasoning = Column(Text)

    # Vendor Classification
    vendor_confidences = Column(Text, default="{}")  # JSON dict
    vendor_other = Column(String)
    vendor_evidence = Column(Text, default="{}")  # JSON dict
    vendor_reasoning = Column(Text)

    # Harm / General
    harm_confidence = Column(Float)
    general_ambiguous_confidence = Column(Float)

    # Severity & Substance
    specificity_level = Column(String)
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


# ============================================================================
# NEW TABLES FOR CLASSIFIER TEST SUITE
# ============================================================================


class DocumentMentionStats(Base):
    """Table for per-document keyword mention statistics."""

    __tablename__ = "document_mention_stats"

    firm_id = Column(String, nullable=False, primary_key=True)
    report_year = Column(Integer, nullable=False, primary_key=True)

    firm_name = Column(String, nullable=False)
    company_number = Column(String, nullable=False)
    sector = Column(String, default="Unknown")

    total_mentions = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    keyword_counts = Column(Text, default="{}")  # JSON dict
    has_ai_mentions = Column(Boolean, default=False)

    source_file = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<DocumentMentionStats {self.firm_name} ({self.firm_id}) - {self.report_year}>"


class ClassificationRun(Base):
    """Table for tracking classifier test runs.

    Each row represents one execution of the test suite.
    """

    __tablename__ = "classification_runs"

    run_id = Column(String, primary_key=True)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    config_json = Column(Text)  # Full config snapshot as JSON
    model_version = Column(String)
    status = Column(String, default="running")  # running, completed, failed
    total_classifications = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    avg_confidence = Column(Float)
    low_confidence_count = Column(Integer, default=0)
    duration_seconds = Column(Float)

    # Relationships
    results = relationship("ClassificationResult", back_populates="run")

    def __repr__(self):
        return f"<ClassificationRun {self.run_id} - {self.status}>"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "config_json": self.config_json,
            "model_version": self.model_version,
            "status": self.status,
            "total_classifications": self.total_classifications,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "avg_confidence": self.avg_confidence,
            "low_confidence_count": self.low_confidence_count,
            "duration_seconds": self.duration_seconds,
        }


class ClassificationResult(Base):
    """Table for storing individual classification results.

    Each row represents one classification (one company/year/classifier combo).
    """

    __tablename__ = "classification_results"

    result_id = Column(String, primary_key=True)
    run_id = Column(String, ForeignKey("classification_runs.run_id"))
    firm_id = Column(String, nullable=False)
    firm_name = Column(String, nullable=False)
    report_year = Column(Integer, nullable=False)
    classifier_type = Column(String, nullable=False)  # harms, adoption, substantiveness, risk, vendor

    # Classification output
    classification_json = Column(Text)  # Full JSON of classification
    primary_label = Column(String)  # Main classification label
    confidence_score = Column(Float)

    # Traceability
    source_file = Column(String)  # Path to preprocessed file
    prompt_hash = Column(String)  # Hash of prompt for reproducibility
    response_raw = Column(Text)  # Raw LLM response (for debugging)
    reasoning = Column(Text)

    # Timing
    api_latency_ms = Column(Integer)
    tokens_used = Column(Integer)
    classified_at = Column(DateTime)

    # Status
    success = Column(Boolean, default=True)
    error_message = Column(Text)

    # Relationships
    run = relationship("ClassificationRun", back_populates="results")
    snippets = relationship("EvidenceSnippet", back_populates="result")

    def __repr__(self):
        return f"<ClassificationResult {self.firm_name} {self.report_year} - {self.classifier_type}>"

    def get_classification(self) -> dict:
        """Get classification as Python dict."""
        return json.loads(self.classification_json) if self.classification_json else {}

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "result_id": self.result_id,
            "run_id": self.run_id,
            "firm_id": self.firm_id,
            "firm_name": self.firm_name,
            "report_year": self.report_year,
            "classifier_type": self.classifier_type,
            "classification": self.get_classification(),
            "primary_label": self.primary_label,
            "confidence_score": self.confidence_score,
            "source_file": self.source_file,
            "reasoning": self.reasoning,
            "api_latency_ms": self.api_latency_ms,
            "tokens_used": self.tokens_used,
            "classified_at": self.classified_at.isoformat() if self.classified_at else None,
            "success": self.success,
            "error_message": self.error_message,
        }


class EvidenceSnippet(Base):
    """Table for storing evidence snippets linked to classifications.

    Each row represents one piece of evidence (quote) supporting a classification.
    """

    __tablename__ = "evidence_snippets"

    snippet_id = Column(String, primary_key=True)
    result_id = Column(String, ForeignKey("classification_results.result_id"))

    # The actual evidence
    text_excerpt = Column(Text, nullable=False)
    excerpt_hash = Column(String)  # For deduplication

    # Location in source
    source_file = Column(String)
    section_name = Column(String)  # e.g., "Principal Risks", "Strategic Report"
    approximate_location = Column(String)  # Page or section reference

    # Classification context
    category = Column(String)  # For categorized evidence (e.g., risk type)

    # For human review
    needs_review = Column(Boolean, default=False)
    review_priority = Column(Integer, default=3)  # 1=high, 2=medium, 3=low
    reviewer_decision = Column(String)  # validated, rejected, unclear
    reviewer_notes = Column(Text)
    reviewed_at = Column(DateTime)
    reviewed_by = Column(String)

    # Relationship
    result = relationship("ClassificationResult", back_populates="snippets")

    def __repr__(self):
        return f"<EvidenceSnippet {self.snippet_id}>"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "snippet_id": self.snippet_id,
            "result_id": self.result_id,
            "text_excerpt": self.text_excerpt,
            "source_file": self.source_file,
            "section_name": self.section_name,
            "category": self.category,
            "needs_review": self.needs_review,
            "review_priority": self.review_priority,
            "reviewer_decision": self.reviewer_decision,
            "reviewer_notes": self.reviewer_notes,
        }


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

    def upsert_processed_document(self, session: Session, data: dict) -> ProcessedDocument:
        """Insert or update a processed markdown document."""
        processed_id = data["processed_id"]
        existing = session.query(ProcessedDocument).filter(
            ProcessedDocument.processed_id == processed_id
        ).first()

        if existing:
            existing.document_id = data.get("document_id", existing.document_id)
            existing.company_id = data.get("company_id", existing.company_id)
            existing.company_name = data.get("company_name", existing.company_name)
            existing.report_year = data.get("report_year", existing.report_year)
            existing.source_format = data.get("source_format")
            existing.preprocess_strategy = data.get("preprocess_strategy")
            existing.markdown_text = data.get("markdown_text", existing.markdown_text)
            existing.run_id = data.get("run_id")
            return existing

        record = ProcessedDocument(
            processed_id=processed_id,
            document_id=data.get("document_id"),
            company_id=data.get("company_id"),
            company_name=data.get("company_name") or data.get("company_id") or processed_id,
            report_year=data.get("report_year"),
            source_format=data.get("source_format"),
            preprocess_strategy=data.get("preprocess_strategy"),
            markdown_text=data.get("markdown_text") or "",
            run_id=data.get("run_id"),
            created_at=datetime.now(),
        )
        session.add(record)
        return record

    def upsert_document_chunk(self, session: Session, data: dict) -> DocumentChunk:
        """Insert or update a document chunk."""
        chunk_id = data["chunk_id"]
        existing = session.query(DocumentChunk).filter(
            DocumentChunk.chunk_id == chunk_id
        ).first()

        keyword_matches = data.get("keyword_matches")
        keyword_matches_json = json.dumps(keyword_matches) if isinstance(keyword_matches, list) else keyword_matches

        if existing:
            existing.processed_id = data.get("processed_id", existing.processed_id)
            existing.document_id = data.get("document_id", existing.document_id)
            existing.company_id = data.get("company_id", existing.company_id)
            existing.company_name = data.get("company_name", existing.company_name)
            existing.report_year = data.get("report_year", existing.report_year)
            existing.report_section = data.get("report_section")
            existing.paragraph_index = data.get("paragraph_index")
            existing.context_before = data.get("context_before")
            existing.context_after = data.get("context_after")
            existing.chunk_text = data.get("chunk_text", existing.chunk_text)
            existing.keyword_matches = keyword_matches_json or existing.keyword_matches
            return existing

        record = DocumentChunk(
            chunk_id=chunk_id,
            processed_id=data.get("processed_id"),
            document_id=data.get("document_id"),
            company_id=data.get("company_id"),
            company_name=data.get("company_name") or data.get("company_id") or chunk_id,
            report_year=data.get("report_year"),
            report_section=data.get("report_section"),
            paragraph_index=data.get("paragraph_index"),
            context_before=data.get("context_before"),
            context_after=data.get("context_after"),
            chunk_text=data.get("chunk_text") or "",
            keyword_matches=keyword_matches_json or "[]",
            created_at=datetime.now(),
        )
        session.add(record)
        return record

    def upsert_company(self, session: Session, data: dict) -> Company:
        """Insert or update a company record."""
        company_id = data["company_id"]
        existing = session.query(Company).filter(
            Company.company_id == company_id
        ).first()

        if existing:
            existing.company_name = data.get("company_name", existing.company_name)
            existing.company_number = data.get("company_number")
            existing.lei = data.get("lei")
            existing.ticker = data.get("ticker")
            existing.sector = data.get("sector")
            existing.index_name = data.get("index")
            existing.company_type = data.get("type")
            return existing

        record = Company(
            company_id=company_id,
            company_name=data.get("company_name") or company_id,
            company_number=data.get("company_number"),
            lei=data.get("lei"),
            ticker=data.get("ticker"),
            sector=data.get("sector"),
            index_name=data.get("index"),
            company_type=data.get("type"),
            created_at=datetime.now(),
        )
        session.add(record)
        return record

    def upsert_document(self, session: Session, data: dict) -> Document:
        """Insert or update a document record."""
        document_id = data["document_id"]
        existing = session.query(Document).filter(
            Document.document_id == document_id
        ).first()

        if existing:
            existing.company_id = data.get("company_id", existing.company_id)
            existing.company_name = data.get("company_name", existing.company_name)
            existing.company_number = data.get("company_number")
            existing.lei = data.get("lei")
            existing.ticker = data.get("ticker")
            existing.sector = data.get("sector")
            existing.report_year = data.get("report_year", existing.report_year)
            existing.source_format = data.get("source_format")
            existing.raw_path = data.get("raw_path")
            existing.checksum_sha256 = data.get("checksum_sha256")
            existing.source = data.get("source")
            existing.status = data.get("status")
            existing.error = data.get("error")
            existing.run_id = data.get("run_id")
            return existing

        record = Document(
            document_id=document_id,
            company_id=data["company_id"],
            company_name=data.get("company_name") or data["company_id"],
            company_number=data.get("company_number"),
            lei=data.get("lei"),
            ticker=data.get("ticker"),
            sector=data.get("sector"),
            report_year=data.get("report_year"),
            source_format=data.get("source_format"),
            raw_path=data.get("raw_path"),
            checksum_sha256=data.get("checksum_sha256"),
            source=data.get("source"),
            status=data.get("status"),
            error=data.get("error"),
            run_id=data.get("run_id"),
            created_at=datetime.now(),
        )
        session.add(record)
        return record

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

    def save_mention_record(
        self,
        session: Session,
        candidate,
        mention_type_result: dict,
        model_version: str,
        adoption_result: Optional[dict] = None,
        risk_result: Optional[dict] = None,
        vendor_result: Optional[dict] = None,
    ) -> Mention:
        """Save a mention record with multi-stage classification output."""
        mention_types = mention_type_result.get("mention_types", [])
        mention_confidences = mention_type_result.get("confidence_scores", {})
        mention_reasoning = mention_type_result.get("reasoning", "")

        adoption_result = adoption_result or {}
        risk_result = risk_result or {}
        vendor_result = vendor_result or {}

        mention = Mention(
            mention_id=candidate.span_id,
            firm_id=candidate.firm_id,
            firm_name=candidate.firm_name,
            sector=candidate.sector,
            report_year=candidate.report_year,
            report_section=candidate.report_section,
            text_excerpt=candidate.text,
            page_number=candidate.page_number,
            keyword=candidate.keyword,
            keyword_text=candidate.keyword_text,
            match_start=candidate.match_start,
            match_end=candidate.match_end,
            mention_types=json.dumps(mention_types),
            mention_type_confidences=json.dumps(mention_confidences),
            mention_reasoning=mention_reasoning,
            risk_types=json.dumps(risk_result.get("risk_types", [])),
            risk_confidences=json.dumps(_normalize_risk_signals(risk_result)),
            risk_evidence=json.dumps(risk_result.get("evidence", {})),
            risk_key_snippets=json.dumps(risk_result.get("key_snippets", {})),
            risk_substantiveness=risk_result.get("substantiveness_score"),
            risk_reasoning=risk_result.get("reasoning", ""),
            adoption_confidences=json.dumps(
                _normalize_adoption_signals(adoption_result)
            ),
            adoption_evidence=json.dumps(adoption_result.get("evidence", {})),
            adoption_reasoning=adoption_result.get("reasoning", ""),
            vendor_confidences=json.dumps(
                {v["vendor"]: v["signal"] for v in vendor_result.get("vendors", [])}
                if vendor_result.get("vendors") else {}
            ),
            vendor_other=vendor_result.get("other_vendor", ""),
            vendor_evidence=json.dumps(vendor_result.get("evidence", {})),
            vendor_reasoning=vendor_result.get("reasoning", ""),
            harm_confidence=mention_confidences.get("harm"),
            general_ambiguous_confidence=mention_confidences.get("general_ambiguous"),
            confidence_score=_max_confidence(mention_confidences),
            reasoning_summary=mention_reasoning,
            model_version=model_version,
            extraction_date=datetime.now().date(),
            review_status="unreviewed",
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

    def save_document_stats(
        self,
        firm_id: str,
        firm_name: str,
        company_number: str,
        report_year: int,
        sector: str,
        total_mentions: int,
        total_chunks: int,
        keyword_counts: dict,
        source_file: Optional[str] = None,
    ) -> DocumentMentionStats:
        """Save per-document mention statistics.

        Args:
            firm_id: Firm identifier
            firm_name: Company name
            company_number: Companies House number
            report_year: Report year
            sector: Company sector
            total_mentions: Total keyword mentions
            total_chunks: Total chunks created
            keyword_counts: Dict of keyword counts
            source_file: Optional path to source report
        """
        session = self.get_session()
        try:
            existing = session.query(DocumentMentionStats).filter(
                DocumentMentionStats.firm_id == firm_id,
                DocumentMentionStats.report_year == report_year,
            ).first()

            if existing:
                session.delete(existing)
                session.flush()

            record = DocumentMentionStats(
                firm_id=firm_id,
                firm_name=firm_name,
                company_number=company_number,
                report_year=report_year,
                sector=sector,
                total_mentions=total_mentions,
                total_chunks=total_chunks,
                keyword_counts=json.dumps(keyword_counts),
                has_ai_mentions=total_mentions > 0,
                source_file=source_file,
                created_at=datetime.now(),
            )
            session.add(record)
            session.commit()
            return record
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving document stats: {e}")
            raise
        finally:
            session.close()

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
                confidence_scores=json.dumps(_normalize_risk_signals(classification_data)),
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


    # ========================================================================
    # Classification Run Methods
    # ========================================================================

    def create_run(
        self,
        run_id: str,
        config: dict,
        model_version: str
    ) -> ClassificationRun:
        """Create a new classification run record.

        Args:
            run_id: Unique run identifier
            config: Run configuration dict
            model_version: Model version being used

        Returns:
            ClassificationRun object
        """
        session = self.get_session()
        try:
            run = ClassificationRun(
                run_id=run_id,
                started_at=datetime.now(),
                config_json=json.dumps(config),
                model_version=model_version,
                status="running",
            )
            session.add(run)
            session.commit()
            logger.info(f"Created classification run: {run_id}")
            return run
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating run: {e}")
            raise
        finally:
            session.close()

    def complete_run(
        self,
        run_id: str,
        total_classifications: int,
        success_count: int,
        error_count: int,
        avg_confidence: float,
        low_confidence_count: int,
        duration_seconds: float,
        status: str = "completed"
    ) -> None:
        """Mark a run as completed with summary stats.

        Args:
            run_id: Run identifier
            total_classifications: Total number of classifications
            success_count: Number of successful classifications
            error_count: Number of failed classifications
            avg_confidence: Average confidence score
            low_confidence_count: Number of low-confidence results
            duration_seconds: Total duration
            status: Final status (completed/failed)
        """
        session = self.get_session()
        try:
            run = session.query(ClassificationRun).filter(
                ClassificationRun.run_id == run_id
            ).first()

            if run:
                run.completed_at = datetime.now()
                run.status = status
                run.total_classifications = total_classifications
                run.success_count = success_count
                run.error_count = error_count
                run.avg_confidence = avg_confidence
                run.low_confidence_count = low_confidence_count
                run.duration_seconds = duration_seconds
                session.commit()
                logger.info(f"Completed run {run_id}: {success_count}/{total_classifications} successful")
        except Exception as e:
            session.rollback()
            logger.error(f"Error completing run: {e}")
            raise
        finally:
            session.close()

    def get_run(self, run_id: str) -> Optional[ClassificationRun]:
        """Get a classification run by ID.

        Args:
            run_id: Run identifier

        Returns:
            ClassificationRun object or None
        """
        session = self.get_session()
        try:
            return session.query(ClassificationRun).filter(
                ClassificationRun.run_id == run_id
            ).first()
        finally:
            session.close()

    def get_all_runs(self, limit: int = 50) -> List[ClassificationRun]:
        """Get recent classification runs.

        Args:
            limit: Maximum number to return

        Returns:
            List of ClassificationRun objects
        """
        session = self.get_session()
        try:
            return session.query(ClassificationRun).order_by(
                ClassificationRun.started_at.desc()
            ).limit(limit).all()
        finally:
            session.close()

    # ========================================================================
    # Classification Result Methods
    # ========================================================================

    def save_classification_result(
        self,
        result_data: dict
    ) -> ClassificationResult:
        """Save a classification result.

        Args:
            result_data: Dict with result fields (from ClassificationResult.to_dict())

        Returns:
            Saved ClassificationResult object
        """
        session = self.get_session()
        try:
            result = ClassificationResult(
                result_id=result_data["result_id"],
                run_id=result_data["run_id"],
                firm_id=result_data["firm_id"],
                firm_name=result_data["firm_name"],
                report_year=result_data["report_year"],
                classifier_type=result_data["classifier_type"],
                classification_json=json.dumps(result_data.get("classification", {})),
                primary_label=result_data.get("primary_label"),
                confidence_score=result_data.get("confidence_score"),
                source_file=result_data.get("source_file"),
                prompt_hash=result_data.get("prompt_hash"),
                response_raw=result_data.get("response_raw"),
                reasoning=result_data.get("reasoning"),
                api_latency_ms=result_data.get("api_latency_ms"),
                tokens_used=result_data.get("tokens_used"),
                classified_at=datetime.now(),
                success=result_data.get("success", True),
                error_message=result_data.get("error_message"),
            )
            session.add(result)
            session.commit()
            return result
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving classification result: {e}")
            raise
        finally:
            session.close()

    def save_classification_results_batch(
        self,
        results: List[dict]
    ) -> int:
        """Save a batch of classification results.

        Args:
            results: List of result dicts

        Returns:
            Number of results saved
        """
        session = self.get_session()
        count = 0
        try:
            for result_data in results:
                result = ClassificationResult(
                    result_id=result_data["result_id"],
                    run_id=result_data["run_id"],
                    firm_id=result_data["firm_id"],
                    firm_name=result_data["firm_name"],
                    report_year=result_data["report_year"],
                    classifier_type=result_data["classifier_type"],
                    classification_json=json.dumps(result_data.get("classification", {})),
                    primary_label=result_data.get("primary_label"),
                    confidence_score=result_data.get("confidence_score"),
                    source_file=result_data.get("source_file"),
                    prompt_hash=result_data.get("prompt_hash"),
                    response_raw=result_data.get("response_raw"),
                    reasoning=result_data.get("reasoning"),
                    api_latency_ms=result_data.get("api_latency_ms"),
                    tokens_used=result_data.get("tokens_used"),
                    classified_at=datetime.now(),
                    success=result_data.get("success", True),
                    error_message=result_data.get("error_message"),
                )
                session.add(result)
                count += 1

            session.commit()
            logger.info(f"Saved {count} classification results")
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving results batch: {e}")
            raise
        finally:
            session.close()

    def get_results_for_run(
        self,
        run_id: str,
        classifier_type: Optional[str] = None
    ) -> List[ClassificationResult]:
        """Get all results for a run.

        Args:
            run_id: Run identifier
            classifier_type: Optional filter by classifier

        Returns:
            List of ClassificationResult objects
        """
        session = self.get_session()
        try:
            query = session.query(ClassificationResult).filter(
                ClassificationResult.run_id == run_id
            )
            if classifier_type:
                query = query.filter(ClassificationResult.classifier_type == classifier_type)
            return query.all()
        finally:
            session.close()

    # ========================================================================
    # Evidence Snippet Methods
    # ========================================================================

    def save_evidence_snippet(
        self,
        snippet_data: dict
    ) -> EvidenceSnippet:
        """Save an evidence snippet.

        Args:
            snippet_data: Dict with snippet fields

        Returns:
            Saved EvidenceSnippet object
        """
        session = self.get_session()
        try:
            # Generate hash for deduplication
            import hashlib
            excerpt_hash = hashlib.sha256(
                snippet_data.get("text_excerpt", "").encode()
            ).hexdigest()[:16]

            snippet = EvidenceSnippet(
                snippet_id=snippet_data["snippet_id"],
                result_id=snippet_data["result_id"],
                text_excerpt=snippet_data["text_excerpt"],
                excerpt_hash=excerpt_hash,
                source_file=snippet_data.get("source_file"),
                section_name=snippet_data.get("section_name"),
                approximate_location=snippet_data.get("approximate_location"),
                category=snippet_data.get("category"),
                needs_review=snippet_data.get("needs_review", False),
                review_priority=snippet_data.get("review_priority", 3),
            )
            session.add(snippet)
            session.commit()
            return snippet
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving evidence snippet: {e}")
            raise
        finally:
            session.close()

    def save_evidence_snippets_batch(
        self,
        snippets: List[dict]
    ) -> int:
        """Save a batch of evidence snippets.

        Args:
            snippets: List of snippet dicts

        Returns:
            Number of snippets saved
        """
        import hashlib

        session = self.get_session()
        count = 0
        try:
            for snippet_data in snippets:
                excerpt_hash = hashlib.sha256(
                    snippet_data.get("text_excerpt", "").encode()
                ).hexdigest()[:16]

                snippet = EvidenceSnippet(
                    snippet_id=snippet_data["snippet_id"],
                    result_id=snippet_data["result_id"],
                    text_excerpt=snippet_data["text_excerpt"],
                    excerpt_hash=excerpt_hash,
                    source_file=snippet_data.get("source_file"),
                    section_name=snippet_data.get("section_name"),
                    approximate_location=snippet_data.get("approximate_location"),
                    category=snippet_data.get("category"),
                    needs_review=snippet_data.get("needs_review", False),
                    review_priority=snippet_data.get("review_priority", 3),
                )
                session.add(snippet)
                count += 1

            session.commit()
            logger.info(f"Saved {count} evidence snippets")
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving snippets batch: {e}")
            raise
        finally:
            session.close()

    def get_snippets_for_review(
        self,
        run_id: Optional[str] = None,
        priority: Optional[int] = None,
        reviewed: Optional[bool] = None
    ) -> List[EvidenceSnippet]:
        """Get evidence snippets for review.

        Args:
            run_id: Optional filter by run
            priority: Optional filter by review priority (1, 2, or 3)
            reviewed: Optional filter by review status

        Returns:
            List of EvidenceSnippet objects
        """
        session = self.get_session()
        try:
            query = session.query(EvidenceSnippet)

            if run_id:
                query = query.join(ClassificationResult).filter(
                    ClassificationResult.run_id == run_id
                )

            if priority:
                query = query.filter(EvidenceSnippet.review_priority == priority)

            if reviewed is not None:
                if reviewed:
                    query = query.filter(EvidenceSnippet.reviewer_decision.isnot(None))
                else:
                    query = query.filter(EvidenceSnippet.reviewer_decision.is_(None))

            return query.order_by(
                EvidenceSnippet.review_priority,
                EvidenceSnippet.snippet_id
            ).all()
        finally:
            session.close()

    def update_snippet_review(
        self,
        snippet_id: str,
        decision: str,
        notes: Optional[str] = None,
        reviewer: Optional[str] = None
    ) -> None:
        """Update the review status of a snippet.

        Args:
            snippet_id: Snippet identifier
            decision: Review decision (validated, rejected, unclear)
            notes: Optional reviewer notes
            reviewer: Optional reviewer name
        """
        session = self.get_session()
        try:
            snippet = session.query(EvidenceSnippet).filter(
                EvidenceSnippet.snippet_id == snippet_id
            ).first()

            if snippet:
                snippet.reviewer_decision = decision
                snippet.reviewer_notes = notes
                snippet.reviewed_by = reviewer
                snippet.reviewed_at = datetime.now()
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating snippet review: {e}")
            raise
        finally:
            session.close()


def get_database() -> Database:
    """Get database instance.

    Returns:
        Database instance
    """
    return Database()
