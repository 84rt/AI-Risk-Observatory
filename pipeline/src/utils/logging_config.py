"""Centralized logging configuration for AIRO classifier test suite.

Provides structured logging with:
- Console output (INFO level)
- File output (DEBUG level) with rotation
- JSON structured logs for analysis
- Per-classifier log files
- Run ID tracking across all logs
"""

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from ..config import get_settings

# Base paths
settings = get_settings()
LOGS_DIR = settings.logs_dir / "pipeline"
CLASSIFIER_LOGS_DIR = LOGS_DIR / "classifier_runs"
MODEL_COMPARISON_LOGS_DIR = LOGS_DIR / "model_comparison"

# Ensure log directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CLASSIFIER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_COMPARISON_LOGS_DIR.mkdir(parents=True, exist_ok=True)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "run_id"):
            log_data["run_id"] = record.run_id
        if hasattr(record, "firm_id"):
            log_data["firm_id"] = record.firm_id
        if hasattr(record, "classifier"):
            log_data["classifier"] = record.classifier
        if hasattr(record, "confidence"):
            log_data["confidence"] = record.confidence
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ClassifierLogAdapter(logging.LoggerAdapter):
    """Logger adapter that adds run_id, classifier, and firm_id context."""

    def __init__(
        self,
        logger: logging.Logger,
        run_id: str,
        classifier: str,
        firm_id: Optional[str] = None,
    ):
        super().__init__(logger, {})
        self.run_id = run_id
        self.classifier = classifier
        self.firm_id = firm_id

    def process(self, msg, kwargs):
        # Add context to extra
        extra = kwargs.get("extra", {})
        extra["run_id"] = self.run_id
        extra["classifier"] = self.classifier
        if self.firm_id:
            extra["firm_id"] = self.firm_id
        kwargs["extra"] = extra
        return msg, kwargs

    def set_firm(self, firm_id: str):
        """Update the firm context for subsequent log messages."""
        self.firm_id = firm_id


def setup_logging(
    log_level: str = "INFO",
    enable_json_file: bool = True,
    run_id: Optional[str] = None,
) -> logging.Logger:
    """Set up centralized logging for the classifier test suite.

    Args:
        log_level: Logging level for console output (DEBUG, INFO, WARNING, ERROR)
        enable_json_file: Whether to create JSON structured log files
        run_id: Optional run ID to include in log filenames

    Returns:
        Root logger for the airo.classifiers namespace
    """
    # Create root logger for classifiers
    root_logger = logging.getLogger("airo.classifiers")
    root_logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler (human-readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler (detailed, rotating)
    date_str = datetime.now().strftime("%Y%m%d")
    log_filename = f"classifier_{date_str}.log"
    if run_id:
        log_filename = f"classifier_{run_id}.log"

    file_handler = RotatingFileHandler(
        LOGS_DIR / log_filename,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | "
        "[run=%(run_id)s, classifier=%(classifier)s, firm=%(firm_id)s] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Set defaults for extra fields
    file_handler.addFilter(_ExtraFieldsFilter())
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

    # JSON structured log file (for analysis)
    if enable_json_file:
        json_filename = f"classifier_{date_str}.json"
        if run_id:
            json_filename = f"classifier_{run_id}.json"

        json_handler = RotatingFileHandler(
            LOGS_DIR / json_filename,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(json_handler)

    return root_logger


class _ExtraFieldsFilter(logging.Filter):
    """Filter that adds default values for extra fields."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = "N/A"
        if not hasattr(record, "classifier"):
            record.classifier = "N/A"
        if not hasattr(record, "firm_id"):
            record.firm_id = "N/A"
        return True


def get_classifier_logger(
    classifier_name: str,
    run_id: str,
    create_separate_file: bool = True,
) -> ClassifierLogAdapter:
    """Get a logger for a specific classifier.

    Args:
        classifier_name: Name of the classifier (harms, adoption, etc.)
        run_id: Run ID for tracking
        create_separate_file: Whether to create a separate log file for this classifier

    Returns:
        ClassifierLogAdapter with context set
    """
    logger = logging.getLogger(f"airo.classifiers.{classifier_name}")

    if create_separate_file:
        # Create per-classifier log file
        date_str = datetime.now().strftime("%Y%m%d")
        classifier_log_path = CLASSIFIER_LOGS_DIR / f"{classifier_name}_{date_str}.log"

        # Check if handler already exists
        handler_exists = any(
            isinstance(h, logging.FileHandler)
            and h.baseFilename == str(classifier_log_path)
            for h in logger.handlers
        )

        if not handler_exists:
            file_handler = logging.FileHandler(classifier_log_path)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)

    return ClassifierLogAdapter(logger, run_id, classifier_name)


def get_run_logger(run_id: str) -> logging.Logger:
    """Get a logger for a specific test run.

    Args:
        run_id: Run ID for this test execution

    Returns:
        Logger configured for the run
    """
    logger = logging.getLogger(f"airo.classifiers.run.{run_id}")
    return logger


def log_classification_start(
    logger: ClassifierLogAdapter,
    firm_id: str,
    report_year: int,
    source_file: str,
) -> None:
    """Log the start of a classification task."""
    logger.set_firm(firm_id)
    logger.info(
        f"Starting classification: {firm_id} ({report_year})",
        extra={"extra_data": {"year": report_year, "source": source_file}},
    )


def log_classification_result(
    logger: ClassifierLogAdapter,
    firm_id: str,
    result: str,
    confidence: float,
    evidence_count: int,
    latency_ms: int,
) -> None:
    """Log the result of a classification."""
    logger.set_firm(firm_id)
    level = logging.INFO if confidence >= 0.7 else logging.WARNING

    logger.log(
        level,
        f"Classified {firm_id}: {result} (confidence={confidence:.2f}, evidence={evidence_count})",
        extra={
            "confidence": confidence,
            "extra_data": {
                "result": result,
                "evidence_count": evidence_count,
                "latency_ms": latency_ms,
            },
        },
    )


def log_api_call(
    logger: ClassifierLogAdapter,
    model: str,
    prompt_tokens: int,
    response_tokens: int,
    latency_ms: int,
    prompt_chars: int | None = None,
    response_chars: int | None = None,
) -> None:
    """Log an API call to the LLM."""
    logger.debug(
        f"API call: model={model}, tokens={prompt_tokens}+{response_tokens}, latency={latency_ms}ms",
        extra={
            "extra_data": {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "prompt_chars": prompt_chars,
                "response_chars": response_chars,
                "latency_ms": latency_ms,
            }
        },
    )


def log_error(
    logger: ClassifierLogAdapter,
    firm_id: str,
    error_type: str,
    error_message: str,
    attempt_number: int = 1,
) -> None:
    """Log an error during classification."""
    logger.set_firm(firm_id)
    logger.error(
        f"Error classifying {firm_id}: {error_type} - {error_message} (attempt {attempt_number})",
        extra={
            "extra_data": {
                "error_type": error_type,
                "error_message": error_message,
                "attempt": attempt_number,
            }
        },
    )


def log_run_summary(
    logger: logging.Logger,
    run_id: str,
    total_classified: int,
    success_count: int,
    error_count: int,
    avg_confidence: float,
    low_confidence_count: int,
    duration_seconds: float,
) -> None:
    """Log the summary of a test run."""
    success_rate = success_count / total_classified if total_classified > 0 else 0

    logger.info(
        f"Run {run_id} complete: "
        f"{success_count}/{total_classified} successful ({success_rate:.1%}), "
        f"avg_confidence={avg_confidence:.2f}, "
        f"low_confidence={low_confidence_count}, "
        f"errors={error_count}, "
        f"duration={duration_seconds:.1f}s"
    )
