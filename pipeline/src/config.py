"""Configuration management for AIRO pipeline."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables from base repo directory
# Look in parent directory first (base repo folder)
base_repo_dir = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=base_repo_dir / ".env.local", override=True)
load_dotenv(dotenv_path=base_repo_dir / ".env", override=False)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    companies_house_api_key: str = Field(..., alias="COMPANIES_HOUSE_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")

    # Model Configuration
    gemini_model: str = Field(
        default="gemini-2.0-flash",  # 2K RPM vs 10 RPM for -exp
        alias="GEMINI_MODEL"
    )
    
    # OpenRouter Configuration
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Paths
    pipeline_root: Path = Field(default=base_repo_dir / "pipeline")
    data_dir: Path = Field(default=base_repo_dir / "data")
    raw_dir: Path = Field(default=base_repo_dir / "data" / "raw")
    processed_dir: Path = Field(default=base_repo_dir / "data" / "processed")
    results_dir: Path = Field(default=base_repo_dir / "data" / "results")
    annotations_dir: Path = Field(default=base_repo_dir / "data" / "annotations")
    logs_dir: Path = Field(default=base_repo_dir / "data" / "logs")

    # Database
    database_path: Path = Field(
        default=base_repo_dir / "data" / "db" / "airo.db",
        alias="DATABASE_PATH"
    )

    # Backwards compatibility: place to store generated artifacts
    output_dir: Path = Field(default=base_repo_dir / "data" / "results")

    # Companies House API
    companies_house_base_url: str = "https://api.company-information.service.gov.uk"

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # LLM Parameters
    max_tokens: int = 4096
    temperature: float = 0.0  # Deterministic for classification

    # Processing Parameters
    max_retries: int = 3
    retry_delay: int = 2  # seconds

    class Config:
        env_file = [".env.local", ".env"]  # Check .env.local first, then .env
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
