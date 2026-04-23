"""Central config module using pydantic-settings."""

import logging
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Paths
    chroma_persist_dir: str = str(Path(__file__).parent.parent.parent / "artifacts" / "chroma")
    rag_data_path: str = str(Path(__file__).parent.parent.parent / "data" / "knowledge" / "rag_case_base_v1.csv")

    # Logging
    log_level: str = "INFO"

    # RAG defaults
    retrieval_k: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get or create cached settings instance."""
    return Settings()


# Configure root logger
def setup_logging(settings: Settings) -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


# Initialize on import
_settings = get_settings()
setup_logging(_settings)
logger = logging.getLogger(__name__)
