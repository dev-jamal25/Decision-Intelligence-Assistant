"""Central config module using pydantic-settings."""

import logging
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Paths
    chroma_persist_dir: str = Field(
        default=str(Path(__file__).parent.parent.parent / "artifacts" / "chroma"),
        alias="CHROMA_PERSIST_DIR",
    )
    rag_data_path: str = str(Path(__file__).parent.parent.parent / "data" / "knowledge" / "rag_case_base_v1.csv")

    # Logging
    log_level: str = "INFO"

    # RAG defaults
    retrieval_k: int = 5

    # OpenRouter embedding settings
    openrouter_api_key: str = ""
    openrouter_embedding_model: str = Field(
        default="nvidia/llama-nemotron-embed-vl-1b-v2:free",
        alias="OPENROUTER_EMBEDDING_MODEL",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )

    # OpenRouter LLM settings
    openrouter_llm_model: str = Field(
        default="openai/gpt-oss-120b:free",
        alias="OPENROUTER_LLM_MODEL",
    )

    class Config:
        # Resolve .env from project root
        env_file = str(Path(__file__).parent.parent.parent / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields from .env without failing


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

