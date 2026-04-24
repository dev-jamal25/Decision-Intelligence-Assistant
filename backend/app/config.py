"""Central config module using pydantic-settings."""

import logging
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


# Define PROJECT_ROOT once at module level (repo root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
assert PROJECT_ROOT.name == "Decision-Intelligence-Assistant", f"PROJECT_ROOT mismatch: {PROJECT_ROOT}"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Paths (relative or absolute)
    chroma_persist_dir: str = Field(
        default="artifacts/chroma",
        alias="CHROMA_PERSIST_DIR",
    )
    rag_data_path: str = Field(
        default="data/knowledge/rag_case_base_v1.csv",
        alias="RAG_DATA_PATH",
    )

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
        env_file = str(PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields from .env without failing

    @field_validator("chroma_persist_dir", "rag_data_path", mode="after")
    @classmethod
    def resolve_paths(cls, v: str) -> str:
        """
        Resolve relative paths relative to PROJECT_ROOT.

        - If path is absolute, return as-is
        - If path is relative, resolve relative to PROJECT_ROOT, normalizing .. references
        """
        p = Path(v)
        # If already absolute, return as-is
        if p.is_absolute():
            return str(p)
        # For relative paths: combine with PROJECT_ROOT then resolve to normalize ".."
        combined = PROJECT_ROOT / p
        # Use resolve() to normalize ".." but save CWD first
        import os
        saved_cwd = os.getcwd()
        try:
            os.chdir(str(PROJECT_ROOT))
            resolved = combined.resolve()
        finally:
            os.chdir(saved_cwd)
        return str(resolved)


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
