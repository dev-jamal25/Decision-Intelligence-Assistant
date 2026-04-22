"""Offline data preparation skeleton.

Mirrors the sections in ``notebooks/notebook.ipynb`` so the same preprocessing
can later be promoted from exploration to a reproducible script.

Scope: SKELETON ONLY. No heavy processing, no embedding, no API calls, no
training. Function bodies are intentionally placeholders.

Dataset interpretation:
    inbound = True  -> customer / user tweet sent to support
    inbound = False -> support / company-side tweet or reply
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "twcs.csv"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RELEVANT_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


def load_raw(path: Path = RAW_CSV, nrows: int | None = None) -> pd.DataFrame:
    """Load the raw twcs CSV. Use ``nrows`` for light exploration."""
    return pd.read_csv(path, nrows=nrows)


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[RELEVANT_COLUMNS].copy()


def clean_text(text: str) -> str:
    """Placeholder text normalization. Final rules TBD."""
    return text


def weak_label(text: str) -> str:
    """Placeholder weak-labeling function. Returns priority string."""
    return "medium"


def build_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """ML base dataset from ``inbound = True`` customer tweets. Placeholder."""
    raise NotImplementedError


def link_first_support_reply(df: pd.DataFrame) -> pd.DataFrame:
    """Link first ``inbound = False`` reply to each customer tweet. Placeholder."""
    raise NotImplementedError


def build_rag_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Build RAG case records with a narrative embedding text field. Placeholder."""
    raise NotImplementedError


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Data preparation skeleton — nothing is executed yet.")
    logger.info("Raw CSV: %s", RAW_CSV)
    logger.info("Interim dir: %s", INTERIM_DIR)
    logger.info("Processed dir: %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()
