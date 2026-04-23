"""Dataset loader for RAG cases from CSV."""

import csv
import logging
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


class RAGCase:
    """Single RAG case: customer ticket + optional support reply."""

    def __init__(
        self,
        customer_tweet_id: int,
        customer_author_id: str,
        customer_created_at: str,
        customer_text: str,
        support_reply_tweet_id: str | None = None,
        support_reply_author_id: str | None = None,
        support_reply_created_at: str | None = None,
        support_reply_text: str | None = None,
    ):
        self.customer_tweet_id = customer_tweet_id
        self.customer_author_id = customer_author_id
        self.customer_created_at = customer_created_at
        self.customer_text = customer_text
        self.support_reply_tweet_id = support_reply_tweet_id
        self.support_reply_author_id = support_reply_author_id
        self.support_reply_created_at = support_reply_created_at
        self.support_reply_text = support_reply_text

    def get_combined_text(self) -> str:
        """Return combined customer + support text for embedding."""
        if self.support_reply_text:
            return f"Customer: {self.customer_text}\nSupport: {self.support_reply_text}"
        return f"Customer: {self.customer_text}"

    def get_metadata(self) -> dict:
        """Return metadata dict for Chroma storage."""
        return {
            "customer_tweet_id": str(self.customer_tweet_id),
            "customer_author_id": self.customer_author_id,
            "customer_created_at": self.customer_created_at,
            "has_support_reply": str(bool(self.support_reply_text)),
        }


def load_rag_cases(csv_path: str | Path) -> Generator[RAGCase, None, None]:
    """Stream RAG cases from CSV file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error(f"RAG data file not found: {csv_path}")
        raise FileNotFoundError(f"RAG data file not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle NaN/None values from CSV
            support_reply_text = row["support_reply_text"]
            if support_reply_text and support_reply_text.lower() != "nan":
                support_text = support_reply_text
            else:
                support_text = None

            case = RAGCase(
                customer_tweet_id=int(row["customer_tweet_id"]),
                customer_author_id=row["customer_author_id"],
                customer_created_at=row["customer_created_at"],
                customer_text=row["customer_text"],
                support_reply_tweet_id=row.get("support_reply_tweet_id"),
                support_reply_author_id=row.get("support_reply_author_id"),
                support_reply_created_at=row.get("support_reply_created_at"),
                support_reply_text=support_text,
            )
            yield case
