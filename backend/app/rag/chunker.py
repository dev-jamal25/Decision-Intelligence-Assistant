"""Case chunking/pairing strategy for RAG."""

import logging
from app.rag.loader import RAGCase

logger = logging.getLogger(__name__)


class RAGCaseChunk:
    """Prepared chunk for insertion into vector store."""

    def __init__(self, chunk_id: str, text: str, metadata: dict):
        """
        Initialize a prepared chunk.

        Args:
            chunk_id: Unique ID for this chunk
            text: Combined text to embed
            metadata: Dict with source information
        """
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata


def chunk_rag_case(case: RAGCase) -> RAGCaseChunk:
    """
    Convert a RAG case into a prepared chunk.

    Pairs customer tweet with support reply (if available).
    Currently 1:1 mapping (one chunk per case).

    Args:
        case: RAGCase from loader

    Returns:
        RAGCaseChunk ready for insertion
    """
    chunk_id = str(case.customer_tweet_id)
    text = case.get_combined_text()
    metadata = case.get_metadata()

    return RAGCaseChunk(chunk_id=chunk_id, text=text, metadata=metadata)
