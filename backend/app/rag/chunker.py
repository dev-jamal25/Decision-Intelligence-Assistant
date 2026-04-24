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
            text: Text to embed (already prepared)
            metadata: Dict with source information
        """
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata


def chunk_rag_case(case: RAGCase) -> RAGCaseChunk:
    """
    Convert a RAG case into a prepared chunk.

    Uses pre-computed document_text from the case.

    Args:
        case: RAGCase from loader

    Returns:
        RAGCaseChunk ready for insertion
    """
    chunk_id = str(case.customer_tweet_id)
    text = case.document_text
    metadata = case.get_metadata()

    return RAGCaseChunk(chunk_id=chunk_id, text=text, metadata=metadata)
