"""Embedding function adapter for Chroma using sentence-transformers."""

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Wrapper around sentence-transformers for embedding texts."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedder with specified model."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=False)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings


@lru_cache(maxsize=1)
def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Get or create cached embedder instance."""
    return Embedder(model_name=model_name)
