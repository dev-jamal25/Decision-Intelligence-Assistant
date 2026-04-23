"""Embedded Chroma client and collection management."""

import logging
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


class ChromaStore:
    """Manages persistent Chroma client and collection access."""

    COLLECTION_NAME = "rag_cases"

    def __init__(self, persist_dir: str):
        """
        Initialize Chroma store with persistent directory.

        Args:
            persist_dir: Path to directory where Chroma will persist data
        """
        self.persist_dir = persist_dir
        chroma_settings = ChromaSettings(
            is_persistent=True,
            persist_directory=persist_dir,
            anonymized_telemetry=False,
        )
        self.client = chromadb.Client(chroma_settings)
        logger.info(f"Initialized Chroma client with persist_dir: {persist_dir}")

    def get_or_create_collection(self, embedding_function=None):
        """
        Get or create the RAG cases collection.

        Args:
            embedding_function: Optional Chroma embedding function.
                If provided, collection is created with it.
                If not provided and collection exists, returns existing collection.

        Returns:
            Chroma collection object
        """
        if embedding_function:
            collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Created/retrieved collection: {self.COLLECTION_NAME}")
        else:
            collection = self.client.get_collection(name=self.COLLECTION_NAME)
            logger.info(f"Retrieved existing collection: {self.COLLECTION_NAME}")
        return collection

    def collection_count(self) -> int:
        """Get document count in collection."""
        try:
            collection = self.client.get_collection(name=self.COLLECTION_NAME)
            return collection.count()
        except Exception as e:
            logger.warning(f"Could not get collection count: {e}")
            return 0


@lru_cache(maxsize=1)
def get_chroma_store(persist_dir: str) -> ChromaStore:
    """Get or create cached Chroma store instance."""
    return ChromaStore(persist_dir=persist_dir)
