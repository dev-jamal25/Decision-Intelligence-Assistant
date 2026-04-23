"""RAG facade: retrieve grounded cases and format context."""

import logging
from app.config import get_settings
from app.rag.embedder import get_embedder
from app.rag.store import get_chroma_store
from app.rag.retriever import Retriever, RetrievedCase

logger = logging.getLogger(__name__)


class RAGService:
    """Wrapper around RAG retrieval for context generation."""

    def __init__(self):
        """Initialize RAG service with dependencies (not collection - checked dynamically)."""
        self.settings = get_settings()
        self.store = get_chroma_store(self.settings.chroma_persist_dir)
        self.embedder = get_embedder(
            api_key=self.settings.openrouter_api_key,
            model=self.settings.openrouter_embedding_model,
            base_url=self.settings.openrouter_base_url,
        )
        logger.info("Initialized RAG service")

    def retrieve_context(self, query: str, k: int = 5) -> tuple[list[RetrievedCase], str]:
        """
        Retrieve context cases for a query.

        Checks collection availability dynamically (supports post-ingest calls).

        Args:
            query: User query
            k: Number of cases to retrieve

        Returns:
            Tuple of (retrieved_cases, formatted_context_string)
        """
        # Check collection availability dynamically
        try:
            collection = self.store.client.get_collection(name=self.store.COLLECTION_NAME)
        except Exception as e:
            logger.warning(f"RAG collection not available: {e}; returning empty context")
            return [], ""

        retriever = Retriever(collection=collection, embedder=self.embedder)

        try:
            cases = retriever.retrieve(query=query, k=k)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return [], ""

        # Format context for LLM
        context_lines = []
        if cases:
            context_lines.append("Retrieved support cases:")
            for i, case in enumerate(cases, 1):
                context_lines.append(f"\nCase {i} (confidence: {case.score:.2f}):")
                context_lines.append(case.text)
        else:
            context_lines.append("No relevant support cases found in knowledge base.")

        context = "\n".join(context_lines)
        logger.info(f"Retrieved {len(cases)} cases for RAG context")
        return cases, context
