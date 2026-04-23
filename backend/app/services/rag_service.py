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
        """Initialize RAG service with retriever."""
        settings = get_settings()
        store = get_chroma_store(settings.chroma_persist_dir)
        embedder = get_embedder(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_embedding_model,
            base_url=settings.openrouter_base_url,
        )
        try:
            collection = store.client.get_collection(name=store.COLLECTION_NAME)
        except Exception as e:
            logger.warning(f"RAG collection not available: {e}")
            collection = None

        self.retriever = Retriever(collection=collection, embedder=embedder) if collection else None
        logger.info("Initialized RAG service")

    def retrieve_context(self, query: str, k: int = 5) -> tuple[list[RetrievedCase], str]:
        """
        Retrieve context cases for a query.

        Args:
            query: User query
            k: Number of cases to retrieve

        Returns:
            Tuple of (retrieved_cases, formatted_context_string)
        """
        if not self.retriever:
            logger.warning("Retriever not initialized; returning empty context")
            return [], ""

        try:
            cases = self.retriever.retrieve(query=query, k=k)
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
