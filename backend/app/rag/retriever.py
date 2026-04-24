import logging

logger = logging.getLogger(__name__)


class RetrievedCase:
    """Result from similarity search."""

    def __init__(self, case_id: str, text: str, score: float, metadata: dict):
        """
        Initialize a retrieved case.

        Args:
            case_id: Unique identifier for the case
            text: The combined customer + support text
            score: Similarity score (0-1)
            metadata: Dict with customer_tweet_id, author_id, etc.
        """
        self.case_id = case_id
        self.text = text
        self.score = score
        self.metadata = metadata

    def to_dict(self) -> dict:
        """Convert to dict for JSON response."""
        return {
            "case_id": self.case_id,
            "text": self.text,
            "score": float(self.score),
            "metadata": self.metadata,
        }


class Retriever:
    """Wrapper around Chroma collection for similarity search."""

    def __init__(self, collection, embedder):
        """
        Initialize retriever with a Chroma collection and embedder.

        Args:
            collection: Chroma collection object (created without embedding function)
            embedder: OpenRouterEmbedder instance for explicit query embedding
        """
        self.collection = collection
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedCase]:
        """
        Retrieve top-k similar cases from the collection.

        Embeds the query explicitly using OpenRouter, then queries Chroma.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of RetrievedCase objects
        """
        try:
            # Embed query explicitly
            query_embedding = self.embedder.embed([query])
            if not query_embedding:
                logger.error(f"Failed to embed query: {query}")
                return []

            # Query with embeddings (collection has no embedding function)
            results = self.collection.query(query_embeddings=query_embedding, n_results=k)
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            return []

        # Unpack Chroma results
        retrieved = []
        if results["ids"] and len(results["ids"]) > 0:
            ids = results["ids"][0]
            documents = results["documents"][0]
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]

            for case_id, doc, distance, metadata in zip(ids, documents, distances, metadatas):
                # Convert distance to similarity (cosine distance to similarity)
                similarity = 1 - distance
                case = RetrievedCase(
                    case_id=case_id,
                    text=doc,
                    score=similarity,
                    metadata=metadata,
                )
                retrieved.append(case)

        logger.info(f"Retrieved {len(retrieved)} cases for query (k={k})")
        return retrieved
