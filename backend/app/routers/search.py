"""Low-level retrieval endpoints for debugging/inspection."""

import logging
from fastapi import APIRouter, HTTPException, Query

from app.config import get_settings
from app.rag.embedder import get_embedder
from app.rag.store import get_chroma_store
from app.rag.retriever import Retriever
from app.schemas.retrieval import RetrievalResultSchema

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=RetrievalResultSchema)
async def search(
    query: str = Query(..., min_length=1, description="Query text"),
    k: int = Query(5, ge=1, le=50, description="Number of results to return"),
) -> RetrievalResultSchema:
    """
    Raw similarity search against the Chroma collection.

    Args:
        query: Search query text
        k: Number of results to return (1-50)

    Returns:
        RetrievalResultSchema with matched cases
    """
    try:
        settings = get_settings()
        store = get_chroma_store(settings.chroma_persist_dir)
        embedder = get_embedder(
            api_key=settings.openrouter_api_key,
            model=settings.embedding_model,
            base_url=settings.embedding_base_url,
        )

        try:
            collection = store.client.get_collection(name=store.COLLECTION_NAME)
        except Exception as e:
            logger.error(f"Collection not found: {e}")
            raise HTTPException(
                status_code=404,
                detail="RAG collection not found.",
            )

        retriever = Retriever(collection=collection, embedder=embedder)
        retrieved_cases = retriever.retrieve(query=query, k=k)

        return RetrievalResultSchema(
            query=query,
            k=k,
            cases=[case.to_dict() for case in retrieved_cases],
            total_retrieved=len(retrieved_cases),
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
