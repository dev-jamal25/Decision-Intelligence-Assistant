"""Inspection endpoints (collection stats, model info, config snapshot)."""

import logging
from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.rag.store import get_chroma_store
from app.schemas.retrieval import StoreInspectResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/inspect", tags=["inspect"])


@router.get("/store", response_model=StoreInspectResponse)
async def inspect_store() -> StoreInspectResponse:
    """
    Inspect the Chroma collection status and statistics.

    Returns:
        StoreInspectResponse with collection details
    """
    try:
        settings = get_settings()
        store = get_chroma_store(settings.chroma_persist_dir)

        # Try to get collection count
        try:
            count = store.collection_count()
            status = "ready" if count > 0 else "empty"
        except Exception as e:
            logger.warning(f"Could not retrieve collection: {e}")
            count = 0
            status = "not_initialized"

        return StoreInspectResponse(
            collection_name=store.COLLECTION_NAME,
            document_count=count,
            status=status,
        )

    except Exception as e:
        logger.error(f"Store inspection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inspection failed: {str(e)}")
