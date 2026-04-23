"""Ingestion endpoints for building the RAG index."""

import logging
from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.rag.loader import load_rag_cases
from app.rag.chunker import chunk_rag_case
from app.rag.embedder import get_embedder
from app.rag.store import get_chroma_store
from app.schemas.retrieval import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/build", response_model=IngestResponse)
async def build_collection(request: IngestRequest = IngestRequest()) -> IngestResponse:
    """
    Build or refresh the Chroma collection from RAG cases CSV.

    Args:
        request: Ingest configuration (overwrite flag)

    Returns:
        IngestResponse with status and counts
    """
    try:
        settings = get_settings()
        store = get_chroma_store(settings.chroma_persist_dir)
        embedder = get_embedder()

        # Get or create collection with embedding function
        chroma_embedding_fn = store.client.get_embedding_function()
        collection = store.get_or_create_collection(embedding_function=chroma_embedding_fn)

        # Clear if requested
        if request.overwrite:
            logger.info("Clearing collection before rebuild")
            collection.delete(where={})

        # Load, chunk, and insert cases
        cases_processed = 0
        batch_size = 100
        batch_ids = []
        batch_texts = []
        batch_metadatas = []

        for rag_case in load_rag_cases(settings.rag_data_path):
            chunk = chunk_rag_case(rag_case)

            batch_ids.append(chunk.chunk_id)
            batch_texts.append(chunk.text)
            batch_metadatas.append(chunk.metadata)
            cases_processed += 1

            # Insert in batches
            if len(batch_ids) >= batch_size:
                embeddings = embedder.embed(batch_texts)
                collection.upsert(
                    ids=batch_ids,
                    embeddings=embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                )
                batch_ids = []
                batch_texts = []
                batch_metadatas = []
                logger.info(f"Processed {cases_processed} cases...")

        # Insert remaining batch
        if batch_ids:
            embeddings = embedder.embed(batch_texts)
            collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
            )

        collection_count = store.collection_count()
        logger.info(f"Collection built: {cases_processed} cases processed, {collection_count} total in collection")

        return IngestResponse(
            success=True,
            cases_processed=cases_processed,
            collection_count=collection_count,
            message=f"Successfully ingested {cases_processed} cases",
        )

    except FileNotFoundError as e:
        logger.error(f"RAG data file not found: {e}")
        raise HTTPException(status_code=404, detail=f"RAG data file not found: {e}")
    except Exception as e:
        logger.error(f"Ingest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")
