"""Answer generation endpoints (debug/temporary)."""

import logging
from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.schemas.analyze import RAGAnswerRequest, NonRAGAnswerRequest, AnswerResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["analyze"])

# Initialize services
_rag_service = None
_llm_service = None


def get_rag_service() -> RAGService:
    """Get or create RAG service."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_llm_service() -> LLMService:
    """Get or create LLM service."""
    global _llm_service
    if _llm_service is None:
        settings = get_settings()
        _llm_service = LLMService(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_llm_model,
            base_url=settings.openrouter_base_url,
        )
    return _llm_service


@router.post("/rag", response_model=AnswerResponse)
async def rag_answer(request: RAGAnswerRequest) -> AnswerResponse:
    """
    Generate answer using RAG-retrieved context.

    Args:
        request: RAGAnswerRequest with query and optional k

    Returns:
        AnswerResponse with answer, model, and context info
    """
    try:
        settings = get_settings()
        rag_service = get_rag_service()
        llm_service = get_llm_service()

        # Retrieve context
        retrieved_cases, context = rag_service.retrieve_context(query=request.query, k=request.k)

        # Generate answer with context
        system_prompt = (
            "You are a helpful support assistant for a customer support team. "
            "Answer the user's question concisely and accurately based on the retrieved support cases provided. "
            "If the retrieved cases are relevant, cite them. If they are not relevant, answer based on general knowledge. "
            "Be cautious and do not make claims about policies you are not certain about."
        )
        user_message = f"Retrieved support cases for context:\n\n{context}\n\nUser question: {request.query}"

        llm_response = llm_service.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.7,
            max_tokens=500,
        )

        return AnswerResponse(
            query=request.query,
            answer=llm_response.text,
            model=llm_service.model,
            context_available=len(retrieved_cases) > 0,
            retrieved_count=len(retrieved_cases),
            usage=llm_response.usage,
        )

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"RAG answer generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")


@router.post("/non-rag", response_model=AnswerResponse)
async def non_rag_answer(request: NonRAGAnswerRequest) -> AnswerResponse:
    """
    Generate answer without RAG context (zero-shot).

    Args:
        request: NonRAGAnswerRequest with query

    Returns:
        AnswerResponse with answer and model info
    """
    try:
        settings = get_settings()
        llm_service = get_llm_service()

        # Generate answer without context
        system_prompt = (
            "You are a helpful support assistant for a customer support team. "
            "Answer the user's question concisely and accurately based on your knowledge. "
            "Be cautious and do not make claims about policies you are not certain about."
        )
        user_message = f"User question: {request.query}"

        llm_response = llm_service.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.7,
            max_tokens=500,
        )

        return AnswerResponse(
            query=request.query,
            answer=llm_response.text,
            model=llm_service.model,
            context_available=False,
            retrieved_count=None,
            usage=llm_response.usage,
        )

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"Non-RAG answer generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")
