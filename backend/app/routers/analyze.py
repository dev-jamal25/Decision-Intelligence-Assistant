"""Answer generation and orchestration endpoints."""

import logging
from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.priority_service import get_priority_service
from app.schemas.analyze import (
    RAGAnswerRequest,
    NonRAGAnswerRequest,
    AnswerResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    RetrievedCaseInfo,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["analyze"])

# Initialize services
_rag_service = None
_llm_service = None
_priority_service = None


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


@router.post("/", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Unified analysis endpoint: combines RAG, LLM, and ML priority.

    Args:
        request: AnalyzeRequest with query and optional k

    Returns:
        AnalyzeResponse with priority, answers, and retrieval results
    """
    try:
        rag_service = get_rag_service()
        llm_service = get_llm_service()
        priority_service = get_priority_service()

        # Run retrieval
        retrieved_cases, rag_context = rag_service.retrieve_context(query=request.query, k=request.k)

        # Extract retrieval diagnostics
        retrieved_count = len(retrieved_cases)
        top_score = retrieved_cases[0].score if retrieved_cases else None
        retrieval_is_weak = top_score is None or top_score < 0.3  # v1: threshold = 0.3

        # Format retrieved cases for response
        retrieved_cases_info = [
            RetrievedCaseInfo(case_id=case.case_id, text=case.text, score=case.score)
            for case in retrieved_cases
        ]

        # Run ML priority prediction
        priority_pred = priority_service.predict(request.query)

        # Generate RAG answer
        system_prompt_rag = (
            "You are a helpful support assistant for a customer support team. "
            "Answer the user's question concisely and accurately based on the retrieved support cases provided. "
            "If the retrieved cases are relevant, cite them. If they are not relevant, answer based on general knowledge. "
            "Be cautious and do not make claims about policies you are not certain about."
        )
        user_message_rag = f"Retrieved support cases for context:\n\n{rag_context}\n\nUser question: {request.query}"
        rag_answer_resp = llm_service.generate(
            system_prompt=system_prompt_rag,
            user_message=user_message_rag,
            temperature=0.7,
            max_tokens=500,
        )

        # Generate non-RAG answer
        system_prompt_non_rag = (
            "You are a helpful support assistant for a customer support team. "
            "Answer the user's question concisely and accurately based on your knowledge. "
            "Be cautious and do not make claims about policies you are not certain about."
        )
        user_message_non_rag = f"User question: {request.query}"
        non_rag_answer_resp = llm_service.generate(
            system_prompt=system_prompt_non_rag,
            user_message=user_message_non_rag,
            temperature=0.7,
            max_tokens=500,
        )

        return AnalyzeResponse(
            query=request.query,
            priority_prediction=priority_pred.priority,
            priority_confidence=priority_pred.confidence,
            priority_model=priority_pred.model_name,
            retrieved_cases=retrieved_cases_info,
            retrieved_count=retrieved_count,
            top_score=top_score,
            retrieval_is_weak=retrieval_is_weak,
            retrieval_threshold=0.3,
            rag_answer=rag_answer_resp.text,
            non_rag_answer=non_rag_answer_resp.text,
            answer_model=llm_service.model,
        )

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ========== Debug endpoints (kept for backward compatibility) ==========


@router.post("/rag", response_model=AnswerResponse)
async def rag_answer(request: RAGAnswerRequest) -> AnswerResponse:
    """
    Generate answer using RAG-retrieved context (debug endpoint).

    Args:
        request: RAGAnswerRequest with query and optional k

    Returns:
        AnswerResponse with answer, model, and context info
    """
    try:
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
    Generate answer without RAG context (zero-shot, debug endpoint).

    Args:
        request: NonRAGAnswerRequest with query

    Returns:
        AnswerResponse with answer and model info
    """
    try:
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
