"""Answer generation and orchestration endpoints."""

import json
import logging
import re
import time

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.services.llm_service import LLMService, LLMResponse
from app.services.rag_service import RAGService
from app.services.priority_service import get_priority_service
from app.services.pricing_service import PricingService, normalize_usage
from app.schemas.analyze import (
    RAGAnswerRequest,
    NonRAGAnswerRequest,
    AnswerResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    LLMCallUsage,
    LatencyMs,
    RetrievedCaseInfo,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["analyze"])

_rag_service = None
_llm_service = None
_pricing_service = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_llm_service() -> LLMService:
    """Get or create LLM service (OpenRouter → OpenRouter fallback → Gemini)."""
    global _llm_service
    if _llm_service is None:
        settings = get_settings()
        _llm_service = LLMService(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_llm_model,
            base_url=settings.openrouter_base_url,
            gemini_api_key=settings.gemini_api_key,
            gemini_model=settings.gemini_llm_model,
            gemini_base_url=settings.gemini_base_url,
            fallback_model=settings.openrouter_llm_fallback_model,
        )
    return _llm_service


def get_pricing_service() -> PricingService:
    global _pricing_service
    if _pricing_service is None:
        settings = get_settings()
        _pricing_service = PricingService(
            openrouter_api_key=settings.openrouter_api_key,
            openrouter_base_url=settings.openrouter_base_url,
        )
    return _pricing_service


# ── LLM zero-shot priority helpers ────────────────────────────────────────────

_PRIORITY_SYSTEM_PROMPT = (
    "You are a customer support ticket priority classifier. "
    "Classify the following customer query as exactly 'urgent' or 'normal'. "
    "Urgent means the customer faces account loss, billing error, service outage, "
    "security issue, or severe immediate impact. Normal means general questions or "
    "minor issues. "
    "Respond with ONLY valid JSON, no markdown fences, no explanation:\n"
    '{"priority": "urgent" or "normal", "confidence": <float 0.0-1.0>, "rationale": "<one sentence>"}'
)


def _parse_priority_response(text: str) -> tuple[str, float, str]:
    """Parse LLM priority JSON. Returns (priority, confidence, rationale)."""
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    try:
        data = json.loads(cleaned)
        priority = str(data.get("priority", "normal")).lower().strip()
        if priority not in ("urgent", "normal"):
            priority = "normal"
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        rationale = str(data.get("rationale", "")).strip()
        return priority, confidence, rationale
    except Exception:
        priority = "urgent" if "urgent" in text.lower() else "normal"
        return priority, 0.5, ""


# ── Per-call usage builder ─────────────────────────────────────────────────────

def _build_call_usage(resp: LLMResponse, pricing: PricingService) -> LLMCallUsage:
    """Build a LLMCallUsage from an LLMResponse using the pricing service."""
    prompt, completion, total = normalize_usage(resp.usage)
    estimated = pricing.estimate_cost(resp.provider, resp.model, prompt, completion)
    return LLMCallUsage(
        provider=resp.provider,
        model=resp.model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        estimated_cost_usd=estimated,
    )


def _usage_summary(calls: list[LLMCallUsage]) -> dict:
    """Total tokens and cost across a list of per-call objects."""
    total_prompt = sum(c.prompt_tokens or 0 for c in calls)
    total_completion = sum(c.completion_tokens or 0 for c in calls)
    costs = [c.estimated_cost_usd for c in calls if c.estimated_cost_usd is not None]
    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "estimated_cost_usd": round(sum(costs), 8) if costs else None,
    }


# ── Main orchestration endpoint ────────────────────────────────────────────────

@router.post("", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        t_total_start = time.perf_counter()

        rag_service = get_rag_service()
        llm_service = get_llm_service()
        priority_service = get_priority_service()
        pricing = get_pricing_service()

        # 1. Retrieval
        t0 = time.perf_counter()
        retrieved_cases, rag_context = rag_service.retrieve_context(
            query=request.query, k=request.k
        )
        latency_retrieval = (time.perf_counter() - t0) * 1000

        retrieved_count = len(retrieved_cases)
        top_score = retrieved_cases[0].score if retrieved_cases else None
        retrieval_is_weak = top_score is None or top_score < 0.3
        retrieved_cases_info = [
            RetrievedCaseInfo(case_id=c.case_id, text=c.text, score=c.score)
            for c in retrieved_cases
        ]

        # 2. ML priority
        t0 = time.perf_counter()
        priority_pred = priority_service.predict(request.query)
        latency_ml = (time.perf_counter() - t0) * 1000

        # 3. LLM zero-shot priority
        t0 = time.perf_counter()
        llm_priority_resp = llm_service.generate(
            system_prompt=_PRIORITY_SYSTEM_PROMPT,
            user_message=f"Customer query: {request.query}",
            temperature=0.2,
            max_tokens=120,
        )
        latency_llm_priority = (time.perf_counter() - t0) * 1000
        llm_priority, llm_priority_conf, llm_priority_rationale = _parse_priority_response(
            llm_priority_resp.text
        )

        # 4. RAG answer
        t0 = time.perf_counter()
        rag_answer_resp = llm_service.generate(
            system_prompt=(
                "You are a helpful support assistant for a customer support team. "
                "Answer the user's question concisely and accurately based on the retrieved "
                "support cases provided. If the retrieved cases are relevant, cite them. "
                "If they are not relevant, answer based on general knowledge. "
                "Be cautious and do not make claims about policies you are not certain about."
            ),
            user_message=(
                f"Retrieved support cases for context:\n\n{rag_context}\n\n"
                f"User question: {request.query}"
            ),
            temperature=0.7,
            max_tokens=500,
        )
        latency_rag = (time.perf_counter() - t0) * 1000

        # 5. Non-RAG answer
        t0 = time.perf_counter()
        non_rag_answer_resp = llm_service.generate(
            system_prompt=(
                "You are a helpful support assistant for a customer support team. "
                "Answer the user's question concisely and accurately based on your knowledge. "
                "Be cautious and do not make claims about policies you are not certain about."
            ),
            user_message=f"User question: {request.query}",
            temperature=0.7,
            max_tokens=500,
        )
        latency_non_rag = (time.perf_counter() - t0) * 1000

        latency_total = (time.perf_counter() - t_total_start) * 1000

        fallback_used = (
            rag_answer_resp.fallback_used
            or non_rag_answer_resp.fallback_used
            or llm_priority_resp.fallback_used
        )

        # Per-call usage objects
        priority_call_usage = _build_call_usage(llm_priority_resp, pricing)
        rag_call_usage = _build_call_usage(rag_answer_resp, pricing)
        non_rag_call_usage = _build_call_usage(non_rag_answer_resp, pricing)

        return AnalyzeResponse(
            query=request.query,
            # backward-compat ML fields
            priority_prediction=priority_pred.priority,
            priority_confidence=priority_pred.confidence,
            priority_model=priority_pred.model_name,
            # explicit ML
            ml_priority_prediction=priority_pred.priority,
            ml_priority_confidence=priority_pred.confidence,
            # LLM zero-shot priority
            llm_zero_shot_priority_prediction=llm_priority,
            llm_zero_shot_priority_confidence=llm_priority_conf,
            llm_zero_shot_priority_rationale=llm_priority_rationale or None,
            # retrieval
            retrieved_cases=retrieved_cases_info,
            retrieved_count=retrieved_count,
            top_score=top_score,
            retrieval_is_weak=retrieval_is_weak,
            retrieval_threshold=0.3,
            # answers
            rag_answer=rag_answer_resp.text,
            non_rag_answer=non_rag_answer_resp.text,
            answer_model=rag_answer_resp.model,
            answer_provider=rag_answer_resp.provider,
            fallback_used=fallback_used,
            # latency
            latency_ms=LatencyMs(
                retrieval=round(latency_retrieval, 1),
                ml=round(latency_ml, 1),
                llm_zero_shot_priority=round(latency_llm_priority, 1),
                rag=round(latency_rag, 1),
                non_rag=round(latency_non_rag, 1),
                total=round(latency_total, 1),
            ),
            # per-call usage
            llm_zero_shot_priority_usage=priority_call_usage,
            rag_answer_usage=rag_call_usage,
            non_rag_answer_usage=non_rag_call_usage,
            usage_summary=_usage_summary([priority_call_usage, rag_call_usage, non_rag_call_usage]),
        )

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ── Debug endpoints (kept for backward compatibility) ─────────────────────────


@router.post("/rag", response_model=AnswerResponse)
async def rag_answer(request: RAGAnswerRequest) -> AnswerResponse:
    """Generate answer using RAG-retrieved context (debug endpoint)."""
    try:
        rag_service = get_rag_service()
        llm_service = get_llm_service()
        retrieved_cases, context = rag_service.retrieve_context(
            query=request.query, k=request.k
        )
        llm_response = llm_service.generate(
            system_prompt=(
                "You are a helpful support assistant for a customer support team. "
                "Answer the user's question concisely and accurately based on the retrieved support cases provided. "
                "If the retrieved cases are relevant, cite them. If they are not relevant, answer based on general knowledge. "
                "Be cautious and do not make claims about policies you are not certain about."
            ),
            user_message=f"Retrieved support cases for context:\n\n{context}\n\nUser question: {request.query}",
            temperature=0.7,
            max_tokens=500,
        )
        return AnswerResponse(
            query=request.query,
            answer=llm_response.text,
            model=llm_response.model,
            provider=llm_response.provider,
            fallback_used=llm_response.fallback_used,
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
    """Generate answer without RAG context (zero-shot, debug endpoint)."""
    try:
        llm_service = get_llm_service()
        llm_response = llm_service.generate(
            system_prompt=(
                "You are a helpful support assistant for a customer support team. "
                "Answer the user's question concisely and accurately based on your knowledge. "
                "Be cautious and do not make claims about policies you are not certain about."
            ),
            user_message=f"User question: {request.query}",
            temperature=0.7,
            max_tokens=500,
        )
        return AnswerResponse(
            query=request.query,
            answer=llm_response.text,
            model=llm_response.model,
            provider=llm_response.provider,
            fallback_used=llm_response.fallback_used,
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
