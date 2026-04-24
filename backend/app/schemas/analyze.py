"""Pydantic models for answer generation and orchestration endpoints."""

from pydantic import BaseModel, Field
from typing import Optional


class RAGAnswerRequest(BaseModel):
    """Request for RAG-grounded answer generation."""

    query: str = Field(..., min_length=1, description="User query/support question")
    k: int = Field(5, ge=1, le=20, description="Number of similar cases to retrieve")


class NonRAGAnswerRequest(BaseModel):
    """Request for non-RAG answer generation."""

    query: str = Field(..., min_length=1, description="User query/support question")


class AnswerResponse(BaseModel):
    """Response with generated answer."""

    query: str = Field(..., description="The user query")
    answer: str = Field(..., description="Generated answer")
    model: str = Field(..., description="LLM model used")
    provider: str = Field(default="openrouter", description="LLM provider used (openrouter or gemini)")
    fallback_used: bool = Field(default=False, description="Whether Gemini fallback was used")
    context_available: bool = Field(default=False, description="Whether RAG context was used")
    retrieved_count: Optional[int] = Field(None, description="Number of retrieved cases (for RAG)")
    usage: Optional[dict] = Field(None, description="Token usage from LLM")


# ========== Unified Orchestration ==========


class AnalyzeRequest(BaseModel):
    """Request for unified analysis (orchestration)."""

    query: str = Field(..., min_length=1, description="User query/support question")
    k: int = Field(5, ge=1, le=20, description="Number of retrieved cases for RAG context")


class RetrievedCaseInfo(BaseModel):
    """Summary of a retrieved case."""

    case_id: str = Field(..., description="Unique case identifier")
    text: str = Field(..., description="Case text")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")


class LLMCallUsage(BaseModel):
    """Token usage and estimated cost for a single LLM call."""

    provider: str = Field(..., description="Provider that answered (openrouter or gemini)")
    model: str = Field(..., description="Model that answered")
    prompt_tokens: Optional[int] = Field(None, description="Input token count")
    completion_tokens: Optional[int] = Field(None, description="Output token count")
    total_tokens: Optional[int] = Field(None, description="Total token count")
    estimated_cost_usd: Optional[float] = Field(
        None, description="Estimated USD cost; null if pricing unavailable"
    )


class LatencyMs(BaseModel):
    """Per-step latency breakdown in milliseconds."""

    retrieval: float = Field(..., description="Vector retrieval time (ms)")
    ml: float = Field(..., description="ML priority prediction time (ms)")
    llm_zero_shot_priority: float = Field(..., description="LLM zero-shot priority call time (ms)")
    rag: float = Field(..., description="RAG answer generation time (ms)")
    non_rag: float = Field(..., description="Non-RAG answer generation time (ms)")
    total: float = Field(..., description="Total analyze wall-clock time (ms)")


class AnalyzeResponse(BaseModel):
    """Unified orchestration response combining RAG, LLM, and ML priority."""

    query: str = Field(..., description="The user query")

    # ── Backward-compatible priority fields (ML) ──────────────────────────────
    priority_prediction: str = Field(..., description="ML priority prediction (normal/urgent)")
    priority_confidence: float = Field(..., ge=0.0, le=1.0, description="ML priority confidence")
    priority_model: str = Field(..., description="ML model name")

    # ── Explicit ML priority (same values, explicit naming for comparison) ─────
    ml_priority_prediction: str = Field(..., description="ML priority prediction (normal/urgent)")
    ml_priority_confidence: float = Field(..., ge=0.0, le=1.0, description="ML priority confidence")

    # ── LLM zero-shot priority ─────────────────────────────────────────────────
    llm_zero_shot_priority_prediction: str = Field(
        ..., description="LLM zero-shot priority prediction (normal/urgent)"
    )
    llm_zero_shot_priority_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="LLM self-reported confidence for priority"
    )
    llm_zero_shot_priority_rationale: Optional[str] = Field(
        None, description="LLM one-sentence rationale for priority decision"
    )

    # ── Retrieval results ──────────────────────────────────────────────────────
    retrieved_cases: list[RetrievedCaseInfo] = Field(
        default_factory=list, description="Top-k retrieved cases"
    )
    retrieved_count: int = Field(..., ge=0, description="Number of cases retrieved")
    top_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Highest similarity score")
    retrieval_is_weak: bool = Field(..., description="Whether retrieval is weak (below threshold)")
    retrieval_threshold: float = Field(default=0.3, description="Weak-retrieval threshold")

    # ── Answer generation ──────────────────────────────────────────────────────
    rag_answer: str = Field(..., description="RAG-grounded answer")
    non_rag_answer: str = Field(..., description="Zero-shot answer (without RAG context)")
    answer_model: str = Field(..., description="LLM model used for answers")
    answer_provider: str = Field(default="openrouter", description="LLM provider (openrouter or gemini)")
    fallback_used: bool = Field(default=False, description="Whether a fallback LLM was used")

    # ── Latency ───────────────────────────────────────────────────────────────
    latency_ms: LatencyMs = Field(..., description="Per-step latency breakdown in ms")

    # ── Per-call usage / cost ─────────────────────────────────────────────────
    rag_answer_usage: Optional[LLMCallUsage] = Field(None, description="Usage for RAG answer call")
    non_rag_answer_usage: Optional[LLMCallUsage] = Field(None, description="Usage for non-RAG answer call")
    llm_zero_shot_priority_usage: Optional[LLMCallUsage] = Field(
        None, description="Usage for LLM zero-shot priority call"
    )
    usage_summary: Optional[dict] = Field(None, description="Totals across all three LLM calls")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    model_info: Optional[dict] = Field(None, description="Additional model/diagnostics info")
