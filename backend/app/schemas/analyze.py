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


class AnalyzeResponse(BaseModel):
    """Unified orchestration response combining RAG, LLM, and ML priority."""

    query: str = Field(..., description="The user query")

    # Priority prediction
    priority_prediction: str = Field(..., description="Predicted priority (normal/urgent)")
    priority_confidence: float = Field(..., ge=0.0, le=1.0, description="Priority prediction confidence")
    priority_model: str = Field(..., description="Priority model used")

    # Retrieval results
    retrieved_cases: list[RetrievedCaseInfo] = Field(
        default_factory=list, description="Top-k retrieved cases"
    )
    retrieved_count: int = Field(..., ge=0, description="Number of cases retrieved")
    top_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Highest similarity score")
    retrieval_is_weak: bool = Field(
        ..., description="Whether retrieval is weak (below threshold)"
    )
    retrieval_threshold: float = Field(
        default=0.3, description="Threshold for weak retrieval (top_score < threshold)"
    )

    # Answer generation
    rag_answer: str = Field(..., description="RAG-grounded answer")
    non_rag_answer: str = Field(..., description="Zero-shot answer (without RAG context)")
    answer_model: str = Field(..., description="LLM model used for answers")

    # Diagnostics
    model_info: Optional[dict] = Field(
        None, description="Additional model/diagnostics info"
    )
