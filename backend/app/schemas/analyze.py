"""Pydantic models for answer generation endpoints."""

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
