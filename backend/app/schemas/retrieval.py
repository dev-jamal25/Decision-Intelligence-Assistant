"""Pydantic models for retrieval results."""

from pydantic import BaseModel, Field


class RetrievedCaseSchema(BaseModel):
    """Single retrieved case from similarity search."""

    case_id: str = Field(..., description="Unique case identifier")
    text: str = Field(..., description="Combined customer + support text")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0-1)")
    metadata: dict = Field(default_factory=dict, description="Case metadata")


class RetrievalResultSchema(BaseModel):
    """Result of a retrieval query."""

    query: str = Field(..., description="The query that was executed")
    k: int = Field(..., ge=1, description="Number of results requested")
    cases: list[RetrievedCaseSchema] = Field(default_factory=list, description="Retrieved cases")
    total_retrieved: int = Field(..., ge=0, description="Total number of cases retrieved")


class IngestRequest(BaseModel):
    """Request to build/refresh the Chroma collection."""

    overwrite: bool = Field(default=False, description="If true, clear collection before building")


class IngestResponse(BaseModel):
    """Response from ingest operation."""

    success: bool = Field(..., description="Whether ingest succeeded")
    cases_processed: int = Field(..., ge=0, description="Number of cases processed")
    collection_count: int = Field(..., ge=0, description="Total documents in collection after ingest")
    message: str = Field(..., description="Status message")


class StoreInspectResponse(BaseModel):
    """Response from store inspection endpoint."""

    collection_name: str = Field(..., description="Name of the collection")
    document_count: int = Field(..., ge=0, description="Number of documents in collection")
    status: str = Field(..., description="Collection status")
