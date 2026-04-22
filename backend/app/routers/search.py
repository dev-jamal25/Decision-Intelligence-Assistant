"""Low-level retrieval endpoints for debugging/inspection. Skeleton only."""

from fastapi import APIRouter

router = APIRouter(prefix="/search", tags=["search"])

# TODO: GET / — raw similarity search against the Chroma collection.
