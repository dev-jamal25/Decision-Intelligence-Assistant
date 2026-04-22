"""Inspection endpoints (collection stats, model info, etc). Skeleton only."""

from fastapi import APIRouter

router = APIRouter(prefix="/inspect", tags=["inspect"])

# TODO: endpoints that expose index size, loaded model info, config snapshot.
