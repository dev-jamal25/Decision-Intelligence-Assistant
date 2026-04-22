"""FastAPI application entry point. Skeleton only — no business logic yet."""

from fastapi import FastAPI

from app.routers import analyze, health, ingest, inspect, search

app = FastAPI(title="Decision Intelligence Assistant")

app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(inspect.router)
