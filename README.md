# Decision-Intelligence-Assistant

A containerized Decision Intelligence Assistant that evaluates RAG, non-RAG, ML baselines, and zero-shot LLMs for customer support ticket prioritization.

## Architecture

- **Backend**: FastAPI + embedded Chroma
- **Frontend**: React
- **Data**: Twitter Customer Support dataset (`data/raw/twcs.csv`)
- **Orchestration entry point**: `POST /analyze`

## Project layout

```
backend/     FastAPI app (routers, schemas, services, rag, utils)
frontend/    React app
artifacts/   Trained ML models and Chroma persistence
data/        raw / interim / processed (gitignored)
notebooks/   Exploration (gitignored)
logs/        analyze / errors / eval
```

## Getting started

TODO — fill in once tooling is wired up.

## Environment

Copy `.env.example` to `.env` and fill in values.
