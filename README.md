# Decision Intelligence Assistant

A containerized multi-model evaluation system for customer support ticket prioritization. Compares machine learning baselines, zero-shot LLM reasoning, and RAG-augmented retrieval to predict support ticket priority.

## Features

- **Multi-model evaluation**: ML logistic regression, OpenAI LLM, OpenRouter embeddings & fallback, Gemini final fallback
- **RAG retrieval**: Embedded support cases with Chroma vector database
- **Priority prediction**: Compare ML vs. LLM approaches in real time
- **Production containerized**: Docker Compose with persistent storage
- **Web interface**: React SPA with nginx reverse proxy
- **API-first**: RESTful endpoints for programmatic access

## Architecture

```
Frontend (React)
      ↓
nginx (reverse proxy, SPA routing)
      ↓ (HTTP)
Backend (FastAPI)
      ├→ Priority Service (ML logistic regression)
      ├→ LLM Service (OpenAI, with OpenRouter + Gemini fallback)
      ├→ RAG Service (Chroma embeddings + retrieval)
      └→ Health & Inspection endpoints
      ↓
[Persistent Storage]
├─ artifacts/chroma/       (Vector DB)
├─ artifacts/models/       (ML artifacts)
└─ data/knowledge/         (RAG case base)
```

## Tech Stack

- **Frontend**: React + Vite, served by nginx
- **Backend**: FastAPI, uvicorn, Pydantic
- **RAG**: Chroma vector DB, OpenRouter embeddings
- **ML**: scikit-learn (logistic regression, TF-IDF)
- **LLMs**: OpenAI (primary), OpenRouter (fallback), Gemini (final fallback)
- **Packaging**: Docker, Docker Compose

## Quick Start — Docker (Recommended)

**Prerequisites**: Docker, Docker Compose

```bash
cd Decision-Intelligence-Assistant

# 1. Configure environment
cp .env.example .env
# Edit .env and add your API keys:
#   OPENAI_API_KEY=sk-...
#   OPENROUTER_API_KEY=sk-...
#   GEMINI_API_KEY=...

# 2. Build and run
docker-compose up -d --build

# 3. Ingest the case base (one-time setup, ~2 minutes)
curl -X POST "http://localhost:8000/ingest?overwrite=true" \
  -H "Content-Type: application/json"

# 4. Access the app
# Frontend: http://localhost:5173
# API docs: http://localhost:8000/docs
```

That's it. The app is now running and data is ingested.

## Quick Start — Local Development

For backend/frontend development without Docker:

```bash
# Backend
cd backend
uv sync
uv run uvicorn app.main:app --reload

# Frontend (in another terminal)
cd frontend
npm install
npm run dev
```

The frontend will proxy `/api/*` requests to `http://localhost:8000` (see `frontend/vite.config.js`).

## Environment Variables

Create a `.env` file in the project root:

```env
# LLM API Keys (required)
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-...
GEMINI_API_KEY=...

# Optional: RAG retrieval size (default: 5)
RETRIEVAL_K=5

# Optional: CORS origins (defaults work for localhost)
CORS_ORIGINS=http://localhost:8000,http://localhost:5173
```

## Using the App

### 1. Start with a ticket

Open http://localhost:8000/docs and try the **POST /analyze** endpoint with a sample ticket:

```json
{
  "query": "Customer says app keeps crashing on login. Very frustrated.",
  "context": []
}
```

### 2. See the comparison

The response includes:
- `ml_priority`: Priority from trained logistic regression model
- `llm_priority`: Priority from zero-shot LLM reasoning
- `rag_answer`: Top-5 similar support cases from the knowledge base
- `non_rag_answer`: Direct LLM response without retrieval
- `metrics`: Latency, token usage, cost estimate

### 3. Analyze diagnostics

Use **GET /inspect/store** to see how many cases are in the knowledge base:

```bash
curl http://localhost:8000/inspect/store
```

Use **GET /search** to manually query the case base:

```bash
curl "http://localhost:8000/search?query=payment%20issue"
```

## API Overview

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` | Service health check |
| `POST` | `/analyze` | Compare all 4 approaches (ML, LLM, RAG, non-RAG) |
| `POST` | `/analyze/rag` | RAG retrieval only |
| `POST` | `/analyze/non-rag` | LLM-only reasoning |
| `POST` | `/ingest` | Load case base into Chroma (one-time) |
| `GET` | `/search` | Search cases by query |
| `GET` | `/inspect/store` | Inspect vector DB stats |

All endpoints expect JSON and return JSON. See http://localhost:8000/docs for interactive docs.

## Comparison Methodology

The app compares four approaches for ticket prioritization:

1. **ML (Logistic Regression)**: Fast, trained on labeled data, production-ready
2. **Zero-Shot LLM**: OpenAI GPT reasoning without context, slower but nuanced
3. **RAG + LLM**: LLM with top-5 similar cases injected as context
4. **Non-RAG LLM**: LLM-only response (baseline for RAG effectiveness)

Each request returns priority scores, latency, and estimated API costs. Use this to understand the trade-off between speed, accuracy, and cost.

## Project Structure

```
backend/
  ├─ app/
  │  ├─ main.py              # FastAPI app setup
  │  ├─ config.py            # Config (API keys, paths)
  │  ├─ routers/             # Route handlers (analyze, ingest, search, health)
  │  ├─ schemas/             # Pydantic request/response models
  │  ├─ services/            # Business logic (priority, LLM, RAG)
  │  ├─ rag/                 # RAG pipeline (loader, chunker, embedder, store, retriever)
  │  └─ prompts/             # LLM prompt templates
  ├─ pyproject.toml          # Dependencies
  └─ Dockerfile

frontend/
  ├─ src/
  │  ├─ components/          # React components (Input, Results, Panels)
  │  ├─ services/            # API client
  │  └─ App.jsx
  ├─ package.json
  ├─ vite.config.js          # Vite config (proxies /api to backend)
  ├─ nginx.conf              # nginx reverse proxy rules
  └─ Dockerfile

artifacts/
  ├─ chroma/                 # ChromaDB persistence (created at runtime)
  └─ models/                 # ML artifacts (logistic regression, TF-IDF vectorizer)

data/
  ├─ raw/                    # Original Twitter Customer Support dataset (gitignored)
  ├─ interim/                # Processed datasets
  └─ knowledge/
     └─ rag_case_base_v1.csv # Support cases for RAG

tests/
docker-compose.yml           # Multi-service orchestration
.env.example                 # Template for environment variables
```

## Performance & Fallbacks

### Latency
- **ML**: <50ms (local inference)
- **LLM (zero-shot)**: 1–3s (API call + reasoning)
- **RAG + LLM**: 3–5s (retrieval + context injection + reasoning)

### LLM Fallback Chain
1. **Primary**: OpenAI (`gpt-4-turbo`)
2. **Fallback**: OpenRouter (free tier alternatives)
3. **Final**: Google Gemini (if both fail)

All models generate a priority score (1–5) and confidence estimate.

### Cost
- OpenAI: ~$0.01 per analysis
- OpenRouter: ~$0.002 per analysis (free tier available)
- Embeddings (OpenRouter): ~$0.0001 per query

## Known Limitations

1. **ML model**: Trained on Twitter customer support data; may not generalize well to other domains
2. **RAG case base**: ~2000 curated cases; quality depends on relevance and diversity
3. **LLM reasoning**: Subject to prompt injection and hallucination; use with caution in production
4. **Educational demo**: This version is built for learning; production deployment would need input validation, rate limiting, and monitoring
5. **No authentication**: All endpoints are public; add API key validation for real use

## Recommendation

**For production deployment:**

- **Use the ML model** (logistic regression) as the primary priority scorer
  - ✅ Fast, deterministic, low cost
  - ✅ Trained on real support data
  - ✅ Works without external API calls
  - ⚠️ Requires periodic retraining on new data

- **Use LLM for escalations and nuance**
  - When ML confidence is low (<0.6), escalate to human review
  - Or use zero-shot LLM for high-variance edge cases
  - RAG retrieval can augment human reviewers with similar past cases

- **Monitor and retrain**
  - Log all predictions and outcomes
  - Retrain ML model monthly on new labeled data
  - A/B test LLM vs. ML on holdout tickets

## Getting Help

- **API docs**: http://localhost:8000/docs (when running)
- **Logs**: Check `docker logs decision-intelligence-backend`
- **Data issues**: Run `/ingest` again with `?overwrite=true` to reload

---

**Status**: v0.1 — Educational demonstration. Production hardening required.
