# Placeholder backend Dockerfile. Not final.
FROM python:3.12-slim

WORKDIR /app

# uv for dependency management (not pip)
RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv sync --no-dev || true

COPY app ./app

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
