FROM python:3.12-slim



# Copy the uv binary from the official image — no curl/install script needed.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/


WORKDIR /app

# System deps for chromadb / pypdf (slim images are minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Layer 1: install dependencies only (cached on pyproject changes) ---
COPY backend/pyproject.toml ./
COPY backend/uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project


# --- Layer 2: copy source code ---
COPY backend/app ./app
COPY data ./data

# Create symlinks so code can find data and artifacts at expected paths
RUN ln -s /app/data /data && \
    ln -s /app/artifacts /artifacts

# Put the uv-managed venv on PATH so "uvicorn" works directly.
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
