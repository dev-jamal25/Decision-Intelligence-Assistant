# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Project Rules

## Workflow
- Architecture and design choices are decided first; then implement only the requested slice.
- Work in small, isolated steps. Never implement a major subsystem in one shot.
- Do not redesign the agreed project structure unless explicitly asked.
- Prefer simple, defensible solutions over clever or overengineered ones.
- Keep prompts and outputs concise and context-efficient.

## Required output after every task
Always end with:
1. what you changed
2. files created
3. files modified
4. verification performed
5. unresolved issues or assumptions

## Python / environment
- Use `uv`, not `pip`, for Python dependency and environment management.
- Keep Python dependencies in `pyproject.toml`.
- Do not commit `.venv`, `env`, or `.env`.
- Read configuration from a central config/settings module, not scattered `os.getenv(...)` calls.

## Backend rules
- Keep FastAPI code split into `routers/`, `schemas/`, `services/`, `utils/`, and `rag/`.
- Every endpoint must use Pydantic request/response models.
- Use structured outputs for LLM calls when applicable; do not parse brittle free-form text with regex.
- Use `logging`, not `print`.
- Use proper `HTTPException` status codes; do not return `200 OK` with error dicts.
- Use `lru_cache` only where it clearly makes sense (config, model loading, persistent clients).

## Project-specific rules
- Main frontend-facing API entry point is a single orchestration endpoint: `POST /analyze`.
- Chroma is embedded in the backend.
- RAG knowledge base comes from the Twitter support dataset, not an external document set.
- ML training uses `inbound = TRUE` customer tweets.
- RAG cases are built from `inbound = TRUE` customer tweets plus the first linked `inbound = FALSE` support reply if available.
- Do not touch `data/raw/twcs.csv` unless the task is explicitly about preprocessing or dataset exploration.

## Verification
- For every coding task, verify the result with the smallest relevant check: imports, tests, build, or endpoint startup.
- Do not claim success without stating what was actually verified.

## Testing
- When a slice introduces real logic, prefer adding or updating small focused tests under `tests/` using `pytest`.
- Prefer small, slice-specific tests over broad test suites.
- If formal `pytest` tests are not appropriate yet, explicitly state what lightweight verification was performed instead.
- Do not claim something works unless you actually verified it.

## Data

- [data/raw/twcs.csv](data/raw/twcs.csv) — the Twitter Customer Support (twcs) dataset, ~516 MB. This is the working corpus for ticket prioritization experiments.
- The entire `data/` tree is gitignored; do not commit derived datasets, embeddings, or indices. Write processed artifacts to `data/interim/` or `data/processed/` (conventional) rather than alongside source.

## Gitignore quirks worth knowing

[.gitignore](.gitignore) excludes `data/`, `notebooks/`, and lowercase `claude.md`. Notebooks are intentionally kept out of version control — if exploration work needs to be shared, export to a script or a rendered artifact rather than un-ignoring the directory. 

