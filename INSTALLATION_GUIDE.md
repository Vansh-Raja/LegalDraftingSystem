# Installation Guide

This guide covers running the Legal Drafting System locally or via Docker.

## Prerequisites
- Python 3.12+
- (Optional) Docker and docker compose
- Postgres with `vector` extension (compose includes pgvector)
- API keys as needed:
  - OpenAI: `OPENAI_KEY` or `OPENAI_API_KEY`
  - OpenRouter: `OPENROUTER_API_KEY`
  - Ollama (optional, for local models): `OLLAMA_HOST`

## Option A: Docker Compose (app + pgvector)
1) Ensure Docker is running.
2) From repo root:
```bash
docker compose up --build
```
3) Access Streamlit at `http://localhost:8501`. Postgres is exposed on host port `${PGVECTOR_PORT:-5433}`.
4) The pgvector service auto-creates the `vector` extension via `db-init/00-create-vector-ext.sql`. If you need to rerun manually:
```bash
docker compose exec pgvector psql -U postgres -d LegalDraftingSystemDB -c "CREATE EXTENSION IF NOT EXISTS vector;"
```
5) Environment handling:
- The image build excludes `.env` via `.dockerignore`. Compose will read host env vars (or a `.env` file in the project root) at runtime; secrets are not baked into the image.
- Set provider keys (e.g., `OPENAI_KEY`, `OPENROUTER_API_KEY`) and DB overrides before running `docker compose up`, or use `environment:` entries in `docker-compose.yml`.
5) If using host-side Ollama, export `OLLAMA_HOST` (e.g., `http://host.docker.internal:11434`) before `docker compose up`.

## Option B: Local (venv)
1) Create venv and install deps:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
2) Set environment variables (example):
```bash
export DB_NAME=LegalDraftingSystemDB
export DB_USER=postgres
export DB_PASSWORD=password
export DB_HOST=localhost
export DB_PORT=5433
# Optional:
# export OPENAI_KEY=...
# export OPENROUTER_API_KEY=...
# export OLLAMA_HOST=http://127.0.0.1:11434
```
3) Start services:
- Ensure Postgres is running with vector extension (or use compose for pgvector only: `docker compose up pgvector`).
4) Run the app:
```bash
streamlit run app.py
```
5) (Optional) Ingest data after processing:
```bash
python process.py
python ingest.py
```

## Notes on Ollama vs API
- The system works without Ollama; you can use OpenAI or OpenRouter for both chat and embeddings if keys are provided.
- Ingestion: set provider/model at prompt; if using API embeddings, ensure `OPENAI_KEY` (or OpenRouter) is set. If using Ollama, ensure `OLLAMA_HOST` is reachable.

## Environment summary
- Database: `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT` (or `PGVECTOR_CONNECTION`)
- Models/embeddings: `OPENAI_KEY`/`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `OLLAMA_HOST` (optional)


