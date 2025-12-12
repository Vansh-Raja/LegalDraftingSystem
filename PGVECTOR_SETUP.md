# PGVector Setup Guide (Legal Drafting System) — Docker-first

The app expects PostgreSQL with the `vector` extension reachable via `DB_*` or `PGVECTOR_CONNECTION` (see `rag.py` `_get_pg_connection_string`). Below uses Docker for a clean, reproducible setup.

## 1) docker-compose (recommended)

Create `docker-compose.yml`:
```yaml
version: "3.9"
services:
  pgvector:
    image: ankane/pgvector:latest
    container_name: pgvector
    environment:
      POSTGRES_USER: lds_user
      POSTGRES_PASSWORD: your_password
      POSTGRES_DB: lds_db
    ports:
      - "5432:5432"
    volumes:
      - pgvector_data:/var/lib/postgresql/data
volumes:
  pgvector_data:
```

Bring it up:
```bash
docker compose up -d
```

## 2) Enable extension (one-time)
The image includes pgvector. Create the extension inside the DB:
```bash
docker compose exec pgvector psql -U lds_user -d lds_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## 3) App connection env (.env)
Use either the granular vars or single URL:
```bash
DB_NAME=lds_db
DB_USER=lds_user
DB_PASSWORD=your_password
DB_HOST=127.0.0.1
DB_PORT=5432
# or
PGVECTOR_CONNECTION=postgresql://lds_user:your_password@127.0.0.1:5432/lds_db
```
`python-dotenv` loads this for all modules.

## 4) What the code will create
- **Vector store**: `langchain_postgres.PGVector` in `rag.py` creates the embeddings table/collection (default collection `langchain`) with the right dimension for your chosen embedding (OpenAI or Ollama).
- **FTS summaries**: `ingest.py` → `upsert_all_case_summaries_from_metadata` → `ensure_case_summaries_schema` creates `case_summaries` and its GIN index on `ts`.
You do not need to create tables manually if the user has privileges.

## 5) Tuning (optional, inside container via `postgresql.conf`)
- `shared_buffers`: ~25% RAM
- `work_mem`: 16–64MB
- `maintenance_work_mem`: 512MB–1GB
- `wal_level = replica`, `max_wal_size = 1GB`, `checkpoint_timeout = 15min`
- `effective_io_concurrency = 200` (SSD)

## 6) Networking
- Local dev: publish `5432:5432` as above.
- Remote host: open the port and set `DB_HOST` to that host/IP. Keep strong passwords.

## 7) Validation checklist
- `docker compose exec pgvector psql -U lds_user -d lds_db -c "SELECT extname FROM pg_extension WHERE extname='vector';"`
- `python ingest.py` finishes, logs case summaries upsert + ingestion complete.
- `streamlit run app.py` answers queries without DB errors.

## 8) Backup basics
```bash
docker compose exec pgvector pg_dump -U lds_user -d lds_db -Fc > lds_db.dump
```
Restore:
```bash
docker compose exec -T pgvector pg_restore -U lds_user -d lds_db < lds_db.dump
```

## 9) Defaults the code relies on
- Collection name: `langchain`
- Chunking: 2000/400 in `ingest.py` → `rag.py`
- Embeddings: defaults to Ollama `nomic-embed-text:latest` (or OpenAI if chosen at ingest)
- Env sourcing: `.env` via `python-dotenv`

With the above, running `process.py`, then `ingest.py`, then `streamlit run app.py` will use the Dockerized PGVector instance. Adjust credentials/ports if you change the compose file.
