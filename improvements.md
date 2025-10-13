Improvements backlog

- Idempotent ingestion
  - Generate deterministic chunk IDs (file_stem + chunk_index + content hash) and pass `ids=` to PGVector to avoid duplicates on re-runs.
  - Optional DB pre-check: fetch existing IDs for the collection and only add missing ones.
  - Optional manifest per source file (hash + chunk count) to skip unchanged files and reindex changed ones.

- Retrieval quality
  - Add full-case fallback: when top-K chunks are dominated by one `file_stem`, load the entire case (or all its chunks) and re-answer.
  - Add query rewriting/standalone question generation for follow-ups to improve recall.
  - Expose tunables (k, fetch_k, filters) per query; start strict (court filter) then relax iteratively.
  - Consider hybrid retrieval (BM25 + vector) or metadata-boosting fields for proper names.

- Prompting and answers
  - Use RAG-focused system prompt (case-focused vs statute-focused synthesis) with self-check and source listing.
  - Include short quotes (≤ 2 sentences) with citations (case name + chunk metadata).
  - Enforce “only use provided context; don’t invent”.

- Observability
  - Keep lightweight debug: show retrieved `file_stem`, preview, context length, streamed tokens.
  - Optional: add a verbose mode flag to toggle debug.

- Collections and consistency
  - Standardize a single collection name (env-configurable). Migrate old unnamed data if needed.
  - Add ingestion stats and verification SQL (counts per collection, sample previews).

- Embeddings / models
  - Use robust local embedding model (e.g., `nomic-embed-text`) and ensure it’s pulled.
  - Allow model selection for both embeddings and chat via env.

- Performance / UX
  - Progress bars for PDF→TXT, metadata extraction, and ingestion (done).
  - Streaming responses in chat (done).
  - Batch sizes configurable; backoff on rate limits.


