Improvements backlog

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


