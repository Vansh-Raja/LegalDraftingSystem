# Legal Drafting System – Project Overview

A retrieval-augmented generation (RAG) system for Indian legal judgments. It ingests court PDFs, extracts structured metadata with LLMs, stores embeddings in PGVector, and serves answers/drafts through a Streamlit UI (plus CLI/debug tools).

## What It Does
- End-to-end pipeline: PDF/DOCX/TXT → cleaned text → LLM metadata → chunking → embeddings → PGVector.
- Hybrid retrieval: vector search plus Postgres full-text summaries; fusion and diversity guards to avoid overfitting on one case.
- LLM-driven query planner: classifies queries (new/follow-up/general-law/chat), rewrites for retrieval, sets top‑K/min-docs, and chooses bridging strategies.
- Filtration and context assembly: LLM selects full cases and chunks with reasoning, assembles context within model limits, and falls back to full-case windows when needed.
- Petition drafting mode: petition-specific filtration plus structured respondent submission drafting with strict “no invented facts” rules and placeholders for missing data.

## Core Components
- `functions.py`: PDF/text extraction, cleaning, stable manifest numbering for judgments.
- `ai.py`: Metadata extraction via OpenAI, OpenRouter rotation, or local Ollama; template/duplicate guards; retries and rate-limit handling.
- `rag.py`: Chunking (2000 chars, 400 overlap with metadata prefixes), embeddings (`nomic-embed-text` via Ollama), PGVector operations, Postgres FTS summaries, fusion/diversity, filtration planner, context assembler.
- `orchestrator.py`: Query planner (QueryPlan) using LLM + heuristics for type/breadth/top‑K/min-docs/statutes/bridging; follow-up and general-law handling.
- `app.py`: Streamlit UI with chat, drafting, and debug tabs; model/runtime controls; session state; streaming answers.
- `petition_rag.py`: Petition-aware filtration and drafting prompts.
- `process.py`: Interactive ETL (choose year folders, run text + metadata extraction).
- `ingest.py`: Idempotent ingestion of chunked cases with metadata-only gating and summary upserts.
- `debug.py` / `st_debug.py`: DB maintenance, duplicate checks, on-page debug console.
- `archive/chat.py`: CLI chat fallback.

## Data & Model Flow
1) PDFs in `judgements/` → text files with stable IDs (`processed_data/txt_data` + manifest).  
2) LLM metadata → JSON per case (`processed_data/metadata`), with duplicate checks.  
3) Chunking with metadata prefixes → embeddings via Ollama → PGVector (JSONB metadata).  
4) Postgres summary index for fast FTS over titles/summaries/statutes/years.  
5) Query path: classify/plan → retrieve (vector + optional FTS) → LLM filtration → context assembly (budget-aware, full-case guard) → streamed answer via selected LLM.  
6) Petition path: upload petition + fighting points → petition filtration → draft structured respondent submissions with references/anchors.

## Techniques to Highlight
- Multi-stage RAG: vector + FTS retrieval, LLM filtration, budgeted context assembly, full-document fallback.
- Query-intent routing: LLM + heuristic planner for new/follow-up/general-law/chat; automatic top‑K/min-doc sizing and statute filters.
- Robust ingestion: manifest-stable numbering, template leakage guards, duplicate detection, idempotent embeddings, rate-limit handling.
- Hybrid model support: OpenAI, OpenRouter, local Ollama for both chat and metadata; configurable at runtime.
- Petition-specific prompting: rebuttal-focused filtration and drafting with citation anchors and placeholder enforcement.

## Tech Stack
- Python, LangChain, Streamlit.
- PostgreSQL + PGVector; Postgres FTS for summaries.
- Models: `nomic-embed-text` (Ollama) for embeddings; chat/filtration via OpenAI `gpt-5-nano-2025-08-07`, OpenRouter variants, or Ollama `qwen3:latest`.
- Utilities: pdfminer, python-docx, tqdm, dotenv.

## How to Run (developer flow)
- `python process.py` → convert PDFs and extract metadata (choose backend).
- `python ingest.py` → chunk, embed, and ingest (metadata-gated) + update summary index.
- `streamlit run app.py` → web UI for chat/drafting/debug; or `python archive/chat.py` for CLI.
- `python debug.py` → inspect/clear collection, counts, duplicate checks.

## Resume Snippet
“Built a legal RAG system that ingests Indian judgments (PDF→text→LLM metadata→PGVector), performs hybrid retrieval with LLM-driven filtration and context assembly, and serves a Streamlit chat plus petition-drafting UI backed by OpenAI/OpenRouter/Ollama models and Postgres/PGVector.”

