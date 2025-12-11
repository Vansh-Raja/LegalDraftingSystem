# Legal Drafting System

A retrieval-augmented generation (RAG) assistant for Indian legal judgments. It converts PDFs to cleaned text and LLM metadata, stores embeddings in PGVector, and serves answers and petition drafts via Streamlit (plus a CLI fallback).

## 🏗️ Architecture Overview

```
Sources (judgements/) → PDF→Text (functions.py, manifest) → LLM Metadata (ai.py)
                     → Chunks + Embeddings (rag.py) → PGVector + Postgres FTS summaries

User Query → Planner (orchestrator.py) → Retrieval (vector + FTS) → LLM Filtration
          → Context Assembler (full-doc guard) → Chat/Draft (app.py / archive/chat.py)
```

The system follows a modular architecture with clear separation of concerns:

### Core Components

1. **PDF Processing** (`functions.py`)
   - Converts PDF legal documents to text
   - Maintains stable file numbering with manifest system
   - Handles both "judgements" and "judgments" directory names

2. **Metadata Extraction** (`ai.py`)
   - Uses LLMs to extract structured metadata from legal text
   - Supports multiple backends: OpenAI, OpenRouter, Ollama
   - Extracts case numbers, parties, court names, dates, summaries, and legal provisions

3. **RAG Operations** (`rag.py`)
   - Document chunking and embedding generation
   - PGVector database integration for similarity search
   - Intelligent filtration and context assembly
   - Named case detection and prioritization

4. **Query Processing** (`orchestrator.py`)
   - Classifies queries (general law, new, follow-up)
   - Creates execution plans with bridging strategies
   - Handles context reuse and statute-based refills

5. **User Interfaces**
   - **Web Interface** (`app.py`): Streamlit-based chat with debug tools
   - **CLI Interface** (`chat.py`): Command-line chat with memory
   - **Debug Tools** (`debug.py`, `st_debug.py`): Database management and logging

## 🔄 System Flow

**Offline pipeline:** PDF → text (`functions.py`, manifest-stable IDs) → LLM metadata (`ai.py`) → chunking (2000/400) → embeddings (`rag.py` using Ollama `nomic-embed-text`) → PGVector + Postgres FTS summaries.  
**Online path:** Query → planner (`orchestrator.py`) → retrieval (vector + FTS with filters) → LLM filtration → context assembly (full-doc guard) → streamed answer/draft (`app.py`).

## 🛠️ Technology Stack

- Python 3.8+, LangChain, Streamlit  
- PostgreSQL + PGVector; Postgres FTS for summaries  
- Models: Ollama (`nomic-embed-text`, `qwen3`), OpenAI `gpt-5-nano-2025-08-07`, OpenRouter variants  
- Utilities: pdfminer, python-docx, tqdm, dotenv

## 📁 Project Structure

```
LegalDraftingSystem_VG/
├── app.py          # Streamlit web (chat + drafting + debug)
├── ai.py           # LLM metadata extraction
├── rag.py          # Chunking, embeddings, retrieval, filtration
├── functions.py    # PDF/text processing, manifest
├── orchestrator.py # Query planner/classifier
├── ingest.py       # Vector + FTS ingestion
├── process.py      # Interactive PDF→text→metadata
├── petition_rag.py # Petition-specific filtration/drafting
├── archive/chat.py # CLI fallback
├── debug.py        # DB debug utilities
├── st_debug.py     # Streamlit debug panel
├── judgements/     # Input PDFs
└── processed_data/
    ├── txt_data/   # Cleaned text (manifest-stable numbering)
    └── metadata/   # LLM metadata JSONs
```

## 🚀 Quick Start

**Prerequisites:** PostgreSQL with PGVector; Python 3.8+; OpenAI/OpenRouter key or Ollama (optional/local).  

**Install**
```bash
pip install -r requirements.txt
cp .env.example .env
```

**Environment variables (.env)**
```bash
# Database (pick one style)
PGVECTOR_CONNECTION=postgresql://user:password@localhost:5432/legaldraftingsystemdb
# or granular
DB_NAME=legaldraftingsystemdb
DB_USER=your_user
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432

# LLM providers (at least one)
OPENAI_KEY=...
OPENROUTER_API_KEY=...

# Optional local models
OLLAMA_HOST=http://localhost:11434
```

**Run (full pipeline)**
```bash
# 1) Process PDFs -> text + metadata (interactive backend choice)
python process.py

# 2) Ingest chunks into PGVector + FTS summaries
python ingest.py

# 3) Launch UI (chat + petition drafting + debug)
streamlit run app.py

# Optional: CLI fallback
python archive/chat.py
```

## 🎯 Highlights
- Query planner: new/follow-up/general-law/chat with auto top-K/min-doc sizing.
- Hybrid retrieval: PGVector + Postgres FTS; RRF fusion, diversity caps.
- LLM filtration & context assembly with full-document fallback when needed.
- Petition drafting mode with isolated prompts and fighting-point guidance.
- Multiple LLM backends (OpenAI/OpenRouter/Ollama) with fallbacks and cost control.
- Debug tooling: Streamlit debug pane, DB utilities, duplicate checks.

## ⭐ Key Features
- Intelligent query classification and context bridging.
- Statute filtering and named-case prioritization.
- Multi-stage filtration and context budgeting.
- Full-document loading when many chunks cluster on one case.
- Real-time streaming responses with session history.

## 🧭 Operational Notes
- Rate limits: falls back to Ollama if OpenAI/OpenRouter are missing/limited.
- Ingestion is idempotent; manifest keeps numbering stable for `txt_data`/`metadata`.
- Debug utilities: `debug.py` (PGVector ops, duplicates), `st_debug.py` (in-app debug).

## 🐛 Debugging Issues
- No embeddings found: run `python ingest.py` after processing.
- API rate limits: switch to Ollama or wait/reset.
- Database connection errors: verify PostgreSQL + PGVector and env vars.
- Empty responses: check Streamlit debug logs.

## 🎮 Demo Branch

A `demo` branch is available with the following differences for hosted deployment:

- **Ollama disabled**: Local models removed from the UI; requires OpenAI or OpenRouter API keys.
- **OpenAI embeddings**: Set `DEMO_MODE=1` to use OpenAI embeddings instead of Ollama.
- **2025 corpus only**: Multi-year support exists but only 2025 Supreme Court judgments are included to keep the repository size manageable.
- **Git LFS**: Large PDF files stored via Git LFS for easy cloning.

**Setup for demo branch:**
```bash
git checkout demo
git lfs pull              # Download PDFs via LFS
pip install -r requirements.txt
# Set OPENAI_KEY and optionally OPENROUTER_API_KEY in .env
# Set DEMO_MODE=1 in .env for OpenAI embeddings
python process.py         # Process PDFs
python ingest.py          # Ingest to PGVector
streamlit run app.py      # Launch UI
```

