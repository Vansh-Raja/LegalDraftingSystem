"""
RAG utilities: loading cases, chunking, embeddings, retrieval, filtration, and
context assembly. This module contains the building blocks that both the CLI
and Streamlit app use for Retrieval-Augmented Generation over legal judgments.
"""

from pathlib import Path
import json
import os
from typing import List, Optional, Tuple, Dict, Any

from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from models import get_embeddings


# Load environment variables (e.g., OLLAMA_HOST, DB_* values) once at import time
load_dotenv()


# Default collection name for PGVector
COLLECTION_NAME = "langchain"  # Default PGVector collection name


def load_case_docs(txt_dir: str = "processed_data/txt_data", meta_dir: str = "processed_data/metadata", only_with_metadata: bool = False) -> List[Document]:
    """
    Load full-text case files and merge available JSON metadata per file.

    - Each text file N.txt will try to load N.json metadata and attach it.
    - Adds a `file_stem` to every `Document.metadata` for downstream grouping.

    Args:
        txt_dir: Directory containing case text files.
        meta_dir: Directory containing case metadata JSON files.
        only_with_metadata: If True, skip cases missing JSON metadata.

    Returns:
        List of LangChain `Document` with merged metadata.
    """
    docs: List[Document] = []
    txt_path = Path(txt_dir)
    meta_path = Path(meta_dir)
    if not txt_path.exists():
        return docs
    for txt in sorted(txt_path.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        loader = TextLoader(str(txt), encoding="utf-8")
        loaded = loader.load()
        meta = {}
        meta_file = meta_path / f"{txt.stem}.json"
        if only_with_metadata and not meta_file.exists():
            continue
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        for d in loaded:
            d.metadata.update(meta)
            d.metadata["file_stem"] = txt.stem
        docs.extend(loaded)
    return docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 2000,
    chunk_overlap: int = 400,
    add_chunk_index: bool = True,
) -> List[Document]:
    """
    Split documents into overlapping chunks and lightly enrich content for recall.

    - Uses a recursive splitter with newline/space boundaries.
    - Optionally assigns a running `chunk_index` across all chunks.
    - Prepends compact metadata hints (case number, parties, court, date, statutes)
      to each chunk's text to help retrieval.

    Args:
        documents: Base documents to split.
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlap between adjacent chunks.
        add_chunk_index: Whether to tag chunks with an index.

    Returns:
        List of chunked `Document` objects.
    """
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    # Use create_documents to preserve metadata alignment exactly like the example
    contents = [d.page_content for d in documents]
    metadatas = [d.metadata for d in documents]
    chunks: List[Document] = splitter.create_documents(contents, metadatas)
    if add_chunk_index:
        for idx, doc in enumerate(chunks):
            doc.metadata["chunk_index"] = idx
    # Lightweight augmentation: include key metadata terms in content to boost recall
    for doc in chunks:
        m = doc.metadata or {}
        parts = []
        if m.get("case_number"):
            parts.append(f"case_number: {m.get('case_number')}")
        parties = m.get("parties") or {}
        if parties.get("petitioner") or parties.get("respondent"):
            parts.append(
                f"parties: {parties.get('petitioner', '')} vs {parties.get('respondent', '')}"
            )
        if m.get("court_name"):
            parts.append(f"court_name: {m.get('court_name')}")
        if m.get("date_of_judgment"):
            parts.append(f"date_of_judgment: {m.get('date_of_judgment')}")
        if m.get("legal_provisions_cited"):
            parts.append(
                "legal_provisions_cited: "
                + ", ".join(m.get("legal_provisions_cited", []))
            )
        if parts:
            prefix = " | ".join(parts)
            doc.page_content = f"{prefix}\n\n" + (doc.page_content or "")
    return chunks


def load_and_chunk_cases(
    txt_dir: str = "processed_data/txt_data",
    meta_dir: str = "processed_data/metadata",
    chunk_size: int = 2000,
    chunk_overlap: int = 400,
    only_with_metadata: bool = False,
) -> List[Document]:
    """
    Convenience helper: load cases then split into chunks with given params.
    """
    base_docs = load_case_docs(txt_dir=txt_dir, meta_dir=meta_dir, only_with_metadata=only_with_metadata)
    return chunk_documents(base_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _get_pg_connection_string() -> str:
    """
    Build a Postgres connection string from environment variables with fallbacks.
    Prefers DB_* variables, then common single-URL envs, then a local default.
    """
    load_dotenv()
    # Prefer explicit DB_* variables; else fall back to common single-URL envs, then default
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    if db_name and db_user and db_password and db_host and db_port:
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return (
        os.getenv("PGVECTOR_CONNECTION")
        or os.getenv("POSTGRES_URL")
        or os.getenv("DATABASE_URL")
        or "postgresql://localhost:5432/legaldraftingsystemdb"
    )


def ensure_case_summaries_schema(connection_string: str | None = None) -> None:
    """
    Ensure the Postgres FTS-backed case_summaries table and indexes exist.
    Schema stores one row per case (file_stem) with a generated tsvector.
    """
    conn = connection_string or _get_pg_connection_string()
    with psycopg.connect(conn) as cx:
        with cx.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS case_summaries (
                    file_stem TEXT PRIMARY KEY,
                    title TEXT,
                    summary TEXT,
                    final_judgment TEXT,
                    court_name TEXT,
                    year INT,
                    statutes TEXT[],
                    ts tsvector
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS case_summaries_ts_idx
                ON case_summaries USING GIN (ts);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS case_summaries_filters_idx
                ON case_summaries (court_name, year);
                """
            )
        cx.commit()


def _derive_title_from_parties(meta: dict) -> str:
    parties = (meta or {}).get("parties") or {}
    pet = parties.get("petitioner") or parties.get("appellant") or ""
    resp = parties.get("respondent") or parties.get("respondents") or ""
    if pet or resp:
        return f"{pet} vs {resp}".strip()
    return (meta or {}).get("case_number") or ""


def _derive_year(meta: dict) -> Optional[int]:
    date_str = (meta or {}).get("date_of_judgment") or ""
    if isinstance(date_str, str) and len(date_str) >= 4 and date_str[:4].isdigit():
        try:
            return int(date_str[:4])
        except Exception:
            return None
    return None


def upsert_all_case_summaries_from_metadata(meta_dir: str = "processed_data/metadata", connection_string: str | None = None) -> int:
    """
    Read metadata JSONs and upsert one summary row per case into case_summaries.
    Returns the number of rows upserted.
    """
    ensure_case_summaries_schema(connection_string)
    conn = connection_string or _get_pg_connection_string()
    count = 0
    with psycopg.connect(conn) as cx:
        with cx.cursor() as cur:
            for p in sorted(Path(meta_dir).glob("*.json"), key=lambda q: int(q.stem) if q.stem.isdigit() else q.stem):
                try:
                    meta = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
                file_stem = p.stem
                title = _derive_title_from_parties(meta)
                summary = (meta or {}).get("summary") or ""
                final_judgment = (meta or {}).get("final_judgment") or ""
                court_name = (meta or {}).get("court_name") or None
                year = _derive_year(meta)
                statutes_list = (meta or {}).get("legal_provisions_cited") or []
                if statutes_list and not isinstance(statutes_list, list):
                    statutes_list = [str(statutes_list)]
                try:
                    cur.execute(
                        """
                        INSERT INTO case_summaries
                            (file_stem, title, summary, final_judgment, court_name, year, statutes, ts)
                        VALUES (%s, %s, %s, %s, %s, %s, %s,
                            to_tsvector('english',
                                coalesce(%s,'') || ' ' || coalesce(%s,'') || ' ' || coalesce(%s,'') || ' ' ||
                                array_to_string(coalesce(%s, ARRAY[]::text[]), ' ')
                            )
                        )
                        ON CONFLICT (file_stem) DO UPDATE SET
                            title = EXCLUDED.title,
                            summary = EXCLUDED.summary,
                            final_judgment = EXCLUDED.final_judgment,
                            court_name = EXCLUDED.court_name,
                            year = EXCLUDED.year,
                            statutes = EXCLUDED.statutes,
                            ts = EXCLUDED.ts
                        """,
                        (
                            file_stem, title, summary, final_judgment, court_name, year, statutes_list,
                            title, summary, final_judgment, statutes_list,
                        ),
                    )
                    count += 1
                except Exception:
                    # Skip problematic rows but continue overall
                    continue
        cx.commit()
    return count


def summary_search_pg(
    user_q: str,
    court_name: Optional[str] = None,
    statutes: Optional[List[str]] = None,
    year: Optional[int] = None,
    limit: int = 50,
    connection_string: str | None = None,
) -> List[Tuple[str, float]]:
    """
    Run Postgres FTS over case summaries/titles/final results and return
    (file_stem, rank) pairs ordered by rank desc.
    Applies optional filters for court_name, year, and statutes.
    """
    ensure_case_summaries_schema(connection_string)
    conn = connection_string or _get_pg_connection_string()
    where_clauses = ["ts @@ websearch_to_tsquery('english', %s)"]
    # First param will be used by ts_rank (SELECT), second by WHERE ts @@ ...
    params: List[Any] = [user_q, user_q]
    if court_name:
        where_clauses.append("court_name = %s")
        params.append(court_name)
    if isinstance(year, int):
        where_clauses.append("year = %s")
        params.append(year)
    if statutes:
        # overlap with any of the provided statutes; cast to text[] for safety
        where_clauses.append("statutes && %s::text[]")
        params.append(statutes)
    sql = (
        "SELECT file_stem, ts_rank(ts, websearch_to_tsquery('english', %s)) AS rank "
        "FROM case_summaries "
        f"WHERE {' AND '.join(where_clauses)} "
        "ORDER BY rank DESC "
        "LIMIT %s"
    )
    params2 = params + [limit]
    results: List[Tuple[str, float]] = []
    with psycopg.connect(conn, row_factory=dict_row) as cx:
        with cx.cursor() as cur:
            cur.execute(sql, params2)
            for row in cur.fetchall():
                fs = row.get("file_stem")
                rk = float(row.get("rank") or 0.0)
                if fs:
                    results.append((fs, rk))
    return results


def fuse_cases_by_rrf(
    summary_ranked: List[Tuple[str, float]],
    chunk_docs: List[Document],
    k: int = 60,
    top_n: int = 30,
) -> List[str]:
    """
    Reciprocal Rank Fusion at case-level.
    - summary_ranked: list of (file_stem, rank_score_desc)
    - chunk_docs: vector results at chunk level; we collapse to best rank per file_stem
    Returns a list of fused file_stem ordered by descending RRF score.
    """
    # Build rank positions (1-based) for summaries
    s_rankpos: Dict[str, int] = {}
    for idx, (fs, _rk) in enumerate(summary_ranked):
        if fs not in s_rankpos:
            s_rankpos[fs] = idx + 1
    # Build rank positions (1-based) for chunks collapsed to cases
    c_rankpos: Dict[str, int] = {}
    seen: Dict[str, int] = {}
    for idx, d in enumerate(chunk_docs):
        fs = (d.metadata or {}).get("file_stem")
        if not fs:
            continue
        pos = idx + 1
        prev = seen.get(fs)
        if prev is None or pos < prev:
            seen[fs] = pos
    c_rankpos = seen
    # Union of cases
    all_cases = set(s_rankpos) | set(c_rankpos)
    scored: List[Tuple[str, float]] = []
    for fs in all_cases:
        s_pos = s_rankpos.get(fs)
        c_pos = c_rankpos.get(fs)
        score = 0.0
        if s_pos is not None:
            score += 1.0 / (k + s_pos)
        if c_pos is not None:
            score += 1.0 / (k + c_pos)
        scored.append((fs, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [fs for fs, _ in scored[:top_n]]


def fetch_top_chunks_for_cases(
    vectorstore: PGVector,
    user_q: str,
    file_stems: List[str],
    per_case_k: int = 6,
) -> List[Document]:
    """
    For each case in file_stems order, fetch up to per_case_k chunks via the
    vectorstore constrained by file_stem.
    """
    docs: List[Document] = []
    for fs in file_stems:
        try:
            batch = vectorstore.similarity_search(user_q, k=per_case_k, filter={"file_stem": fs})
            docs.extend(batch)
        except Exception:
            continue
    return docs


def interleave_docs_by_case(
    docs: List[Document],
    per_case_limit: int = 2,
    max_total: Optional[int] = None,
) -> List[Document]:
    """
    Round-robin interleave documents across cases (by file_stem) to improve
    diversity shown to downstream steps and debugging. Limits per-case and total.
    """
    if not docs:
        return []
    by_case: Dict[str, List[Document]] = {}
    for d in docs:
        fs = (d.metadata or {}).get("file_stem")
        if not fs:
            fs = "__unknown__"
        lst = by_case.get(fs)
        if lst is None:
            lst = []
            by_case[fs] = lst
        if per_case_limit <= 0 or len(lst) < per_case_limit:
            lst.append(d)
    # Round-robin emit
    queues = list(by_case.values())
    out: List[Document] = []
    idx = 0
    while queues:
        i = idx % len(queues)
        bucket = queues[i]
        if bucket:
            out.append(bucket.pop(0))
            if max_total is not None and len(out) >= max_total:
                break
        if not bucket:
            queues.pop(i)
            # do not increment idx here to not skip next bucket
        else:
            idx += 1
    return out

def ingest_chunks_to_pgvector(
    chunks: List[Document],
    connection_string: str | None = None,
    embedding_model: str = "nomic-embed-text:latest",
    embedding_provider: str | None = None,
    use_jsonb: bool = True,
    create_extension: bool = False,
) -> PGVector:
    """
    Ingest a list of chunk documents into PGVector in one call.

    Filters out empty chunks; embeds with an Ollama embeddings model; creates or
    uses the given collection.
    """
    if not chunks:
        raise ValueError("No chunks provided for ingestion")
    conn = connection_string or _get_pg_connection_string()
    # Filter out empty/whitespace-only chunks to avoid embedding errors
    non_empty_chunks = [d for d in chunks if (d.page_content or "").strip()]
    embeddings = get_embeddings(embedding_model, provider=embedding_provider)
    vectorstore = PGVector.from_documents(
        non_empty_chunks,
        embedding=embeddings,
        connection=conn,
        use_jsonb=use_jsonb,
        create_extension=create_extension,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def ingest_chunks_to_pgvector_batched(
    chunks: List[Document],
    connection_string: str | None = None,
    embedding_model: str = "nomic-embed-text:latest",
    embedding_provider: str | None = None,
    use_jsonb: bool = True,
    create_extension: bool = False,
    batch_size: int = 128,
):
    """
    Ingest in batches: first call creates the collection, subsequent calls add.

    Useful for very large corpora to keep memory usage bounded.
    """
    from tqdm import tqdm  # local import to avoid hard dep when unused

    if not chunks:
        raise ValueError("No chunks provided for ingestion")
    conn = connection_string or _get_pg_connection_string()
    # Filter out empty/whitespace-only chunks
    non_empty = [d for d in chunks if (d.page_content or "").strip()]
    embeddings = get_embeddings(embedding_model, provider=embedding_provider)

    def _batches(items: List[Document]):
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    total = len(non_empty)
    if total == 0:
        return None

    first = True
    vs = None
    with tqdm(total=total, desc="Ingesting chunks", unit="chunk") as pbar:
        for batch in _batches(non_empty):
            if first:
                vs = PGVector.from_documents(
                    batch,
                    embedding=embeddings,
                    connection=conn,
                    use_jsonb=use_jsonb,
                    create_extension=create_extension,
                    collection_name=COLLECTION_NAME,
                )
                first = False
                pbar.update(len(batch))
            else:
                vs.add_documents(batch)
                pbar.update(len(batch))
    return vs


def get_vectorstore(
    connection_string: str | None = None,
    embedding_model: str | None = None,
    embedding_provider: str | None = None,
    use_jsonb: bool = True,
    collection_name: str | None = None,
) -> PGVector:
    """
    Return a PGVector handle bound to an embeddings model.

    Use when an index/collection already exists, or to add/query documents.
    """
    conn = connection_string or _get_pg_connection_string()
    # Choose default embedding model based on provider if not specified
    if embedding_model is None:
        if embedding_provider in {"openai", "openrouter", "groq", "ollama_cloud"}:
            embed_model = None  # let get_embeddings pick API default
        else:
            embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
    else:
        embed_model = embedding_model

    embeddings = get_embeddings(embed_model, provider=embedding_provider)
    # When collection_name is None, default internal collection is used
    return PGVector(
        connection=conn,
        embeddings=embeddings,
        use_jsonb=use_jsonb,
        collection_name=collection_name or COLLECTION_NAME,
    )


def build_retriever(vectorstore: PGVector, statute_filters: list[str] | None = None, court_name: str | None = "Supreme Court of India", k: int = 6):
    """
    Create a retriever with optional JSONB metadata filters and top-K control.
    """
    filter_dict: dict = {}
    if statute_filters:
        filter_dict["legal_provisions_cited"] = {"$in": statute_filters}
    if court_name:
        filter_dict["court_name"] = court_name
    # Stricter retriever (original behavior)
    return vectorstore.as_retriever(search_kwargs={"k": k, "filter": filter_dict})


def build_qa_chain(
    retriever,
    llm_model: str = "qwen3:latest",
    temperature: float = 0.0,
    return_sources: bool = True,
):
    """
    Build a simple RetrievalQA chain around an Ollama chat model and retriever.
    """
    llm = ChatOllama(model=llm_model, temperature=temperature)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=return_sources,
    )


def ask(question: str, qa_chain) -> str:
    """Helper to execute a retrieval QA chain with a question."""
    return qa_chain.run(question)


def maybe_full_case(
    chosen_docs: List[Document],
    txt_dir: str = "processed_data/txt_data",
    threshold: int = 4,
):
    """
    If many retrieved chunks belong to the same case, suggest loading that full
    case. Returns a Path to the full file if it exists, else None.
    """
    if not chosen_docs:
        return None
    freq: dict[str, int] = {}
    for d in chosen_docs:
        fs = d.metadata.get("file_stem")
        if fs:
            freq[fs] = freq.get(fs, 0) + 1
    if not freq:
        return None
    fs, cnt = max(freq.items(), key=lambda x: x[1])
    if cnt >= threshold:
        p = Path(txt_dir) / f"{fs}.txt"
        return p if p.exists() else None
    return None


# ---------------------- Filtering LLM: Planner & Assembler ----------------------

class ChunkRef(BaseModel):
    """Reference to a specific chunk in a case file (by stem and chunk index)."""
    file_stem: str
    chunk_index: int


class ReasonFullDoc(BaseModel):
    """Why a full document should be included in the assembled context."""
    file_stem: str
    reason: str


class ReasonChunk(BaseModel):
    """Why a particular chunk is relevant to the user question."""
    file_stem: str
    chunk_index: int
    reason: str


class FiltrationPlan(BaseModel):
    """
    Output schema from filtration LLM: what full docs and chunks to include,
    estimated budget, and per-item reasoning (optional but preferred).
    """
    selected_full_docs: List[str] = []
    selected_chunks: List[ChunkRef] = []
    drop_chunks: List[ChunkRef] = []
    # Unused; assembler uses model context window. Kept for backward compatibility.
    context_budget_tokens: int = 0
    reasoning_full_docs: List[ReasonFullDoc] = []
    reasoning_chunks: List[ReasonChunk] = []
    overall_reasoning: Optional[str] = None
    notes: Optional[str] = None
    request_more_cases: bool = False
    expansion_reason: Optional[str] = None


def _build_chunk_previews(docs: List[Document], max_chars: int = 800) -> List[dict]:
    """Summarize docs into compact previews for the filtration LLM."""
    previews = []
    for d in docs:
        m = d.metadata or {}
        previews.append({
            "file_stem": m.get("file_stem"),
            "chunk_index": m.get("chunk_index", -1),
            "court_name": m.get("court_name"),
            "case_number": m.get("case_number"),
            "date_of_judgment": m.get("date_of_judgment"),
            "legal_provisions_cited": m.get("legal_provisions_cited", []),
            "preview": (d.page_content or "").strip().replace("\n", " ")[:max_chars],
        })
    return previews


def filtration_retriever(
    user_q: str,
    docs: List[Document],
    mode: str = "chunk",
    case_metadatas: Optional[List[dict]] = None,
    desired_min_full_docs: int = 3,
    desired_max_chunks_per_case: int = 2,
    query_context: Optional[dict] = None,
) -> FiltrationPlan:
    """
    Select most relevant documents/chunks using an LLM planning step.

    mode:
      - "chunk": pass chunk previews (and optionally case metadata) to the LLM
      - "metadata": pass only case metadata, no chunk previews
    query_context:
      - Optional dict with planner hints (breadth, fused counts, etc.)
    """
    if not docs:
        return FiltrationPlan(selected_full_docs=[], selected_chunks=[], drop_chunks=[], context_budget_tokens=6000)

    load_dotenv()
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        # No OpenAI key; return empty plan to trigger fallback upstream
        return FiltrationPlan(selected_full_docs=[], selected_chunks=[], drop_chunks=[], context_budget_tokens=6000)

    previews = _build_chunk_previews(docs) if mode == "chunk" else []
    system_msg = (
        "You are a legal expert filtration retriever. You will receive a user question, chunk previews, and case-level metadata from a legal RAG pipeline. "
        "Your job is to (1) select the most relevant chunks, (2) request full-document retrieval for cases that must be loaded in full, and (3) signal when additional fused cases should be fetched.\n\n"
        "Context hints:\n"
        "- `query_context` contains planner guidance (breadth classification, fused-case count, retrieval_k).\n"
        "- Each entry in `cases` includes the fused rank, metadata summary, final judgment, statutes, parties, and court. Read summaries before discarding a case.\n\n"
        "Strict rules:\n"
        "- Prefer cases whose previews or metadata contain specific mentions from the question (parties, case numbers, courts, statutes, time frames).\n"
        "- If the user names a specific case, ensure that case is selected (full doc if necessary) and keep focus tight.\n"
        "- You may request ANY number of full documents, but include only those needed to answer thoroughly.\n"
        "- Remove irrelevant chunks; select concise spans that best support the answer.\n"
        "- Output STRICT JSON matching the provided schema (no prose outside JSON fields).\n"
        "- Provide reasoning for every selected_full_docs and selected_chunks entry, and set overall_reasoning with a 1–2 sentence summary.\n"
        "- When metadata summaries or final judgments indicate relevance, use them as justification to retain the case even if the preview snippet looks weak.\n"
        "- Set `request_more_cases` to true only when the fused list still has clearly relevant cases that should be fetched; include a short `expansion_reason`. Otherwise leave it false.\n"
        "- Do NOT use knowledge beyond the provided previews, metadata, and hints.\n\n"
        "Breadth-aware coverage directives:\n"
        "- If `query_context.breadth` is \"broad\", assemble a diverse set of cases (aim ≥ desired_min_full_docs, often 5–8) covering the requested time span/statutes. Consider mid-ranked fused cases that add new angles.\n"
        "- If breadth is \"narrow\", choose the strongest 2–4 cases covering the requested statute/topic; you may expand if summaries show distinct fact patterns needed for comparison.\n"
        "- If breadth is \"specific\", focus on the named case (plus closely related ones only if they are essential for contrast or procedural history).\n"
        "- Use fused ranks and metadata summaries to justify selections; avoid dropping higher-ranked cases without a clear reason.\n"
        "- Balance chunk picks across selected cases; favor overview/headnote chunks before deep procedural detail unless the query demands it."
    )
    effective_qc = dict(query_context or {})
    if "breadth" not in effective_qc:
        effective_qc["breadth"] = "unknown"
    effective_qc.setdefault("fused_case_count", len(case_metadatas or []))
    effective_qc.setdefault("retrieval_k", len(docs))

    user_payload = {
        "question": user_q,
        "mode": mode,
        "chunks": previews,
        "cases": (case_metadatas or []),
        "query_context": effective_qc,
        "preferences": {
            "desired_min_full_docs": max(1, int(desired_min_full_docs or 1)),
            "desired_max_chunks_per_case": max(1, int(desired_max_chunks_per_case or 1)),
        },
        "schema": {
            "selected_full_docs": ["<file_stem>"],
            "selected_chunks": [{"file_stem": "<file_stem>", "chunk_index": 0}],
            "drop_chunks": [{"file_stem": "<file_stem>", "chunk_index": 0}],
            "reasoning_full_docs": [{"file_stem": "<file_stem>", "reason": "why this case is needed"}],
            "reasoning_chunks": [{"file_stem": "<file_stem>", "chunk_index": 0, "reason": "why this chunk"}],
            "overall_reasoning": "optional global rationale",
            "notes": "optional short note",
            "request_more_cases": False,
            "expansion_reason": "optional short reason when requesting more cases"
        }
    }

    try:
        llm = ChatOpenAI(model="gpt-5-nano-2025-08-07", temperature=0)
        structured = llm.with_structured_output(FiltrationPlan)
        plan: FiltrationPlan = structured.invoke([
            ("system", system_msg),
            ("user", json.dumps(user_payload)),
        ])
        # Ensure reasoning arrays are populated
        if not plan.reasoning_full_docs and plan.selected_full_docs:
            plan.reasoning_full_docs = [ReasonFullDoc(file_stem=fs, reason="matches query / necessary context") for fs in plan.selected_full_docs]
        if not plan.reasoning_chunks and plan.selected_chunks:
            plan.reasoning_chunks = [ReasonChunk(file_stem=c.file_stem, chunk_index=c.chunk_index, reason="directly relevant to question") for c in plan.selected_chunks]
        if not plan.overall_reasoning:
            plan.overall_reasoning = "Selected items optimize relevance to the question while controlling context size."
        return plan
    except Exception:
        # Default safe heuristic: dominant case + top 2 chunks
        from collections import Counter
        stems = [d.metadata.get("file_stem") for d in docs if d.metadata.get("file_stem")]
        sel_full: List[str] = []
        sel_chunks: List[ChunkRef] = []
        if stems:
            dominant, _ = Counter(stems).most_common(1)[0]
            sel_full = [dominant]
        for d in docs[:2]:
            fs = d.metadata.get("file_stem")
            ci = d.metadata.get("chunk_index", 0)
            if fs is not None:
                sel_chunks.append(ChunkRef(file_stem=fs, chunk_index=ci))
        return FiltrationPlan(
            selected_full_docs=sel_full,
            selected_chunks=sel_chunks,
            drop_chunks=[],
            context_budget_tokens=0,
            request_more_cases=False,
        )


def enforce_case_diversity(
    plan: FiltrationPlan,
    fused_cases: Optional[List[str]] = None,
    desired_min_full_docs: int = 3,
    desired_max_chunks_per_case: int = 2,
    breadth: str = "unknown",
    allow_expansion: bool = False,
) -> FiltrationPlan:
    """
    Deterministic guard: ensure a minimum number of distinct full docs and cap
    chunks per case to avoid domination by a single case.
    """
    fused_list = [fs for fs in (fused_cases or []) if fs]
    breadth_norm = (breadth or "unknown").lower()
    target_min = max(1, int(desired_min_full_docs or 1))
    if breadth_norm == "broad" and fused_list:
        broad_floor = max(5, int(len(fused_list) * 0.6))
        target_min = max(target_min, min(len(fused_list), min(broad_floor, 12)))
    elif breadth_norm == "narrow":
        target_min = max(target_min, 3)
    elif breadth_norm == "specific":
        target_min = max(1, min(target_min, 3))

    expansion_requested = allow_expansion or bool(getattr(plan, "request_more_cases", False))

    existing: Dict[str, bool] = {}

    def _top_up(limit: int) -> None:
        if not fused_list:
            return
        for fs in fused_list:
            if fs in existing:
                continue
            plan.selected_full_docs.append(fs)
            existing[fs] = True
            if len(plan.selected_full_docs) >= limit:
                break

    for fs in plan.selected_full_docs:
        if fs:
            existing[fs] = True

    if len(plan.selected_full_docs) < target_min:
        _top_up(target_min)

    if expansion_requested and breadth_norm == "broad":
        expanded_limit = min(len(fused_list), min(max(target_min, len(plan.selected_full_docs)) + 2, 12))
        if len(plan.selected_full_docs) < expanded_limit:
            _top_up(expanded_limit)
            extra = len(plan.selected_full_docs) - target_min
            if extra > 0:
                note_msg = plan.expansion_reason or "LLM requested broader coverage"
                addition = f"Expanded by {extra} case(s) due to request_more_cases ({note_msg})."
                plan.notes = f"{plan.notes}; {addition}" if plan.notes else addition

    # Deduplicate while preserving order
    if plan.selected_full_docs:
        seen_order = set()
        deduped = []
        for fs in plan.selected_full_docs:
            if fs and fs not in seen_order:
                deduped.append(fs)
                seen_order.add(fs)
        plan.selected_full_docs = deduped

    # Cap chunks per case
    if plan.selected_chunks:
        capped: List[ReasonChunk] = []
        per_case_count: Dict[str, int] = {}
        for c in plan.selected_chunks:
            cnt = per_case_count.get(c.file_stem, 0)
            if cnt < max(1, int(desired_max_chunks_per_case or 1)):
                capped.append(c)
                per_case_count[c.file_stem] = cnt + 1
        plan.selected_chunks = capped
    return plan


def _token_estimate_from_chars(chars: int) -> int:
    """Very rough token estimate: ~4 chars per token."""
    return max(1, chars // 4)


def _windows_around_terms(text: str, query: str, budget_tokens: int) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Extract multiple windows around occurrences of query terms, within budget.
    Falls back to head+tail if no term hits are found.
    """
    budget_chars = budget_tokens * 4
    raw_lc = text.lower()
    qtokens = [t for t in (query.lower().replace(" v. ", " vs "))
               .split() if t.isalpha() and len(t) > 2]
    positions = []
    for t in set(qtokens):
        start = 0
        hits = 0
        while True:
            pos = raw_lc.find(t, start)
            if pos == -1 or hits >= 5:
                break
            positions.append(pos)
            start = pos + max(1, len(t))
            hits += 1
    positions = sorted(set(positions))
    if not positions:
        # Head + tail fallback
        head = text[: budget_chars // 2]
        tail = text[- budget_chars // 2 :]
        used = head + "\n\n...\n\n" + tail
        return used, [(0, len(head)), (len(text) - len(tail), len(text))]

    win_half = max(2000, budget_chars // 8)
    windows: List[Tuple[int, int]] = []
    for p in positions:
        s = max(0, p - win_half)
        e = min(len(text), p + win_half)
        windows.append((s, e))

    # Merge windows
    merged: List[Tuple[int, int]] = []
    for s, e in sorted(windows):
        if not merged or s > merged[-1][1] + 50:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    parts = []
    total = 0
    spans: List[Tuple[int, int]] = []
    for s, e in merged:
        seg_len = e - s
        if total + seg_len > budget_chars:
            if budget_chars - total <= 0:
                break
            e = s + (budget_chars - total)
            seg_len = e - s
        parts.append(text[s:e])
        spans.append((s, e))
        total += seg_len
        if total >= budget_chars:
            break
    used = "\n\n...\n\n".join(parts)
    return used, spans


def assemble_context_from_plan(
    plan: FiltrationPlan,
    user_q: str,
    vectorstore: PGVector,
    txt_dir: str = "processed_data/txt_data",
    budget_tokens: Optional[int] = None,
    initial_docs: Optional[List[Document]] = None,
) -> Tuple[str, List[Document], Dict[str, Any]]:
    """
    Assemble a context string within the suggested budget using the plan.

    Strategy:
      1) Include full docs (whole or windowed) in priority order
      2) Add specific selected chunks if budget remains
      3) Return context, source docs (unused placeholder), and debug spans
    """
    budget = plan.context_budget_tokens if (budget_tokens is None) else budget_tokens
    used_docs: List[Document] = []
    assembled_parts: List[str] = []
    spans_debug: List[Tuple[int, int, str]] = []  # (start, end, file)
    included_full_docs: List[Dict[str, Any]] = []

    # 1) Full documents windows (include whole doc if it fits budget)
    for fs in plan.selected_full_docs:
        fp = Path(txt_dir) / f"{fs}.txt"
        if not fp.exists():
            included_full_docs.append({
                "file": f"{fs}.txt",
                "included": False,
                "mode": "missing_text_file",
                "chars_used": 0,
                "available_chars": 0,
            })
            continue
        raw = fp.read_text(encoding="utf-8")
        header = f"[file: {fs}.txt]\n"
        budget_chars = budget * 4
        current_chars = sum(len(p) for p in assembled_parts)
        remaining = max(0, budget_chars - current_chars)
        doc_info: Dict[str, Any] = {"file": f"{fs}.txt", "available_chars": len(raw)}
        if len(header) + len(raw) <= remaining:
            assembled_parts.append(header + raw)
            # record span as whole file
            spans_debug.append((0, len(raw), f"{fs}.txt"))
            doc_info.update({"included": True, "mode": "full", "chars_used": len(raw)})
        else:
            segment, spans = _windows_around_terms(raw, user_q, max(1, remaining // 4))
            if segment:
                assembled_parts.append(header + segment)
                for s, e in spans:
                    spans_debug.append((s, e, f"{fs}.txt"))
                doc_info.update({"included": True, "mode": "window", "chars_used": len(segment)})
            else:
                doc_info.update({"included": False, "mode": "skipped_no_budget", "chars_used": 0})
        if doc_info.get("included") or doc_info.get("mode") == "skipped_no_budget":
            included_full_docs.append(doc_info)

    # 2) Selected chunks: pull from initial_docs when available, else query vectorstore by file_stem and filter by chunk_index
    selected_map = {(c.file_stem, c.chunk_index) for c in plan.selected_chunks}
    selected_chunks_text: List[str] = []
    if selected_map:
        # try to include from initial_docs to preserve exact chunks without extra calls
        if initial_docs:
            for d in initial_docs:
                fs = (d.metadata or {}).get("file_stem")
                ci = (d.metadata or {}).get("chunk_index")
                if fs is not None and ci is not None and (fs, ci) in selected_map:
                    selected_chunks_text.append(d.page_content or "")
        # if still missing, try pulling via similarity search filtered by file_stem, then filter by chunk_index
        remaining = selected_map.copy()
        if initial_docs:
            for d in initial_docs:
                fs = (d.metadata or {}).get("file_stem")
                ci = (d.metadata or {}).get("chunk_index")
                if fs is not None and ci is not None and (fs, ci) in remaining:
                    remaining.discard((fs, ci))
        for (fs, ci) in list(remaining):
            try:
                # fetch a reasonably large set from that file and filter locally
                batch = vectorstore.similarity_search(user_q, k=50, filter={"file_stem": fs})
                for bd in batch:
                    if (bd.metadata or {}).get("chunk_index") == ci:
                        selected_chunks_text.append(bd.page_content or "")
                        break
            except Exception:
                continue
    # add selected chunks respecting remaining budget
    if selected_chunks_text:
        budget_chars = budget * 4
        current_chars = sum(len(p) for p in assembled_parts)
        for txt in selected_chunks_text:
            if current_chars + len(txt) + 2 > budget_chars:
                break
            # best effort: try to infer file name from chunk header if present upstream, else omit
            assembled_parts.append(txt)
            current_chars += len(txt) + 2

    # Build metadata overview (unique stems from full docs first, then chunk sources)
    metadata_stems: List[str] = []
    for fs in plan.selected_full_docs:
        if fs not in metadata_stems:
            metadata_stems.append(fs)
    for fs, _ci in selected_map:
        if fs not in metadata_stems:
            metadata_stems.append(fs)

    metadata_section_lines: List[str] = []
    metadata_debug: List[Dict[str, Any]] = []
    if metadata_stems:
        meta_dir = Path(txt_dir).parent / "metadata"
        for idx, fs in enumerate(metadata_stems, start=1):
            meta_path = meta_dir / f"{fs}.json"
            meta_payload: Dict[str, Any] = {}
            if meta_path.exists():
                try:
                    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta_payload = {}
            parties = meta_payload.get("parties") or {}
            petitioner = parties.get("petitioner") or ""
            respondent = parties.get("respondent") or ""
            case_number = meta_payload.get("case_number") or "Unknown case number"
            court_name = meta_payload.get("court_name") or "Unknown court"
            date_of_judgment = meta_payload.get("date_of_judgment") or "Unknown date"
            summary = meta_payload.get("summary") or ""
            if summary and len(summary) > 400:
                summary = summary[:397] + "..."
            parties_label = ""
            if petitioner or respondent:
                parties_label = f"{petitioner} vs {respondent}".strip()
            line_parts = [
                f"{idx}. file_stem={fs}",
                f"case={case_number}",
                f"court={court_name}",
                f"date={date_of_judgment}",
            ]
            if parties_label:
                line_parts.append(f"parties={parties_label}")
            if summary:
                line_parts.append(f"summary={summary}")
            metadata_section_lines.append(" | ".join(line_parts))
            metadata_debug.append({
                "file_stem": fs,
                "case_number": case_number,
                "court_name": court_name,
                "date_of_judgment": date_of_judgment,
                "has_summary": bool(meta_payload.get("summary")),
            })

    metadata_section = ""
    if metadata_section_lines:
        metadata_section = "Case Metadata Overview (read this first):\n" + "\n".join(metadata_section_lines)

    body_section = "\n\n---\n\n".join(assembled_parts) if assembled_parts else ""
    context_parts: List[str] = []
    if metadata_section:
        context_parts.append(metadata_section)
    if body_section:
        context_parts.append("Detailed Context Excerpts:\n" + body_section)
    context = "\n\n====\n\n".join(context_parts) if context_parts else ""
    est_tokens = _token_estimate_from_chars(len(context))
    debug: Dict[str, Any] = {
        "spans": spans_debug,
        "est_tokens": est_tokens,
        "selected_full_docs": plan.selected_full_docs,
        "selected_chunks_count": len(plan.selected_chunks),
        "included_full_docs": included_full_docs,
        "metadata_cases": metadata_debug,
    }
    return context, used_docs, debug


# ---------------------- Named-case guard helpers ----------------------

def _named_case_candidates(user_q: str, docs: List[Document]) -> List[str]:
    """
    Heuristic scoring of which case stems best match user query tokens.
    Returns stems ordered by descending match score.
    """
    q = user_q.lower().replace(" v. ", " vs ")
    toks = [t for t in q.split() if t.isalpha() and len(t) > 2]
    if not toks:
        return []
    scored: Dict[str, int] = {}
    for d in docs:
        fs = (d.metadata or {}).get("file_stem")
        if not fs:
            continue
        txt = ((d.page_content or "") + " " + json.dumps(d.metadata or {})).lower()
        hits = sum(1 for t in set(toks) if t in txt)
        if hits:
            scored[fs] = max(scored.get(fs, 0), hits)
    return [fs for fs, _ in sorted(scored.items(), key=lambda x: -x[1])]


def apply_named_case_guard(plan: FiltrationPlan, user_q: str, docs: List[Document]) -> FiltrationPlan:
    """
    Ensure that if the user mentions a case, that case gets prioritized by
    inserting it at the front of `selected_full_docs` when missing.
    """
    must = _named_case_candidates(user_q, docs)
    if not must:
        return plan
    top = must[0]
    if top not in plan.selected_full_docs:
        plan.selected_full_docs = [top] + plan.selected_full_docs
    return plan
