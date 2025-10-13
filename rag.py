from pathlib import Path
import json
import os
from typing import List, Optional, Tuple, Dict, Any

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


# Default collection name for PGVector
COLLECTION_NAME = "langchain"


def load_case_docs(txt_dir: str = "processed_data/txt_data", meta_dir: str = "processed_data/metadata", only_with_metadata: bool = False) -> List[Document]:
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
    base_docs = load_case_docs(txt_dir=txt_dir, meta_dir=meta_dir, only_with_metadata=only_with_metadata)
    return chunk_documents(base_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _get_pg_connection_string() -> str:
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


def ingest_chunks_to_pgvector(
    chunks: List[Document],
    connection_string: str | None = None,
    embedding_model: str = "nomic-embed-text:latest",
    use_jsonb: bool = True,
    create_extension: bool = False,
) -> PGVector:
    if not chunks:
        raise ValueError("No chunks provided for ingestion")
    conn = connection_string or _get_pg_connection_string()
    # Filter out empty/whitespace-only chunks to avoid embedding errors
    non_empty_chunks = [d for d in chunks if (d.page_content or "").strip()]
    embeddings = OllamaEmbeddings(model=embedding_model)
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
    use_jsonb: bool = True,
    create_extension: bool = False,
    batch_size: int = 128,
):
    """Ingest in batches with a single initial PGVector.from_documents and subsequent add_documents calls."""
    from tqdm import tqdm  # local import to avoid hard dep when unused

    if not chunks:
        raise ValueError("No chunks provided for ingestion")
    conn = connection_string or _get_pg_connection_string()
    # Filter out empty/whitespace-only chunks
    non_empty = [d for d in chunks if (d.page_content or "").strip()]
    embeddings = OllamaEmbeddings(model=embedding_model)

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
    embedding_model: str = "nomic-embed-text:latest",
    use_jsonb: bool = True,
    collection_name: str | None = None,
) -> PGVector:
    """Return a PGVector handle (use when an index already exists)."""
    conn = connection_string or _get_pg_connection_string()
    embeddings = OllamaEmbeddings(model=embedding_model)
    # When collection_name is None, default internal collection is used
    return PGVector(
        connection=conn,
        embeddings=embeddings,
        use_jsonb=use_jsonb,
        collection_name=collection_name or COLLECTION_NAME,
    )


def build_retriever(vectorstore: PGVector, statute_filters: list[str] | None = None, court_name: str | None = "Supreme Court of India", k: int = 6):
    """Create a retriever with metadata filters (JSONB)."""
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
    llm = ChatOllama(model=llm_model, temperature=temperature)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=return_sources,
    )


def ask(question: str, qa_chain) -> str:
    return qa_chain.run(question)


def maybe_full_case(
    chosen_docs: List[Document],
    txt_dir: str = "processed_data/txt_data",
    threshold: int = 4,
):
    """Return full-text Path for most frequent case if frequency exceeds threshold; else None."""
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
    file_stem: str
    chunk_index: int


class ReasonFullDoc(BaseModel):
    file_stem: str
    reason: str


class ReasonChunk(BaseModel):
    file_stem: str
    chunk_index: int
    reason: str


class FiltrationPlan(BaseModel):
    selected_full_docs: List[str] = []
    selected_chunks: List[ChunkRef] = []
    drop_chunks: List[ChunkRef] = []
    context_budget_tokens: int = 6000
    reasoning_full_docs: List[ReasonFullDoc] = []
    reasoning_chunks: List[ReasonChunk] = []
    overall_reasoning: Optional[str] = None
    notes: Optional[str] = None


def _build_chunk_previews(docs: List[Document], max_chars: int = 800) -> List[dict]:
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


def filtration_retriever(user_q: str, docs: List[Document], mode: str = "chunk", case_metadatas: Optional[List[dict]] = None) -> FiltrationPlan:
    """Use GPT-5-Nano as filtration retriever to select full docs and chunks.
    mode: "chunk" uses chunk previews (+ optional case metadata); "metadata" uses only case metadata.
    Ollama is not used here.
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
        "You are a legal expert filtration retriever. You will receive a user question and previews of chunks "
        "retrieved by a RAG system from a legal judgments database (and sometimes case-level metadata). Your job is to: (1) select only the most relevant chunks, "
        "(2) request full-document retrieval for specific cases if the question likely requires full-case context (e.g., a full summary is requested, or previews don't contain the key answer but clearly belong to the target case), and (3) propose a context budget.\n\n"
        "Strict rules:\n"
        "- Prefer chunks/cases whose previews contain exact or near-exact mentions from the question (party names, case number, court, date).\n"
        "- If the question clearly names a specific case, prioritize that case.\n"
        "- You may request ANY number of full documents, but be mindful and include only those truly necessary.\n"
        "- Remove irrelevant chunks to keep the context concise.\n"
        "- Output STRICT JSON matching the schema only (no prose).\n"
        "- You MUST include concise reasoning for each selected_full_docs item and each selected_chunks item (short phrase per item), and set overall_reasoning with a 1-2 sentence summary. Do not leave reasoning arrays empty.\n"
        "- Suggest an appropriate context_budget_tokens (e.g., 6000-20000) considering model limits.\n"
        "- Do NOT use external knowledge beyond the provided previews or metadata."
        "- When case-level metadata is provided, use it to disambiguate parties/court/sections and prefer exact matches."
    )
    user_payload = {
        "question": user_q,
        "mode": mode,
        "chunks": previews,
        "cases": (case_metadatas or []),
        "schema": {
            "selected_full_docs": ["<file_stem>"],
            "selected_chunks": [{"file_stem": "<file_stem>", "chunk_index": 0}],
            "drop_chunks": [{"file_stem": "<file_stem>", "chunk_index": 0}],
            "context_budget_tokens": 6000,
            "reasoning_full_docs": [{"file_stem": "<file_stem>", "reason": "why this case is needed"}],
            "reasoning_chunks": [{"file_stem": "<file_stem>", "chunk_index": 0, "reason": "why this chunk"}],
            "overall_reasoning": "optional global rationale",
            "notes": "optional short note"
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
        return FiltrationPlan(selected_full_docs=sel_full, selected_chunks=sel_chunks, drop_chunks=[], context_budget_tokens=6000)


def _token_estimate_from_chars(chars: int) -> int:
    return max(1, chars // 4)


def _windows_around_terms(text: str, query: str, budget_tokens: int) -> Tuple[str, List[Tuple[int, int]]]:
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
    """Assemble a budgeted context string from the plan; return context, source docs, and debug info."""
    budget = plan.context_budget_tokens if (budget_tokens is None) else budget_tokens
    used_docs: List[Document] = []
    assembled_parts: List[str] = []
    spans_debug: List[Tuple[int, int, str]] = []  # (start, end, file)

    # 1) Full documents windows (include whole doc if it fits budget)
    for fs in plan.selected_full_docs:
        fp = Path(txt_dir) / f"{fs}.txt"
        if not fp.exists():
            continue
        raw = fp.read_text(encoding="utf-8")
        header = f"[file: {fs}.txt]\n"
        budget_chars = budget * 4
        current_chars = sum(len(p) for p in assembled_parts)
        remaining = max(0, budget_chars - current_chars)
        if len(header) + len(raw) <= remaining:
            assembled_parts.append(header + raw)
            # record span as whole file
            spans_debug.append((0, len(raw), f"{fs}.txt"))
        else:
            segment, spans = _windows_around_terms(raw, user_q, max(1, remaining // 4))
            if segment:
                assembled_parts.append(header + segment)
                for s, e in spans:
                    spans_debug.append((s, e, f"{fs}.txt"))

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

    context = "\n\n---\n\n".join(assembled_parts) if assembled_parts else ""
    est_tokens = _token_estimate_from_chars(len(context))
    debug: Dict[str, Any] = {
        "spans": spans_debug,
        "est_tokens": est_tokens,
        "selected_full_docs": plan.selected_full_docs,
        "selected_chunks_count": len(plan.selected_chunks),
    }
    return context, used_docs, debug


# ---------------------- Named-case guard helpers ----------------------

def _named_case_candidates(user_q: str, docs: List[Document]) -> List[str]:
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
    must = _named_case_candidates(user_q, docs)
    if not must:
        return plan
    top = must[0]
    if top not in plan.selected_full_docs:
        plan.selected_full_docs = [top] + plan.selected_full_docs
    return plan

