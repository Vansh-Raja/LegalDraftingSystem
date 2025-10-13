from pathlib import Path
import json
import os
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA


# Default collection name for PGVector
COLLECTION_NAME = "langchain"


def load_case_docs(txt_dir: str = "processed_data/txt_data", meta_dir: str = "processed_data/metadata") -> List[Document]:
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
) -> List[Document]:
    base_docs = load_case_docs(txt_dir=txt_dir, meta_dir=meta_dir)
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


def build_retriever(vectorstore: PGVector, statute_filters: list[str] | None = None, court_name: str | None = "Supreme Court", k: int = 6):
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


