"""
Data ingestion module for the Legal Drafting System.
Processes text files and metadata, then ingests them into the PGVector database.
"""

import os
from rag import load_and_chunk_cases, ingest_chunks_to_pgvector_batched, get_vectorstore, upsert_all_case_summaries_from_metadata
from pathlib import Path
import json


def _list_stems_with_metadata(meta_dir: str) -> set[str]:
    """
    Get set of file stems that have corresponding metadata JSON files.
    
    Args:
        meta_dir (str): Directory containing metadata JSON files
        
    Returns:
        set[str]: Set of file stems (without extension)
    """
    stems: set[str] = set()
    for p in Path(meta_dir).glob("*.json"):
        stems.add(p.stem)
    return stems


def run_ingest(batch_size: int = 32) -> None:
    """
    Ingest processed legal documents into the PGVector database.
    
    This function:
    1. Only processes documents that have metadata JSON files
    2. Loads and chunks the text files
    3. Checks for existing embeddings to avoid duplicates
    4. Ingests new chunks in batches for efficiency
    
    Args:
        batch_size (int): Number of chunks to process in each batch
    """
    # Prompt for embedding provider/model
    print("Select embedding backend:")
    print("  [1] OpenAI (text-embedding-3-small)")
    print("  [2] Ollama local (nomic-embed-text:latest) [default]")
    choice = input("Enter choice [1-2, default 2]: ").strip()
    if choice == "1":
        embedding_provider = "openai"
        embedding_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    else:
        embedding_provider = "ollama"
        embedding_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
    print(f"Using embeddings: provider={embedding_provider}, model={embedding_model}")

    # Step 1: Find all documents that have metadata
    stems = _list_stems_with_metadata("processed_data/metadata")
    if not stems:
        print("No metadata JSONs found; nothing to ingest.")
        print("Run 'python process.py' first to extract metadata.")
        return

    print(f"Found {len(stems)} documents with metadata.")
    # Ensure summary index is up to date
    try:
        up_cnt = upsert_all_case_summaries_from_metadata("processed_data/metadata")
        print(f"Upserted {up_cnt} case summaries into Postgres FTS index.")
    except Exception as e:
        print(f"Warning: could not upsert case summaries ({e}). Continuing with vector ingestion.")

    # Step 2: Load and chunk only documents with metadata
    print("Loading and chunking documents...")
    chunks = load_and_chunk_cases(
        txt_dir="processed_data/txt_data",
        meta_dir="processed_data/metadata",
        chunk_size=2000,
        chunk_overlap=400,
        only_with_metadata=True,
    )
    if not chunks:
        print("No chunks to ingest after filtering for metadata.")
        return

    print(f"Created {len(chunks)} chunks from {len(stems)} documents.")

    # Step 3: Check for existing embeddings to avoid duplicates
    skip_existing = os.getenv("SKIP_EXISTING_CHECK", "1") == "1"
    existing = set()
    if skip_existing:
        print("Skipping existing-embedding check (SKIP_EXISTING_CHECK=1).")
    else:
        print("Checking for existing embeddings (this can take time)...")
        vs = get_vectorstore(embedding_model=embedding_model, embedding_provider=embedding_provider)
        try:
            for idx, fs in enumerate(sorted(stems), 1):
                docs = vs.similarity_search("seed", k=50, filter={"file_stem": fs})
                for d in docs:
                    ci = (d.metadata or {}).get("chunk_index")
                    if ci is not None:
                        existing.add((fs, ci))
                if idx % 100 == 0:
                    print(f"  checked {idx}/{len(stems)} stems...")
        except Exception:
            print("Warning: Could not check existing embeddings. Proceeding with full ingestion.")

    # Step 4: Filter out chunks that already exist
    new_chunks = []
    for d in chunks:
        fs = (d.metadata or {}).get("file_stem")
        ci = (d.metadata or {}).get("chunk_index")
        if fs is None or ci is None:
            continue
        if (fs, ci) not in existing:
            new_chunks.append(d)

    if not new_chunks:
        print("All chunks already ingested; nothing new to add.")
        return

    print(f"Found {len(new_chunks)} new chunks to ingest.")

    # Step 5: Ingest new chunks in batches
    print(f"Ingesting {len(new_chunks)} chunks in batches of {batch_size}...")
    if batch_size > 32 and embedding_provider == "ollama":
        print("Tip: If you see EOF errors with Ollama, reduce batch_size (e.g., to 5 or 8).")
    # Warm-up embedding to show base_url/dimensions for diagnostics
    try:
        from models import get_embeddings
        emb = get_embeddings(embedding_model, provider=embedding_provider)
        sample_vec = emb.embed_query("ping")
        base_url = getattr(emb, "base_url", None)
        client = getattr(emb, "_client", None)
        if client and hasattr(client, "base_url"):
            base_url = getattr(client, "base_url")
        print(f"[Embeddings] provider={embedding_provider} model={embedding_model} base_url={base_url} dim={len(sample_vec)}")
    except Exception as e:
        print(f"[Embeddings] warm-up failed: {e}")

    vs2 = ingest_chunks_to_pgvector_batched(
        new_chunks,
        batch_size=batch_size,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
    )
    if vs2 is None:
        print("No non-empty chunks to ingest.")
    else:
        print(f"Ingestion complete! Added {len(new_chunks)} chunks to the vector database.")
        print("You can now run the chat applications.")


if __name__ == "__main__":
    run_ingest()


