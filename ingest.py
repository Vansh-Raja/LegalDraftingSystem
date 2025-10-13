from rag import load_and_chunk_cases, ingest_chunks_to_pgvector_batched, get_vectorstore
from pathlib import Path
import json


def _list_stems_with_metadata(meta_dir: str) -> set[str]:
    stems: set[str] = set()
    for p in Path(meta_dir).glob("*.json"):
        stems.add(p.stem)
    return stems


def run_ingest(batch_size: int = 128) -> None:
    # Only ingest items that have metadata JSON
    stems = _list_stems_with_metadata("processed_data/metadata")
    if not stems:
        print("No metadata JSONs found; nothing to ingest.")
        return

    # Load and chunk only those with metadata
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

    # Idempotence: skip chunks whose (file_stem, chunk_index) already exist in vector store
    vs = get_vectorstore()
    existing = set()
    try:
        # Pull a sample per file_stem via filter and collect chunk_index
        for fs in sorted(stems):
            docs = vs.similarity_search("seed", k=1000, filter={"file_stem": fs})
            for d in docs:
                ci = (d.metadata or {}).get("chunk_index")
                if ci is not None:
                    existing.add((fs, ci))
    except Exception:
        pass

    new_chunks = []
    for d in chunks:
        fs = (d.metadata or {}).get("file_stem")
        ci = (d.metadata or {}).get("chunk_index")
        if fs is None or ci is None:
            continue
        if (fs, ci) not in existing:
            new_chunks.append(d)

    if not new_chunks:
        print("All chunks already ingested; nothing new.")
        return

    vs2 = ingest_chunks_to_pgvector_batched(new_chunks, batch_size=batch_size)
    if vs2 is None:
        print("No non-empty chunks to ingest.")
    else:
        print(f"Ingestion complete. Added {len(new_chunks)} chunks.")


if __name__ == "__main__":
    run_ingest()


