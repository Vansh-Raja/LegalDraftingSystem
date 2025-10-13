from rag import load_and_chunk_cases, ingest_chunks_to_pgvector_batched


def run_ingest(batch_size: int = 128) -> None:
    # Load and chunk from processed_data only
    chunks = load_and_chunk_cases(
        txt_dir="processed_data/txt_data",
        meta_dir="processed_data/metadata",
        chunk_size=2000,
        chunk_overlap=400,
    )
    if not chunks:
        print("No chunks to ingest. Ensure processed_data/txt_data and metadata exist.")
        return
    vs = ingest_chunks_to_pgvector_batched(chunks, batch_size=batch_size)
    if vs is None:
        print("No non-empty chunks to ingest.")
    else:
        print("Ingestion complete.")


if __name__ == "__main__":
    run_ingest()


