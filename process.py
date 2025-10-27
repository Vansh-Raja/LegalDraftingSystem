"""
Interactive processing pipeline for legal documents.
Converts PDFs to text and extracts metadata using user-selected LLM backend.
"""

from pathlib import Path

from functions import save_all_judgements_to_text
from ai import save_metadata_for_all_texts


def _select_judgement_directory() -> str | None:
    """
    Prompt the user to choose which judgments folder (by year) to ingest.
    
    Returns:
        str | None: Path to the selected directory, or None if none available.
    """
    candidate_roots = [Path("judgements"), Path("judgments")]
    existing_roots = [root for root in candidate_roots if root.exists()]
    if not existing_roots:
        print("No 'judgements' or 'judgments' directory found. Please add PDFs before running.")
        return None

    root = existing_roots[0]
    subdirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.lower())

    if not subdirs:
        print(f"No year sub-folders detected under '{root}'. Processing all PDFs in this directory.")
        return str(root)

    while True:
        print("Available judgment folders:")
        for idx, folder in enumerate(subdirs, start=1):
            print(f"  [{idx}] {folder.name}")
        all_option = len(subdirs) + 1
        print(f"  [{all_option}] All years")
        print("Enter choice (number or year name, 'all' for every folder): ", end="")
        selection = input().strip()

        if not selection:
            print("No input detected. Please try again.\n")
            continue

        if selection.lower() == "all":
            return str(root)

        if selection.isdigit():
            idx = int(selection)
            if 1 <= idx <= len(subdirs):
                return str(subdirs[idx - 1])
            if idx == all_option:
                return str(root)
        else:
            matches = [folder for folder in subdirs if folder.name.lower() == selection.lower()]
            if matches:
                return str(matches[0])

        print("Invalid selection. Please choose one of the listed options.\n")


def run_pdf_to_text_and_metadata_interactive() -> None:
    """
    Interactive pipeline to process legal documents.
    
    Steps:
    1. Convert PDFs to text files
    2. Prompt user to select metadata extraction backend
    3. Extract metadata using selected LLM backend
    """
    judgements_path = _select_judgement_directory()
    if not judgements_path:
        return

    # Step 1: Convert PDFs to text files
    print(f"\nStep 1: Converting PDFs from '{judgements_path}' to text...")
    save_all_judgements_to_text(judgements_dir=judgements_path)

    # Step 2: Ask user to choose metadata extraction backend
    print("\nStep 2: Extracting metadata...")
    print("Select metadata backend:")
    print("  [1] OpenAI Nano (default) - Recommended, requires OPENAI_KEY")
    print("  [2] OpenRouter - Free tier available, requires OPENROUTER_API_KEY")
    print("  [3] Ollama qwen3:latest - Local, free, requires Ollama installation")
    print("  [4] OpenRouter qwen/qwen3-235b-a22b-2507 - Stable tier, requires OPENROUTER_API_KEY")
    print("Enter choice [1-4]: ", end="")
    
    choice = input().strip()
    if choice == "3":
        print("Using Ollama qwen3:latest...")
        save_metadata_for_all_texts(backend="ollama", ollama_model="qwen3:latest")
    elif choice == "4":
        model_name = "qwen/qwen3-235b-a22b-2507"
        print(f"Using OpenRouter ({model_name})...")
        save_metadata_for_all_texts(backend="openrouter", openrouter_model=model_name)
    elif choice == "2":
        print("Using OpenRouter...")
        save_metadata_for_all_texts(backend="openrouter")
    else:
        print("Using OpenAI Nano (default)...")
        save_metadata_for_all_texts(backend="openai_nano")

    try:
        txt_total = sum(1 for _ in Path("processed_data/txt_data").glob("*.txt"))
        metadata_total = sum(1 for _ in Path("processed_data/metadata").glob("*.json"))
        print(f"\nMetadata files present: {metadata_total}/{txt_total}")
        if metadata_total < txt_total:
            print("Some judgments still need metadata. You can re-run Step 2 later or switch to another backend.")
    except Exception:
        pass

    print("\nProcessing complete! You can now run the ingestion and chat applications.")


if __name__ == "__main__":
    run_pdf_to_text_and_metadata_interactive()
