"""
Interactive processing pipeline for legal documents.
Converts PDFs to text and extracts metadata using user-selected LLM backend.
"""

from functions import save_all_judgements_to_text
from ai import save_metadata_for_all_texts


def run_pdf_to_text_and_metadata_interactive() -> None:
    """
    Interactive pipeline to process legal documents.
    
    Steps:
    1. Convert PDFs to text files
    2. Prompt user to select metadata extraction backend
    3. Extract metadata using selected LLM backend
    """
    # Step 1: Convert PDFs to text files
    print("Step 1: Converting PDFs to text...")
    save_all_judgements_to_text()

    # Step 2: Ask user to choose metadata extraction backend
    print("\nStep 2: Extracting metadata...")
    print("Select metadata backend:")
    print("  [1] OpenAI Nano (default) - Recommended, requires OPENAI_KEY")
    print("  [2] OpenRouter - Free tier available, requires OPENROUTER_API_KEY")
    print("  [3] Ollama qwen3:latest - Local, free, requires Ollama installation")
    print("Enter choice [1-3]: ", end="")
    
    choice = input().strip()
    if choice == "3":
        print("Using Ollama qwen3:latest...")
        save_metadata_for_all_texts(backend="ollama", ollama_model="qwen3:latest")
    elif choice == "2":
        print("Using OpenRouter...")
        save_metadata_for_all_texts(backend="openrouter")
    else:
        print("Using OpenAI Nano (default)...")
        save_metadata_for_all_texts(backend="openai_nano")
    
    print("\nProcessing complete! You can now run the ingestion and chat applications.")


if __name__ == "__main__":
    run_pdf_to_text_and_metadata_interactive()


