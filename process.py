from functions import save_all_judgements_to_text
from ai import save_metadata_for_all_texts


def run_pdf_to_text_and_metadata_interactive() -> None:
    """Convert PDFs to text, then prompt for metadata backend and process to JSON."""
    # Step 1: PDFs -> text
    save_all_judgements_to_text()

    # Step 2: Ask user for metadata backend
    print("Select metadata backend: [1] OpenAI Nano (default), [2] OpenRouter, [3] Ollama qwen3:latest > ", end="")
    choice = input().strip()
    if choice == "3":
        save_metadata_for_all_texts(backend="ollama", ollama_model="qwen3:latest")
    elif choice == "2":
        save_metadata_for_all_texts(backend="openrouter")
    else:
        save_metadata_for_all_texts(backend="openai_nano")


if __name__ == "__main__":
    run_pdf_to_text_and_metadata_interactive()


