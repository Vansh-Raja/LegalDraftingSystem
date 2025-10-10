from functions import save_all_judgements_to_text
from ai import save_metadata_for_all_texts


if __name__ == "__main__":
    save_all_judgements_to_text()
    choice = input("Select metadata backend: [1] OpenRouter (default), [2] Ollama qwen3:latest > ").strip()
    if choice == "2":
        save_metadata_for_all_texts(backend="ollama", ollama_model="qwen3:latest")
    else:
        save_metadata_for_all_texts(backend="openrouter")


