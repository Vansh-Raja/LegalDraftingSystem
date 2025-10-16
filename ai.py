"""
AI-powered metadata extraction for legal documents.
Extracts structured metadata (case number, parties, court, date, etc.) from legal judgment text
using various LLM backends (OpenAI, OpenRouter, Ollama).
"""

import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import ollama

# Configure logging to suppress verbose SDK debug messages
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
for _lib in ("httpx", "httpcore", "openai"):
    logging.getLogger(_lib).setLevel(logging.WARNING)


def _ensure_json_dict(text: str) -> dict:
    """
    Parse JSON from LLM output with fallback strategies.
    
    Args:
        text (str): Raw text output from LLM
        
    Returns:
        dict: Parsed metadata or empty schema if parsing fails
    """
    # Try direct JSON parsing first
    try:
        parsed = json.loads(text)
        logging.debug("JSON parsed directly from model output.")
        return parsed
    except Exception:
        logging.debug("Direct JSON parse failed; attempting substring extraction.")
    
    # Try to extract JSON substring between first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            logging.debug("JSON parsed from substring of model output.")
            return parsed
        except Exception:
            logging.debug("Substring JSON parse also failed.")
    
    # Fallback: return empty schema if all parsing attempts fail
    logging.warning("Falling back to empty metadata schema.")
    return {
        "case_number": "",
        "parties": {"petitioner": "", "respondent": ""},
        "court_name": "",
        "date_of_judgment": "",
        "summary": "",
        "legal_provisions_cited": [],
        "final_judgment": "",
    }


def _truncate_text(text: str, max_chars: int = 12000) -> str:
    """
    Truncate long text to fit within model token limits.
    
    Args:
        text (str): Input text to truncate
        max_chars (int): Maximum characters allowed
        
    Returns:
        str: Truncated text
    """
    original_len = len(text)
    cleaned = text.strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
        logging.debug(f"Input truncated from {original_len} to {len(cleaned)} chars")
    else:
        logging.debug(f"Input length {len(cleaned)} within limit")
    return cleaned


def _is_empty_metadata(data: dict) -> bool:
    """
    Check if metadata dictionary matches the empty fallback schema.
    
    Args:
        data (dict): Metadata dictionary to check
        
    Returns:
        bool: True if data is empty/fallback schema
    """
    try:
        return (
            data.get("case_number", "") == ""
            and isinstance(data.get("parties", {}), dict)
            and data.get("parties", {}).get("petitioner", "") == ""
            and data.get("parties", {}).get("respondent", "") == ""
            and data.get("court_name", "") == ""
            and data.get("date_of_judgment", "") == ""
            and data.get("summary", "") == ""
            and isinstance(data.get("legal_provisions_cited", []), list)
            and len(data.get("legal_provisions_cited", [])) == 0
            and data.get("final_judgment", "") == ""
        )
    except Exception:
        return False


def _extract_text_from_responses(resp) -> str:
    """
    Extract text content from OpenAI Responses API reply.
    
    Args:
        resp: Response object from OpenAI Responses API
        
    Returns:
        str: Extracted text content or empty string if extraction fails
    """
    # Try the convenience method first (openai>=1.40)
    try:
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt:
            return txt
    except Exception:
        pass
    
    # Fall back to walking the response structure
    try:
        outputs = getattr(resp, "output", None) or []
        for out in outputs:
            content = getattr(out, "content", None) or []
            for c in content:
                if getattr(c, "type", None) == "output_text":
                    return getattr(c, "text", "")
                # Some SDKs use {type: "text"}
                if getattr(c, "type", None) == "text":
                    return getattr(c, "text", "")
    except Exception:
        pass
    return ""


def extract_metadata_with_openrouter(text: str) -> dict:
    """
    Extract metadata using OpenRouter API (free tier available).
    
    Args:
        text (str): Legal judgment text to extract metadata from
        
    Returns:
        dict: Extracted metadata dictionary
        
    Raises:
        RuntimeError: If rate limit exceeded (stops processing loop)
    """
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    # Initialize OpenRouter client
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    
    # System prompt for metadata extraction
    system_msg = (
        "You are a legal document metadata extractor. Respond with strictly valid JSON ONLY (no prose, no angle-bracket tags). "
        "Schema with exact keys: {\"case_number\": string, \"parties\": {\"petitioner\": string, \"respondent\": string}, "
        "\"court_name\": string, \"date_of_judgment\": string, \"summary\": string, \"legal_provisions_cited\": [string], \"final_judgment\": string}. "
        "Summary style: concise, keyword-dense, factual; prioritize issues, standards, reasoning, and disposition; "
        "avoid redundant restatement of case_number/parties/court/date; minimize adjectives/adverbs; no quotations/citations inside the summary; "
        "prefer tight phrasing (commas/semicolons) over verbose sentences; length may be extended if content remains crisp. "
        "Rules for legal_provisions_cited: list statute sections and landmark case citations only (short strings). "
        "If any field is unknown, use empty strings or an empty array."
    )
    
    # Example JSON for the model to follow
    example_json = (
        "{"
        "\"case_number\": \"2025 INSC 149; CIVIL APPEAL NO(S). 8414 OF 2017\", "
        "\"parties\": {\"petitioner\": \"Airports Authority of India\", \"respondent\": \"Pradip Kumar Banerjee\"}, "
        "\"court_name\": \"Supreme Court of India\", \"date_of_judgment\": \"2025-02-04\", "
        "\"summary\": \"Disciplinary enquiry vs criminal acquittal; standard: preponderance, not beyond reasonable doubt; confession admissible in departmental proceedings; HC reappreciation improper in intra-court appeal; dismissal penalty sustained; impugned judgment set aside.\", "
        "\"legal_provisions_cited\": [\"Prevention of Corruption Act, 1988 s.7, s.13(2), s.13(1)(d)\", \"Factories Act, 1948\"], "
        "\"final_judgment\": \"Appeal allowed; High Court judgment set aside; dismissal restored.\""
        "}"
    )
    
    # Prepare user message with truncated text
    processed_text = _truncate_text(text)
    user_msg = (
        "Extract the metadata for the following judgment. Return JSON only, matching the schema.\n\n"
        "Example JSON (follow format, adapt content):\n" + example_json + "\n\n"
        "Text (truncated if long):\n" + processed_text + "\n\nJSON:"
    )
    
    try:
        # Call OpenRouter API
        resp = client.chat.completions.create(
            model="qwen/qwen3-235b-a22b",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=1000,
            top_p=1.0,
            response_format={"type": "json_object"},
            extra_headers={
                "HTTP-Referer": "local-dev",
                "X-Title": "LegalDraftingSystem",
            },
            extra_body={
                "provider": {"only": ["deepinfra/fp8"]} 
            }
        )
        content = resp.choices[0].message.content or ""
        return _ensure_json_dict(content)
    except Exception as e:
        # Handle rate limits and other errors
        msg = str(e)
        if "429" in msg or "Rate limit" in msg:
            print("the openrouter limit exceed try after sometime.")
            # Raise sentinel to stop processing loop
            raise RuntimeError("OPENROUTER_RATE_LIMIT")
        logging.error(f"Error during OpenRouter call: {e}")
        return _ensure_json_dict("")


def extract_metadata_with_openai_nano(text: str) -> dict:
    """
    Extract metadata using OpenAI's gpt-5-nano model (preferred method).
    
    Args:
        text (str): Legal judgment text to extract metadata from
        
    Returns:
        dict: Extracted metadata dictionary
        
    Raises:
        Exception: If API call fails
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_KEY")
    client = OpenAI(api_key=api_key)
    
    # System prompt for metadata extraction
    system_msg = (
        "You are a legal document metadata extractor. Respond with strictly valid JSON ONLY (no prose). "
        "Schema with exact keys: {\"case_number\": string, \"parties\": {\"petitioner\": string, \"respondent\": string}, "
        "\"court_name\": string, \"date_of_judgment\": string, \"summary\": string, \"legal_provisions_cited\": [string], \"final_judgment\": string}. "
        "Summary style: concise, keyword-dense, factual; prioritize issues, standards, reasoning, and disposition; minimize adjectives. "
        "If any field is unknown, use empty strings or an empty array."
    )
    
    # Example JSON template
    example_json = (
        "{"
        "\"case_number\": \"\", \"parties\": {\"petitioner\": \"\", \"respondent\": \"\"}, "
        "\"court_name\": \"\", \"date_of_judgment\": \"\", \"summary\": \"\", \"legal_provisions_cited\": [], \"final_judgment\": \"\""
        "}"
    )
    
    # Prepare user message
    processed_text = _truncate_text(text)
    user_msg = (
        "Extract the metadata for the following judgment. Return JSON only, matching the schema.\n\n"
        "Example JSON (follow format, adapt content):\n" + example_json + "\n\n"
        "Text (truncated if long):\n" + processed_text + "\n\nJSON:"
    )
    
    try:
        # Try Responses API first (preferred for gpt-5-nano)
        resp = client.responses.create(
            model="gpt-5-nano-2025-08-07",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_msg}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_msg}]},
            ],
            # Structured JSON output for Responses API
            text={"format": {"type": "json_object"}},
            max_output_tokens=700,
        )
        content = _extract_text_from_responses(resp)
        
        if not content:
            # Fallback to Chat Completions API
            resp_cc = client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[
                    {"role": "system", "content": system_msg + " Return strictly valid JSON only."},
                    {"role": "user", "content": user_msg},
                ],
            )
            content = (resp_cc.choices[0].message.content or "")

        # Debug: show raw model output for diagnostics
        try:
            preview = content[:800]
            _log_debug(f"[DEBUG][OpenAI Nano] Raw reply preview (len={len(content)}):\n{preview}")
        except Exception:
            pass
        
        parsed = _ensure_json_dict(content)
        if _is_empty_metadata(parsed):
            _log_debug("[DEBUG][OpenAI Nano] Parsed empty metadata. See raw preview above.")
        return parsed
    except Exception as e:
        logging.error(f"Error during OpenAI Nano call: {e}")
        raise


def save_metadata_for_all_texts(txt_dir: str = "processed_data/txt_data", output_dir: str = "processed_data/metadata", backend: str = "openrouter", ollama_model: str | None = None) -> None:
    """
    Process all legal judgment text files and extract metadata using specified LLM backend.
    
    Args:
        txt_dir (str): Directory containing .txt files
        output_dir (str): Directory to save .json metadata files
        backend (str): LLM backend to use ("openrouter", "openai_nano", "ollama")
        ollama_model (str, optional): Ollama model name if using Ollama backend
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(txt_dir):
        # Silent return if input directory doesn't exist
        return
    
    # Find all numeric .txt files (1.txt, 2.txt, etc.) for consistent processing order
    txt_files = sorted(
        [f for f in os.listdir(txt_dir) if f.lower().endswith(".txt") and os.path.splitext(f)[0].isdigit()],
        key=lambda name: int(os.path.splitext(name)[0])
    )
    if not txt_files:
        # Silent return if no files found
        return
    
    # Process files with progress bar
    with tqdm(total=len(txt_files), desc="Extracting metadata", unit="file") as pbar:
        for name in txt_files:
            txt_path = os.path.join(txt_dir, name)
            base = os.path.splitext(name)[0]
            json_output_path = os.path.join(output_dir, base + ".json")
            
            # Skip if metadata already exists
            if os.path.exists(json_output_path):
                pbar.update(1)
                continue
            
            # Read text file
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            try:
                # Extract metadata using specified backend
                if backend == "ollama":
                    metadata = extract_metadata_with_ollama(text, model=ollama_model or "qwen3:latest")
                elif backend == "openai_nano":
                    metadata = extract_metadata_with_openai_nano(text)
                else:
                    metadata = extract_metadata_with_openrouter(text)
            except RuntimeError as stop_exc:
                if str(stop_exc) == "OPENROUTER_RATE_LIMIT":
                    print("the openrouter limit exceed try after sometime.")
                    break
                # Any other runtime error: stop the loop
                print(f"Metadata extraction error: {stop_exc}. Stopping.")
                break
            except Exception as e:
                print(f"Metadata extraction error: {e}. Stopping.")
                break

            # Validate that we got meaningful metadata
            if not isinstance(metadata, dict) or _is_empty_metadata(metadata):
                print(f"Received empty/invalid metadata for {name}. Stopping without writing JSON.")
                break

            # Write metadata to JSON file
            try:
                with open(json_output_path, "w", encoding="utf-8") as jf:
                    json.dump(metadata, jf, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Failed to write metadata for {name}: {e}. Stopping.")
                break

            pbar.update(1)


def extract_metadata_with_ollama(text: str, model: str = "qwen3:latest") -> dict:
    """
    Extract metadata using local Ollama model (free, runs locally).
    
    Args:
        text (str): Legal judgment text to extract metadata from
        model (str): Ollama model name to use
        
    Returns:
        dict: Extracted metadata dictionary
    """
    load_dotenv()
    # Note: User can set OLLAMA_HOST env variable; defaults to localhost:11434
    
    # System prompt for metadata extraction
    system_msg = (
        "You are a legal document metadata extractor. Respond with strictly valid JSON ONLY (no prose, no angle-bracket tags). "
        "Schema with exact keys: {\"case_number\": string, \"parties\": {\"petitioner\": string, \"respondent\": string}, "
        "\"court_name\": string, \"date_of_judgment\": string, \"summary\": string, \"legal_provisions_cited\": [string], \"final_judgment\": string}. "
        "Summary style: concise, keyword-dense, factual; prioritize issues, standards, reasoning, and disposition; avoid redundancy; no quotes. "
        "If any field is unknown, use empty strings or an empty array."
    )
    
    # Example JSON template
    example_json = (
        "{\"case_number\": \"\", \"parties\": {\"petitioner\": \"\", \"respondent\": \"\"}, \"court_name\": \"\", \"date_of_judgment\": \"\", "
        "\"summary\": \"\", \"legal_provisions_cited\": [], \"final_judgment\": \"\"}"
    )
    
    # Prepare messages for Ollama
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "Extract the metadata for the following judgment. Return JSON only, matching the schema.\n\nExample JSON:\n" + example_json + "\n\nText:\n" + _truncate_text(text) + "\n\nJSON:"},
    ]
    
    # Call Ollama API
    response = ollama.chat(model=model, messages=messages, options={"temperature": 0})
    content = (response.get("message") or {}).get("content") or ""
    return _ensure_json_dict(content)

