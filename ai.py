"""
AI-powered metadata extraction for legal documents.
Extracts structured metadata (case number, parties, court, date, etc.) from legal judgment text
using various LLM backends (OpenAI, OpenRouter, Ollama).
"""

import os
import json
import logging
import time
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import ollama

# Configure logging to suppress verbose SDK debug messages
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
for _lib in ("httpx", "httpcore", "openai"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

_TEMPLATE_CASE_NUMBER = "2025 INSC 149; CIVIL APPEAL NO(S). 8414 OF 2017"
_TEMPLATE_PETITIONER = "Airports Authority of India"
_TEMPLATE_RESPONDENT = "Pradip Kumar Banerjee"
_TEMPLATE_SUMMARY_SNIPPET = "Disciplinary enquiry vs criminal acquittal"
_TEMPLATE_TRUSTED_STEMS = {"7"}

_OPENROUTER_MODEL_REGISTRY = {
    "qwen/qwen3-235b-a22b": {"provider": {"only": ["deepinfra/fp8"]}},
    "qwen/qwen3-235b-a22b:free": None,
    "qwen/qwen3-235b-a22b-2507": None,
}
_OPENROUTER_MODEL_ROTATION = [
    "qwen/qwen3-235b-a22b",
    "qwen/qwen3-235b-a22b:free",
]
_openrouter_model_cursor = 0


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


def _looks_like_template(metadata: dict, source_text: str | None = None, stem: str | None = None) -> bool:
    """
    Heuristically determine whether metadata looks like the prompt example instead of fresh extraction.
    """
    if not isinstance(metadata, dict):
        return True

    summary = metadata.get("summary", "") or ""
    case_number = metadata.get("case_number", "") or ""
    parties = metadata.get("parties") or {}
    petitioner = parties.get("petitioner", "") or ""
    respondent = parties.get("respondent", "") or ""

    lowered_summary = summary.lower()

    if stem and stem in _TEMPLATE_TRUSTED_STEMS:
        # Allow original seed documents even if they resemble the template.
        return False
    if _TEMPLATE_SUMMARY_SNIPPET.lower() in lowered_summary:
        return True

    if case_number.strip() == _TEMPLATE_CASE_NUMBER:
        return True

    if petitioner.strip() == _TEMPLATE_PETITIONER and respondent.strip() == _TEMPLATE_RESPONDENT:
        return True

    placeholder_markers = ("<case citation>", "<petitioner name>", "<respondent name>", "<key issues")
    combined = " ".join([case_number.lower(), summary.lower(), petitioner.lower(), respondent.lower()])
    if any(marker in combined for marker in placeholder_markers):
        return True

    return False


def _metadata_preview(metadata: dict) -> str:
    """
    Build a concise one-line preview of metadata content for logging.
    """
    if not isinstance(metadata, dict):
        return "<non-dict metadata>"
    case_number = (metadata.get("case_number") or "").strip() or "<empty case_number>"
    parties = metadata.get("parties") or {}
    petitioner = (parties.get("petitioner") or "").strip()
    respondent = (parties.get("respondent") or "").strip()
    summary = (metadata.get("summary") or "").strip()
    if len(summary) > 160:
        summary = summary[:157].rstrip() + "..."
    parts = [f"case_number={case_number}"]
    if petitioner or respondent:
        parts.append(f"parties=({petitioner or '??'} vs {respondent or '??'})")
    if summary:
        parts.append(f"summary='{summary}'")
    return "; ".join(parts)


def extract_metadata_with_openrouter(text: str, model_override: str | None = None) -> dict:
    """
    Extract metadata using OpenRouter API (free tier available).
    
    Args:
        text (str): Legal judgment text to extract metadata from
        model_override (str | None): Explicit OpenRouter model name to use; when provided disables rotation
        
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
        "You are a legal document metadata extractor. Respond with strictly valid JSON ONLY (no prose, no comments). "
        "Use the schema with exact keys: {\"case_number\": string, \"parties\": {\"petitioner\": string, \"respondent\": string}, "
        "\"court_name\": string, \"date_of_judgment\": string, \"summary\": string, \"legal_provisions_cited\": [string], \"final_judgment\": string}. "
        "Always extract details from the supplied judgment text—never copy example placeholders or fabricate content. "
        "Summary tone: concise, keyword-dense, fact-focused; highlight issues, legal standards, reasoning, and disposition; avoid redundant headers or flowery language. "
        "List statutes or landmark cases only in legal_provisions_cited. "
        "If a value is genuinely unavailable, return an empty string (or empty list) for that field."
    )
    
    # Example JSON for the model to follow
    example_json = (
        "{"
        "\"case_number\": \"<case citation>\", "
        "\"parties\": {\"petitioner\": \"<petitioner name>\", \"respondent\": \"<respondent name>\"}, "
        "\"court_name\": \"<court name>\", "
        "\"date_of_judgment\": \"<YYYY-MM-DD>\", "
        "\"summary\": \"<key issues; standards applied; reasoning; outcome>\", "
        "\"legal_provisions_cited\": [\"<statute section>\", \"<leading case>\"] , "
        "\"final_judgment\": \"<disposition>\""
        "}"
    )
    
    # Prepare user message with truncated text
    processed_text = _truncate_text(text)
    user_msg = (
        "Extract the metadata for the following judgment. Return JSON only, matching the schema.\n\n"
        "Example JSON (format only—replace placeholders with real values):\n" + example_json + "\n\n"
        "Text (truncated if long):\n" + processed_text + "\n\nJSON:"
    )
    global _openrouter_model_cursor

    rotation_len = len(_OPENROUTER_MODEL_ROTATION)
    if model_override:
        candidate_names = [model_override]
    elif rotation_len:
        start_idx = _openrouter_model_cursor % rotation_len
        candidate_names = _OPENROUTER_MODEL_ROTATION[start_idx:] + _OPENROUTER_MODEL_ROTATION[:start_idx]
    else:
        candidate_names = []

    if not candidate_names:
        logging.error("No OpenRouter models configured for metadata extraction.")
        return _ensure_json_dict("")

    last_exception: Exception | None = None
    rate_limit_hit = False

    for attempt_pos, model_name in enumerate(candidate_names, start=1):
        extra_body = _OPENROUTER_MODEL_REGISTRY.get(model_name)
        attempt_note = f"(attempt {attempt_pos}/{len(candidate_names)})"
        print(f"Using OpenRouter model {model_name} {attempt_note}")
        try:
            request_kwargs = dict(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=5000,
                top_p=1.0,
                response_format={"type": "json_object"},
                extra_headers={
                    "HTTP-Referer": "local-dev",
                    "X-Title": "LegalDraftingSystem",
                },
            )
            if extra_body:
                request_kwargs["extra_body"] = extra_body

            resp = client.chat.completions.create(**request_kwargs)
            content = resp.choices[0].message.content or ""
            if not model_override and rotation_len:
                try:
                    base_idx = _OPENROUTER_MODEL_ROTATION.index(model_name)
                except ValueError:
                    base_idx = _openrouter_model_cursor
                _openrouter_model_cursor = (base_idx + 1) % rotation_len
                print(f"OpenRouter model {model_name} succeeded; next default index is {_openrouter_model_cursor}")
            else:
                print(f"OpenRouter model {model_name} succeeded.")
            return _ensure_json_dict(content)
        except Exception as e:
            msg = str(e)
            last_exception = e
            msg_lower = msg.lower()
            if "429" in msg or "rate limit" in msg_lower or "temporarily rate-limited" in msg_lower:
                print(f"OpenRouter rate limit for {model_name}: {msg}. Trying alternate model...")
                rate_limit_hit = True
                continue
            logging.error(f"OpenRouter error for {model_name}: {e}")
            continue

    if rate_limit_hit:
        raise RuntimeError("OPENROUTER_RATE_LIMIT")

    if last_exception is not None:
        logging.error(f"OpenRouter metadata extraction failed after trying all models: {last_exception}")

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


def _purge_template_metadata_files(txt_dir: str, output_dir: str) -> None:
    """
    Remove previously generated metadata files that match the known template shape.
    """
    txt_root = Path(txt_dir)
    out_root = Path(output_dir)
    if not txt_root.exists() or not out_root.exists():
        return

    removed = 0
    for json_path in out_root.glob("*.json"):
        base = json_path.stem
        txt_path = txt_root / f"{base}.txt"
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        source = ""
        if txt_path.exists():
            try:
                source = txt_path.read_text(encoding="utf-8")
            except Exception:
                source = ""
        if _looks_like_template(payload, source):
            try:
                json_path.unlink()
                removed += 1
            except Exception:
                continue
    if removed:
        print(f"Removed {removed} template metadata file(s); they will be regenerated.")


def _report_duplicate_case_numbers(output_dir: str, sample_limit: int = 5) -> None:
    """
    Scan generated metadata for duplicate case numbers and warn the operator.
    """
    out_root = Path(output_dir)
    if not out_root.exists():
        return

    mapping: dict[str, list[str]] = defaultdict(list)
    for json_path in out_root.glob("*.json"):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        case_number = (payload.get("case_number") or "").strip()
        if case_number:
            mapping[case_number].append(json_path.stem)

    duplicates = {case: ids for case, ids in mapping.items() if len(ids) > 1}
    if not duplicates:
        return

    print(f"Warning: detected {len(duplicates)} duplicated case_number entries in metadata output.")
    for idx, (case, ids) in enumerate(duplicates.items()):
        if idx >= sample_limit:
            break
        preview = ", ".join(sorted(ids)[:6])
        print(f"  - '{case}' present in files: {preview}")
    print("Review affected files; duplicates may indicate prompt/template leakage.")


def save_metadata_for_all_texts(
    txt_dir: str = "processed_data/txt_data",
    output_dir: str = "processed_data/metadata",
    backend: str = "openrouter",
    ollama_model: str | None = None,
    openrouter_model: str | None = None,
    throttle_seconds: float = 0.75,
    max_rate_limit_retries: int = 5,
    rate_limit_backoff: float = 60.0,
) -> None:
    """
    Process all legal judgment text files and extract metadata using specified LLM backend.
    
    Args:
        txt_dir (str): Directory containing .txt files
        output_dir (str): Directory to save .json metadata files
        backend (str): LLM backend to use ("openrouter", "openai_nano", "ollama")
        ollama_model (str, optional): Ollama model name if using Ollama backend
        openrouter_model (str, optional): Preferred OpenRouter model name; if provided, disables rotation fallback
        throttle_seconds (float): Delay between OpenRouter calls to avoid rate limits
        max_rate_limit_retries (int): Maximum retries when OpenRouter responds with rate limit
        rate_limit_backoff (float): Base seconds to wait between rate-limit retries (multiplied by attempt count)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Clean up stale template JSON files before processing.
    _purge_template_metadata_files(txt_dir, output_dir)

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

    entries: list[tuple[str, str, str]] = []
    for name in txt_files:
        txt_path = os.path.join(txt_dir, name)
        base = os.path.splitext(name)[0]
        json_output_path = os.path.join(output_dir, base + ".json")
        entries.append((name, txt_path, json_output_path))

    pending_entries = [item for item in entries if not os.path.exists(item[2])]
    already_processed = len(entries) - len(pending_entries)

    if already_processed:
        print(f"{already_processed} metadata file(s) already processed; skipping them.")

    if not pending_entries:
        print("All available metadata files already exist. Nothing new to extract.")
        _report_duplicate_case_numbers(output_dir)
        return

    # Process files with progress bar
    sleep_between_calls = throttle_seconds if backend == "openrouter" else 0.0

    total_files = len(pending_entries)
    processed_files = 0

    with tqdm(total=total_files, desc="Extracting metadata", unit="file") as pbar:
        for name, txt_path, json_output_path in pending_entries:
            base = os.path.splitext(name)[0]

            # Read text file
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            metadata = None
            retries = 0
            while metadata is None:
                try:
                    # Extract metadata using specified backend
                    if backend == "ollama":
                        candidate = extract_metadata_with_ollama(text, model=ollama_model or "qwen3:latest")
                    elif backend == "openai_nano":
                        candidate = extract_metadata_with_openai_nano(text)
                    else:
                        candidate = extract_metadata_with_openrouter(text, model_override=openrouter_model)
                except RuntimeError as stop_exc:
                    if str(stop_exc) == "OPENROUTER_RATE_LIMIT" and backend == "openrouter":
                        retries += 1
                        if retries > max_rate_limit_retries:
                            print(
                                f"OpenRouter rate limit persists after {max_rate_limit_retries} retries. "
                                "Stopping metadata extraction; rerun later or choose another backend."
                            )
                            return
                        wait_time = rate_limit_backoff * retries
                        print(
                            f"OpenRouter rate limit hit while processing {name}. "
                            f"Retrying in {int(wait_time)} seconds (attempt {retries}/{max_rate_limit_retries})."
                        )
                        time.sleep(wait_time)
                        continue
                    print(f"Metadata extraction error for {name}: {stop_exc}. Stopping.")
                    return
                except Exception as e:
                    print(f"Metadata extraction error for {name}: {e}. Stopping.")
                    return
                else:
                    metadata = candidate

                if metadata is None:
                    continue

                if not isinstance(metadata, dict) or _is_empty_metadata(metadata):
                    retries += 1
                    if retries > 3:
                        print(
                            f"Received empty/invalid metadata for {name} after {retries} attempts. Stopping without writing JSON."
                        )
                        return
                    print(f"Empty metadata for {name}; retrying ({retries}/3) after short delay.")
                    time.sleep(5)
                    metadata = None
                    continue

                if _looks_like_template(metadata, text, base):
                    retries += 1
                    if retries > 3:
                        print(
                            f"Detected template-like metadata for {name} after {retries} attempts. Stopping to avoid misalignment.\n"
                            f"  Preview: {_metadata_preview(metadata)}"
                        )
                        return
                    print(
                        f"Template-like metadata detected for {name}; retrying ({retries}/3) after short delay.\n"
                        f"  Preview: {_metadata_preview(metadata)}"
                    )
                    time.sleep(5)
                    metadata = None
                    continue

            # Write metadata to JSON file
            try:
                with open(json_output_path, "w", encoding="utf-8") as jf:
                    json.dump(metadata, jf, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Failed to write metadata for {name}: {e}. Stopping.")
                return

            if sleep_between_calls > 0:
                time.sleep(sleep_between_calls)

            pbar.update(1)
            processed_files += 1

    if processed_files < total_files:
        print(
            f"Metadata extraction finished early ({processed_files}/{total_files}). "
            "Rerun later to complete remaining files."
        )

    _report_duplicate_case_numbers(output_dir)


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
