import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import ollama

# Configure module-level logging (suppress SDK/http debug that may include full request bodies)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
for _lib in ("httpx", "httpcore", "openai"):
    logging.getLogger(_lib).setLevel(logging.WARNING)


def _ensure_json_dict(text: str) -> dict:
    """Best-effort: parse JSON from model output. Falls back to empty schema if needed."""
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
    # Fallback empty schema
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
    """Trim very long inputs to reduce latency and avoid model limits."""
    original_len = len(text)
    cleaned = text.strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
        logging.debug(f"Input truncated from {original_len} to {len(cleaned)} chars")
    else:
        logging.debug(f"Input length {len(cleaned)} within limit")
    return cleaned


def _is_empty_metadata(data: dict) -> bool:
    """Return True if data matches the empty fallback schema."""
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
    """Best-effort extraction of text from Responses API reply."""
    # openai>=1.40 exposes output_text convenience
    try:
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt:
            return txt
    except Exception:
        pass
    # Fall back to walking the structure
    try:
        outputs = getattr(resp, "output", None) or []
        for out in outputs:
            content = getattr(out, "content", None) or []
            for c in content:
                if getattr(c, "type", None) == "output_text":
                    return getattr(c, "text", "")
                # some SDKs use {type: "text"}
                if getattr(c, "type", None) == "text":
                    return getattr(c, "text", "")
    except Exception:
        pass
    return ""


def extract_metadata_with_openrouter(text: str) -> dict:
    """Use OpenRouter via OpenAI client to extract structured metadata and return a dict."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    # Debug suppressed
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
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
    processed_text = _truncate_text(text)
    user_msg = (
        "Extract the metadata for the following judgment. Return JSON only, matching the schema.\n\n"
        "Example JSON (follow format, adapt content):\n" + example_json + "\n\n"
        "Text (truncated if long):\n" + processed_text + "\n\nJSON:"
    )
    try:
        resp = client.chat.completions.create(
            model="qwen/qwen3-235b-a22b:free",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=700,
            top_p=1.0,
            response_format={"type": "json_object"},
            extra_headers={
                "HTTP-Referer": "local-dev",
                "X-Title": "LegalDraftingSystem",
            },
        )
        content = resp.choices[0].message.content or ""
        return _ensure_json_dict(content)
    except Exception as e:
        # Gracefully handle rate limits and abort outer loop via sentinel
        msg = str(e)
        if "429" in msg or "Rate limit" in msg:
            print("the openrouter limit exceed try after sometime.")
            # Raise a sentinel to stop processing loop
            raise RuntimeError("OPENROUTER_RATE_LIMIT")
        logging.error(f"Error during OpenRouter call: {e}")
        return _ensure_json_dict("")


def extract_metadata_with_openai_nano(text: str) -> dict:
    """Use OpenAI gpt-5-nano-2025-08-07 to extract structured metadata and return a dict."""
    load_dotenv()
    api_key = os.getenv("OPENAI_KEY")
    client = OpenAI(api_key=api_key)
    system_msg = (
        "You are a legal document metadata extractor. Respond with strictly valid JSON ONLY (no prose). "
        "Schema with exact keys: {\"case_number\": string, \"parties\": {\"petitioner\": string, \"respondent\": string}, "
        "\"court_name\": string, \"date_of_judgment\": string, \"summary\": string, \"legal_provisions_cited\": [string], \"final_judgment\": string}. "
        "Summary style: concise, keyword-dense, factual; prioritize issues, standards, reasoning, and disposition; minimize adjectives. "
        "If any field is unknown, use empty strings or an empty array."
    )
    example_json = (
        "{"
        "\"case_number\": \"\", \"parties\": {\"petitioner\": \"\", \"respondent\": \"\"}, "
        "\"court_name\": \"\", \"date_of_judgment\": \"\", \"summary\": \"\", \"legal_provisions_cited\": [], \"final_judgment\": \"\""
        "}"
    )
    processed_text = _truncate_text(text)
    user_msg = (
        "Extract the metadata for the following judgment. Return JSON only, matching the schema.\n\n"
        "Example JSON (follow format, adapt content):\n" + example_json + "\n\n"
        "Text (truncated if long):\n" + processed_text + "\n\nJSON:"
    )
    try:
        # Prefer Responses API for gpt-5-nano
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
            # Fallback to Chat Completions without unsupported params
            resp_cc = client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[
                    {"role": "system", "content": system_msg + " Return strictly valid JSON only."},
                    {"role": "user", "content": user_msg},
                ],
            )
            content = (resp_cc.choices[0].message.content or "")

        # Debug: show raw model output (truncated) for diagnostics
        try:
            preview = content[:800]
            print(f"[DEBUG][OpenAI Nano] Raw reply preview (len={len(content)}):\n{preview}")
        except Exception:
            pass
        parsed = _ensure_json_dict(content)
        if _is_empty_metadata(parsed):
            print("[DEBUG][OpenAI Nano] Parsed empty metadata. See raw preview above.")
        return parsed
    except Exception as e:
        logging.error(f"Error during OpenAI Nano call: {e}")
        raise


def save_metadata_for_all_texts(txt_dir: str = "processed_data/txt_data", output_dir: str = "processed_data/metadata", backend: str = "openrouter", ollama_model: str | None = None) -> None:
    """Process numeric .txt files in order and write corresponding .json files. Skips existing JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(txt_dir):
        # silent if missing
        return
    # Only process top-level numeric files like 1.txt, 2.txt, ... for consistent ordering
    txt_files = sorted(
        [f for f in os.listdir(txt_dir) if f.lower().endswith(".txt") and os.path.splitext(f)[0].isdigit()],
        key=lambda name: int(os.path.splitext(name)[0])
    )
    if not txt_files:
        # silent if none
        return
    with tqdm(total=len(txt_files), desc="Extracting metadata", unit="file") as pbar:
        for name in txt_files:
            txt_path = os.path.join(txt_dir, name)
            base = os.path.splitext(name)[0]
            json_output_path = os.path.join(output_dir, base + ".json")
            if os.path.exists(json_output_path):
                pbar.update(1)
                continue
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            try:
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

            # Validate non-empty structured output
            if not isinstance(metadata, dict) or _is_empty_metadata(metadata):
                print(f"Received empty/invalid metadata for {name}. Stopping without writing JSON.")
                break

            try:
                with open(json_output_path, "w", encoding="utf-8") as jf:
                    json.dump(metadata, jf, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Failed to write metadata for {name}: {e}. Stopping.")
                break

            pbar.update(1)


def extract_metadata_with_ollama(text: str, model: str = "qwen3:latest") -> dict:
    """Use local Ollama qwen3:latest to extract structured metadata and return a dict."""
    load_dotenv()
    # If needed, user can set OLLAMA_HOST env for the SDK; otherwise it defaults to localhost:11434
    system_msg = (
        "You are a legal document metadata extractor. Respond with strictly valid JSON ONLY (no prose, no angle-bracket tags). "
        "Schema with exact keys: {\"case_number\": string, \"parties\": {\"petitioner\": string, \"respondent\": string}, "
        "\"court_name\": string, \"date_of_judgment\": string, \"summary\": string, \"legal_provisions_cited\": [string], \"final_judgment\": string}. "
        "Summary style: concise, keyword-dense, factual; prioritize issues, standards, reasoning, and disposition; avoid redundancy; no quotes. "
        "If any field is unknown, use empty strings or an empty array."
    )
    example_json = (
        "{\"case_number\": \"\", \"parties\": {\"petitioner\": \"\", \"respondent\": \"\"}, \"court_name\": \"\", \"date_of_judgment\": \"\", "
        "\"summary\": \"\", \"legal_provisions_cited\": [], \"final_judgment\": \"\"}"
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "Extract the metadata for the following judgment. Return JSON only, matching the schema.\n\nExample JSON:\n" + example_json + "\n\nText:\n" + _truncate_text(text) + "\n\nJSON:"},
    ]
    # Use the SDK's chat to get a single response
    response = ollama.chat(model=model, messages=messages, options={"temperature": 0})
    content = (response.get("message") or {}).get("content") or ""
    return _ensure_json_dict(content)


