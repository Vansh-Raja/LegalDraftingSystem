"""
PDF processing functions for legal document extraction.
Converts PDF files to text format for further processing by the RAG system.
"""

import os
from io import BytesIO
try:
    import docx  # python-docx
except Exception:
    docx = None
from pdfminer.high_level import extract_text
from pathlib import Path
from tqdm import tqdm

# Note: tqdm progress bar shows elapsed time and ETA during processing

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text content from a single PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content, or empty string if extraction fails
    """
    # Extract all text from the PDF using pdfminer
    return extract_text(pdf_path) or ""


def _clean_text(content: str) -> str:
    """
    Normalise whitespace and drop common boilerplate artefacts from extracted text.
    """
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = []
    previous_blank = False
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            if previous_blank:
                continue
            previous_blank = True
            cleaned_lines.append("")
            continue

        # Skip common scraped footer noise
        if line.lower().startswith("indian kanoon -"):
            continue

        cleaned_lines.append(line)
        previous_blank = False

    return "\n".join(cleaned_lines).strip()


def _extract_text_from_docx_bytes(raw_bytes: bytes) -> str:
    """
    Extract text content from a DOCX file provided as raw bytes.
    Returns empty string if python-docx is unavailable or parsing fails.
    """
    if not raw_bytes:
        return ""
    if docx is None:
        return ""
    try:
        bio = BytesIO(raw_bytes)
        doc = docx.Document(bio)
        parts = []
        for p in doc.paragraphs:
            txt = (p.text or "").rstrip()
            if txt:
                parts.append(txt)
            else:
                parts.append("")
        return "\n".join(parts)
    except Exception:
        return ""


def _extract_text_from_txt_bytes(raw_bytes: bytes) -> str:
    """
    Decode text from raw bytes with utf-8 fallback; return empty string on failure.
    """
    if not raw_bytes:
        return ""
    try:
        return raw_bytes.decode("utf-8", errors="strict")
    except Exception:
        try:
            return raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            try:
                return raw_bytes.decode("latin-1", errors="replace")
            except Exception:
                return ""


def extract_text_from_upload(file_name: str | None, mime_type: str | None, raw_bytes: bytes | None) -> str:
    """
    Route an uploaded file (PDF/DOCX/TXT) to the appropriate text extractor.
    Returns cleaned plain text, or empty string when extraction fails.
    """
    if not raw_bytes:
        return ""
    name = (file_name or "").lower()
    mt = (mime_type or "").lower()

    # Try by mime-type first
    if "pdf" in mt:
        try:
            # pdfminer expects a path; for bytes, write to a temporary file is required.
            # Avoid disk IO in this helper; fall back to extension routing below.
            pass
        except Exception:
            pass
    if "word" in mt or "docx" in mt:
        txt = _extract_text_from_docx_bytes(raw_bytes)
        return _clean_text(txt) if txt else ""
    if mt.startswith("text/") or "plain" in mt:
        txt = _extract_text_from_txt_bytes(raw_bytes)
        return _clean_text(txt) if txt else ""

    # Route by extension
    if name.endswith(".docx"):
        txt = _extract_text_from_docx_bytes(raw_bytes)
        return _clean_text(txt) if txt else ""
    if name.endswith(".txt"):
        txt = _extract_text_from_txt_bytes(raw_bytes)
        return _clean_text(txt) if txt else ""
    if name.endswith(".pdf"):
        # For PDFs from uploads, simple in-memory parsing isn't available with pdfminer.six.
        # Caller should save to disk if PDF support for uploads is required; return empty for now.
        return ""
    # Unknown type: attempt text decode
    txt = _extract_text_from_txt_bytes(raw_bytes)
    return _clean_text(txt) if txt else ""

def save_all_judgements_to_text(judgements_dir: str = "judgements", output_dir: str = "processed_data/txt_data") -> None:
    """
    Convert all PDF files to text and save as sequentially numbered files (1.txt, 2.txt, etc.).
    
    Features:
    - Maintains a manifest to keep numbering stable across runs
    - Skips already processed files
    - Handles both "judgements" and "judgments" directory names
    - Uses progress bar for user feedback
    
    Args:
        judgements_dir (str): Directory containing PDF files (default: "judgements")
        output_dir (str): Directory to save text files (default: "processed_data/txt_data")
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load existing manifest to maintain stable numbering
    manifest_path = output_path / "manifest.json"
    try:
        import json as _json
    except Exception:
        _json = None

    # Load existing manifest or create empty one
    manifest = {}
    if _json and manifest_path.exists():
        try:
            manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    # Find all currently used numbers
    used_numbers = set(manifest.values()) if manifest else set()
    for txt_file in output_path.glob("*.txt"):
        if txt_file.stem.isdigit():
            used_numbers.add(int(txt_file.stem))
    current_max = max(used_numbers) if used_numbers else 0

    # Helper function to collect all PDF files recursively
    def collect_pdfs(root: Path) -> list[Path]:
        return [
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() == ".pdf"
            and not p.name.startswith("._")
        ]

    # Find PDF files in the specified directory
    root_path = Path(judgements_dir)
    pdf_files = collect_pdfs(root_path) if root_path.exists() else []
    
    # Fallback to US spelling if UK spelling directory doesn't exist
    if not pdf_files and judgements_dir == "judgements":
        alt_root = Path("judgments")
        if alt_root.exists():
            pdf_files = collect_pdfs(alt_root)

    # Sort PDF files for consistent processing order
    pdf_files = sorted(pdf_files, key=lambda p: str(p).lower())

    if not pdf_files:
        return

    # Process PDFs with progress bar
    total = len(pdf_files)
    with tqdm(total=total, desc="Converting PDFs", unit="file") as pbar:
        for pdf in pdf_files:
            pdf_key = str(pdf)
            assigned = manifest.get(pdf_key) if manifest else None
            
            # Skip if already processed
            if assigned is not None and (output_path / f"{assigned}.txt").exists():
                pbar.update(1)
                continue

            # Find next available number
            current_max += 1
            target_txt = output_path / f"{current_max}.txt"
            if target_txt.exists():
                # Find the next free slot
                n = current_max
                while (output_path / f"{n}.txt").exists():
                    n += 1
                current_max = n
                target_txt = output_path / f"{current_max}.txt"

            # Extract text and save to file
            try:
                raw_content = extract_text_from_pdf(str(pdf))
            except Exception as exc:
                pbar.write(f"[WARN] Failed to extract '{pdf}': {exc}")
                current_max -= 1
                continue

            cleaned = _clean_text(raw_content)
            if not cleaned:
                pbar.write(f"[WARN] No text extracted from '{pdf}'. Skipping.")
                current_max -= 1
                continue

            target_txt.write_text(cleaned, encoding="utf-8")

            # Update manifest
            if _json:
                manifest[pdf_key] = current_max
                manifest_path.write_text(_json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            pbar.update(1)
    
    # Note: AI-related metadata extraction functions are in ai.py
