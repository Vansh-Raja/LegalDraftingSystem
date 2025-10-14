"""
PDF processing functions for legal document extraction.
Converts PDF files to text format for further processing by the RAG system.
"""

import os
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
        return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]

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
            content = extract_text_from_pdf(str(pdf))
            target_txt.write_text(content, encoding="utf-8")

            # Update manifest
            if _json:
                manifest[pdf_key] = current_max
                manifest_path.write_text(_json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            pbar.update(1)
    
    # Note: AI-related metadata extraction functions are in ai.py
