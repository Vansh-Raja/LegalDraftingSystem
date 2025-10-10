import os
from pdfminer.high_level import extract_text
from pathlib import Path
from tqdm import tqdm

# Note: tqdm progress is used for elapsed time and ETA

# This function takes the path to a PDF file and returns its text content.
def extract_text_from_pdf(pdf_path: str) -> str:
    """Return full text content of a single PDF file."""
    # Get all text from the PDF, or return an empty string if nothing found
    return extract_text(pdf_path) or ""

# This function goes through all PDF files in the 'judgements' folder,
# extracts their text, and saves each as a .txt file in another folder.
def save_all_judgements_to_text(judgements_dir: str = "judgements", output_dir: str = "processed_data/txt_data") -> None:
    """
    Convert PDFs to text and save as sequentially numbered files 1.txt, 2.txt, ...
    A manifest is maintained to keep numbering stable across runs and skipping already processed files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_path / "manifest.json"
    try:
        import json as _json
    except Exception:
        _json = None

    manifest = {}
    if _json and manifest_path.exists():
        try:
            manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    used_numbers = set(manifest.values()) if manifest else set()
    for txt_file in output_path.glob("*.txt"):
        if txt_file.stem.isdigit():
            used_numbers.add(int(txt_file.stem))
    current_max = max(used_numbers) if used_numbers else 0

    # Collect PDFs case-insensitively
    def collect_pdfs(root: Path) -> list[Path]:
        return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]

    root_path = Path(judgements_dir)
    pdf_files = collect_pdfs(root_path) if root_path.exists() else []
    if not pdf_files and judgements_dir == "judgements":
        # fallback to US spelling if UK spelling missing or empty
        alt_root = Path("judgments")
        if alt_root.exists():
            pdf_files = collect_pdfs(alt_root)

    pdf_files = sorted(pdf_files, key=lambda p: str(p).lower())

    if not pdf_files:
        return

    total = len(pdf_files)
    with tqdm(total=total, desc="Converting PDFs", unit="file") as pbar:
        for pdf in pdf_files:
            pdf_key = str(pdf)
            assigned = manifest.get(pdf_key) if manifest else None
            if assigned is not None and (output_path / f"{assigned}.txt").exists():
                pbar.update(1)
                continue

            current_max += 1
            target_txt = output_path / f"{current_max}.txt"
            if target_txt.exists():
                # find the next free slot
                n = current_max
                while (output_path / f"{n}.txt").exists():
                    n += 1
                current_max = n
                target_txt = output_path / f"{current_max}.txt"

            content = extract_text_from_pdf(str(pdf))
            target_txt.write_text(content, encoding="utf-8")

            if _json:
                manifest[pdf_key] = current_max
                manifest_path.write_text(_json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            pbar.update(1)
    # AI-related metadata functions moved to ai.py
