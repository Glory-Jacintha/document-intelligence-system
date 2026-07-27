from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(file_path: str) -> tuple[str, int]:
    reader = PdfReader(Path(file_path))
    page_texts = []

    for page in reader.pages:
        page_texts.append(page.extract_text() or "")

    return "\n\n".join(page_texts).strip(), len(reader.pages)
