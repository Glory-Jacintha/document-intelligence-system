from pathlib import Path
import re

from pypdf import PdfReader


def clean_text(text: str) -> str:
    """
    Preserve headings and paragraphs while removing noisy whitespace.
    """

    text = text.replace("\r", "")

    # Remove trailing spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove 3+ blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf_text(file_path: str) -> tuple[str, int]:

    reader = PdfReader(Path(file_path))

    pages = []

    for page_number, page in enumerate(reader.pages):

        page_text = page.extract_text()

        if not page_text:
            continue

        page_text = clean_text(page_text)

        pages.append(page_text)

    full_text = "\n\n".join(pages)

    print("=" * 80)
    print("EXTRACTED PDF")
    print("=" * 80)
    print(full_text[:3000])
    print("=" * 80)

    return full_text, len(reader.pages)