import re


def chunk_text(
    text: str,
    chunk_size: int = 700,
    overlap: int = 100,
):
    if not text:
        return []

    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        # Split oversized paragraphs
        if len(paragraph) > chunk_size:

            if current:
                chunks.append(current.strip())
                current = ""

            start = 0

            while start < len(paragraph):

                end = start + chunk_size

                chunks.append(paragraph[start:end])

                start += chunk_size - overlap

            continue

        if len(current) + len(paragraph) <= chunk_size:

            if current:
                current += "\n\n"

            current += paragraph

        else:

            chunks.append(current.strip())

            current = paragraph

    if current:
        chunks.append(current.strip())

    return [
        c
        for c in chunks
        if len(c.strip()) > 40
    ]