import re


def _split_sentences(text: str) -> list[str]:
    cleaned_text = " ".join(text.split())
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned_text)
        if sentence.strip()
    ]


def chunk_text(text, chunk_size=800, overlap=150):

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []

    current = ""

    for paragraph in paragraphs:

        if len(current) + len(paragraph) < chunk_size:

            current += "\n\n" + paragraph

        else:

            chunks.append(current.strip())

            current = paragraph

    if current:

        chunks.append(current)

    return chunks
