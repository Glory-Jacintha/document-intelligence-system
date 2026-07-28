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
            step = chunk_size - overlap

            # Avoid emitting a tiny trailing chunk that is almost entirely
            # a duplicate of the previous window's overlap region. If the
            # remaining new content after the next step would be smaller
            # than half a step, just extend the current window to the end
            # of the paragraph instead of starting another window.
            min_new_content = max(1, step // 2)

            while start < len(paragraph):

                end = min(start + chunk_size, len(paragraph))

                if end == len(paragraph):
                    chunks.append(paragraph[start:end])
                    break

                remaining_new_content = len(paragraph) - (start + step)

                if remaining_new_content < min_new_content:
                    chunks.append(paragraph[start:])
                    break

                chunks.append(paragraph[start:end])
                start += step

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