import re
from typing import List


class TextChunker:
    def __init__(self,
                 chunk_size: int = 700,
                 overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:

        if not text:
            return []

        # Normalize whitespace
        text = text.replace("\r", "")
        text = re.sub(r"[ \t]+", " ", text)

        # Split into paragraphs while preserving document structure
        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if p.strip()
        ]

        chunks = []
        current = ""

        for paragraph in paragraphs:

            # Extremely long paragraph
            if len(paragraph) > self.chunk_size:

                if current:
                    chunks.append(current.strip())
                    current = ""

                start = 0

                while start < len(paragraph):

                    end = start + self.chunk_size

                    chunks.append(paragraph[start:end])

                    start += self.chunk_size - self.overlap

                continue

            # Normal paragraph

            if len(current) + len(paragraph) < self.chunk_size:

                current += "\n\n" + paragraph

            else:

                chunks.append(current.strip())

                current = paragraph

        if current:
            chunks.append(current.strip())

        # Remove tiny chunks

        chunks = [
            chunk
            for chunk in chunks
            if len(chunk.strip()) > 40
        ]

        return chunks