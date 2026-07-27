import re


def _split_sentences(text: str) -> list[str]:
    cleaned_text = " ".join(text.split())
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned_text)
        if sentence.strip()
    ]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = _split_sentences(text)

    if not sentences:
        return []

    chunks = []
    current_sentences = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)

        if current_sentences and current_size + sentence_size + 1 > chunk_size:
            chunks.append(" ".join(current_sentences))

            overlap_sentences = []
            overlap_size = 0

            for previous_sentence in reversed(current_sentences):
                previous_size = len(previous_sentence)

                if overlap_sentences and overlap_size + previous_size + 1 > overlap:
                    break

                overlap_sentences.insert(0, previous_sentence)
                overlap_size += previous_size + 1

            current_sentences = overlap_sentences
            current_size = overlap_size

        current_sentences.append(sentence)
        current_size += sentence_size + 1

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
