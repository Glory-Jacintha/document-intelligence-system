import re

from google import genai

from app.core.config import GEMINI_API_KEY, GEMINI_MODEL
from app.schemas.document import DocumentSearchResult
from app.services.vector_store import STOP_WORDS


SNIPPET_WINDOW_CHARS = 300


def _find_keyword_position(question: str, text: str) -> int | None:
    query_words = [
        word for word in re.findall(r"\w+", question.lower())
        if word not in STOP_WORDS
    ]
    lower_text = text.lower()

    positions = [
        position
        for position in (lower_text.find(word) for word in query_words)
        if position != -1
    ]

    return min(positions) if positions else None


def _build_answer_snippet(question: str, top_source: DocumentSearchResult) -> str:
    text = top_source.text.strip()
    keyword_position = _find_keyword_position(question, text)

    start = 0 if keyword_position is None else max(0, keyword_position - SNIPPET_WINDOW_CHARS)
    end = start + (SNIPPET_WINDOW_CHARS * 2)

    snippet = text[start:end].strip()

    if start > 0:
        snippet = f"...{snippet}"

    if end < len(text):
        snippet = snippet.rsplit(" ", 1)[0] + "..."

    return snippet


def _build_sources_list(sources: list[DocumentSearchResult]) -> str:
    return "\n".join(
        f"- {source.filename}, chunk {source.chunk_index} (score: {source.score:.2f})"
        for source in sources
    )


def create_extractive_answer(question: str, sources: list[DocumentSearchResult]) -> str:
    if not sources:
        return "I could not find relevant information in the indexed documents."

    return _build_answer_snippet(question, sources[0])


def _build_context(sources: list[DocumentSearchResult]) -> str:
    return "\n\n".join(
        (
            f"Source {index + 1}\n"
            f"File: {source.filename}\n"
            f"Chunk: {source.chunk_index}\n"
            f"Text: {source.text}"
        )
        for index, source in enumerate(sources)
    )


def create_gemini_answer(question: str, sources: list[DocumentSearchResult]) -> str:
    if not sources:
        return "I could not find relevant information in the indexed documents."

    if not GEMINI_API_KEY:
        return create_extractive_answer(question, sources)

    prompt = f"""
You are a document question-answering assistant.
Answer the user's question using only the provided document sources.
If the sources do not contain the answer, say: "I could not find that information in the provided documents."
Keep the answer concise and factual.
Return only the final answer text.
Do not include source labels, chunk text, JSON, markdown tables, or a "Sources" section.

Question:
{question}

Document sources:
{_build_context(sources)}
"""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
    except Exception:
        return create_extractive_answer(question, sources)

    if not response.text:
        return create_extractive_answer(question, sources)

    return response.text.strip()
