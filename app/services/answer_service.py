import re

from app.core.config import GEMINI_API_KEY, GEMINI_MODEL
from app.schemas.document import DocumentSearchResult
from app.services.vector_store import query_similar_chunks


SNIPPET_WINDOW_CHARS = 300

STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "with", "this", "that", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "can", "need", "dare", "ought", "used", "what", "which", "who",
    "how", "when", "where", "why", "policy", "policies",
}


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

    answer_snippet = _build_answer_snippet(question, sources[0])
    sources_list = _build_sources_list(sources)

    return f"Answer:\n{answer_snippet}\n\nSources:\n{sources_list}"


def create_gemini_answer(question: str, sources: list[DocumentSearchResult]) -> str:
    if not sources:
        return "I could not find relevant information in the indexed documents."

    if not GEMINI_API_KEY:
        return create_extractive_answer(question, sources)

    try:
        from google import genai

        # Highest scoring chunks first
        sources = sorted(sources, key=lambda s: s.score, reverse=True)

        context_parts = []

        for source in sources:
            context_parts.append(
                f"""
DOCUMENT: {source.filename}
CHUNK: {source.chunk_index}
RELEVANCE SCORE: {source.score:.3f}

{source.text}
"""
            )

        context = "\n\n" + ("=" * 80 + "\n\n").join(context_parts)

        prompt = f"""
You are a Document Intelligence assistant.

Your job is to answer questions ONLY using the supplied document excerpts.

Rules:

1. Read ALL document excerpts.
2. Pay close attention to section headings such as:
   - EDUCATION
   - EXPERIENCE
   - PROJECTS
   - SKILLS
   - PROFILE
3. If the answer appears anywhere in the excerpts, answer it directly.
4. Quote names, organizations and locations exactly as written.
5. Do NOT invent information.
6. Only answer "The document does not contain that information."
   if the answer truly does not exist in ANY excerpt.

QUESTION

{question}

DOCUMENT EXCERPTS

{context}

ANSWER
"""

        print("\n========== PROMPT SENT TO GEMINI ==========\n")
        print(prompt)
        print("\n==========================================\n")

        client = genai.Client(api_key=GEMINI_API_KEY)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        if response.text:
            return response.text.strip()

        return create_extractive_answer(question, sources)

    except Exception as e:
        print("Gemini Error:", e)
        return create_extractive_answer(question, sources)