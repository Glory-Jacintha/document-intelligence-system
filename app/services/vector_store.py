import chromadb
import re

from app.core.config import CHROMA_COLLECTION_NAME, CHROMA_DIR, MIN_RELEVANCE_SCORE
from app.schemas.document import DocumentChunk


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "has",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
}

TERM_EXPANSIONS = {
    "certifications": {"certification", "certificate", "trainings"},
    "completed": {"duration", "issued", "title"},
    "enhancements": {"enhancement", "improvements", "scope"},
    "future": {"planned", "scope", "enhancement", "improvements"},
    "planned": {"future", "scope", "enhancement"},
}


_collection = None


def _get_collection():
    global _collection

    if _collection is not None:
        return _collection

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def _chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index}"


def _query_terms(question: str) -> set[str]:
    terms = {
        word for word in re.findall(r"\w+", question.lower())
        if word not in STOP_WORDS
    }

    for term in list(terms):
        terms.update(TERM_EXPANSIONS.get(term, set()))

    return terms


def _keyword_score(question: str, text: str) -> float:
    terms = _query_terms(question)

    if not terms:
        return 0.0

    text_words = set(re.findall(r"\w+", text.lower()))
    return len(terms & text_words) / len(terms)


def _heading_score(question: str, text: str) -> float:
    terms = _query_terms(question)

    if not terms:
        return 0.0

    heading_candidates = re.findall(
        r"(?:^|\s)(?:\d+(?:\.\d+)*\.?\s+)?([A-Z][A-Z0-9 &/():-]{8,})",
        text,
    )

    if not heading_candidates:
        return 0.0

    best_score = 0.0

    for heading in heading_candidates:
        heading_words = set(re.findall(r"\w+", heading.lower()))
        score = len(terms & heading_words) / len(terms)
        best_score = max(best_score, score)

    return best_score


def _combined_score(question: str, text: str, vector_score: float = 0.0) -> float:
    keyword_score = _keyword_score(question, text)
    heading_score = _heading_score(question, text)
    return (heading_score * 0.45) + (keyword_score * 0.35) + (vector_score * 0.20)


def index_chunks(document_id: str, chunks: list[DocumentChunk]) -> int:
    """Replace a document's chunks in the vector store with new ones."""
    collection = _get_collection()
    collection.delete(where={"document_id": document_id})

    if not chunks:
        return 0

    collection.add(
        ids=[_chunk_id(document_id, chunk.chunk_index) for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {"document_id": chunk.document_id, "chunk_index": chunk.chunk_index}
            for chunk in chunks
        ],
    )
    return len(chunks)


def delete_document(document_id: str) -> None:
    collection = _get_collection()
    collection.delete(where={"document_id": document_id})


def query_similar_chunks(
    question: str,
    top_k: int,
    filenames_by_document_id: dict[str, str],
    document_id: str | None = None,
) -> list[dict]:
    collection = _get_collection()
    available = collection.count()

    if available == 0:
        return []

    candidate_count = min(max(top_k * 5, 20), available)
    where_filter = {"document_id": document_id} if document_id else None
    query_kwargs = {
        "query_texts": [question],
        "n_results": candidate_count,
    }

    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    matches_by_key = {}

    for _, text, metadata, distance in zip(ids, documents, metadatas, distances):
        vector_score = 1 - distance
        score = _combined_score(question, text, vector_score)

        if score < MIN_RELEVANCE_SCORE:
            continue

        document_id = metadata["document_id"]
        key = (document_id, metadata["chunk_index"])
        matches_by_key[key] = {
            "document_id": document_id,
            "filename": filenames_by_document_id.get(document_id, "unknown"),
            "chunk_index": metadata["chunk_index"],
            "score": score,
            "text": text,
        }

    get_kwargs = {"include": ["documents", "metadatas"]}

    if where_filter:
        get_kwargs["where"] = where_filter

    stored_chunks = collection.get(**get_kwargs)

    for text, metadata in zip(stored_chunks["documents"], stored_chunks["metadatas"]):
        score = _combined_score(question, text)

        if score < MIN_RELEVANCE_SCORE:
            continue

        document_id = metadata["document_id"]
        key = (document_id, metadata["chunk_index"])
        existing_match = matches_by_key.get(key)

        if existing_match is not None and existing_match["score"] >= score:
            continue

        matches_by_key[key] = {
            "document_id": document_id,
            "filename": filenames_by_document_id.get(document_id, "unknown"),
            "chunk_index": metadata["chunk_index"],
            "score": score,
            "text": text,
        }

    matches = list(matches_by_key.values())
    matches.sort(key=lambda match: match["score"], reverse=True)
    return matches[:top_k]
