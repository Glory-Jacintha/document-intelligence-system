import chromadb

from app.core.config import CHROMA_COLLECTION_NAME, CHROMA_DIR, MIN_RELEVANCE_SCORE
from app.schemas.document import DocumentChunk


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

    where_filter = {"document_id": document_id} if document_id else None

    results = collection.query(
        query_texts=[question],
        n_results=min(max(1, top_k), available),
        where=where_filter,
    )

    print("\n========== RETRIEVED CHUNKS ==========")

    for i, doc in enumerate(results["documents"][0]):
        print(f"\nChunk {i+1}")
        print(doc)
        print("-" * 60)

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    matches = []

    for _, text, metadata, distance in zip(ids, documents, metadatas, distances):
        score = 1 - distance

        doc_id = metadata["document_id"]
        matches.append(
            {
                "document_id": doc_id,
                "filename": filenames_by_document_id.get(doc_id, "unknown"),
                "chunk_index": metadata["chunk_index"],
                "score": score,
                "text": text,
            }
        )

    return matches
