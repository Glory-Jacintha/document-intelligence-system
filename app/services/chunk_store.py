import json

from app.core.config import CHUNK_STORE_PATH
from app.schemas.document import DocumentChunk


def load_chunks() -> list[DocumentChunk]:
    if not CHUNK_STORE_PATH.exists():
        return []

    with CHUNK_STORE_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    return [DocumentChunk(**chunk) for chunk in chunks]


def save_chunks(chunks: list[DocumentChunk]) -> None:
    CHUNK_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CHUNK_STORE_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            [chunk.model_dump() for chunk in chunks],
            file,
            indent=2,
        )


def replace_document_chunks(document_id: str, document_chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    other_chunks = [
        chunk for chunk in load_chunks()
        if chunk.document_id != document_id
    ]
    save_chunks(other_chunks + document_chunks)
    return document_chunks


def get_document_chunks(document_id: str) -> list[DocumentChunk]:
    return [
        chunk for chunk in load_chunks()
        if chunk.document_id == document_id
    ]


def delete_document_chunks(document_id: str) -> int:
    chunks = load_chunks()
    remaining_chunks = [
        chunk for chunk in chunks
        if chunk.document_id != document_id
    ]
    deleted_count = len(chunks) - len(remaining_chunks)
    save_chunks(remaining_chunks)
    return deleted_count
