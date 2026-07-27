import json

from app.core.config import DOCUMENT_STORE_PATH
from app.schemas.document import DocumentMetadata


def load_documents() -> list[DocumentMetadata]:
    if not DOCUMENT_STORE_PATH.exists():
        return []

    with DOCUMENT_STORE_PATH.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    return [DocumentMetadata(**document) for document in documents]


def save_documents(documents: list[DocumentMetadata]) -> None:
    DOCUMENT_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with DOCUMENT_STORE_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            [document.model_dump() for document in documents],
            file,
            indent=2,
        )


def add_document(document: DocumentMetadata) -> DocumentMetadata:
    documents = load_documents()
    documents.append(document)
    save_documents(documents)
    return document


def get_document(document_id: str) -> DocumentMetadata | None:
    documents = load_documents()

    for document in documents:
        if document.document_id == document_id:
            return document

    return None


def delete_document_metadata(document_id: str) -> DocumentMetadata | None:
    documents = load_documents()
    deleted_document = None
    remaining_documents = []

    for document in documents:
        if document.document_id == document_id:
            deleted_document = document
        else:
            remaining_documents.append(document)

    if deleted_document is None:
        return None

    save_documents(remaining_documents)
    return deleted_document
