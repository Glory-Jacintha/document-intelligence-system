from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import (
    ALLOWED_EXTENSIONS,
    ANSWER_SOURCE_MIN_SCORE,
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    MAX_UPLOAD_SIZE_BYTES,
    UPLOAD_DIR,
)
from app.schemas.document import (
    DocumentAnswerRequest,
    DocumentAnswerResponse,
    DocumentAnswerSource,
    DocumentChunk,
    DocumentChunks,
    DocumentDeleteResponse,
    DocumentMetadata,
    DocumentProcessingResponse,
    DocumentSearchResult,
    DocumentText,
    DocumentUploadProcessingResponse,
)
from app.services.answer_service import create_gemini_answer
from app.services.chunk_store import (
    delete_document_chunks,
    get_document_chunks,
    replace_document_chunks,
)
from app.services.document_store import (
    add_document,
    delete_document_metadata,
    get_document,
    load_documents,
)
from app.services.pdf_extractor import extract_pdf_text
from app.services.text_chunker import chunk_text
from app.services.vector_store import (
    delete_document as delete_document_vectors,
    index_chunks,
    query_similar_chunks,
)


router = APIRouter(prefix="/documents", tags=["documents"])


def search_indexed_chunks(
    question: str,
    top_k: int,
    document_id: str | None = None,
) -> list[DocumentSearchResult]:
    documents_by_id = {
        document.document_id: document.filename
        for document in load_documents()
    }

    if not documents_by_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload and process at least one document before searching.",
        )

    if document_id and document_id not in documents_by_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    matches = query_similar_chunks(
        question,
        max(1, top_k),
        documents_by_id,
        document_id=document_id,
    )

    return [DocumentSearchResult(**match) for match in matches]


def expand_answer_context(sources: list[DocumentSearchResult]) -> list[DocumentSearchResult]:
    if not sources:
        return []

    selected_sources = [
        source for source in sources
        if source.score >= ANSWER_SOURCE_MIN_SCORE
    ]

    if not selected_sources:
        selected_sources = [sources[0]]

    expanded_sources = []
    seen_keys = set()

    for source in selected_sources:
        document_chunks = {
            chunk.chunk_index: chunk
            for chunk in get_document_chunks(source.document_id)
        }
        neighbor_window = 4 if "scope for future" in source.text.lower() else 1

        for chunk_index in range(
            source.chunk_index,
            source.chunk_index + neighbor_window + 1,
        ):
            chunk = document_chunks.get(chunk_index)

            if chunk is None:
                continue

            key = (chunk.document_id, chunk.chunk_index)

            if key in seen_keys:
                continue

            seen_keys.add(key)
            expanded_sources.append(
                DocumentSearchResult(
                    document_id=source.document_id,
                    filename=source.filename,
                    chunk_index=chunk.chunk_index,
                    score=source.score,
                    text=chunk.text,
                )
            )

    return expanded_sources


def process_document_for_search(document_id: str) -> DocumentProcessingResponse:
    document = get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    text, _ = extract_pdf_text(document.storage_path)
    chunk_texts = chunk_text(text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
    chunks = [
        DocumentChunk(document_id=document.document_id, chunk_index=index, text=chunk)
        for index, chunk in enumerate(chunk_texts)
    ]
    saved_chunks = replace_document_chunks(document.document_id, chunks)
    indexed_count = index_chunks(document.document_id, saved_chunks)

    return DocumentProcessingResponse(
        document_id=document.document_id,
        filename=document.filename,
        chunk_count=len(saved_chunks),
        embedding_count=indexed_count,
        message="Document processed successfully.",
    )


async def save_uploaded_pdf(file: UploadFile) -> DocumentMetadata:
    file_extension = Path(file.filename or "").suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed.",
        )

    file_content = await file.read()

    if len(file_content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size must be 20 MB or smaller.",
        )

    document_id = str(uuid4())
    saved_filename = f"{document_id}{file_extension}"
    upload_path = UPLOAD_DIR / saved_filename

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(file_content)

    document = DocumentMetadata(
        document_id=document_id,
        filename=file.filename or saved_filename,
        content_type=file.content_type or "application/pdf",
        size_bytes=len(file_content),
        storage_path=str(upload_path),
    )

    return add_document(document)


@router.get("", response_model=list[DocumentMetadata])
def list_documents() -> list[DocumentMetadata]:
    return load_documents()



@router.post(
    "/upload-and-process",
    response_model=DocumentUploadProcessingResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_and_process_document(
    file: UploadFile = File(...),
) -> DocumentUploadProcessingResponse:
    document = await save_uploaded_pdf(file)
    processing = process_document_for_search(document.document_id)

    return DocumentUploadProcessingResponse(
        document=document,
        processing=processing,
    )



@router.get("/{document_id}", response_model=DocumentMetadata)
def read_document(document_id: str) -> DocumentMetadata:
    document = get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return document



@router.get("/{document_id}/text", response_model=DocumentText)
def read_document_text(document_id: str) -> DocumentText:
    document = get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    text, page_count = extract_pdf_text(document.storage_path)

    return DocumentText(
        document_id=document.document_id,
        filename=document.filename,
        page_count=page_count,
        text=text,
    )



@router.get("/{document_id}/chunks", response_model=DocumentChunks)
def read_document_chunks(document_id: str) -> DocumentChunks:
    document = get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    chunks = get_document_chunks(document.document_id)

    return DocumentChunks(
        document_id=document.document_id,
        filename=document.filename,
        chunk_count=len(chunks),
        chunks=chunks,
    )


@router.post("/ask", response_model=DocumentAnswerResponse)
def ask_documents(request: DocumentAnswerRequest) -> DocumentAnswerResponse:
    sources = search_indexed_chunks(
        request.question,
        request.top_k,
        document_id=request.document_id,
    )
    answer_sources_for_model = expand_answer_context(sources)
    answer = create_gemini_answer(request.question, answer_sources_for_model)
    answer_sources = [
        DocumentAnswerSource(
            document_id=source.document_id,
            filename=source.filename,
            chunk_index=source.chunk_index,
            score=source.score,
        )
        for source in answer_sources_for_model
    ]

    return DocumentAnswerResponse(
        question=request.question,
        answer=answer,
        sources=answer_sources,
    )



@router.delete("/{document_id}/delete", response_model=DocumentDeleteResponse)
def delete_document(document_id: str) -> DocumentDeleteResponse:
    document = delete_document_metadata(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    storage_path = Path(document.storage_path)
    deleted_file = False

    if storage_path.exists():
        storage_path.unlink()
        deleted_file = True

    deleted_chunks = delete_document_chunks(document.document_id)
    delete_document_vectors(document.document_id)

    return DocumentDeleteResponse(
        document_id=document.document_id,
        filename=document.filename,
        deleted_file=deleted_file,
        deleted_chunks=deleted_chunks,
        deleted_from_vector_store=True,
        message="Document deleted successfully.",
    )
