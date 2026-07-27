from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    content_type: str
    size_bytes: int
    storage_path: str


class DocumentText(BaseModel):
    document_id: str
    filename: str
    page_count: int
    text: str


class DocumentChunk(BaseModel):
    document_id: str
    chunk_index: int
    text: str


class DocumentChunks(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    chunks: list[DocumentChunk]


class DocumentSearchRequest(BaseModel):
    question: str
    top_k: int = 3


class DocumentSearchResult(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text: str


class DocumentSearchResponse(BaseModel):
    question: str
    result_count: int
    results: list[DocumentSearchResult]


class DocumentAnswerRequest(BaseModel):
    question: str
    top_k: int = 3
    document_id: str | None = None


class DocumentAnswerSource(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float


class DocumentAnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[DocumentAnswerSource]


class DocumentProcessingResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    embedding_count: int
    message: str


class DocumentUploadProcessingResponse(BaseModel):
    document: DocumentMetadata
    processing: DocumentProcessingResponse


class DocumentDeleteResponse(BaseModel):
    document_id: str
    filename: str
    deleted_file: bool
    deleted_chunks: int
    deleted_from_vector_store: bool
    message: str
