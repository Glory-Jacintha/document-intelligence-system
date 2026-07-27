import os
from pathlib import Path
from uuid import uuid4

import streamlit as st


def _load_streamlit_secrets() -> None:
    try:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        gemini_api_key = None

    if gemini_api_key and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = gemini_api_key


_load_streamlit_secrets()

from app.api.routes.documents import ask_documents, process_document_for_search
from app.core.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES, UPLOAD_DIR
from app.schemas.document import DocumentAnswerRequest, DocumentMetadata
from app.services.chunk_store import delete_document_chunks
from app.services.document_store import (
    add_document,
    delete_document_metadata,
    load_documents,
)
from app.services.vector_store import delete_document as delete_document_vectors


st.set_page_config(
    page_title="Document Intelligence System",
    page_icon="📄",
    layout="wide",
)


def _save_uploaded_pdf(uploaded_file) -> DocumentMetadata:
    file_extension = Path(uploaded_file.name or "").suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Only PDF files are allowed.")

    file_content = uploaded_file.getvalue()

    if len(file_content) > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError("File size must be 20 MB or smaller.")

    document_id = str(uuid4())
    saved_filename = f"{document_id}{file_extension}"
    upload_path = UPLOAD_DIR / saved_filename

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(file_content)

    document = DocumentMetadata(
        document_id=document_id,
        filename=uploaded_file.name or saved_filename,
        content_type=uploaded_file.type or "application/pdf",
        size_bytes=len(file_content),
        storage_path=str(upload_path),
    )

    return add_document(document)


def _delete_document(document_id: str) -> None:
    document = delete_document_metadata(document_id)

    if document is None:
        return

    storage_path = Path(document.storage_path)

    if storage_path.exists():
        storage_path.unlink()

    delete_document_chunks(document.document_id)
    delete_document_vectors(document.document_id)


def _document_options() -> dict[str, str | None]:
    documents = load_documents()
    options = {"All documents": None}

    for document in documents:
        label = f"{document.filename} ({document.document_id[:8]})"
        options[label] = document.document_id

    return options


st.title("Document Intelligence System")

with st.sidebar:
    st.header("Documents")

    uploaded_file = st.file_uploader("Upload and process a PDF", type=["pdf"])

    if uploaded_file and st.button("Upload and process", type="primary"):
        try:
            with st.spinner("Uploading and processing document..."):
                document = _save_uploaded_pdf(uploaded_file)
                processing = process_document_for_search(document.document_id)
            st.success(
                f"Processed {document.filename}: "
                f"{processing.chunk_count} chunks indexed."
            )
            st.rerun()
        except Exception as error:
            st.error(str(error))

    st.divider()

    documents = load_documents()

    if documents:
        for document in documents:
            with st.expander(document.filename):
                st.caption(document.document_id)
                st.write(f"Size: {document.size_bytes:,} bytes")

                if st.button(
                    "Delete",
                    key=f"delete-{document.document_id}",
                    type="secondary",
                ):
                    _delete_document(document.document_id)
                    st.success(f"Deleted {document.filename}")
                    st.rerun()
    else:
        st.info("No documents uploaded yet.")


document_options = _document_options()
selected_label = st.selectbox(
    "Search scope",
    options=list(document_options.keys()),
)

question = st.text_area(
    "Ask a question",
    placeholder="Example: What future enhancements are planned for Data Spectra?",
)
top_k = st.slider("Number of source chunks", min_value=1, max_value=10, value=3)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Thinking..."):
                response = ask_documents(
                    DocumentAnswerRequest(
                        question=question.strip(),
                        top_k=top_k,
                        document_id=document_options[selected_label],
                    )
                )

            st.subheader("Answer")
            st.write(response.answer)

            if response.sources:
                st.subheader("Sources")
                st.dataframe(
                    [
                        {
                            "filename": source.filename,
                            "document_id": source.document_id,
                            "chunk_index": source.chunk_index,
                            "score": round(source.score, 3),
                        }
                        for source in response.sources
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
        except Exception as error:
            st.error(str(error))
