"""Auto-seeds default documents that should always be available to every user.

Files placed in the `seed_documents/` folder (tracked in git, unlike
`data/uploads/`) are automatically uploaded and indexed on startup if they
aren't already present. This matters because `data/documents.json`,
`data/chunks.json`, and `data/chroma/` are all gitignored, so on Streamlit
Cloud the indexed documents disappear on every redeploy/restart. Seeding
guarantees documents like the Employee Handbook come back automatically for
every user without anyone needing to manually re-upload them.
"""

from pathlib import Path
from uuid import uuid4

from app.core.config import BASE_DIR, UPLOAD_DIR
from app.schemas.document import DocumentMetadata
from app.services.document_store import add_document, load_documents

SEED_DOCUMENTS_DIR = BASE_DIR / "seed_documents"


def seed_default_documents() -> list[str]:
    """Index any PDF in seed_documents/ that isn't already indexed.

    Matching is by filename, so re-running this is a no-op once a seed file
    has been indexed. Returns the list of filenames that were newly seeded.
    """
    if not SEED_DOCUMENTS_DIR.exists():
        return []

    # Import here (not at module load time) to avoid a circular import,
    # since documents.py already imports from this module's sibling services.
    from app.api.routes.documents import process_document_for_search

    already_indexed_filenames = {
        document.filename for document in load_documents()
    }

    newly_seeded = []

    for seed_path in sorted(SEED_DOCUMENTS_DIR.glob("*.pdf")):
        if seed_path.name in already_indexed_filenames:
            continue

        file_content = seed_path.read_bytes()
        document_id = str(uuid4())
        saved_filename = f"{document_id}{seed_path.suffix.lower()}"
        upload_path = UPLOAD_DIR / saved_filename

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        upload_path.write_bytes(file_content)

        document = DocumentMetadata(
            document_id=document_id,
            filename=seed_path.name,
            content_type="application/pdf",
            size_bytes=len(file_content),
            storage_path=str(upload_path),
        )
        add_document(document)
        process_document_for_search(document.document_id)
        newly_seeded.append(seed_path.name)

    return newly_seeded
