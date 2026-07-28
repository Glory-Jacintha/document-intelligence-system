from fastapi import FastAPI

from app.api.routes.documents import router as documents_router
from app.services.document_seed import seed_default_documents


app = FastAPI(
    title="Document Intelligence System",
    description="Backend system for uploading and querying documents.",
    version="0.1.0",
)

app.include_router(documents_router)


@app.on_event("startup")
def _seed_documents_on_startup() -> None:
    seed_default_documents()


@app.get("/")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "project": "Document Intelligence System",
    }
