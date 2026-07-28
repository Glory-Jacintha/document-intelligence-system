import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DOCUMENT_STORE_PATH = DATA_DIR / "documents.json"
CHUNK_STORE_PATH = DATA_DIR / "chunks.json"
EMBEDDING_STORE_PATH = DATA_DIR / "embeddings.json"
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_COLLECTION_NAME = "document_chunks"

ALLOWED_EXTENSIONS = {".pdf"}
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 200
EMBEDDING_DIMENSIONS = 384

# Minimum score for a chunk to be returned from ChromaDB at all
MIN_RELEVANCE_SCORE = 0.15

# Minimum score for a chunk to be included in the LLM answer context
# (chunks below this fall back to top-1 only)
ANSWER_SOURCE_MIN_SCORE = 0.3

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
