import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DOCUMENT_STORE_PATH = DATA_DIR / "documents.json"
CHUNK_STORE_PATH = DATA_DIR / "chunks.json"
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_COLLECTION_NAME = "document_chunks"

ALLOWED_EXTENSIONS = {".pdf"}
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 200

# Matches below this similarity score are treated as "not confident enough"
# and dropped from search/ask results, instead of always returning top_k
# regardless of quality.
MIN_RELEVANCE_SCORE = 0.3
ANSWER_SOURCE_MIN_SCORE = 0.4

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
