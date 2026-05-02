import os
from pathlib import Path

class Settings:
    BASE_DIR = Path(__file__).resolve().parent.parent

    DWANI_API_BASE_URL = os.getenv("DWANI_API_BASE_URL", "https://gemma4-api.dwani.ai/v1")
    if not DWANI_API_BASE_URL:
        raise RuntimeError("DWANI_API_BASE_URL environment variable is required.")
    DWANI_LLM_MODEL = os.getenv("DWANI_LLM_MODEL", "gemma4")
    DWANI_EMBEDDING_MODEL = os.getenv("DWANI_EMBEDDING_MODEL", "google/embeddinggemma-300m")

    FONT_PATH = BASE_DIR / "fonts" / "DejaVuSans.ttf"

    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "12000"))
    MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "3000"))

    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./files.db")

settings = Settings()