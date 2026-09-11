import os
from pathlib import Path

from dotenv import load_dotenv


# =========================
# Project root directory
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent


# =========================
# Load .env
# =========================

load_dotenv(BASE_DIR / ".env")


# =========================
# Mistral / LLM Configuration
# =========================

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

MISTRAL_MODEL = os.getenv(
    "MISTRAL_MODEL",
    "ministral-3b-2512",
)


# =========================
# SQLite Configuration
# =========================

# SQLite database file will be created automatically
# in the project root.
DATABASE_URL = "sqlite:///./aws_agent.db"


# =========================
# ChromaDB Configuration
# =========================

CHROMA_PATH = str(
    BASE_DIR / "chroma_db"
)


# =========================
# AWS Configuration
# =========================

AWS_SESSION_DURATION = int(
    os.getenv(
        "AWS_SESSION_DURATION",
        "3600",
    )
)