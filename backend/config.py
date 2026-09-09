import os
from pathlib import Path

from dotenv import load_dotenv


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from the project root
load_dotenv(BASE_DIR / ".env")


# =========================
# GROQ / LLM Configuration
# =========================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "meta-llama/llama-4-scout",
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