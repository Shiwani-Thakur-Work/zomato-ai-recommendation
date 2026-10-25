"""
config.py
---------
Centralised configuration loader for the Zomato AI Recommendation System.

All environment variables are loaded from a .env file (via python-dotenv).
Import from this module instead of calling os.getenv() directly anywhere else.

Usage:
    from src.config import GROQ_API_KEY, DATA_PROCESSED_PATH, MAX_CANDIDATES_FOR_LLM
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env from project root ───────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


# ── Groq LLM Provider ─────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Also check Streamlit Cloud secrets fallback
if not GROQ_API_KEY:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            GROQ_API_KEY = str(st.secrets["GROQ_API_KEY"])
        if hasattr(st, "secrets") and "GROQ_MODEL" in st.secrets:
            GROQ_MODEL = str(st.secrets["GROQ_MODEL"])
    except Exception:
        pass


# ── Data Paths ────────────────────────────────────────────────────────────────
DATA_RAW_DIR: Path = BASE_DIR / os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_PATH: Path = BASE_DIR / os.getenv(
    "DATA_PROCESSED_PATH", "data/processed/zomato_cleaned.parquet"
)


# ── Recommendation Engine Defaults ────────────────────────────────────────────
DEFAULT_MAX_RESULTS: int = int(os.getenv("DEFAULT_MAX_RESULTS", "5"))
DEFAULT_MIN_RATING: float = float(os.getenv("DEFAULT_MIN_RATING", "3.5"))
MAX_CANDIDATES_FOR_LLM: int = int(os.getenv("MAX_CANDIDATES_FOR_LLM", "15"))


# ── Startup Validation ────────────────────────────────────────────────────────
def validate_config() -> None:
    """
    Raise early with a clear message if critical configuration is missing.
    Call this once at application startup (e.g., top of app.py or main.py).

    Raises:
        EnvironmentError: If GROQ_API_KEY is not set.
    """
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Copy .env.example to .env and add your Groq API key. "
            "Get one at https://console.groq.com/keys"
        )
