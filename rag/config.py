"""App settings: models, paths, retrieval knobs, and the OpenAI API key."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "app.db"

load_dotenv(ROOT / ".env")

# Models the user can pick in the sidebar (API id -> label). The first one is the default.
MODELS = {
    "gpt-6-luna": "GPT-6 Luna — fast, cheapest",
    "gpt-6.1-sol": "GPT-6.1 Sol — better answers",
}
DEFAULT_MODEL = "gpt-6-luna"

# Model that reads scanned pages and images (OCR).
OCR_MODEL = "gpt-6-luna"

# Retrieval settings for OpenAI file search.
MAX_SEARCH_RESULTS = 8
# 0..1. Higher = only very relevant passages (more "No information"); lower = looser matches.
SCORE_THRESHOLD = 0.4

# How many earlier chat messages to send along so follow-up questions make sense.
HISTORY_MESSAGES = 12

NO_INFO = "No information"

UPLOAD_TYPES = ["pdf", "docx", "txt", "md", "png", "jpg", "jpeg"]


def api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY") or None
