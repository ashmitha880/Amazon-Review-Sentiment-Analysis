"""
Shared configuration: paths and constants used across notebooks, src modules,
and the Streamlit app. Credentials are read from environment variables (set
in your .env file, loaded via python-dotenv) — never hardcoded here.
"""

import os
from pathlib import Path

# Project root = one level up from src/
ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# API credentials — set these in your .env file, loaded via load_dotenv()
AMAZON_ASIN = os.environ.get("AMAZON_ASIN")
AMAZON_COOKIE = os.environ.get("AMAZON_COOKIE")
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
API_URL = os.environ.get("AMAZON_REVIEWS_API_URL", "https://REPLACE-WITH-YOUR-API-HOST/reviews")

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
