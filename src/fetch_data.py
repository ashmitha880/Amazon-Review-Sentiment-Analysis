"""
Calls the Amazon Reviews API and saves each page as raw JSON into
data/raw/. Imported by notebooks/01_data_acquisition.ipynb.
"""

import json
import time
import requests

from src.config import RAW_DATA_DIR, AMAZON_ASIN, AMAZON_COOKIE, RAPIDAPI_KEY, API_URL


def fetch_page(cursor: str = None, country: str = "IN"):
    params = {
        "asin": AMAZON_ASIN,
        "cookie": AMAZON_COOKIE,
        "country": country,
        "sort_by": "TOP_REVIEWS",
    }
    if cursor:
        params["cursor"] = cursor

    headers = {"X-RapidAPI-Key": RAPIDAPI_KEY}
    response = requests.get(API_URL, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_all_pages(num_pages: int = 30, country: str = "IN", delay: float = 1.0):
    """Fetches up to `num_pages` pages, saving each as raw_data/page_NNN.json.
    Stops early if the API runs out of pages (no more cursor)."""
    assert AMAZON_ASIN and AMAZON_COOKIE and RAPIDAPI_KEY, (
        "Missing credentials. Set AMAZON_ASIN, AMAZON_COOKIE, RAPIDAPI_KEY "
        "in your .env file."
    )

    cursor = None
    saved_paths = []

    for page_num in range(1, num_pages + 1):
        print(f"Fetching page {page_num}/{num_pages}...")
        payload = fetch_page(cursor, country=country)

        out_path = RAW_DATA_DIR / f"page_{page_num:03d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        saved_paths.append(out_path)

        cursor = payload.get("data", {}).get("cursor")
        if not cursor:
            print("No more pages available — stopping early.")
            break

        time.sleep(delay)

    print(f"Done. Saved {len(saved_paths)} pages to {RAW_DATA_DIR}")
    return saved_paths
