"""
Parses raw JSON pages from data/raw/ into a single flat DataFrame.
Imported by notebooks/02_eda_preprocessing.ipynb.
"""

import json
import glob
import pandas as pd

from src.config import RAW_DATA_DIR


def parse_response(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    reviews = payload.get("data", {}).get("reviews", [])
    rows = []
    for r in reviews:
        rows.append({
            "review_id": r.get("review_id"),
            "title": r.get("review_title") or "",
            "comment": r.get("review_comment") or "",
            "star_rating": int(r.get("review_star_rating", 0)),
            "verified_purchase": r.get("is_verified_purchase"),
            "date": r.get("review_date"),
            "has_images": len(r.get("review_images") or []) > 0,
        })
    return pd.DataFrame(rows)


def load_all_raw_reviews() -> pd.DataFrame:
    files = glob.glob(str(RAW_DATA_DIR / "*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON files found in {RAW_DATA_DIR}. Run notebook "
            f"01_data_acquisition.ipynb first."
        )
    dfs = [parse_response(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset="review_id")
    return df
