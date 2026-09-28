"""
Preprocessing utilities: cleaning text, filtering to English, mapping star
ratings to sentiment labels, and balancing classes. Imported by
notebooks/02_eda_preprocessing.ipynb (and reused by the app for consistent
cleaning at inference time).
"""

import re
import pandas as pd
from sklearn.utils import resample

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


def star_to_sentiment(stars: int) -> str:
    if stars <= 2:
        return "negative"
    elif stars == 3:
        return "neutral"
    return "positive"


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_english(text: str) -> bool:
    if not LANGDETECT_AVAILABLE or not text.strip():
        return True
    try:
        return detect(text) == "en"
    except Exception:
        return False


def balance_classes(df: pd.DataFrame, label_col: str = "sentiment") -> pd.DataFrame:
    groups = [g for _, g in df.groupby(label_col)]
    min_size = min(len(g) for g in groups)
    balanced = [
        resample(g, replace=False, n_samples=min_size, random_state=42)
        for g in groups
    ]
    return pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)


def build_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Raw parsed DataFrame -> cleaned, labeled, English-only DataFrame."""
    df = raw_df.copy()
    df["full_text"] = (df["title"] + " " + df["comment"]).str.strip()
    df = df[df["full_text"].str.len() > 5]

    print(f"{len(df)} reviews before language filtering")
    df = df[df["full_text"].apply(is_english)]
    print(f"{len(df)} reviews remain after filtering to English")

    df["clean_text"] = df["full_text"].apply(clean_text)
    df["sentiment"] = df["star_rating"].apply(star_to_sentiment)

    return df
