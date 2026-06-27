import pandas as pd
from langdetect import detect, DetectorFactory

from src.config import DATA_PATH, TARGET_LANGUAGE, REVIEWS_COL, LABEL_COL

DetectorFactory.seed = 0


def safe_detect(text: str) -> str:
    """Detect the language of text, returning 'unknown' on any exception."""
    try:
        return detect(text)
    except Exception:
        return "unknown"


def load_data(path) -> pd.DataFrame:
    """Read the CSV at path into a DataFrame."""
    return pd.read_csv(path)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully duplicate rows from df."""
    return df.drop_duplicates()


def filter_dutch(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose Reviews column is detected as TARGET_LANGUAGE."""
    mask = df[REVIEWS_COL].apply(safe_detect) == TARGET_LANGUAGE
    return df[mask]


def get_data(path=DATA_PATH) -> tuple[pd.Series, pd.Series]:
    """Load, deduplicate, and filter to Dutch reviews; return (X, y)."""
    df = load_data(path)
    df = drop_duplicates(df)
    df = filter_dutch(df)
    return df[REVIEWS_COL], df[LABEL_COL]
