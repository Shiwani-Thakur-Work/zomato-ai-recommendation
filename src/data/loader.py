"""
loader.py
---------
Handles downloading and caching the Zomato dataset from Hugging Face.

The dataset is fetched once and cached under DATA_RAW_DIR. Subsequent calls
reuse the local cache — no network call is made unless the cache is missing.

Public API
----------
    load_raw_dataset() -> pd.DataFrame
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW_DIR

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_REPO = "ManikaSaini/zomato-restaurant-recommendation"
DATASET_SPLIT = "train"


# ── Custom Exceptions ─────────────────────────────────────────────────────────
class DatasetLoadError(RuntimeError):
    """Raised when the Zomato dataset cannot be downloaded or loaded."""


class EmptyDatasetError(ValueError):
    """Raised when the downloaded dataset contains zero rows."""


# ── Public API ────────────────────────────────────────────────────────────────
def load_raw_dataset() -> pd.DataFrame:
    """Download the Zomato dataset from Hugging Face and return as a DataFrame.

    On the first call the dataset is streamed and cached to DATA_RAW_DIR.
    On subsequent calls the local cache is used directly.

    Returns:
        pd.DataFrame: Raw dataset with original column names intact.

    Raises:
        DatasetLoadError: If the network is unavailable or the repo is not found.
        EmptyDatasetError: If the downloaded dataset contains zero rows.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Import here so the rest of the module is usable even when `datasets`
        # is not installed (e.g. during isolated unit tests with a mock).
        from datasets import load_dataset  # type: ignore[import]

        logger.info("Loading dataset '%s' (split=%s) …", DATASET_REPO, DATASET_SPLIT)
        dataset = load_dataset(
            DATASET_REPO,
            split=DATASET_SPLIT,
            cache_dir=str(DATA_RAW_DIR),
            trust_remote_code=False,
        )
        df: pd.DataFrame = dataset.to_pandas()  # type: ignore[union-attr]

    except Exception as exc:
        raise DatasetLoadError(
            f"Failed to download the Zomato dataset from Hugging Face.\n"
            f"Repo: {DATASET_REPO!r}\n"
            f"Cache dir: {DATA_RAW_DIR}\n"
            f"Original error: {exc}"
        ) from exc

    if df.empty:
        raise EmptyDatasetError(
            f"The downloaded dataset '{DATASET_REPO}' is empty (0 rows). "
            "Please check the dataset on Hugging Face."
        )

    logger.info("Raw dataset loaded: %d rows, %d columns.", len(df), len(df.columns))
    return df
