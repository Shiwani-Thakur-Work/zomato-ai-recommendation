"""
preprocessor.py
---------------
ETL pipeline: cleans, normalises, and persists the Zomato dataset as Parquet.

The main entry point is build_processed_dataset(), which:
  1. Returns the cached Parquet if it already exists on disk.
  2. Otherwise, downloads the raw dataset, applies all transformations,
     and saves the result to DATA_PROCESSED_PATH.

Edge-case handling follows the spec in docs/edge-cases.md §1.

Public API
----------
    build_processed_dataset() -> pd.DataFrame
    clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_PROCESSED_PATH
from src.data.loader import DatasetLoadError, EmptyDatasetError, load_raw_dataset

logger = logging.getLogger(__name__)


# ── Column alias map ──────────────────────────────────────────────────────────
# Maps known raw column names (from different HuggingFace dataset versions) to
# the canonical internal names used throughout the project.
COLUMN_ALIASES: dict[str, str] = {
    # Restaurant identity
    "Restaurant Name": "name",
    "restaurant_name": "name",
    "Name": "name",
    "name": "name",
    # Location
    "City": "city",
    "city": "city",
    "Location": "locality",
    "location": "locality",
    "Locality": "locality",
    "locality": "locality",
    "Subzone": "locality",
    "subzone": "locality",
    "Address": "address",
    "address": "address",
    # Cuisine
    "Cuisines": "cuisines_raw",
    "cuisines": "cuisines_raw",
    "Cuisine": "cuisines_raw",
    "cuisine": "cuisines_raw",
    # Cost
    "approx_cost(for two people)": "cost_raw",
    "approx_cost_for_two_people": "cost_raw",
    "Average Cost for two": "cost_raw",
    "average_cost_for_two": "cost_raw",
    "Cost for two": "cost_raw",
    "cost_for_two": "cost_raw",
    "Cost": "cost_raw",
    "cost": "cost_raw",
    # Rating
    "rate": "rating_raw",
    "Rate": "rating_raw",
    "Aggregate rating": "rating_raw",
    "aggregate_rating": "rating_raw",
    "Rating": "rating_raw",
    "rating": "rating_raw",
    # Votes
    "Votes": "votes_raw",
    "votes": "votes_raw",
    # Highlights / tags
    "Highlights": "highlights_raw",
    "highlights": "highlights_raw",
    "Tags": "highlights_raw",
    "tags": "highlights_raw",
    "dish_liked": "highlights_raw",
}

# Required canonical columns that must exist after alias renaming.
REQUIRED_COLUMNS: list[str] = ["name", "city", "cuisines_raw", "cost_raw", "rating_raw"]


# ── Budget tier boundaries (INR) ──────────────────────────────────────────────
BUDGET_LOW_MAX = 400
BUDGET_MEDIUM_MAX = 1200


# ── Private helper functions ──────────────────────────────────────────────────

def _strip_accents(text: str) -> str:
    """Normalise unicode accents to their base ASCII equivalents."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _parse_rating(val: Any) -> float:
    """Parse a raw rating value to a float clamped within [0.0, 5.0].

    Handles: "NEW", "-", "", NaN, "4.1/5", "4", 4.1
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    try:
        cleaned = str(val).strip().split("/")[0].strip()
        r = float(cleaned)
        if math.isnan(r):
            return 0.0
        return min(max(r, 0.0), 5.0)
    except (ValueError, TypeError):
        return 0.0


def _parse_cost(val: Any) -> int | None:
    """Parse a raw cost string/value to a non-negative integer (INR).

    Handles: "₹800", "1,200", "750.0", 800, "Free", "N/A", NaN, negative values.
    Returns None for unparseable or invalid values.
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        # Strip currency symbols, commas, and whitespace
        cleaned = re.sub(r"[₹,\s]", "", str(val).strip())
        cost_f = float(cleaned)
        if math.isnan(cost_f):
            return None
        cost = int(cost_f)
        return cost if cost > 0 else None
    except (ValueError, TypeError):
        return None


def _parse_cuisines(val: Any) -> list[str]:
    """Parse a raw cuisine string into a normalised lowercase list.

    Handles: "North Indian, Chinese", NaN, "", single values, extra whitespace.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ["unknown"]
    raw = str(val).strip()
    if not raw:
        return ["unknown"]
    items = [_strip_accents(c.strip()).lower() for c in raw.split(",")]
    return [c for c in items if c] or ["unknown"]


def _parse_highlights(val: Any) -> list[str]:
    """Parse a raw highlights/tags string into a normalised lowercase list."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    raw = str(val).strip()
    if not raw:
        return []
    # Highlights may be comma-separated or pipe-separated
    sep = "|" if "|" in raw else ","
    items = [h.strip().lower() for h in raw.split(sep)]
    return [h for h in items if h]


def _parse_city(val: Any) -> str:
    """Normalise a city/location string to lowercase stripped form."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "unknown"
    # Strip trailing punctuation, extra whitespace, normalise case
    cleaned = re.sub(r"[^\w\s,\-]", "", str(val).strip())
    return cleaned.strip().lower() or "unknown"


def _parse_votes(val: Any) -> int:
    """Parse a raw votes value to a non-negative integer."""
    try:
        v = int(str(val).strip().replace(",", ""))
        return max(v, 0)
    except (ValueError, TypeError):
        return 0


def _assign_budget_tier(cost: int | None) -> str:
    """Map an integer cost (INR) to a budget tier label."""
    if cost is None:
        return "medium"  # fallback for unknown cost
    if cost <= BUDGET_LOW_MAX:
        return "low"
    elif cost <= BUDGET_MEDIUM_MAX:
        return "medium"
    else:
        return "high"


# ── Column renaming ───────────────────────────────────────────────────────────

def _apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw dataset columns to canonical internal names.

    Only renames columns that exist in the DataFrame; ignores unknown columns.
    If 'city' is missing after renaming but 'locality' or 'address' is available,
    it defaults to 'bangalore' (reflecting the HuggingFace Zomato dataset origin).
    Raises ValueError if any REQUIRED_COLUMNS are missing after renaming.
    """
    rename_map = {
        raw: canonical
        for raw, canonical in COLUMN_ALIASES.items()
        if raw in df.columns
    }
    df = df.rename(columns=rename_map)

    # If city column is missing, infer or default to 'bangalore' if locality/address is present
    if "city" not in df.columns:
        if "locality" in df.columns or "address" in df.columns:
            logger.info("Column 'city' not present; defaulting to 'bangalore' based on locality/address.")
            df["city"] = "bangalore"

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns after alias mapping: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            "Update COLUMN_ALIASES in preprocessor.py to match the dataset schema."
        )
    return df


# ── Main cleaning pipeline ────────────────────────────────────────────────────

def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all ETL transformations to the raw DataFrame.

    Steps:
        1. Apply column aliases.
        2. Drop rows with no restaurant name.
        3. Parse each field with edge-case handling.
        4. Fill missing cost values with the median.
        5. Assign budget tier.
        6. Deduplicate on (name, city) — keep highest-vote row.
        7. Select and order final output columns.

    Args:
        df: Raw DataFrame from load_raw_dataset().

    Returns:
        Cleaned, normalised DataFrame ready for Parquet persistence.
    """
    logger.info("Starting ETL cleaning pipeline on %d rows …", len(df))

    # ── Step 1: Column aliases ────────────────────────────────────────────────
    df = _apply_column_aliases(df)

    # ── Step 2: Drop empty name rows ──────────────────────────────────────────
    df = df.dropna(subset=["name"]).copy()
    df = df[df["name"].str.strip() != ""].copy()
    logger.info("After dropping empty name rows: %d rows.", len(df))

    # ── Step 3: Parse each field ──────────────────────────────────────────────
    df["name"] = df["name"].astype(str).str.strip()

    df["city"] = df["city"].apply(_parse_city)

    # Keep the locality and address columns as-is if present (useful for UI)
    if "locality" in df.columns:
        df["locality"] = df["locality"].fillna("").astype(str).str.strip().str.lower()
    if "address" in df.columns:
        df["address"] = df["address"].fillna("").astype(str).str.strip()

    df["cuisines"] = df["cuisines_raw"].apply(_parse_cuisines)

    # Rating: apply parse then convert to float64
    df["rating"] = df["rating_raw"].apply(_parse_rating).astype("float64")

    # Votes
    df["votes"] = df["votes_raw"].apply(_parse_votes) if "votes_raw" in df.columns else 0

    # Highlights
    df["highlights"] = (
        df["highlights_raw"].apply(_parse_highlights)
        if "highlights_raw" in df.columns
        else [[] for _ in range(len(df))]
    )

    # Cost: parse to int | None, then fill NaN with median
    df["cost_for_two"] = df["cost_raw"].apply(_parse_cost)
    cost_median = int(
        df["cost_for_two"].dropna().median()
        if df["cost_for_two"].notna().any()
        else 600
    )
    df["cost_for_two"] = df["cost_for_two"].fillna(cost_median).astype(int)
    logger.info("Cost NaN fill: median = ₹%d", cost_median)

    # ── Step 4: Budget tier ───────────────────────────────────────────────────
    df["budget_tier"] = df["cost_for_two"].apply(_assign_budget_tier)

    # ── Step 5: Deduplicate — keep highest votes row per (name, city, locality) ─────────
    dedup_cols = ["name", "city"]
    if "locality" in df.columns:
        dedup_cols.append("locality")
    before_dedup = len(df)
    df = (
        df.sort_values("votes", ascending=False)
        .drop_duplicates(subset=dedup_cols, keep="first")
        .reset_index(drop=True)
    )
    logger.info(
        "Deduplication (%s): %d → %d rows (removed %d duplicates).",
        ", ".join(dedup_cols), before_dedup, len(df), before_dedup - len(df),
    )

    # ── Step 6: Select final columns ──────────────────────────────────────────
    output_cols = ["name", "city", "cuisines", "cost_for_two", "budget_tier",
                   "rating", "votes", "highlights"]
    for optional in ("locality", "address"):
        if optional in df.columns:
            output_cols.append(optional)

    df = df[output_cols].copy()

    logger.info(
        "ETL complete. Final shape: %d rows × %d columns.", len(df), len(df.columns)
    )
    return df


# ── Parquet persistence ───────────────────────────────────────────────────────

def build_processed_dataset(force_refresh: bool = False) -> pd.DataFrame:
    """Build and cache the cleaned Zomato dataset as a Parquet file.

    On the first call (or when force_refresh=True) the raw dataset is
    downloaded, cleaned, and written to DATA_PROCESSED_PATH.
    Subsequent calls simply read the cached Parquet file.

    Args:
        force_refresh: If True, re-download and re-process even if the
                       Parquet cache already exists.

    Returns:
        Cleaned pd.DataFrame with canonical schema.

    Raises:
        DatasetLoadError: If the HuggingFace download fails.
        EmptyDatasetError: If the cleaned dataset is empty.
        IOError: If the Parquet file cannot be written (e.g. disk full).
    """
    # ── Try loading from cache ─────────────────────────────────────────────────
    if not force_refresh and DATA_PROCESSED_PATH.exists():
        try:
            logger.info("Loading cleaned dataset from cache: %s", DATA_PROCESSED_PATH)
            df = pd.read_parquet(DATA_PROCESSED_PATH)
            logger.info("Cache hit: %d rows loaded.", len(df))
            return df
        except Exception as exc:
            logger.warning(
                "Parquet cache is corrupt (%s). Re-running ETL …", exc
            )
            DATA_PROCESSED_PATH.unlink(missing_ok=True)

    # ── Run full ETL ──────────────────────────────────────────────────────────
    logger.info("Cache miss or force_refresh=True — running full ETL pipeline …")
    raw_df = load_raw_dataset()
    clean_df = clean_and_normalize(raw_df)

    if clean_df.empty:
        raise EmptyDatasetError(
            "ETL pipeline produced an empty DataFrame. "
            "Check the raw dataset and cleaning logic."
        )

    # ── Write to Parquet (atomic: write to .tmp then rename) ──────────────────
    DATA_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = DATA_PROCESSED_PATH.with_suffix(".parquet.tmp")
    try:
        clean_df.to_parquet(tmp_path, index=False)
        tmp_path.rename(DATA_PROCESSED_PATH)
        logger.info("Cleaned dataset saved to: %s", DATA_PROCESSED_PATH)
    except IOError as exc:
        tmp_path.unlink(missing_ok=True)
        raise IOError(
            f"Failed to write Parquet file to {DATA_PROCESSED_PATH}. "
            f"Check disk space and directory permissions.\nOriginal error: {exc}"
        ) from exc

    return clean_df
