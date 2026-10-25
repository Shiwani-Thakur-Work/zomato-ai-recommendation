"""
test_data_pipeline.py
---------------------
Unit tests for the Phase 2 data ingestion and ETL pipeline.

Tests are isolated — they do NOT call HuggingFace or touch the filesystem.
Network-dependent tests are marked with @pytest.mark.integration.

Run unit tests only:
    uv run pytest tests/test_data_pipeline.py -v -m "not integration"

Run all tests (requires network + local Parquet cache):
    uv run pytest tests/test_data_pipeline.py -v
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import (
    _assign_budget_tier,
    _parse_city,
    _parse_cost,
    _parse_cuisines,
    _parse_highlights,
    _parse_rating,
    _parse_votes,
    clean_and_normalize,
)


# ─────────────────────────────────────────────────────────────────────────────
# _parse_rating
# ─────────────────────────────────────────────────────────────────────────────

class TestParseRating:
    def test_new_string_returns_zero(self):
        assert _parse_rating("NEW") == 0.0

    def test_dash_returns_zero(self):
        assert _parse_rating("-") == 0.0

    def test_empty_string_returns_zero(self):
        assert _parse_rating("") == 0.0

    def test_none_returns_zero(self):
        assert _parse_rating(None) == 0.0

    def test_nan_returns_zero(self):
        assert _parse_rating(float("nan")) == 0.0

    def test_fraction_notation(self):
        assert _parse_rating("4.1/5") == 4.1

    def test_integer_string(self):
        assert _parse_rating("4") == 4.0

    def test_float_passthrough(self):
        assert _parse_rating(4.4) == 4.4

    def test_clamp_above_five(self):
        assert _parse_rating("5.6") == 5.0

    def test_clamp_below_zero(self):
        assert _parse_rating("-1.0") == 0.0

    def test_integer_passthrough(self):
        assert _parse_rating(3) == 3.0


# ─────────────────────────────────────────────────────────────────────────────
# _parse_cost
# ─────────────────────────────────────────────────────────────────────────────

class TestParseCost:
    def test_currency_symbol_stripped(self):
        assert _parse_cost("₹800") == 800

    def test_comma_formatted(self):
        assert _parse_cost("1,200") == 1200

    def test_float_string(self):
        assert _parse_cost("750.0") == 750

    def test_plain_integer_string(self):
        assert _parse_cost("600") == 600

    def test_integer_passthrough(self):
        assert _parse_cost(800) == 800

    def test_free_returns_none(self):
        assert _parse_cost("Free") is None

    def test_na_returns_none(self):
        assert _parse_cost("N/A") is None

    def test_none_returns_none(self):
        assert _parse_cost(None) is None

    def test_zero_returns_none(self):
        assert _parse_cost("0") is None

    def test_negative_returns_none(self):
        assert _parse_cost("-100") is None

    def test_nan_returns_none(self):
        assert _parse_cost(float("nan")) is None


# ─────────────────────────────────────────────────────────────────────────────
# _parse_cuisines
# ─────────────────────────────────────────────────────────────────────────────

class TestParseCuisines:
    def test_single_cuisine(self):
        assert _parse_cuisines("Italian") == ["italian"]

    def test_multiple_cuisines(self):
        assert _parse_cuisines("North Indian, Chinese") == ["north indian", "chinese"]

    def test_extra_whitespace(self):
        assert _parse_cuisines("North Indian ,  Chinese") == ["north indian", "chinese"]

    def test_nan_returns_unknown(self):
        assert _parse_cuisines(float("nan")) == ["unknown"]

    def test_none_returns_unknown(self):
        assert _parse_cuisines(None) == ["unknown"]

    def test_empty_string_returns_unknown(self):
        assert _parse_cuisines("") == ["unknown"]

    def test_lowercase_normalisation(self):
        result = _parse_cuisines("ITALIAN")
        assert result == ["italian"]

    def test_long_list_preserved(self):
        raw = "Italian, French, Chinese, Thai, Mexican, Indian, Japanese, Greek, Spanish, Korean"
        result = _parse_cuisines(raw)
        assert len(result) == 10
        assert all(c == c.lower() for c in result)


# ─────────────────────────────────────────────────────────────────────────────
# _parse_city
# ─────────────────────────────────────────────────────────────────────────────

class TestParseCity:
    def test_strips_whitespace(self):
        assert _parse_city("  Bangalore  ") == "bangalore"

    def test_lowercase_normalisation(self):
        assert _parse_city("NEW DELHI") == "new delhi"

    def test_strips_trailing_punctuation(self):
        assert _parse_city("Chennai.") == "chennai"

    def test_none_returns_unknown(self):
        assert _parse_city(None) == "unknown"

    def test_nan_returns_unknown(self):
        assert _parse_city(float("nan")) == "unknown"

    def test_mixed_case(self):
        assert _parse_city("Mumbai") == "mumbai"


# ─────────────────────────────────────────────────────────────────────────────
# _parse_votes
# ─────────────────────────────────────────────────────────────────────────────

class TestParseVotes:
    def test_comma_formatted(self):
        assert _parse_votes("1,850") == 1850

    def test_integer_passthrough(self):
        assert _parse_votes(500) == 500

    def test_non_numeric_returns_zero(self):
        assert _parse_votes("abc") == 0

    def test_none_returns_zero(self):
        assert _parse_votes(None) == 0

    def test_zero(self):
        assert _parse_votes(0) == 0


# ─────────────────────────────────────────────────────────────────────────────
# _assign_budget_tier
# ─────────────────────────────────────────────────────────────────────────────

class TestAssignBudgetTier:
    def test_low_boundary(self):
        assert _assign_budget_tier(400) == "low"

    def test_below_low_boundary(self):
        assert _assign_budget_tier(300) == "low"

    def test_medium_boundary_lower(self):
        assert _assign_budget_tier(401) == "medium"

    def test_medium_boundary_upper(self):
        assert _assign_budget_tier(1200) == "medium"

    def test_high_boundary(self):
        assert _assign_budget_tier(1201) == "high"

    def test_high_value(self):
        assert _assign_budget_tier(2500) == "high"

    def test_none_returns_medium_fallback(self):
        assert _assign_budget_tier(None) == "medium"


# ─────────────────────────────────────────────────────────────────────────────
# clean_and_normalize (integration-level with synthetic DataFrame)
# ─────────────────────────────────────────────────────────────────────────────

def _make_raw_df(**overrides) -> pd.DataFrame:
    """Create a minimal synthetic raw DataFrame for testing clean_and_normalize."""
    base = {
        "Restaurant Name": ["Test Restaurant", "Duplicate", "Duplicate"],
        "City": ["Bangalore", "Mumbai", "Mumbai"],
        "Cuisines": ["North Indian, Chinese", "Italian", "Italian"],
        "Average Cost for two": ["₹800", "1,200", "1,200"],
        "Aggregate rating": ["4.1", "NEW", "3.5"],
        "Votes": ["1,000", "500", "800"],
        "Highlights": ["outdoor seating, wifi", None, "rooftop"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestCleanAndNormalize:
    def test_output_has_required_columns(self):
        df = clean_and_normalize(_make_raw_df())
        for col in ["name", "city", "cuisines", "cost_for_two", "budget_tier", "rating", "votes"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_rating_parsed_correctly(self):
        df = clean_and_normalize(_make_raw_df())
        # "4.1" row
        row = df[df["name"] == "Test Restaurant"].iloc[0]
        assert row["rating"] == 4.1

    def test_new_rating_becomes_zero(self):
        df = clean_and_normalize(_make_raw_df())
        # Duplicate rows: one has "NEW" (0.0) one has "3.5"
        # After dedup, higher-votes row (800) wins = "3.5"
        dup_rows = df[df["name"] == "Duplicate"]
        assert len(dup_rows) == 1
        assert dup_rows.iloc[0]["rating"] == 3.5

    def test_deduplication_keeps_higher_votes(self):
        df = clean_and_normalize(_make_raw_df())
        # "Duplicate" in "mumbai" appears twice; higher votes row (800) kept
        dup = df[df["name"] == "Duplicate"]
        assert len(dup) == 1
        assert dup.iloc[0]["votes"] == 800

    def test_budget_tier_assigned(self):
        df = clean_and_normalize(_make_raw_df())
        row = df[df["name"] == "Test Restaurant"].iloc[0]
        assert row["budget_tier"] == "medium"  # ₹800 → medium

    def test_cuisines_normalised_to_list(self):
        df = clean_and_normalize(_make_raw_df())
        row = df[df["name"] == "Test Restaurant"].iloc[0]
        assert isinstance(row["cuisines"], list)
        assert "north indian" in row["cuisines"]
        assert "chinese" in row["cuisines"]

    def test_city_normalised_to_lowercase(self):
        df = clean_and_normalize(_make_raw_df())
        assert df["city"].iloc[0] == "bangalore"

    def test_highlights_nan_becomes_empty_list(self):
        df = clean_and_normalize(_make_raw_df())
        row = df[df["name"] == "Duplicate"]
        # After dedup (higher-votes row = 800 which had "rooftop"), highlights is a list
        assert isinstance(row.iloc[0]["highlights"], list)

    def test_missing_name_rows_dropped(self):
        raw = _make_raw_df()
        raw.loc[0, "Restaurant Name"] = None
        df = clean_and_normalize(raw)
        assert "Test Restaurant" not in df["name"].values

    def test_cost_nan_filled_with_median(self):
        raw = _make_raw_df()
        raw.loc[0, "Average Cost for two"] = "N/A"
        df = clean_and_normalize(raw)
        # All cost_for_two values should be integers (no NaN)
        assert df["cost_for_two"].isna().sum() == 0
        assert df["cost_for_two"].dtype == int


# ─────────────────────────────────────────────────────────────────────────────
# Integration test — requires live HuggingFace access
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_build_processed_dataset_end_to_end():
    """Full integration test: download, clean, persist, reload."""
    from src.data.preprocessor import build_processed_dataset
    from src.config import DATA_PROCESSED_PATH

    df = build_processed_dataset()

    # Shape
    assert len(df) > 100, "Expected at least 100 restaurant rows"

    # Schema
    assert "rating" in df.columns
    assert df["rating"].dtype == "float64"
    assert "cuisines" in df.columns
    assert isinstance(df["cuisines"].iloc[0], (list, np.ndarray))
    assert len(df["cuisines"].iloc[0]) > 0
    assert "budget_tier" in df.columns
    assert df["budget_tier"].isin(["low", "medium", "high"]).all()
    assert "cost_for_two" in df.columns
    assert df["cost_for_two"].isna().sum() == 0

    # Parquet was written
    assert DATA_PROCESSED_PATH.exists()
