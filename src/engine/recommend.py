"""
recommend.py
------------
Top-level recommendation orchestrator for the Zomato AI Recommendation System.

Coordinates:
    1. Dataset loading (Parquet cache via build_processed_dataset)
    2. Multi-criteria filtering & candidate ranking (filter.py)
    3. Qualitative LLM inference with Groq JSON mode (llm_client.py)
    4. Heuristic fallback when Groq API key is absent or inference encounters an error
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.config import MAX_CANDIDATES_FOR_LLM
from src.data.preprocessor import build_processed_dataset
from src.engine.filter import get_candidates_with_metadata
from src.engine.llm_client import (
    get_recommendations_from_llm,
    heuristic_fallback,
    no_match_response,
)
from src.engine.schemas import RecommendationResponse, UserPreferences

logger = logging.getLogger(__name__)


def recommend(
    prefs: UserPreferences,
    df: Optional[pd.DataFrame] = None,
) -> RecommendationResponse:
    """Generate curated restaurant recommendations for user preferences.

    Args:
        prefs: Validated user preferences instance.
        df: Optional pre-loaded DataFrame. If None, reads from cached Parquet.

    Returns:
        Structured RecommendationResponse instance (LLM synthesized or heuristic fallback).
    """
    # 1. Ensure dataset is available
    if df is None:
        try:
            df = build_processed_dataset()
        except Exception as exc:
            logger.error("Failed to load processed dataset: %s", exc)
            return RecommendationResponse(
                status="error",
                user_query_summary=f"Query for {prefs.location}",
                recommendations=[],
                curator_summary="Unable to load restaurant database. Please check data files.",
                is_fallback=True,
            )

    # 2. Candidate retrieval & ranking
    candidates, relaxation_note = get_candidates_with_metadata(
        df=df,
        prefs=prefs,
        max_k=MAX_CANDIDATES_FOR_LLM,
    )

    if not candidates:
        logger.info("Zero candidates found for query: %s", prefs.location)
        return no_match_response(prefs)

    # 3. LLM Inference via Groq Cloud
    try:
        response = get_recommendations_from_llm(prefs, candidates)
        if relaxation_note:
            response.relaxation_applied = relaxation_note
        logger.info("Successfully synthesized recommendations via Groq LLM.")
        return response

    except Exception as exc:
        logger.warning(
            "Groq LLM inference unavailable or failed (%s). Engaging heuristic fallback.",
            exc,
        )
        return heuristic_fallback(
            prefs=prefs,
            candidates=candidates,
            relaxation_applied=relaxation_note,
        )
