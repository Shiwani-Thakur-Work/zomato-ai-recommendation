"""
llm_client.py
-------------
Groq Cloud API client, anti-hallucination prompt builder,
structured JSON output parser, and heuristic fallback engine.

Primary Model: llama-3.3-70b-versatile
Fast Alternative: llama-3.1-8b-instant
Enforces structured JSON output via Groq's native JSON mode:
    response_format={"type": "json_object"}
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from groq import Groq

from src.config import GROQ_API_KEY, GROQ_MODEL
from src.engine.schemas import (
    RecommendationResponse,
    RestaurantCandidate,
    RestaurantRecommendation,
    UserPreferences,
)

logger = logging.getLogger(__name__)


# ── System Prompt (Static, Anti-Hallucination) ───────────────────────────────
SYSTEM_PROMPT = """You are an expert culinary guide and restaurant recommendation engine for India.
You possess nuanced knowledge of dining preferences, Indian and international cuisines, ambience, and hospitality.

CRITICAL BEHAVIORAL DIRECTIVES:
1. STRICT ANTI-HALLUCINATION: Recommend ONLY restaurants that appear in the provided candidate list. Never invent or introduce outside restaurant names.
2. DATA GROUNDING: Ground all justifications directly in the candidate's actual cuisines, rating, cost, highlights, and neighborhood.
3. CONTEXTUAL REASONING: Harmonize the user's hard criteria (location, budget, rating) with any subjective vibe cues in their notes (e.g. romantic, quick lunch, rooftop, pet-friendly).
4. STRUCTURED OUTPUT: You MUST respond ONLY with a valid JSON object matching the requested schema. No code fences (no ```json), no preambles, and no closing remarks outside the JSON.
"""


def _get_groq_client() -> Groq:
    """Instantiate and return the Groq client if an API key is available."""
    api_key = GROQ_API_KEY
    if not api_key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                api_key = str(st.secrets["GROQ_API_KEY"])
        except Exception:
            pass

    if not api_key or api_key.strip().startswith("gsk_xxxx") or not api_key.strip():
        raise ValueError(
            "GROQ_API_KEY is missing or invalid. Set a valid Groq API key in .env or the environment."
        )
    return Groq(api_key=api_key.strip())


def build_user_prompt(prefs: UserPreferences, candidates: List[dict]) -> str:
    """Construct the dynamic user prompt with grounded candidates and criteria."""
    # Simplify candidates to keep token usage compact and clean
    condensed_candidates = []
    for c in candidates:
        condensed_candidates.append({
            "name": c.get("name"),
            "locality": c.get("locality"),
            "city": c.get("city"),
            "cuisines": c.get("cuisines", []),
            "cost_for_two": f"₹{c.get('cost_for_two', 0)}",
            "budget_tier": c.get("budget_tier"),
            "rating": c.get("rating"),
            "votes": c.get("votes"),
            "highlights": c.get("highlights", []),
        })

    candidates_json = json.dumps(condensed_candidates, ensure_ascii=False, indent=2)
    cuisine_str = ", ".join(prefs.cuisines) if prefs.cuisines else "Any / Diverse"
    notes_str = prefs.additional_notes.strip() if prefs.additional_notes else "None specified"
    veg_requirement = "\n- Dietary Requirement: STRICTLY 100% PURE VEGETARIAN ONLY (Never recommend meat, chicken, or non-veg dishes)" if prefs.pure_veg else ""

    return f"""CANDIDATE RESTAURANTS (Choose from ONLY these {len(condensed_candidates)} venues):
{candidates_json}

USER REQUIREMENTS:
- Location / Area: {prefs.location}
- Budget Tier: {prefs.budget.value}
- Cuisines Desired: {cuisine_str}
- Minimum Acceptable Rating: {prefs.min_rating}
- Special Notes / Desired Vibe: "{notes_str}"{veg_requirement}
- Desired Number of Recommendations: {prefs.max_results}

INSTRUCTIONS:
1. Review all candidate restaurants above against the user's criteria and subjective notes.
2. Select and rank the top {prefs.max_results} best-matching restaurants.
3. For each selected restaurant:
   - Provide an insightful, personalized 'ai_explanation' (2-3 sentences) explaining precisely why this spot is an ideal choice for this user's specific cravings or occasion.
   - Accurately include its exact restaurant_name, cuisine, rating, estimated_cost (e.g. '₹800 for two'), match_rank, locality, and address.
4. Compose a concise 'curator_summary' (1-2 sentences) summarizing the culinary character of your selections.
5. Provide a 'user_query_summary' encapsulating what the user asked for.

JSON SCHEMA TO RETURN:
{{
  "status": "success",
  "user_query_summary": "string",
  "curator_summary": "string",
  "recommendations": [
    {{
      "restaurant_name": "exact candidate name",
      "cuisine": ["cuisine1", "cuisine2"],
      "rating": 4.5,
      "estimated_cost": "₹800 for two",
      "match_rank": 1,
      "ai_explanation": "Compelling explanation...",
      "locality": "locality name",
      "address": "address"
    }}
  ]
}}"""


def get_recommendations_from_llm(
    prefs: UserPreferences,
    candidates: List[dict],
    model: str = GROQ_MODEL,
) -> RecommendationResponse:
    """Call the Groq Cloud API with structured JSON output enforcement.

    Args:
        prefs: User criteria and subjective notes.
        candidates: Pre-filtered list of candidate restaurant dictionaries.
        model: Groq model identifier (defaults to GROQ_MODEL config).

    Returns:
        Validated RecommendationResponse instance.

    Raises:
        ValueError: If GROQ_API_KEY is not configured or LLM output is hallucinated.
        Exception: Any network, authentication, or validation errors from Groq.
    """
    client = _get_groq_client()

    prompt = build_user_prompt(prefs, candidates)
    logger.info("Dispatching Groq inference call (model=%s, candidates=%d)", model, len(candidates))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=1500,
    )

    raw_json = response.choices[0].message.content or "{}"
    data = json.loads(raw_json)

    # Candidate name validation (Anti-Hallucination Guard)
    valid_names = {c["name"].strip().lower() for c in candidates}
    recs = data.get("recommendations", [])
    grounded_recs = []

    # Map candidate lookup by lowercase name
    cand_by_name = {c["name"].strip().lower(): c for c in candidates}

    for item in recs:
        r_name = str(item.get("restaurant_name", "")).strip()
        if r_name.lower() in valid_names:
            # Enrich locality/address/is_pure_veg from candidate pool if omitted by LLM
            cand = cand_by_name[r_name.lower()]
            if not item.get("locality") and cand.get("locality"):
                item["locality"] = cand.get("locality")
            if not item.get("address") and cand.get("address"):
                item["address"] = cand.get("address")
            item["is_pure_veg"] = cand.get("is_pure_veg", False)
            grounded_recs.append(item)
        else:
            logger.warning("Dropped hallucinated restaurant from LLM response: '%s'", r_name)

    if not grounded_recs:
        raise ValueError("Groq LLM response contained no valid restaurants from the candidate pool.")

    data["recommendations"] = grounded_recs
    data["is_fallback"] = False
    return RecommendationResponse.model_validate(data)


def heuristic_fallback(
    prefs: UserPreferences,
    candidates: List[dict],
    relaxation_applied: Optional[str] = None,
) -> RecommendationResponse:
    """Rule-based ranking engine invoked when Groq API is unavailable or unconfigured.

    Selects the top-ranked restaurants based on composite score and generates
    clean, deterministic justifications.
    """
    logger.info("Using heuristic fallback engine on %d candidates", len(candidates))
    top_candidates = candidates[:prefs.max_results]

    recommendations: List[RestaurantRecommendation] = []
    for i, c in enumerate(top_candidates, 1):
        cuisines_str = ", ".join(c.get("cuisines", [])[:2]).title()
        locality_str = c.get("locality", c.get("city", "")).title()
        cost_val = c.get("cost_for_two", 0)

        # Build natural justification
        justification = (
            f"Top-rated {cuisines_str} destination in {locality_str} "
            f"with a {c.get('rating', 0.0):.1f}★ rating ({c.get('votes', 0):,} votes). "
            f"Priced at approximately ₹{cost_val} for two ({c.get('budget_tier', 'medium')} budget)."
        )
        if c.get("highlights"):
            top_dishes = ", ".join(c["highlights"][:2]).title()
            justification += f" Known for {top_dishes}."

        rec = RestaurantRecommendation(
            restaurant_name=c["name"],
            cuisine=c.get("cuisines", []),
            rating=float(c.get("rating", 0.0)),
            estimated_cost=f"₹{cost_val} for two",
            match_rank=i,
            ai_explanation=justification,
            locality=c.get("locality"),
            address=c.get("address"),
            is_pure_veg=bool(c.get("is_pure_veg", False)),
        )
        recommendations.append(rec)

    notes_part = f' with focus on "{prefs.additional_notes}"' if prefs.additional_notes else ""
    summary_text = (
        f"Search in {prefs.location.title()} for {prefs.budget.value} budget "
        f"({', '.join(prefs.cuisines).title() if prefs.cuisines else 'all cuisines'}){notes_part}"
    )

    curator_note = (
        "Curated via deterministic multi-criteria scoring "
        "(rating, review popularity, and cuisine alignment) as Groq LLM inference was unavailable."
    )

    return RecommendationResponse(
        status="fallback",
        user_query_summary=summary_text,
        recommendations=recommendations,
        curator_summary=curator_note,
        is_fallback=True,
        relaxation_applied=relaxation_applied,
    )


def no_match_response(prefs: UserPreferences) -> RecommendationResponse:
    """Clean empty response when no restaurants match even after relaxation."""
    return RecommendationResponse(
        status="empty",
        user_query_summary=f"Search for {prefs.location} ({prefs.budget.value} budget)",
        recommendations=[],
        curator_summary=(
            f"No dining establishments found matching '{prefs.location}'. "
            "Please try widening your location search (e.g. 'Bangalore') or relaxing your budget and rating filters."
        ),
        is_fallback=False,
    )
