"""
test_engine.py
--------------
Unit tests for the recommendation engine:
    - Pydantic v2 schemas (UserPreferences, BudgetTier, etc.)
    - Candidate filter, city matching, scoring formula, and adaptive relaxation
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from src.engine.filter import (
    _calculate_cuisine_match_ratio,
    _city_matches,
    _score_candidate,
    get_candidates,
    get_candidates_with_metadata,
)
from src.engine.schemas import (
    BudgetTier,
    RecommendationResponse,
    RestaurantCandidate,
    RestaurantRecommendation,
    UserPreferences,
)


# ─────────────────────────────────────────────────────────────────────────────
# Schema Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUserPreferencesSchema:
    def test_valid_preferences(self):
        prefs = UserPreferences(
            location="Bangalore",
            budget=BudgetTier.MEDIUM,
            cuisines=["Italian", "Chinese"],
            min_rating=4.0,
            additional_notes="romantic rooftop",
            max_results=5,
        )
        assert prefs.location == "Bangalore"
        assert prefs.budget == BudgetTier.MEDIUM
        assert prefs.cuisines == ["italian", "chinese"]
        assert prefs.min_rating == 4.0
        assert prefs.additional_notes == "romantic rooftop"
        assert prefs.max_results == 5

    def test_default_values(self):
        prefs = UserPreferences(location="Delhi")
        assert prefs.budget == BudgetTier.ANY
        assert prefs.cuisines == []
        assert prefs.min_rating == 3.5
        assert prefs.additional_notes == ""
        assert prefs.max_results == 5

    def test_empty_location_raises_error(self):
        with pytest.raises(ValueError):
            UserPreferences(location="   ")

    def test_rating_out_of_bounds_raises_error(self):
        with pytest.raises(ValueError):
            UserPreferences(location="Bangalore", min_rating=5.5)
        with pytest.raises(ValueError):
            UserPreferences(location="Bangalore", min_rating=-0.5)

    def test_max_results_out_of_bounds_raises_error(self):
        with pytest.raises(ValueError):
            UserPreferences(location="Bangalore", max_results=0)
        with pytest.raises(ValueError):
            UserPreferences(location="Bangalore", max_results=15)

    def test_cuisines_from_comma_separated_string(self):
        prefs = UserPreferences(location="Bangalore", cuisines="North Indian, Chinese, Continental")
        assert prefs.cuisines == ["north indian", "chinese", "continental"]


class TestResponseSchemas:
    def test_restaurant_candidate_instantiation(self):
        candidate = RestaurantCandidate(
            name="Empire Restaurant",
            city="Bangalore",
            locality="Indiranagar",
            address="100 Feet Road, Indiranagar",
            cuisines=["North Indian", "Mughlai"],
            cost_for_two=800,
            budget_tier="medium",
            rating=4.2,
            votes=1500,
            highlights=["buffet", "late night"],
        )
        assert candidate.name == "Empire Restaurant"
        assert candidate.rating == 4.2

    def test_recommendation_response_structure(self):
        rec = RestaurantRecommendation(
            restaurant_name="Toit",
            cuisine=["Italian", "American"],
            rating=4.7,
            estimated_cost="₹1,500 for two",
            match_rank=1,
            ai_explanation="Renowned microbrewery with excellent wood-fired pizzas matching your vibe.",
        )
        resp = RecommendationResponse(
            user_query_summary="Indiranagar microbrewery with good food",
            recommendations=[rec],
            curator_summary="Selected the top rated brewpub in Indiranagar.",
            is_fallback=False,
        )
        assert resp.status == "success"
        assert len(resp.recommendations) == 1
        assert resp.recommendations[0].restaurant_name == "Toit"


# ─────────────────────────────────────────────────────────────────────────────
# Location & City Matching Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCityMatches:
    def test_exact_city_match(self):
        assert _city_matches("bangalore", "bangalore") is True

    def test_case_insensitive_match(self):
        assert _city_matches("BANGALORE", "bangalore") is True

    def test_transliteration_bengaluru_to_bangalore(self):
        assert _city_matches("Bengaluru", "bangalore") is True
        assert _city_matches("BLR", "bangalore") is True

    def test_locality_match(self):
        assert _city_matches(
            query_location="Indiranagar",
            candidate_city="bangalore",
            candidate_locality="indiranagar",
        ) is True

    def test_address_match(self):
        assert _city_matches(
            query_location="Koramangala 5th Block",
            candidate_city="bangalore",
            candidate_locality="koramangala",
            candidate_address="12, 5th Block, Koramangala, Bangalore",
        ) is True

    def test_misspelling_fuzzy_match(self):
        assert _city_matches("Bangalre", "bangalore") is True

    def test_mismatched_city(self):
        assert _city_matches("Mumbai", "bangalore", "indiranagar") is False


# ─────────────────────────────────────────────────────────────────────────────
# Scoring Formula Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidateScoring:
    def test_cuisine_match_ratio_empty_query(self):
        assert _calculate_cuisine_match_ratio(["italian", "chinese"], []) == 0.5

    def test_cuisine_match_ratio_full_match(self):
        ratio = _calculate_cuisine_match_ratio(["italian", "chinese"], ["italian", "chinese"])
        assert ratio == 1.0

    def test_cuisine_match_ratio_partial_match(self):
        ratio = _calculate_cuisine_match_ratio(["italian", "north indian"], ["italian", "mexican"])
        assert ratio == 0.5

    def test_cuisine_match_ratio_no_match(self):
        ratio = _calculate_cuisine_match_ratio(["italian"], ["japanese"])
        assert ratio == 0.0

    def test_score_candidate_exact_calculation(self):
        # row: rating=4.0, votes=999, cuisines=["italian"]
        # query_cuisines=["italian"]
        # rating part: 4.0 * 0.5 = 2.0
        # votes part: log10(999 + 1) * 0.3 = log10(1000) * 0.3 = 3 * 0.3 = 0.9
        # cuisine part: 1.0 * 0.2 = 0.2
        # Total expected = 2.0 + 0.9 + 0.2 = 3.1
        row = {"rating": 4.0, "votes": 999, "cuisines": ["italian"]}
        score = _score_candidate(row, ["italian"])
        assert math.isclose(score, 3.1, abs_tol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Candidate Filter & Adaptive Relaxation Tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_candidate_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "name": "Rest A",
            "city": "bangalore",
            "locality": "indiranagar",
            "address": "100ft road indiranagar",
            "cuisines": ["italian", "continental"],
            "cost_for_two": 1500,
            "budget_tier": "high",
            "rating": 4.5,
            "votes": 2000,
            "highlights": ["rooftop", "cocktails"],
        },
        {
            "name": "Rest B",
            "city": "bangalore",
            "locality": "indiranagar",
            "address": "12th main indiranagar",
            "cuisines": ["italian", "pizza"],
            "cost_for_two": 800,
            "budget_tier": "medium",
            "rating": 4.2,
            "votes": 1200,
            "highlights": ["outdoor seating"],
        },
        {
            "name": "Rest C",
            "city": "bangalore",
            "locality": "koramangala",
            "address": "5th block koramangala",
            "cuisines": ["chinese", "thai"],
            "cost_for_two": 350,
            "budget_tier": "low",
            "rating": 3.8,
            "votes": 600,
            "highlights": ["quick bites"],
        },
        {
            "name": "Rest D",
            "city": "bangalore",
            "locality": "indiranagar",
            "address": "defence colony indiranagar",
            "cuisines": ["cafe", "bakery"],
            "cost_for_two": 600,
            "budget_tier": "medium",
            "rating": 3.2,
            "votes": 300,
            "highlights": ["wifi"],
        },
    ])


class TestGetCandidates:
    def test_filter_exact_location_and_rating(self):
        df = _make_candidate_df()
        prefs = UserPreferences(
            location="Indiranagar",
            min_rating=4.0,
            budget=BudgetTier.ANY,
        )
        candidates = get_candidates(df, prefs)
        names = [c["name"] for c in candidates]
        assert "Rest A" in names
        assert "Rest B" in names
        assert "Rest C" not in names  # in Koramangala
        assert "Rest D" not in names  # rating 3.2 < 4.0

    def test_ranking_order_by_composite_score(self):
        df = _make_candidate_df()
        prefs = UserPreferences(
            location="Indiranagar",
            min_rating=3.0,
            cuisines=["Italian"],
        )
        candidates = get_candidates(df, prefs)
        # Rest A (rating 4.5, 2000 votes, Italian) should rank #1
        assert candidates[0]["name"] == "Rest A"
        assert candidates[0]["composite_score"] > candidates[1]["composite_score"]

    def test_adaptive_relaxation_triggers_when_few_candidates(self):
        df = _make_candidate_df()
        # Rest D is the only one in Indiranagar with budget low/medium and rating 3.2
        # Asking for min_rating 4.4 and budget low yields 0 candidates initially
        prefs = UserPreferences(
            location="Indiranagar",
            min_rating=4.4,
            budget=BudgetTier.LOW,
            max_results=3,
        )
        candidates, relaxation = get_candidates_with_metadata(df, prefs)
        assert len(candidates) > 0
        assert relaxation is not None
        assert "Relaxed criteria" in relaxation


# ─────────────────────────────────────────────────────────────────────────────
# Real Parquet Dataset Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestRealDatasetCandidates:
    @pytest.fixture
    def real_df(self):
        from src.data.preprocessor import build_processed_dataset
        return build_processed_dataset()

    def test_real_dataset_bangalore_query(self, real_df):
        prefs = UserPreferences(
            location="Bangalore",
            budget=BudgetTier.MEDIUM,
            min_rating=4.0,
            max_results=5,
        )
        candidates = get_candidates(real_df, prefs, max_k=10)
        assert len(candidates) > 0
        assert len(candidates) <= 10
        # All candidates must have required keys
        for c in candidates:
            assert "name" in c
            assert "rating" in c
            assert c["rating"] >= 4.0
            assert c["budget_tier"] == "medium"

    def test_real_dataset_indiranagar_query(self, real_df):
        prefs = UserPreferences(
            location="Indiranagar",
            cuisines=["Italian"],
            min_rating=3.5,
            max_results=5,
        )
        candidates = get_candidates(real_df, prefs, max_k=10)
        assert len(candidates) > 0
        # Should have found top Indiranagar Italian spots (e.g. Toit)
        assert any("toit" in c["name"].lower() or "indiranagar" in str(c.get("locality", "")).lower() for c in candidates)

    def test_candidates_parseable_by_pydantic_schema(self, real_df):
        prefs = UserPreferences(location="Koramangala", min_rating=3.8)
        candidates = get_candidates(real_df, prefs, max_k=5)
        assert len(candidates) > 0
        for c in candidates:
            candidate_obj = RestaurantCandidate(**c)
            assert candidate_obj.name
            assert candidate_obj.rating >= 3.8


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — LLM Client, Prompt Builder & Recommendation Orchestration Tests
# ─────────────────────────────────────────────────────────────────────────────

from unittest.mock import MagicMock, patch
from src.engine.llm_client import (
    build_user_prompt,
    get_recommendations_from_llm,
    heuristic_fallback,
    no_match_response,
)
from src.engine.recommend import recommend


class TestPromptBuilder:
    def test_build_user_prompt_contains_criteria_and_candidates(self):
        prefs = UserPreferences(
            location="Indiranagar",
            budget=BudgetTier.MEDIUM,
            cuisines=["Italian"],
            min_rating=4.0,
            additional_notes="quiet place with outdoor seating",
            max_results=3,
        )
        candidates = [
            {
                "name": "Toit",
                "city": "bangalore",
                "locality": "indiranagar",
                "cuisines": ["italian", "american"],
                "cost_for_two": 1500,
                "budget_tier": "high",
                "rating": 4.6,
                "votes": 12000,
                "highlights": ["craft beer", "wood fired pizza"],
            }
        ]
        prompt = build_user_prompt(prefs, candidates)
        assert "Indiranagar" in prompt
        assert "quiet place with outdoor seating" in prompt
        assert "Toit" in prompt
        assert "JSON SCHEMA TO RETURN" in prompt


class TestHeuristicFallback:
    def test_fallback_generates_valid_response(self):
        prefs = UserPreferences(
            location="Bangalore",
            budget=BudgetTier.MEDIUM,
            cuisines=["Continental"],
            max_results=2,
        )
        candidates = [
            {
                "name": "Rest 1",
                "city": "bangalore",
                "locality": "indiranagar",
                "address": "123 Indiranagar",
                "cuisines": ["continental"],
                "cost_for_two": 800,
                "budget_tier": "medium",
                "rating": 4.5,
                "votes": 500,
                "highlights": ["pasta"],
            },
            {
                "name": "Rest 2",
                "city": "bangalore",
                "locality": "koramangala",
                "address": "456 Koramangala",
                "cuisines": ["continental"],
                "cost_for_two": 700,
                "budget_tier": "medium",
                "rating": 4.2,
                "votes": 300,
                "highlights": ["coffee"],
            },
        ]
        resp = heuristic_fallback(prefs, candidates, relaxation_applied="Relaxed rating")
        assert resp.status == "fallback"
        assert resp.is_fallback is True
        assert len(resp.recommendations) == 2
        assert resp.recommendations[0].restaurant_name == "Rest 1"
        assert resp.recommendations[0].match_rank == 1
        assert "4.5★" in resp.recommendations[0].ai_explanation
        assert resp.relaxation_applied == "Relaxed rating"

    def test_no_match_response_structure(self):
        prefs = UserPreferences(location="NonExistentCity")
        resp = no_match_response(prefs)
        assert resp.status == "empty"
        assert len(resp.recommendations) == 0
        assert "NonExistentCity" in resp.curator_summary


class TestRecommendOrchestrator:
    def test_recommend_empty_candidates_returns_no_match(self):
        df = pd.DataFrame(columns=["name", "city", "rating", "budget_tier", "cuisines", "votes", "cost_for_two"])
        prefs = UserPreferences(location="UnknownPlace")
        resp = recommend(prefs, df=df)
        assert resp.status == "empty"
        assert len(resp.recommendations) == 0

    def test_recommend_gracefully_falls_back_when_llm_fails(self):
        from unittest.mock import patch

        df = pd.DataFrame([
            {
                "name": "Fallback Bistro",
                "city": "bangalore",
                "locality": "indiranagar",
                "address": "12th main indiranagar",
                "cuisines": ["italian"],
                "cost_for_two": 900,
                "budget_tier": "medium",
                "rating": 4.3,
                "votes": 1000,
                "highlights": ["risotto"],
            }
        ])
        prefs = UserPreferences(location="Indiranagar")
        with patch("src.engine.recommend.get_recommendations_from_llm", side_effect=RuntimeError("Simulated Groq Failure")):
            resp = recommend(prefs, df=df)
        assert resp.is_fallback is True
        assert len(resp.recommendations) == 1
        assert resp.recommendations[0].restaurant_name == "Fallback Bistro"

    def test_mock_groq_llm_success(self):
        prefs = UserPreferences(location="Indiranagar", max_results=1)
        candidates = [
            {
                "name": "Sunny Ristorante",
                "city": "bangalore",
                "locality": "indiranagar",
                "address": "100ft road indiranagar",
                "cuisines": ["italian"],
                "cost_for_two": 1200,
                "budget_tier": "medium",
                "rating": 4.5,
                "votes": 2500,
                "highlights": ["tiramisu"],
            }
        ]
        mock_json_content = """{
            "status": "success",
            "user_query_summary": "Authentic Italian in Indiranagar",
            "curator_summary": "Top choice for classic Italian fare.",
            "recommendations": [
                {
                    "restaurant_name": "Sunny Ristorante",
                    "cuisine": ["italian"],
                    "rating": 4.5,
                    "estimated_cost": "₹1200 for two",
                    "match_rank": 1,
                    "ai_explanation": "Ideal Italian spot with homemade pasta and cozy ambience.",
                    "locality": "indiranagar",
                    "address": "100ft road indiranagar"
                }
            ]
        }"""
        with patch("src.engine.llm_client._get_groq_client") as mock_client_factory:
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = mock_json_content
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_factory.return_value = mock_client

            resp = get_recommendations_from_llm(prefs, candidates)
            assert resp.status == "success"
            assert resp.is_fallback is False
            assert len(resp.recommendations) == 1
            assert resp.recommendations[0].restaurant_name == "Sunny Ristorante"

    def test_anti_hallucination_filters_out_invented_restaurant(self):
        prefs = UserPreferences(location="Indiranagar")
        candidates = [
            {
                "name": "Real Restaurant",
                "city": "bangalore",
                "locality": "indiranagar",
                "address": "100ft road",
                "cuisines": ["north indian"],
                "cost_for_two": 600,
                "budget_tier": "medium",
                "rating": 4.1,
                "votes": 800,
                "highlights": [],
            }
        ]
        mock_json_with_hallucination = """{
            "status": "success",
            "user_query_summary": "Indiranagar dinner",
            "curator_summary": "Curated selection.",
            "recommendations": [
                {
                    "restaurant_name": "Fake Invented Cafe",
                    "cuisine": ["cafe"],
                    "rating": 4.9,
                    "estimated_cost": "₹500 for two",
                    "match_rank": 1,
                    "ai_explanation": "Totally made up place.",
                    "locality": "indiranagar",
                    "address": "unknown"
                },
                {
                    "restaurant_name": "Real Restaurant",
                    "cuisine": ["north indian"],
                    "rating": 4.1,
                    "estimated_cost": "₹600 for two",
                    "match_rank": 2,
                    "ai_explanation": "Grounded authentic match.",
                    "locality": "indiranagar",
                    "address": "100ft road"
                }
            ]
        }"""
        with patch("src.engine.llm_client._get_groq_client") as mock_client_factory:
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = mock_json_with_hallucination
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_factory.return_value = mock_client

            resp = get_recommendations_from_llm(prefs, candidates)
            # Only Real Restaurant should survive, Fake Invented Cafe must be dropped!
            assert len(resp.recommendations) == 1
            assert resp.recommendations[0].restaurant_name == "Real Restaurant"


@pytest.mark.integration
def test_e2e_recommend_call():
    """End-to-end integration test validating the full recommendation flow."""
    from src.data.preprocessor import build_processed_dataset
    from src.engine.recommend import recommend
    from src.engine.schemas import BudgetTier, UserPreferences

    df = build_processed_dataset()
    prefs = UserPreferences(
        location="Bangalore",
        budget=BudgetTier.MEDIUM,
        cuisines=["north indian"],
        min_rating=4.0,
        additional_notes="Looking for a cozy family dinner",
        max_results=3,
    )
    result = recommend(prefs, df)
    assert result.status in ("success", "fallback")
    assert len(result.recommendations) <= 3
    assert all(r.ai_explanation for r in result.recommendations)


class TestPureVegFiltering:
    """Tests verifying strict pure vegetarian classification and candidate filtering."""

    def test_pure_veg_schema_flag(self):
        prefs_veg = UserPreferences(location="Faridabad", pure_veg=True)
        assert prefs_veg.pure_veg is True
        prefs_default = UserPreferences(location="Faridabad")
        assert prefs_default.pure_veg is False

    def test_is_pure_veg_classifier(self):
        from src.engine.filter import is_pure_veg_restaurant

        # Pure veg examples
        assert is_pure_veg_restaurant({"name": "Om Sweets & Snacks", "cuisines": ["sweets", "north indian"], "highlights": []}) is True
        assert is_pure_veg_restaurant({"name": "Sagar Ratna", "cuisines": ["south indian", "north indian"], "highlights": []}) is True
        assert is_pure_veg_restaurant({"name": "Haldiram's", "cuisines": ["street food", "sweets"], "highlights": []}) is True
        assert is_pure_veg_restaurant({"name": "Bikanervala", "cuisines": ["sweets", "fast food"], "highlights": ["pure veg"]}) is True
        assert is_pure_veg_restaurant({"name": "Giani's", "cuisines": ["ice cream", "desserts"], "highlights": []}) is True

        # Non-veg examples
        assert is_pure_veg_restaurant({"name": "Bikkgane Biryani", "cuisines": ["biryani", "mughlai"], "highlights": []}) is False
        assert is_pure_veg_restaurant({"name": "Burger King", "cuisines": ["american", "fast food"], "highlights": ["chicken burger"]}) is False
        assert is_pure_veg_restaurant({"name": "KFC", "cuisines": ["american", "snacks"], "highlights": []}) is False
        assert is_pure_veg_restaurant({"name": "Mr Singh's Tandoori Hut", "cuisines": ["north indian", "chinese"], "highlights": []}) is False
        assert is_pure_veg_restaurant({"name": "Al Bake", "cuisines": ["north indian", "mughlai"], "highlights": ["chicken shawarma"]}) is False
        assert is_pure_veg_restaurant({"name": "Fisherman's Wharf", "cuisines": ["seafood", "goan"], "highlights": ["fish curry"]}) is False

    def test_candidate_retrieval_pure_veg_enforcement(self):
        from src.engine.filter import get_candidates_with_metadata
        from src.engine.schemas import BudgetTier, UserPreferences

        test_data = pd.DataFrame([
            {
                "name": "Bikkgane Biryani",
                "city": "faridabad",
                "locality": "sector 16",
                "address": "sector 16",
                "rating": 4.5,
                "votes": 1200,
                "cost_for_two": 350,
                "budget_tier": "low",
                "cuisines": ["biryani", "mughlai"],
                "highlights": ["chicken biryani"],
            },
            {
                "name": "Burger King",
                "city": "faridabad",
                "locality": "sector 16",
                "address": "sector 16",
                "rating": 4.2,
                "votes": 1500,
                "cost_for_two": 350,
                "budget_tier": "low",
                "cuisines": ["american"],
                "highlights": ["crispy chicken"],
            },
            {
                "name": "Om Sweets & Snacks",
                "city": "faridabad",
                "locality": "sector 16",
                "address": "sector 16",
                "rating": 4.3,
                "votes": 1100,
                "cost_for_two": 400,
                "budget_tier": "low",
                "cuisines": ["sweets", "north indian"],
                "highlights": ["chole bhature", "gulab jamun"],
            },
            {
                "name": "Sagar Ratna",
                "city": "faridabad",
                "locality": "sector 16",
                "address": "sector 16",
                "rating": 4.4,
                "votes": 800,
                "cost_for_two": 400,
                "budget_tier": "low",
                "cuisines": ["south indian"],
                "highlights": ["masala dosa", "filter coffee"],
            },
        ])

        # When Pure Veg is True: only Om Sweets & Sagar Ratna must appear
        prefs_veg = UserPreferences(
            location="faridabad",
            budget=BudgetTier.LOW,
            min_rating=3.5,
            pure_veg=True,
            max_results=5,
        )
        cands_veg, _ = get_candidates_with_metadata(test_data, prefs_veg)
        cand_names = [c["name"] for c in cands_veg]
        assert "Om Sweets & Snacks" in cand_names
        assert "Sagar Ratna" in cand_names
        assert "Bikkgane Biryani" not in cand_names
        assert "Burger King" not in cand_names
        assert all(c["is_pure_veg"] is True for c in cands_veg)

        # When Pure Veg is False: non-veg options like Bikkgane Biryani are eligible
        prefs_nonveg = UserPreferences(
            location="faridabad",
            budget=BudgetTier.LOW,
            min_rating=3.5,
            pure_veg=False,
            max_results=5,
        )
        cands_nonveg, _ = get_candidates_with_metadata(test_data, prefs_nonveg)
        nonveg_names = [c["name"] for c in cands_nonveg]
        assert "Bikkgane Biryani" in nonveg_names
        assert "Burger King" in nonveg_names




