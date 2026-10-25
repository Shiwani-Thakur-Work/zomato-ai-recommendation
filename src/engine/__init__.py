# src/engine/__init__.py
# Core recommendation engine sub-package.

from src.engine.filter import (
    get_candidates,
    get_candidates_with_metadata,
    is_pure_veg_restaurant,
)
from src.engine.llm_client import (
    build_user_prompt,
    get_recommendations_from_llm,
    heuristic_fallback,
    no_match_response,
)
from src.engine.recommend import recommend
from src.engine.schemas import (
    BudgetTier,
    RecommendationResponse,
    RestaurantCandidate,
    RestaurantRecommendation,
    UserPreferences,
)

__all__ = [
    "BudgetTier",
    "UserPreferences",
    "RestaurantCandidate",
    "RestaurantRecommendation",
    "RecommendationResponse",
    "get_candidates",
    "get_candidates_with_metadata",
    "is_pure_veg_restaurant",
    "recommend",
    "build_user_prompt",
    "get_recommendations_from_llm",
    "heuristic_fallback",
    "no_match_response",
]

