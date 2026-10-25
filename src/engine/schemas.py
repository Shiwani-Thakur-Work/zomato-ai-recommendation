"""
schemas.py
----------
Pydantic v2 data models for user preferences, candidate restaurants,
and structured recommendation responses.

Data Contract Hierarchy:
    BudgetTier (Enum)
    UserPreferences (Input validation)
    RestaurantCandidate (Intermediate representation for filtering & ranking)
    RestaurantRecommendation (Single recommendation item with AI justification)
    RecommendationResponse (Complete top-level API payload)
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class BudgetTier(str, Enum):
    """Supported budget tiers."""
    LOW = "low"
    MEDIUM = "medium"
    MID = "medium"
    HIGH = "high"
    ANY = "any"


class UserPreferences(BaseModel):
    """End-user search criteria and subjective preferences."""

    location: str = Field(
        ...,
        min_length=1,
        description="Target city or locality (e.g. 'Bangalore', 'Indiranagar', 'Koramangala')",
    )
    budget: BudgetTier = Field(
        default=BudgetTier.ANY,
        description="Budget tier constraint: low (<=400), medium (401-1200), high (>1200), or any",
    )
    cuisines: List[str] = Field(
        default_factory=list,
        description="List of desired cuisines (e.g. ['Italian', 'Continental'])",
    )
    min_rating: float = Field(
        default=3.5,
        ge=0.0,
        le=5.0,
        description="Minimum rating threshold between 0.0 and 5.0",
    )
    additional_notes: Optional[str] = Field(
        default="",
        description="Free-form subjective cues (e.g. 'romantic rooftop ambience with outdoor seating')",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Target number of recommendations to return (1-10)",
    )

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Location cannot be empty or pure whitespace.")
        return cleaned

    @field_validator("cuisines", mode="before")
    @classmethod
    def validate_cuisines(cls, v: list[str] | str | None) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = [c.strip() for c in v.split(",") if c.strip()]
        return [str(c).strip().lower() for c in v if str(c).strip()]


class RestaurantCandidate(BaseModel):
    """Clean candidate restaurant record passed into ranking and LLM context."""

    name: str = Field(..., description="Restaurant name")
    city: str = Field(..., description="City name")
    locality: Optional[str] = Field(default=None, description="Neighborhood or area")
    address: Optional[str] = Field(default=None, description="Full street address")
    cuisines: List[str] = Field(default_factory=list, description="List of cuisines")
    cost_for_two: int = Field(..., description="Approximate cost for two in INR")
    budget_tier: str = Field(..., description="Budget tier label: low, medium, or high")
    rating: float = Field(..., description="Rating on 0.0 - 5.0 scale")
    votes: int = Field(..., description="Total votes count")
    highlights: List[str] = Field(default_factory=list, description="Key features and dishes")


class RestaurantRecommendation(BaseModel):
    """An individual restaurant recommendation synthesized by the AI engine."""

    restaurant_name: str = Field(..., description="Exact restaurant name from candidate pool")
    cuisine: List[str] = Field(default_factory=list, description="Cuisines served")
    rating: float = Field(..., description="Restaurant rating")
    estimated_cost: str = Field(..., description="Estimated cost display string (e.g. '₹800 for two')")
    match_rank: int = Field(..., description="1-based rank position in recommendation list")
    ai_explanation: str = Field(
        ...,
        description="Nuanced, grounded explanation detailing why this restaurant matches the user's specific preferences",
    )
    locality: Optional[str] = Field(default=None, description="Locality or neighborhood")
    address: Optional[str] = Field(default=None, description="Street address")


class RecommendationResponse(BaseModel):
    """Top-level structured response returned to the UI or client."""

    status: str = Field(default="success", description="'success', 'partial', or 'fallback'")
    user_query_summary: str = Field(..., description="Concise synopsis of the interpreted user criteria")
    recommendations: List[RestaurantRecommendation] = Field(
        default_factory=list,
        description="Ranked list of restaurant recommendations",
    )
    curator_summary: str = Field(
        ...,
        description="AI curator commentary explaining the overall selection rationale",
    )
    is_fallback: bool = Field(
        default=False,
        description="True if recommendations were generated via heuristic fallback rather than LLM",
    )
    relaxation_applied: Optional[str] = Field(
        default=None,
        description="Description of relaxation step applied if hard filtering had too few results",
    )
