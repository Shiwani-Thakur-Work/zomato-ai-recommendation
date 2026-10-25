"""
cli.py
------
Terminal CLI for Zomato AI Restaurant Recommendations.
Run without arguments for interactive mode, or pass flags directly.

Usage:
    uv run python cli.py
    uv run python cli.py --location "Faridabad" --cuisine "North Indian" --budget low
    uv run python cli.py --location "Bangalore" --notes "cozy cafe with coffee"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows terminal prints UTF-8 emojis without charmap errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.engine.recommend import recommend
from src.engine.schemas import BudgetTier, UserPreferences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🍽️  Zomato AI Restaurant Recommendation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-l", "--location",
        type=str,
        default=None,
        help="City or locality (e.g. Faridabad, Delhi, Gurgaon, Bangalore, Indiranagar)",
    )
    parser.add_argument(
        "-c", "--cuisine",
        type=str,
        default=None,
        help="Desired cuisine(s), comma-separated (e.g. 'North Indian, Street Food' or 'Cafe')",
    )
    parser.add_argument(
        "-b", "--budget",
        type=str,
        choices=["low", "medium", "high", "any"],
        default="any",
        help="Budget tier: low (<=400), medium (401-1200), high (>1200), or any",
    )
    parser.add_argument(
        "-r", "--rating",
        type=float,
        default=3.5,
        help="Minimum rating threshold (0.0 to 5.0)",
    )
    parser.add_argument(
        "-n", "--notes",
        type=str,
        default="",
        help="Vibe, occasion, or notes (e.g. 'romantic rooftop', 'quick breakfast')",
    )
    parser.add_argument(
        "-v", "--pure-veg",
        action="store_true",
        default=False,
        help="Strictly filter for 100 percent pure vegetarian restaurants",
    )
    parser.add_argument(
        "-k", "--max-results",
        type=int,
        default=5,
        help="Maximum number of recommendations to display",
    )
    return parser.parse_args()


def interactive_prompt() -> UserPreferences:
    print("\n" + "=" * 65)
    print("   🍽️   ZOMATO AI DINING CONCIERGE (TERMINAL MODE)   🍽️")
    print("=" * 65 + "\n")

    # Location
    loc_input = input("📍 Enter City or Locality [Default: Faridabad]: ").strip()
    location = loc_input if loc_input else "Faridabad"

    # Pure Veg
    v_input = input("🌱 Pure Vegetarian Only? (y/n) [Default: n]: ").strip().lower()
    pure_veg = v_input in ["y", "yes", "true", "1"]

    # Cuisine
    c_input = input("🍲 Desired Cuisine(s) (optional, e.g. North Indian, Cafe, Biryani): ").strip()
    cuisines = [c.strip() for c in c_input.split(",") if c.strip()] if c_input else []

    # Budget
    b_input = input("💰 Budget Tier (low / medium / high / any) [Default: any]: ").strip().lower()
    budget = BudgetTier(b_input) if b_input in ["low", "medium", "high", "any"] else BudgetTier.ANY

    # Rating
    r_input = input("⭐ Minimum Rating (0.0 - 5.0) [Default: 3.5]: ").strip()
    try:
        min_rating = float(r_input) if r_input else 3.5
    except ValueError:
        min_rating = 3.5

    # Notes
    notes = input("✨ Dining Vibe / Special Notes (optional, e.g. 'cozy, outdoor seating'): ").strip()

    return UserPreferences(
        location=location,
        cuisines=cuisines,
        budget=budget,
        min_rating=min_rating,
        additional_notes=notes,
        pure_veg=pure_veg,
        max_results=5,
    )


def print_results(response) -> None:
    print("\n" + "=" * 68)
    print(f"  🔍 RESULTS FOR: {response.user_query_summary.upper()}")
    print("=" * 68)

    if response.relaxation_applied:
        print(f"\n⚠️  Note: {response.relaxation_applied}")

    if response.curator_summary:
        print(f"\n💡 AI Curator Insights:")
        print(f"   {response.curator_summary}\n")

    if not response.recommendations:
        print("❌ No matching restaurants found for your criteria. Try widening budget or rating.")
        print("=" * 68 + "\n")
        return

    print("─" * 68)
    for rec in response.recommendations:
        cuisines_str = ", ".join(rec.cuisine) if rec.cuisine else "Various"
        locality_str = rec.locality.title() if rec.locality else "Neighborhood"
        veg_str = "  🌱 Pure Veg" if getattr(rec, "is_pure_veg", False) else ""

        print(f"\n[{rec.match_rank}]  {rec.restaurant_name.upper()}{veg_str}")
        print(f"     ⭐ Rating: {rec.rating:.1f}  |  💰 Cost: {rec.estimated_cost}  |  📍 {locality_str}")
        print(f"     🍲 Cuisines: {cuisines_str}")
        print(f"     ✨ AI Match Reason: {rec.ai_explanation}")
        if rec.address:
            print(f"     🏠 Address: {rec.address}")

    print("\n" + "=" * 68 + "\n")


def main() -> None:
    args = parse_args()

    # If no location flag was provided via CLI, launch interactive mode
    if args.location is None:
        prefs = interactive_prompt()
    else:
        cuisines = [c.strip() for c in args.cuisine.split(",") if c.strip()] if args.cuisine else []
        prefs = UserPreferences(
            location=args.location,
            cuisines=cuisines,
            budget=BudgetTier(args.budget),
            min_rating=args.rating,
            additional_notes=args.notes,
            pure_veg=args.pure_veg,
            max_results=args.max_results,
        )

    print(f"\nSearching 33,476+ venues for '{prefs.location}' (Pure Veg: {'ON' if prefs.pure_veg else 'OFF'})...")
    response = recommend(prefs)
    print_results(response)


if __name__ == "__main__":
    main()
