"""
inspect_data.py
---------------
CLI tool to inspect the Zomato restaurant dataset directly inside the IDE/terminal.

Usage:
    uv run python scripts/inspect_data.py
    uv run python scripts/inspect_data.py --city faridabad
    uv run python scripts/inspect_data.py --city delhi --cuisine "north indian" --min-rating 4.0
"""
from __future__ import annotations

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Inspect Zomato restaurant dataset")
    parser.add_argument("--city", type=str, default="", help="Filter by city (e.g. faridabad, delhi, gurgaon)")
    parser.add_argument("--locality", type=str, default="", help="Filter by locality substring")
    parser.add_argument("--cuisine", type=str, default="", help="Filter by cuisine substring")
    parser.add_argument("--min-rating", type=float, default=0.0, help="Minimum rating (0.0 to 5.0)")
    parser.add_argument("--limit", type=int, default=15, help="Number of rows to display")
    parser.add_argument("--parquet", type=str, default="data/processed/zomato_cleaned.parquet", help="Path to parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"\n=======================================================")
    print(f"  ZOMATO RESTAURANT DATASET (Total: {len(df):,} records)")
    print(f"=======================================================\n")

    if args.city:
        df = df[df["city"].str.lower() == args.city.lower()]
    if args.locality:
        df = df[df["locality"].str.lower().str.contains(args.locality.lower(), na=False)]
    if args.min_rating > 0.0:
        df = df[df["rating"] >= args.min_rating]
    if args.cuisine:
        q_c = args.cuisine.lower()
        df = df[df["cuisines"].apply(lambda cs: any(q_c in str(c).lower() for c in cs) if isinstance(cs, (list, tuple)) else False)]

    print(f"Filtered Results: {len(df):,} matching restaurants (Showing top {min(len(df), args.limit)}):\n")

    if df.empty:
        print("No matching restaurants found with the given filters.")
        return

    # Prepare display columns
    disp = df.head(args.limit).copy()
    disp["cuisines"] = disp["cuisines"].apply(lambda cs: ", ".join([str(c).title() for c in cs][:3]) if isinstance(cs, (list, tuple)) else str(cs))
    disp["cost"] = disp["cost_for_two"].apply(lambda c: f"Rs.{c}")

    cols = ["name", "city", "locality", "rating", "votes", "cost", "budget_tier", "cuisines"]
    print(disp[cols].to_string(index=False))
    print(f"\nTip: For an interactive web table, open 'dataset_viewer.html' or open 'data/processed/zomato_restaurants_preview.csv' in IDE.\n")

if __name__ == "__main__":
    main()
