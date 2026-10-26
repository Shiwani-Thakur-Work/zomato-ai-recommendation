"""
taste_profile.py
----------------
Manages user taste profiles, interactive simulated orders, bookmarked/saved
restaurants, and personalization signals for the Zomato AI Recommendation Engine.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional
import streamlit as st


# ── Signature Menu Catalog by Cuisine ─────────────────────────────────────────
CUISINE_SIGNATURE_MENUS: dict[str, list[dict[str, Any]]] = {
    "north indian": [
        {"item": "Dal Makhani (Slow-Cooked)", "price": 240, "is_veg": True, "badge": "Bestseller"},
        {"item": "Paneer Butter Masala", "price": 290, "is_veg": True, "badge": "Must Try"},
        {"item": "Butter Garlic Naan (2 pcs)", "price": 85, "is_veg": True, "badge": "Popular"},
        {"item": "Kadhai Paneer Special", "price": 310, "is_veg": True, "badge": "Chef Special"},
        {"item": "Gulab Jamun with Rabri", "price": 120, "is_veg": True, "badge": "Dessert"},
    ],
    "south indian": [
        {"item": "Special Ghee Roast Masala Dosa", "price": 160, "is_veg": True, "badge": "Bestseller"},
        {"item": "Steamed Button Idli with Podi Ghee", "price": 110, "is_veg": True, "badge": "Must Try"},
        {"item": "Crispy Medu Vada (2 pcs)", "price": 85, "is_veg": True, "badge": "Crispy"},
        {"item": "Authentic Degree Filter Coffee", "price": 55, "is_veg": True, "badge": "Beverage"},
        {"item": "Rava Onion Dosa", "price": 145, "is_veg": True, "badge": "Classic"},
    ],
    "street food": [
        {"item": "Dahi Bhalla Papdi Chaat", "price": 110, "is_veg": True, "badge": "Bestseller"},
        {"item": "Kurkuri Aloo Tikki with Chole", "price": 120, "is_veg": True, "badge": "Signature"},
        {"item": "Puffed Delhi Chole Bhature (2 pcs)", "price": 150, "is_veg": True, "badge": "Must Try"},
        {"item": "Crispy Gol Gappe (6 Flavored Waters)", "price": 80, "is_veg": True, "badge": "Street Favorite"},
        {"item": "Amritsari Kulcha with Chana", "price": 160, "is_veg": True, "badge": "Authentic"},
    ],
    "italian": [
        {"item": "Wood-Fired Margherita Pizza", "price": 420, "is_veg": True, "badge": "Bestseller"},
        {"item": "Creamy Wild Mushroom Penne", "price": 390, "is_veg": True, "badge": "Chef Special"},
        {"item": "Garlic Sourdough with Melted Mozzarella", "price": 190, "is_veg": True, "badge": "Popular"},
        {"item": "Handcrafted Tiramisu", "price": 260, "is_veg": True, "badge": "Dessert"},
    ],
    "cafe": [
        {"item": "Artisan Cold Brew (Ethiopian Roast)", "price": 195, "is_veg": True, "badge": "Bestseller"},
        {"item": "Avocado & Sundried Tomato Toast", "price": 280, "is_veg": True, "badge": "Healthy"},
        {"item": "Flaky Butter Croissant", "price": 140, "is_veg": True, "badge": "Fresh Baked"},
        {"item": "Spanish Latte / Cappuccino", "price": 180, "is_veg": True, "badge": "Signature"},
    ],
    "sweets": [
        {"item": "Pure Desi Ghee Jalebi (100g)", "price": 95, "is_veg": True, "badge": "Hot & Crisp"},
        {"item": "Royal Kesari Rasmalai (2 pcs)", "price": 130, "is_veg": True, "badge": "Bestseller"},
        {"item": "Silver Vark Kaju Katli (250g)", "price": 280, "is_veg": True, "badge": "Gourmet"},
        {"item": "Moong Dal Halwa (Desi Ghee)", "price": 140, "is_veg": True, "badge": "Must Try"},
    ],
    "chinese": [
        {"item": "Steamed Veg Corn & Water Chestnut Dimsums", "price": 240, "is_veg": True, "badge": "Bestseller"},
        {"item": "Wok-Tossed Hakka Noodles with Veggies", "price": 220, "is_veg": True, "badge": "Popular"},
        {"item": "Crispy Honey Chilli Lotus Stem", "price": 260, "is_veg": True, "badge": "Must Try"},
        {"item": "Paneer Manchurian Gravy", "price": 270, "is_veg": True, "badge": "Classic"},
    ],
}

DEFAULT_MENU = [
    {"item": "Chef's Special Tasting Platter", "price": 320, "is_veg": True, "badge": "Bestseller"},
    {"item": "Signature House Curry with Bread", "price": 260, "is_veg": True, "badge": "Must Try"},
    {"item": "Crispy Starter & House Dips", "price": 180, "is_veg": True, "badge": "Popular"},
    {"item": "Artisan Dessert of the Day", "price": 150, "is_veg": True, "badge": "Dessert"},
]


# ── State Initialization ──────────────────────────────────────────────────────
def init_taste_profile_state() -> None:
    """Ensure session state structures exist for user orders and bookmarks."""
    if "orders" not in st.session_state:
        st.session_state["orders"] = []
    if "saved_restaurants" not in st.session_state:
        st.session_state["saved_restaurants"] = {}
    if "view_mode" not in st.session_state:
        st.session_state["view_mode"] = "explore"  # "explore" | "saved" | "orders" | "taste_profile"


def get_signature_dishes(cuisines: list[str]) -> list[dict[str, Any]]:
    """Retrieve realistic signature dishes based on restaurant cuisines."""
    c_str = " ".join([c.lower() for c in cuisines])
    for key, items in CUISINE_SIGNATURE_MENUS.items():
        if key in c_str:
            return items
    return DEFAULT_MENU


def add_order(
    restaurant_name: str,
    item_name: str,
    price: int,
    cuisine: str,
    locality: str,
) -> None:
    """Record an interactive order to the user's session state and update profile."""
    init_taste_profile_state()
    order_record = {
        "order_id": f"ZOM-{len(st.session_state['orders']) + 101}",
        "restaurant_name": restaurant_name,
        "item_name": item_name,
        "price": price,
        "cuisine": cuisine,
        "locality": locality,
        "timestamp": datetime.datetime.now().strftime("%I:%M %p, %d %b"),
        "status": "Delivered 🛵",
    }
    st.session_state["orders"].insert(0, order_record)


def toggle_saved_restaurant(restaurant: dict[str, Any]) -> bool:
    """Toggle restaurant in the user's saved favorites list."""
    init_taste_profile_state()
    name = restaurant.get("restaurant_name") or restaurant.get("name")
    if not name:
        return False

    if name in st.session_state["saved_restaurants"]:
        del st.session_state["saved_restaurants"][name]
        return False
    else:
        st.session_state["saved_restaurants"][name] = restaurant
        return True


def is_restaurant_saved(restaurant_name: str) -> bool:
    """Check if a restaurant is in the user's saved list."""
    init_taste_profile_state()
    return restaurant_name in st.session_state["saved_restaurants"]


def compute_user_taste_profile() -> dict[str, Any]:
    """Compute aggregate taste metrics, favorite cuisines, and prompt personalization context."""
    init_taste_profile_state()
    orders = st.session_state["orders"]
    saved = st.session_state["saved_restaurants"]

    if not orders and not saved:
        return {
            "has_profile": False,
            "orders_count": 0,
            "saved_count": 0,
            "top_cuisines": [],
            "avg_spend": 0,
            "prompt_hint": "",
            "badge_summary": "Taste Profile: New Diner (Awaiting First Order)",
        }

    # Aggregate cuisine counts
    cuisine_counts: dict[str, int] = {}
    total_spend = 0

    for o in orders:
        c = o.get("cuisine", "North Indian").title()
        cuisine_counts[c] = cuisine_counts.get(c, 0) + 1
        total_spend += o.get("price", 0)

    for r in saved.values():
        c_list = r.get("cuisine", [])
        if isinstance(c_list, str):
            c_list = [c_list]
        for c in c_list:
            c_title = str(c).title()
            cuisine_counts[c_title] = cuisine_counts.get(c_title, 0) + 1

    sorted_cuisines = sorted(cuisine_counts.items(), key=lambda x: x[1], reverse=True)
    top_cuisines = [c[0] for c in sorted_cuisines[:3]]

    avg_spend = int(total_spend / max(len(orders), 1)) if orders else 500

    # Formulate prompt hint for LLM personalization injection
    order_items = [o.get("item_name") for o in orders[:4]]
    items_str = ", ".join(order_items) if order_items else "Traditional Indian Comfort Food"
    cuisines_str = ", ".join(top_cuisines) if top_cuisines else "Diverse"

    prompt_hint = (
        f"[User Taste Profile Context: Diner has {len(orders)} past orders and {len(saved)} saved venues. "
        f"Demonstrates strong preference for: {cuisines_str}. "
        f"Frequently enjoys dishes like: {items_str}. Average item spend: ₹{avg_spend}. "
        f"Prioritize spots and culinary highlights matching this palate.]"
    )

    badge_summary = f"✨ Personalized for your palate: Favors {cuisines_str} ({len(orders)} orders, {len(saved)} saved)"

    return {
        "has_profile": True,
        "orders_count": len(orders),
        "saved_count": len(saved),
        "top_cuisines": top_cuisines,
        "avg_spend": avg_spend,
        "prompt_hint": prompt_hint,
        "badge_summary": badge_summary,
    }
