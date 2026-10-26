"""
app.py
------
Consumer-facing Zomato AI Restaurant Recommendation Application.
Styled to authentically reflect Zomato's web & mobile experience:
- Authentic Zomato visual identity (Zomato Red #E23744, forest green #24963F ratings)
- Top navigation with search omnibar, category tabs (Delivery, Dining Out, Nightlife)
- Horizontal filter pill bar with active Pure Veg green mode (matching zomato.com/ncr/restaurants?veg=1)
- Interactive, clickable restaurant cards with:
    1. Real-time "Save / Bookmark" toggle (stored in user session)
    2. Interactive "View Menu & Quick Order" with clickable signature dishes
    3. Direct Google Maps navigation links
- Closed-loop AI Taste Profile: Learned preferences from orders & saved venues
  dynamically enrich subsequent recommendations
- 1-Click Curated Dining Collections (Pure Veg Thalis, Street Chaat, Cafes, Mithai)
- Powered by Groq LLM (Llama 3.3 70B) with automatic Heuristic Fallback Engine
"""

from __future__ import annotations

import base64
import os
import sys
import textwrap
import urllib.parse
from pathlib import Path

# Add project root to sys.path so 'src' can be resolved
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.config import DEFAULT_MAX_RESULTS, DEFAULT_MIN_RATING, GROQ_MODEL, MAX_CANDIDATES_FOR_LLM
from src.data.preprocessor import build_processed_dataset
from src.engine.recommend import recommend
from src.engine.schemas import BudgetTier, RecommendationResponse, UserPreferences
from src.ui.taste_profile import (
    add_order,
    compute_user_taste_profile,
    get_signature_dishes,
    init_taste_profile_state,
    is_restaurant_saved,
    toggle_saved_restaurant,
)

# ── Streamlit Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="Zomato | Restaurants & AI Dining Concierge",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize Session State for Orders, Bookmarks, and Active Views
init_taste_profile_state()


# ── Clean HTML Rendering Helper (Prevents Markdown Code-Block Leakage) ─────────
def render_html(html_str: str) -> None:
    """Render HTML safely ensuring zero leading 4-space indentations that trigger code blocks."""
    cleaned = textwrap.dedent(html_str).strip()
    st.markdown(cleaned, unsafe_allow_html=True)


# ── Asset Management & Direct Base64 Image Loading ────────────────────────────
def load_base64_image(filename: str) -> str:
    """Load and base64-encode an asset image for instant HTML card rendering without stale caching."""
    asset_path = PROJECT_ROOT / "src" / "ui" / "assets" / filename
    if not asset_path.exists():
        alt_name = filename.replace("_thumb.jpg", ".jpg")
        asset_path = PROJECT_ROOT / "src" / "ui" / "assets" / alt_name
    if asset_path.exists():
        with open(asset_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
    return ""


def get_restaurant_card_image(cuisines: list[str], name: str) -> str:
    """Match restaurant cuisine and name to the most fitting authentic food image."""
    c_str = (" ".join(cuisines) + " " + name).lower()
    if any(k in c_str for k in ["sweet", "mithai", "dessert", "bakery", "cake", "bikaner", "om sweets", "haldiram"]):
        return load_base64_image("mithai_thumb.jpg")
    if any(k in c_str for k in ["cafe", "coffee", "beverage", "tea", "sandwich", "breakfast", "continental", "italian", "pizza"]):
        return load_base64_image("cafe_thumb.jpg")
    if any(k in c_str for k in ["chaat", "street", "fast food", "roll", "snack", "burger", "tikki", "chole", "bhature"]):
        return load_base64_image("chaat_thumb.jpg")
    return load_base64_image("thali_thumb.jpg")


# ── Authentic Zomato Web CSS Design System ───────────────────────────────────
ZOMATO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
    color-scheme: light !important;
    --primary: #E23744;
    --primary-hover: #DB313F;
    --secondary-green: #24963F;
    --green-bg: #EAF5EC;
    --green-border: #CEEAD6;
    --on-surface: #1C1C1C;
    --on-surface-subtle: #4F4F4F;
    --on-surface-muted: #828282;
    --surface-bg: #F8F9FA;
    --surface-card: #FFFFFF;
    --border-color: #E8E8E8;
    --border-subtle: #FCD4D4;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
.main,
body,
html {
    background-color: var(--surface-bg) !important;
    color: var(--on-surface) !important;
}

/* Hide Streamlit Chrome */
#MainMenu { visibility: hidden !important; display: none !important; }
header { visibility: hidden !important; display: none !important; }
footer { visibility: hidden !important; display: none !important; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
div[data-testid="stToolbarActions"] { display: none !important; }
div[data-testid="stHeaderActionElements"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
div[class*="viewerBadge"] { display: none !important; }
[data-testid="stViewerBadge"] { display: none !important; }

/* Container width & typography */
.main .block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1240px !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif !important;
    color: var(--on-surface) !important;
    font-weight: 700 !important;
}

p, span, label, div {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Zomato Top Navigation Bar ── */
.zomato-header-navbar {
    background: #FFFFFF !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    padding: 0.85rem 1.4rem !important;
    margin-bottom: 1.2rem !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04) !important;
}

.zomato-brand-group {
    display: flex !important;
    align-items: center !important;
    gap: 0.85rem !important;
}

.zomato-logo-main {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.3rem !important;
    font-weight: 800 !important;
    color: var(--primary) !important;
    letter-spacing: -0.05em !important;
    font-style: italic !important;
    line-height: 1 !important;
}

.zomato-ai-badge {
    background: #FFF0F2 !important;
    color: var(--primary) !important;
    border: 1px solid var(--border-subtle) !important;
    font-size: 0.75rem !important;
    font-weight: 800 !important;
    padding: 0.22rem 0.65rem !important;
    border-radius: 9999px !important;
    letter-spacing: 0.5px !important;
}

.zomato-nav-actions {
    display: flex !important;
    align-items: center !important;
    gap: 0.65rem !important;
}


/* ── Zomato Filter Pills Row ── */
.zomato-filter-pills-row {
    display: flex !important;
    align-items: center !important;
    gap: 0.65rem !important;
    flex-wrap: wrap !important;
    margin-bottom: 1.6rem !important;
}

.z-pill {
    background: #FFFFFF !important;
    border: 1px solid #CFCFCF !important;
    color: #4F4F4F !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 0.95rem !important;
    border-radius: 8px !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.45rem !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
}

.z-pill-veg-active {
    background: var(--green-bg) !important;
    border: 1.5px solid var(--secondary-green) !important;
    color: #137333 !important;
    font-weight: 700 !important;
}

.z-veg-dot {
    width: 10px !important;
    height: 10px !important;
    border-radius: 50% !important;
    background-color: var(--secondary-green) !important;
    display: inline-block !important;
}

/* ── Authentic Zomato Collections Grid ── */
.collections-title-cluster {
    margin: 1.6rem 0 1rem 0 !important;
}

.collections-title {
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    color: var(--on-surface) !important;
    margin: 0 !important;
}

.collections-subtitle {
    font-size: 0.95rem !important;
    color: var(--on-surface-subtle) !important;
    margin-top: 0.25rem !important;
}

.z-collection-card {
    position: relative !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    height: 220px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
    cursor: pointer !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease !important;
}

.z-collection-card:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 8px 24px rgba(226, 55, 68, 0.16) !important;
}

.z-collection-bg-img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    display: block !important;
}

.z-collection-overlay {
    position: absolute !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    padding: 1.1rem 1rem !important;
    background: linear-gradient(to top, rgba(0, 0, 0, 0.88) 0%, rgba(0, 0, 0, 0.4) 60%, transparent 100%) !important;
    color: #FFFFFF !important;
}

.z-collection-name {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    margin-bottom: 0.2rem !important;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5) !important;
}

.z-collection-places {
    font-size: 0.82rem !important;
    color: #E0E0E0 !important;
    display: flex !important;
    align-items: center !important;
    gap: 4px !important;
}

/* ── Zomato Restaurant Card (Exact Web UI Match) ── */
.zomato-card-container {
    background: #FFFFFF !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    margin-bottom: 1.5rem !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.zomato-card-container:hover {
    box-shadow: 0 8px 26px rgba(0, 0, 0, 0.09) !important;
    border-color: #D6D6D6 !important;
}

.z-card-image-wrap {
    position: relative !important;
    height: 200px !important;
    overflow: hidden !important;
    background: #EEEEEE !important;
}

.z-card-img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    transition: transform 0.4s ease !important;
}

.zomato-card-container:hover .z-card-img {
    transform: scale(1.03) !important;
}

.z-badge-offer {
    position: absolute !important;
    bottom: 12px !important;
    left: 12px !important;
    background: #256FEF !important;
    color: #FFFFFF !important;
    font-size: 0.75rem !important;
    font-weight: 800 !important;
    padding: 0.25rem 0.65rem !important;
    border-radius: 6px !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25) !important;
}

.z-badge-rank {
    position: absolute !important;
    top: 12px !important;
    left: 12px !important;
    background: rgba(28, 28, 28, 0.88) !important;
    backdrop-filter: blur(4px) !important;
    color: #FFFFFF !important;
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    padding: 0.25rem 0.65rem !important;
    border-radius: 6px !important;
}

.z-card-content {
    padding: 1.15rem 1.3rem !important;
}

.z-card-row-1 {
    display: flex !important;
    justify-content: space-between !important;
    align-items: flex-start !important;
    margin-bottom: 0.35rem !important;
}

.z-restaurant-name {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: var(--on-surface) !important;
    margin: 0 !important;
    line-height: 1.25 !important;
}

.z-rating-pill {
    background: var(--secondary-green) !important;
    color: #FFFFFF !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    padding: 0.25rem 0.6rem !important;
    border-radius: 6px !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 3px !important;
    flex-shrink: 0 !important;
}

.z-card-row-2 {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    font-size: 0.92rem !important;
    color: var(--on-surface-subtle) !important;
    margin-bottom: 0.3rem !important;
}

.z-cuisines-text {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    max-width: 70% !important;
}

.z-cost-text {
    font-weight: 600 !important;
    color: var(--on-surface) !important;
}

.z-card-row-3 {
    font-size: 0.85rem !important;
    color: var(--on-surface-muted) !important;
    margin-bottom: 0.75rem !important;
}

/* AI Justification Box */
.z-ai-justification {
    background: #FFFDF9 !important;
    border-left: 3.5px solid var(--primary) !important;
    border: 1px solid #F5ECE0 !important;
    border-left-width: 3.5px !important;
    border-radius: 0 8px 8px 0 !important;
    padding: 0.75rem 1rem !important;
    margin-top: 0.7rem !important;
}

.z-ai-title {
    font-size: 0.75rem !important;
    font-weight: 800 !important;
    color: var(--primary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 0.25rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.3rem !important;
}

.z-ai-text {
    font-size: 0.88rem !important;
    color: #2D2D2D !important;
    line-height: 1.5 !important;
    margin: 0 !important;
}

/* Interactive Food Menu Shelf */
.menu-shelf-container {
    background: #FDFDFD !important;
    border: 1px dashed #E0E0E0 !important;
    border-radius: 10px !important;
    padding: 0.85rem !important;
    margin-top: 0.85rem !important;
}

.menu-shelf-header {
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    color: var(--on-surface) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
    margin-bottom: 0.55rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
}

.menu-item-card {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    background: #FFFFFF !important;
    border: 1px solid #ECECEC !important;
    border-radius: 8px !important;
    padding: 0.55rem 0.85rem !important;
    margin-bottom: 0.45rem !important;
}

.menu-item-info {
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}

.menu-item-name {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: var(--on-surface) !important;
}

.menu-item-price {
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    color: var(--on-surface-subtle) !important;
}

/* Taste Profile Banner */
.taste-profile-banner {
    background: linear-gradient(135deg, #FFF5F5 0%, #FFFFFF 100%) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    padding: 0.85rem 1.25rem !important;
    margin-bottom: 1.25rem !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    box-shadow: 0 2px 8px rgba(226, 55, 68, 0.05) !important;
}

.taste-text-title {
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    color: var(--primary) !important;
}

.taste-text-sub {
    font-size: 0.82rem !important;
    color: var(--on-surface-subtle) !important;
}
</style>
"""
render_html(ZOMATO_CSS)


# ── DATASET CACHE LOADER ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_cached_dataset() -> pd.DataFrame:
    """Load cleaned Parquet dataset into memory."""
    return build_processed_dataset()


try:
    dataset = get_cached_dataset()
except Exception as exc:
    st.error(f"Error loading restaurant dataset: {exc}")
    dataset = pd.DataFrame()


# ── TOP NAVIGATION BAR ────────────────────────────────────────────────────────
taste_stats = compute_user_taste_profile()
saved_count = len(st.session_state["saved_restaurants"])
orders_count = len(st.session_state["orders"])

saved_badge = f"❤️ Saved ({saved_count})" if saved_count > 0 else "❤️ Saved"
orders_badge = f"🛍️ Orders ({orders_count})" if orders_count > 0 else "🛍️ Orders"

nav_html = f"""
<div class="zomato-header-navbar">
    <div class="zomato-brand-group">
        <span class="zomato-logo-main">zomato</span>
        <span class="zomato-ai-badge">AI CONCIERGE</span>
    </div>
    <div class="zomato-nav-actions">
        <span style="font-size:0.85rem; color:#828282; font-weight:600;">Delhi NCR • Bangalore</span>
    </div>
</div>
"""
render_html(nav_html)

# Top Bar Views Switcher (Explore | Saved Favorites | Orders & Taste Profile)
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([2.5, 1.2, 1.2, 1.4])
with nav_col2:
    if st.button("🍽️ Explore Places", use_container_width=True, type="primary" if st.session_state["view_mode"] == "explore" else "secondary"):
        st.session_state["view_mode"] = "explore"
        st.rerun()

with nav_col3:
    if st.button(saved_badge, use_container_width=True, type="primary" if st.session_state["view_mode"] == "saved" else "secondary"):
        st.session_state["view_mode"] = "saved"
        st.rerun()

with nav_col4:
    if st.button(orders_badge, use_container_width=True, type="primary" if st.session_state["view_mode"] == "orders" else "secondary"):
        st.session_state["view_mode"] = "orders"
        st.rerun()


# ── VIEW: SAVED FAVORITES ─────────────────────────────────────────────────────
if st.session_state["view_mode"] == "saved":
    st.markdown("### ❤️ Your Saved Places")
    saved_dict = st.session_state["saved_restaurants"]
    if not saved_dict:
        st.info("You haven't saved any restaurants yet. Click '❤️ Save' on any restaurant card to bookmark your favorite spots!")
    else:
        st.write(f"You have **{len(saved_dict)}** saved spots.")
        for name, r in list(saved_dict.items()):
            col_s1, col_s2 = st.columns([3.5, 1])
            with col_s1:
                st.markdown(f"**{name}** — ⭐ {r.get('rating', '4.0')} | 📍 {r.get('locality', 'City')} | 💰 {r.get('estimated_cost', '₹600 for two')}")
                st.caption(f"Cuisines: {', '.join(r.get('cuisine', [])) if isinstance(r.get('cuisine'), list) else r.get('cuisine')}")
            with col_s2:
                if st.button(f"Remove", key=f"del_saved_{name}"):
                    del st.session_state["saved_restaurants"][name]
                    st.rerun()
            st.divider()
    st.stop()


# ── VIEW: ORDERS & TASTE PROFILE ──────────────────────────────────────────────
if st.session_state["view_mode"] == "orders":
    st.markdown("### 🛍️ Your Food Orders & AI Taste Profile")
    orders = st.session_state["orders"]

    # Taste Profile Summary Card
    if taste_stats["has_profile"]:
        render_html(
            f"""
            <div class="taste-profile-banner">
                <div>
                    <div class="taste-text-title">✨ AI Taste Profile Active</div>
                    <div class="taste-text-sub">{taste_stats['badge_summary']}</div>
                </div>
                <div style="font-size:0.82rem; font-weight:700; color:#1C1C1C;">
                    Avg Item Spend: ₹{taste_stats['avg_spend']}
                </div>
            </div>
            """
        )

    if not orders:
        st.info("No past orders yet. Click **'🍽️ View Menu & Order'** on any restaurant recommendation card to place a simulated order and train your AI taste profile!")
    else:
        for o in orders:
            col_o1, col_o2 = st.columns([3.5, 1])
            with col_o1:
                st.markdown(f"**{o['item_name']}** from **{o['restaurant_name']}**")
                st.caption(f"ID: {o['order_id']} • {o['locality']} • {o['timestamp']} • Cuisine: {o['cuisine']}")
            with col_o2:
                st.markdown(f"**₹{o['price']}**")
                st.caption(f"{o['status']}")
            st.divider()

        if st.button("Clear Order History"):
            st.session_state["orders"] = []
            st.rerun()
    st.stop()


# ── VIEW: EXPLORE & RECOMMENDATIONS (PRIMARY VIEW) ───────────────────────────



# 2. Taste Profile Personalization Callout (If user has past orders/saved venues)
if taste_stats["has_profile"]:
    render_html(
        f"""
        <div class="taste-profile-banner">
            <div>
                <div class="taste-text-title">✨ AI Personalization Feedback Loop Active</div>
                <div class="taste-text-sub">{taste_stats['badge_summary']}</div>
            </div>
            <div style="font-size:0.8rem; background:#EAF5EC; color:#137333; font-weight:700; padding:0.25rem 0.65rem; border-radius:9999px;">
                Taste Profile Synced
            </div>
        </div>
        """
    )


# 3. Zomato Omnisearch & Filters Console
with st.form("zomato_search_console"):
    # Row 1: Unified Search Omnibar
    col_c, col_loc, col_search = st.columns([1.3, 1.2, 2.5])

    with col_c:
        city_display_map = {
            "Delhi NCR": "delhi",
            "Bangalore (Bengaluru)": "bangalore",
            "Gurgaon (Gurugram)": "gurgaon",
            "Noida": "noida",
            "Faridabad": "faridabad",
            "All Cities": "",
        }
        city_selected_label = st.selectbox(
            "📍 Location",
            options=list(city_display_map.keys()),
            index=0,
            help="Select metropolitan dining region",
        )
        resolved_city = city_display_map[city_selected_label]

    with col_loc:
        locality_input = st.text_input(
            "📌 Locality / Area",
            value="",
            placeholder="e.g. Connaught Place, Indiranagar, Cyber Hub",
            help="Filter to a neighborhood or sector",
        )

    with col_search:
        craving_query = st.text_input(
            "🔍 Search for restaurant, cuisine or a dish",
            value="Dal Makhani, crispy paneer tikka & artisan breads with family",
            placeholder="e.g. Pure Veg Thali, Chole Bhature, artisan cold brew, woodfired pizza",
            help="Natural language craving or vibe (e.g. 'romantic rooftop', 'quick breakfast')",
        )

    # Row 2: Zomato Filter Bar (Pure Veg Pill, Rating, Budget)
    col_f1, col_f2, col_f3, col_f4 = st.columns([1.2, 1.8, 1.4, 1.1])

    with col_f1:
        st.write("")
        is_pure_veg = st.checkbox(
            "🌱 Pure Veg Only",
            value=True,
            help="Filter strictly for 100% pure vegetarian restaurants (matching zomato.com?veg=1)",
        )

    with col_f2:
        budget_selection = st.selectbox(
            "💰 Cost for Two",
            options=["Any Budget", "Low (< ₹400)", "Mid (₹400 – ₹1,000)", "High (> ₹1,000)"],
            index=0,
        )

    with col_f3:
        min_rating = st.slider(
            "⭐ Rating Threshold",
            min_value=0.0,
            max_value=5.0,
            value=3.5,
            step=0.1,
            help="Filter restaurants with rating at or above this score",
        )

    with col_f4:
        max_results = st.selectbox(
            "🔢 Max Results",
            options=[3, 5, 8, 10],
            index=1,
        )

    # Row 3: Cuisines Multiselect
    selected_cuisines = st.multiselect(
        "🍽️ Filter by Cuisines (Optional)",
        options=[
            "North Indian",
            "South Indian",
            "Chinese",
            "Street Food & Chaat",
            "Sweets & Mithai",
            "Italian & Pizzas",
            "Cafe & Beverages",
            "Bakery & Desserts",
            "Fast Food & Burgers",
            "Mughlai & Biryani",
            "Continental",
            "Pan-Asian & Thai",
        ],
        default=[],
        help="Filter specific cuisines",
    )

    submitted = st.form_submit_button("✨ Search with AI Concierge", use_container_width=True)


# 4. Filter Pills Display (Exact visual mimicry of zomato.com/ncr/restaurants)
veg_class = "z-pill z-pill-veg-active" if is_pure_veg else "z-pill"
veg_pill_html = f"""
<div class="zomato-filter-pills-row">
    <span class="z-pill">⚙️ Filters</span>
    <span class="{veg_class}"><span class="z-veg-dot"></span> Pure Veg</span>
    <span class="z-pill">⭐ Rating: {min_rating:.1f}+</span>
    <span class="z-pill">💰 {budget_selection}</span>
    <span class="z-pill">📍 {city_selected_label.split(' (')[0]}</span>
</div>
"""
render_html(veg_pill_html)


# ── RESOLVE SEARCH PREFERENCES & PERSONALIZATION ─────────────────────────────
budget_tier_map = {
    "Any Budget": BudgetTier.ANY,
    "Low (< ₹400)": BudgetTier.LOW,
    "Mid (₹400 – ₹1,000)": BudgetTier.MEDIUM,
    "High (> ₹1,000)": BudgetTier.HIGH,
}
target_budget = budget_tier_map.get(budget_selection, BudgetTier.ANY)

city_clean_name = city_selected_label.split(" (")[0]
loc_clean = locality_input.strip()
if loc_clean and resolved_city:
    effective_location = f"{loc_clean}, {resolved_city}"
elif loc_clean:
    effective_location = loc_clean
elif resolved_city:
    effective_location = resolved_city
else:
    effective_location = "Delhi NCR"

cuisine_map = {
    "North Indian": "north indian",
    "South Indian": "south indian",
    "Chinese": "chinese",
    "Street Food & Chaat": "street food",
    "Sweets & Mithai": "sweets",
    "Italian & Pizzas": "italian",
    "Fast Food & Burgers": "fast food",
    "Bakery & Desserts": "bakery",
    "Cafe & Beverages": "cafe",
    "Mughlai & Biryani": "biryani",
    "Continental": "continental",
    "Pan-Asian & Thai": "asian",
}
canonical_cuisines = [cuisine_map.get(c, c.lower()) for c in selected_cuisines]

# Inject user taste profile context into LLM prompt notes if available
effective_query = craving_query.strip()
if taste_stats["has_profile"]:
    effective_query = f"{effective_query} {taste_stats['prompt_hint']}".strip()

if is_pure_veg and "veg" not in effective_query.lower():
    effective_query = f"pure vegetarian {effective_query}".strip()


# ── RECOMMENDATION EXECUTION & INTERACTIVE CARD RENDERING ─────────────────────
if submitted or ("last_response" in st.session_state and st.session_state["last_response"]):
    if submitted:
        prefs = UserPreferences(
            location=effective_location,
            budget=target_budget,
            cuisines=canonical_cuisines,
            min_rating=min_rating,
            additional_notes=effective_query,
            pure_veg=is_pure_veg,
            max_results=max_results,
        )

        with st.spinner(f"🔍 Finding top dining spots in {city_clean_name}..."):
            try:
                response = recommend(prefs, dataset)
                st.session_state["last_response"] = response
                st.session_state["last_city_name"] = city_clean_name
            except Exception as exc:
                st.error(f"Recommendation engine error: {exc}")
                response = None
    else:
        response = st.session_state["last_response"]
        city_clean_name = st.session_state.get("last_city_name", city_clean_name)

    if response and response.recommendations:
        render_html(
            f"""
            <div style="margin: 1.5rem 0 1rem 0; display:flex; justify-content:space-between; align-items:center;">
                <h3 style="margin:0; font-size:1.45rem; font-weight:800;">🍽️ Top Handpicked Restaurants in {city_clean_name}</h3>
                <span style="color:#24963F; font-size:0.9rem; font-weight:700;">✓ {len(response.recommendations)} Verified Matches</span>
            </div>
            """
        )

        if response.curator_summary:
            relaxation_html = ""
            if response.relaxation_applied:
                relaxation_html = f'<div style="margin-top:0.5rem; font-size:0.85rem; color:#DB313F; font-weight:600;">💡 {response.relaxation_applied}</div>'

            render_html(
                f"""
                <div style="background:#FFFDF9; border:1px solid #F5ECE0; border-left:4px solid #E23744; border-radius:10px; padding:0.9rem 1.2rem; margin-bottom:1.5rem;">
                    <div style="font-size:0.8rem; font-weight:800; color:#E23744; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.25rem;">💡 AI Dining Curator Insights</div>
                    <p style="margin:0; font-size:0.92rem; color:#2D2D2D; line-height:1.5;">{response.curator_summary}</p>
                    {relaxation_html}
                </div>
                """
            )

        # Render Interactive Cards
        for rank, r in enumerate(response.recommendations, start=1):
            r_name = r.restaurant_name
            cuisines_list = r.cuisine if isinstance(r.cuisine, list) else [str(r.cuisine)]
            cuisines_str = ", ".join([c.title() for c in cuisines_list[:4]])
            loc_str = r.locality.title() if r.locality else city_clean_name
            addr_str = r.address if r.address else f"{loc_str}, {city_clean_name}"
            card_img = get_restaurant_card_image(cuisines_list, r_name)

            # Check if this restaurant is currently saved
            is_saved = is_restaurant_saved(r_name)
            save_label = "❤️ Saved" if is_saved else "🤍 Save"

            # Google maps query URL
            query_encoded = urllib.parse.quote(f"{r_name} {loc_str} {city_clean_name}")
            maps_url = f"https://www.google.com/maps/search/?api=1&query={query_encoded}"

            # Visual Restaurant Card Layout
            card_visual_html = f"""
            <div class="zomato-card-container">
                <div class="z-card-image-wrap">
                    <img src="{card_img}" class="z-card-img" alt="{r_name}" />
                    <span class="z-badge-rank">#{rank} Top Match</span>
                    <span class="z-badge-offer">FLAT 20% OFF</span>
                </div>
                <div class="z-card-content">
                    <div class="z-card-row-1">
                        <h4 class="z-restaurant-name">{r_name}</h4>
                        <span class="z-rating-pill">★ {r.rating:.1f}</span>
                    </div>
                    <div class="z-card-row-2">
                        <span class="z-cuisines-text">{cuisines_str}</span>
                        <span class="z-cost-text">{r.estimated_cost}</span>
                    </div>
                    <div class="z-card-row-3">📍 {loc_str}, {city_clean_name} • 32 mins</div>
                    <div class="z-ai-justification">
                        <div class="z-ai-title">✨ Why You'll Love This Place</div>
                        <div class="z-ai-text">{r.ai_explanation}</div>
                    </div>
                </div>
            </div>
            """
            render_html(card_visual_html)

            # Interactive Action Bar for this restaurant
            btn_col1, btn_col2, btn_col3 = st.columns([1.2, 2.2, 1.4])

            with btn_col1:
                if st.button(save_label, key=f"save_btn_{r_name}_{rank}", use_container_width=True):
                    now_saved = toggle_saved_restaurant({
                        "restaurant_name": r_name,
                        "rating": r.rating,
                        "locality": loc_str,
                        "estimated_cost": r.estimated_cost,
                        "cuisine": cuisines_list,
                        "address": addr_str,
                    })
                    if now_saved:
                        st.toast(f"❤️ Saved {r_name} to your Favorites!")
                    else:
                        st.toast(f"Removed {r_name} from Favorites.")
                    st.rerun()

            with btn_col2:
                # View Menu & Interactive Quick Order Expander
                menu_expander = st.expander(f"🍽️ View Menu & Order from {r_name}")
                with menu_expander:
                    dishes = get_signature_dishes(cuisines_list)
                    st.markdown("**Popular Signature Dishes:**")
                    for d in dishes:
                        d_col_info, d_col_order = st.columns([3, 1.2])
                        with d_col_info:
                            badge_txt = f"[{d['badge']}] " if "badge" in d else ""
                            st.write(f"🌱 **{d['item']}** — ₹{d['price']}  \n*{badge_txt}Freshly prepared*")
                        with d_col_order:
                            if st.button(f"+ Order", key=f"ord_{r_name}_{d['item']}_{rank}"):
                                add_order(
                                    restaurant_name=r_name,
                                    item_name=d["item"],
                                    price=d["price"],
                                    cuisine=cuisines_list[0] if cuisines_list else "North Indian",
                                    locality=loc_str,
                                )
                                st.toast(f"✅ Ordered {d['item']}! Added to your AI Taste Profile.")
                                st.rerun()
                        st.markdown("<hr style='margin:0.4rem 0; border:none; border-top:1px dashed #EAEAEA;'>", unsafe_allow_html=True)

            with btn_col3:
                st.markdown(
                    f"""
                    <a href="{maps_url}" target="_blank" style="text-decoration:none;">
                        <div style="background:#F8F9FA; border:1px solid #D9D9D9; border-radius:8px; padding:0.45rem 0.75rem; text-align:center; font-size:0.85rem; font-weight:700; color:#1C1C1C;">
                            📍 Directions ↗
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    else:
        st.warning(f"No restaurants matched your exact criteria in {city_clean_name}. Try broadening your rating threshold or budget tier.")


# ── COLLECTIONS CAROUSEL SECTION (Zomato Signature) ──────────────────────────
render_html(
    """
    <div class="collections-title-cluster">
        <h2 class="collections-title">Collections</h2>
        <div class="collections-subtitle">Explore curated lists of top restaurants, cafes, pubs, and bars in your city, based on trends</div>
    </div>
    """
)

thali_b64 = load_base64_image("thali_thumb.jpg")
chaat_b64 = load_base64_image("chaat_thumb.jpg")
cafe_b64 = load_base64_image("cafe_thumb.jpg")
sweets_b64 = load_base64_image("mithai_thumb.jpg")

col_c1, col_c2, col_c3, col_c4 = st.columns(4)

with col_c1:
    render_html(
        f"""
        <div class="z-collection-card">
            <img src="{thali_b64}" class="z-collection-bg-img" alt="Pure Veg Delights" />
            <div class="z-collection-overlay">
                <div class="z-collection-name">Pure Veg Thalis & Feasts</div>
                <div class="z-collection-places">18 Places &gt;</div>
            </div>
        </div>
        """
    )
    if st.button("Explore Pure Veg ↗", key="col_btn_veg", use_container_width=True):
        st.session_state["last_response"] = None
        st.rerun()

with col_c2:
    render_html(
        f"""
        <div class="z-collection-card">
            <img src="{chaat_b64}" class="z-collection-bg-img" alt="Street Food" />
            <div class="z-collection-overlay">
                <div class="z-collection-name">Iconic Street Bites</div>
                <div class="z-collection-places">24 Places &gt;</div>
            </div>
        </div>
        """
    )
    if st.button("Explore Street Chaat ↗", key="col_btn_chaat", use_container_width=True):
        st.session_state["last_response"] = None
        st.rerun()

with col_c3:
    render_html(
        f"""
        <div class="z-collection-card">
            <img src="{cafe_b64}" class="z-collection-bg-img" alt="Cozy Cafes" />
            <div class="z-collection-overlay">
                <div class="z-collection-name">Cozy Cafes & Bakeries</div>
                <div class="z-collection-places">15 Places &gt;</div>
            </div>
        </div>
        """
    )
    if st.button("Explore Cafes ↗", key="col_btn_cafe", use_container_width=True):
        st.session_state["last_response"] = None
        st.rerun()

with col_c4:
    render_html(
        f"""
        <div class="z-collection-card">
            <img src="{sweets_b64}" class="z-collection-bg-img" alt="Heritage Mithai & Sweets" />
            <div class="z-collection-overlay">
                <div class="z-collection-name">Heritage Mithai & Sweets</div>
                <div class="z-collection-places">12 Places &gt;</div>
            </div>
        </div>
        """
    )
    if st.button("Explore Mithai ↗", key="col_btn_sweets", use_container_width=True):
        st.session_state["last_response"] = None
        st.rerun()
