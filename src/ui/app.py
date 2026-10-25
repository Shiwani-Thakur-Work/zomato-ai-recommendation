"""
app.py
------
Consumer-facing Zomato AI Restaurant Recommendation Application.
Styled with the authentic Zomato luxury app visual design language:
- Light theme explicitly enforced with high-contrast, crystal-clear typography
- Front-and-center, highlighted Hero Search & Filter console
- Zero markdown code-block leaking (all HTML properly dedented without 4-space indentation)
- Comprehensive multi-city support: Faridabad, Delhi NCR, Gurgaon, Haryana cities, and Bangalore
- Real food photography for dining collections and recommendation cards
- Strict veg-first curation (pure veg thalis, street chaat, cozy cafes, sweet shops)
- Powered by Groq LLM (qwen/qwen3.8-27b) with robust heuristic fallback

Launch locally:
    uv run streamlit run src/ui/app.py
"""

from __future__ import annotations

import base64
import os
import sys
import textwrap
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

# ── Streamlit Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="Zomato | AI Restaurant Recommendations",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Clean HTML Rendering Helper (Prevents Markdown Code-Block Leakage) ─────────
def render_html(html_str: str) -> None:
    """Render HTML safely ensuring zero leading 4-space indentations that trigger code blocks."""
    cleaned = textwrap.dedent(html_str).strip()
    st.markdown(cleaned, unsafe_allow_html=True)


# ── Asset Management & Base64 Image Caching ──────────────────────────────────
@st.cache_data
def load_base64_image(filename: str) -> str:
    """Load and base64-encode an asset image for instant HTML card rendering."""
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
        return load_base64_image("sweets_thumb.jpg")
    if any(k in c_str for k in ["cafe", "coffee", "beverage", "tea", "sandwich", "breakfast", "continental", "italian", "pizza"]):
        return load_base64_image("cafe_thumb.jpg")
    if any(k in c_str for k in ["chaat", "street", "fast food", "roll", "snack", "burger", "tikki", "chole", "bhature"]):
        return load_base64_image("chaat_thumb.jpg")
    return load_base64_image("thali_thumb.jpg")


# ── Bulletproof Light-Themed Zomato CSS Styling ──────────────────────────────
ZOMATO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

/* Force Light Background & High Contrast Across ALL Systems & OS Modes */
:root {
    color-scheme: light !important;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
.main,
body,
html {
    background-color: #F8F9FA !important;
    color: #1C1C1C !important;
}

/* Hide ALL Streamlit Developer Chrome & Deploy Buttons */
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

/* Base App Container */
.main .block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1200px !important;
}

/* Typography Overrides to Guarantee 100% Readability */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    color: #1C1C1C !important;
    font-weight: 700 !important;
}

p, span, label, div {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #1C1C1C;
}

/* Zomato Top Navigation Bar */
.zomato-nav-bar {
    background: #FFFFFF !important;
    border: 1px solid #EAEAEA !important;
    border-radius: 16px !important;
    padding: 1rem 1.6rem !important;
    margin-bottom: 1.5rem !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04) !important;
}

.brand-section {
    display: flex !important;
    align-items: center !important;
    gap: 1rem !important;
}

.zomato-logo-text {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.3rem !important;
    font-weight: 800 !important;
    color: #E23744 !important;
    letter-spacing: -0.04em !important;
    font-style: italic !important;
    line-height: 1 !important;
}

.nav-city-badge {
    background: #FFF5F5 !important;
    color: #E23744 !important;
    border: 1px solid #FCD4D4 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0.35rem 0.85rem !important;
    border-radius: 20px !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.3rem !important;
}

.concierge-pill {
    background: #24963F !important;
    color: #FFFFFF !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    padding: 0.35rem 0.85rem !important;
    border-radius: 20px !important;
    letter-spacing: 0.4px !important;
    text-transform: uppercase !important;
}

/* HIGHLIGHTED HERO SEARCH CONSOLE */
.search-console-hero {
    background: #FFFFFF !important;
    border: 2px solid #FCD4D4 !important;
    border-radius: 22px !important;
    padding: 2.2rem 2.5rem 1.8rem 2.5rem !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 10px 30px rgba(226, 55, 68, 0.08) !important;
}

.hero-brand-tag {
    display: inline-block !important;
    background: #FFF5F5 !important;
    color: #E23744 !important;
    border: 1px solid #FCD4D4 !important;
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    padding: 0.25rem 0.75rem !important;
    border-radius: 20px !important;
    margin-bottom: 0.6rem !important;
}

.hero-main-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #1C1C1C !important;
    margin: 0 0 0.35rem 0 !important;
    line-height: 1.25 !important;
}

.hero-subtext {
    font-size: 1rem !important;
    color: #555555 !important;
    margin: 0 0 1.5rem 0 !important;
    font-weight: 500 !important;
}

/* Form Controls & Inputs */
.stTextInput > div > div > input {
    background-color: #FFFFFF !important;
    color: #1C1C1C !important;
    border: 1.5px solid #D0D5DD !important;
    border-radius: 10px !important;
    padding: 0.65rem 0.9rem !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}

.stTextInput > div > div > input:focus {
    border-color: #E23744 !important;
    box-shadow: 0 0 0 3px rgba(226, 55, 68, 0.15) !important;
}

.stSelectbox > div > div {
    background-color: #FFFFFF !important;
    color: #1C1C1C !important;
    border: 1.5px solid #D0D5DD !important;
    border-radius: 10px !important;
}

label[data-testid="stWidgetLabel"] p {
    font-weight: 700 !important;
    color: #2D2D2D !important;
    font-size: 0.92rem !important;
}

/* Big Red Zomato Submit Button */
.stButton > button {
    background: linear-gradient(135deg, #E23744 0%, #D32330 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.18rem !important;
    padding: 0.85rem 2rem !important;
    border-radius: 12px !important;
    width: 100% !important;
    box-shadow: 0 4px 16px rgba(226, 55, 68, 0.35) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
    margin-top: 0.7rem !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #D32330 0%, #B81D28 100%) !important;
    box-shadow: 0 6px 22px rgba(226, 55, 68, 0.5) !important;
    transform: translateY(-2px) !important;
}

/* Quick Suggestion Pills Bar */
.quick-chips-wrapper {
    margin-top: 1.2rem !important;
    padding-top: 1rem !important;
    border-top: 1px dashed #EAEAEA !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    flex-wrap: wrap !important;
}

.quick-chip-label {
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    color: #696969 !important;
    margin-right: 0.4rem !important;
}

.quick-chip-item {
    background: #FFF5F5 !important;
    color: #E23744 !important;
    border: 1px solid #FCD4D4 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 0.3rem 0.75rem !important;
    border-radius: 20px !important;
    display: inline-block !important;
}

/* Dining Collections Grid */
.collection-section-header {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.45rem !important;
    font-weight: 800 !important;
    color: #1C1C1C !important;
    margin: 2.5rem 0 1.2rem 0 !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}

.collection-card {
    background: #FFFFFF !important;
    border: 1px solid #EAEAEA !important;
    border-radius: 18px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.25s ease-in-out !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
}

.collection-card:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 10px 26px rgba(226, 55, 68, 0.14) !important;
    border-color: #FCD4D4 !important;
}

.collection-img {
    width: 100% !important;
    height: 165px !important;
    object-fit: cover !important;
    display: block !important;
}

.collection-body {
    padding: 1.2rem 1.35rem !important;
    flex: 1 !important;
    background: #FFFFFF !important;
}

.collection-body h4 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    color: #1C1C1C !important;
    margin: 0 0 0.4rem 0 !important;
}

.collection-body p {
    font-size: 0.88rem !important;
    color: #555555 !important;
    line-height: 1.5 !important;
    margin: 0 !important;
    font-weight: 500 !important;
}

/* Recommendation Results Cards */
.results-header-banner {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin: 2.5rem 0 1.2rem 0 !important;
    padding-bottom: 0.8rem !important;
    border-bottom: 2px solid #EAEAEA !important;
}

.results-header-banner h3 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    color: #1C1C1C !important;
    margin: 0 !important;
}

.zomato-restaurant-card {
    background: #FFFFFF !important;
    border: 1px solid #EAEAEA !important;
    border-radius: 18px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06) !important;
    margin-bottom: 1.6rem !important;
    transition: all 0.2s ease-in-out !important;
}

.zomato-restaurant-card:hover {
    border-color: #E23744 !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 28px rgba(226, 55, 68, 0.14) !important;
}

.card-media-wrap {
    position: relative !important;
    width: 100% !important;
    height: 185px !important;
    overflow: hidden !important;
}

.card-media-wrap img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    transition: transform 0.35s ease !important;
}

.zomato-restaurant-card:hover .card-media-wrap img {
    transform: scale(1.03) !important;
}

.overlay-rank-tag {
    position: absolute !important;
    top: 14px !important;
    left: 14px !important;
    background: #E23744 !important;
    color: #FFFFFF !important;
    font-size: 0.82rem !important;
    font-weight: 800 !important;
    padding: 0.35rem 0.8rem !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    letter-spacing: 0.5px !important;
}

.overlay-discount-tag {
    position: absolute !important;
    bottom: 12px !important;
    left: 14px !important;
    background: rgba(28, 28, 28, 0.9) !important;
    backdrop-filter: blur(4px) !important;
    color: #FFFFFF !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    padding: 0.3rem 0.7rem !important;
    border-radius: 6px !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
}

.overlay-time-tag {
    position: absolute !important;
    bottom: 12px !important;
    right: 14px !important;
    background: #FFFFFF !important;
    color: #1C1C1C !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    padding: 0.28rem 0.65rem !important;
    border-radius: 6px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
}

.card-content-area {
    padding: 1.35rem 1.6rem !important;
    background: #FFFFFF !important;
}

.card-top-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin-bottom: 0.4rem !important;
}

.restaurant-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    color: #1C1C1C !important;
    margin: 0 !important;
}

.rating-badge-pill {
    background: #24963F !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    padding: 0.28rem 0.7rem !important;
    border-radius: 8px !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.25rem !important;
    white-space: nowrap !important;
}

.locality-subrow {
    font-size: 0.92rem !important;
    color: #666666 !important;
    margin-bottom: 0.75rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.4rem !important;
}

.cost-highlight {
    font-weight: 700 !important;
    color: #1C1C1C !important;
    margin-left: 0.4rem !important;
}

.cuisine-badge-item {
    background: #FFF5F5 !important;
    color: #E23744 !important;
    border: 1px solid #FCD4D4 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 0.25rem 0.65rem !important;
    border-radius: 6px !important;
    display: inline-block !important;
    margin-right: 0.35rem !important;
    margin-bottom: 0.35rem !important;
}

.veg-indicator {
    background: #E6F4EA !important;
    color: #137333 !important;
    border: 1px solid #CEEAD6 !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    padding: 0.25rem 0.6rem !important;
    border-radius: 6px !important;
    display: inline-inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    margin-right: 0.4rem !important;
}

/* AI Recommendation Insight Box */
.ai-insight-box {
    background: #FFFDF9 !important;
    border-left: 4px solid #E23744 !important;
    border: 1px solid #F5ECE0 !important;
    border-left-width: 4px !important;
    border-radius: 0 10px 10px 0 !important;
    padding: 0.95rem 1.25rem !important;
    margin-top: 1rem !important;
}

.ai-insight-title {
    font-size: 0.82rem !important;
    font-weight: 800 !important;
    color: #E23744 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 0.25rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.3rem !important;
}

.ai-insight-text {
    font-size: 0.92rem !important;
    color: #2D2D2D !important;
    line-height: 1.55 !important;
    margin: 0 !important;
}
</style>
"""

render_html(ZOMATO_CSS)


# ── Load Dataset ──────────────────────────────────────────────────────────────
@st.cache_data
def get_dataset() -> pd.DataFrame:
    """Load canonical cleaned dataset from parquet."""
    try:
        return build_processed_dataset()
    except Exception as exc:
        st.error(f"Error loading restaurant dataset: {exc}")
        return pd.DataFrame()


dataset = get_dataset()


# ── ZOMATO TOP NAVIGATION BAR ─────────────────────────────────────────────────
render_html(
    """
    <div class="zomato-nav-bar">
        <div class="brand-section">
            <span class="zomato-logo-text">zomato</span>
            <span class="nav-city-badge">📍 Faridabad • Delhi NCR • Haryana • Bangalore (33,476+ Venues)</span>
        </div>
        <div>
            <span class="concierge-pill">AI Dining Concierge</span>
        </div>
    </div>
    """
)


# ── MAIN SCREEN: HIGHLIGHTED HERO SEARCH & FILTER CONSOLE ─────────────────────
render_html(
    """
    <div class="search-console-hero">
        <div class="hero-brand-tag">🔴 ZOMATO AI RESTAURANT DISCOVERY</div>
        <h1 class="hero-main-title">Find the Best Places to Eat & Drink</h1>
        <p class="hero-subtext">Instant personalized recommendations across <b>Faridabad, Delhi NCR, Gurgaon, Haryana & Bangalore</b></p>
    </div>
    """
)

# Search Form placed directly on the main page for maximum visibility
with st.form("main_search_form"):
    # Row 1: City, Area, and Craving Inputs
    col_city, col_locality, col_craving = st.columns([1.3, 1.3, 2.4])

    with col_city:
        city_display_map = {
            "Faridabad (Haryana)": "faridabad",
            "Delhi (NCR)": "delhi",
            "Gurgaon / Gurugram (Haryana)": "gurgaon",
            "Noida (NCR)": "noida",
            "Bangalore / Bengaluru (Karnataka)": "bangalore",
            "Rohtak (Haryana)": "rohtak",
            "Panipat (Haryana)": "panipat",
            "Hisar (Haryana)": "hisar",
            "Ambala (Haryana)": "ambala",
            "Sonipat (Haryana)": "sonipat",
            "Karnal (Haryana)": "karnal",
            "Bahadurgarh (Haryana)": "bahadurgarh",
            "All Regions (Delhi NCR, Haryana & Bangalore)": "",
        }
        city_selected_label = st.selectbox(
            "📍 Select City / Region",
            options=list(city_display_map.keys()),
            index=0,  # Default to Faridabad
            help="Choose your target city across Delhi NCR, Haryana, or Bangalore",
        )
        resolved_city = city_display_map[city_selected_label]

    with col_locality:
        locality_input = st.text_input(
            "📌 Locality / Sector (Optional)",
            value="",
            placeholder="e.g. Sector 16, NIT, Indiranagar, Cyber Hub",
            help="Filter to a specific sector, market, or neighborhood",
        )

    with col_craving:
        craving_query = st.text_input(
            "🔍 Your Food Craving or Dish",
            value="Spicy paneer tikka and dal makhani with family",
            placeholder="e.g. Pure Veg Thali, Dal Makhani & Naan, Chole Bhature, artisan coffee",
            help="Describe your food craving, dishes, or dining mood",
        )

    # Row 2: Instant Filter Controls
    col_f1, col_f2, col_f3, col_f4 = st.columns([1.2, 1.8, 1.4, 1.1])

    with col_f1:
        st.write("")  # Spacing
        is_pure_veg = st.checkbox(
            "🌱 Pure Veg Only",
            value=True,
            help="Strictly filter for pure vegetarian restaurants and family dining",
        )

    with col_f2:
        budget_selection = st.selectbox(
            "💰 Budget for Two",
            options=["Low (< ₹400)", "Mid (₹400 – ₹1,000)", "High (> ₹1,000)"],
            index=0,
        )

    with col_f3:
        min_rating = st.slider(
            "⭐ Minimum Rating",
            min_value=0.0,
            max_value=5.0,
            value=3.5,
            step=0.1,
            help="Filter restaurants with rating at or above this threshold",
        )

    with col_f4:
        max_results = st.selectbox(
            "🔢 Max Results",
            options=[3, 5, 8, 10],
            index=1,
        )

    # Prominent Submit Button
    submitted = st.form_submit_button("🍽️ Discover Top Restaurants", use_container_width=True)

# Quick Cravings Suggestion Bar
render_html(
    """
    <div class="quick-chips-wrapper">
        <span class="quick-chip-label">Popular Cravings:</span>
        <span class="quick-chip-item">🥗 Royal Pure Veg Thali</span>
        <span class="quick-chip-item">🍲 Delhi Chole Bhature</span>
        <span class="quick-chip-item">☕ Cozy Cafe & Bakery</span>
        <span class="quick-chip-item">🍨 Sweets & Mithai</span>
        <span class="quick-chip-item">🍛 Dal Makhani & Naan</span>
    </div>
    """
)


# ── EXECUTION & RECOMMENDATION RESULTS ────────────────────────────────────────

budget_tier_map = {
    "Low (< ₹400)": BudgetTier.LOW,
    "Mid (₹400 – ₹1,000)": BudgetTier.MEDIUM,
    "High (> ₹1,000)": BudgetTier.HIGH,
}
target_budget = budget_tier_map[budget_selection]

# Resolve effective target location for engine
city_clean_name = city_selected_label.split(" (")[0]
effective_location = locality_input.strip() if locality_input.strip() else (resolved_city if resolved_city else "Delhi NCR")

# Pure veg query injection if toggle is enabled
effective_query = craving_query.strip()
if is_pure_veg and "veg" not in effective_query.lower():
    effective_query = f"pure vegetarian {effective_query}".strip()

if submitted:
    prefs = UserPreferences(
        location=effective_location,
        budget=target_budget,
        min_rating=min_rating,
        additional_notes=effective_query,
        max_results=max_results,
    )

    with st.spinner(f"🔍 Curating the best dining matches in {city_clean_name}..."):
        try:
            response: RecommendationResponse = recommend(prefs, dataset)
        except Exception as exc:
            st.error(f"Recommendation engine error: {exc}")
            response = None

    if response and response.recommendations:
        render_html(
            f"""
            <div class="results-header-banner">
                <h3>🍽️ Top Recommendations in {city_clean_name}</h3>
                <span style="color:#24963F; font-size:0.95rem; font-weight:700;">✓ {len(response.recommendations)} Verified Matches</span>
            </div>
            """
        )

        if response.curator_summary:
            render_html(
                f"""
                <div style="background:#FFF9F9; border:1px solid #FCD4D4; border-radius:14px; padding:1.2rem 1.5rem; margin-bottom:1.6rem;">
                    <div style="font-weight:800; color:#E23744; font-size:0.95rem; margin-bottom:0.3rem;">ZOMATO CURATOR'S VERDICT</div>
                    <p style="margin:0; font-size:0.94rem; color:#2D2D2D; line-height:1.55;">{response.curator_summary}</p>
                </div>
                """
            )

        # Render Authentic Zomato Cards WITHOUT Markdown 4-space code block leakage
        for rank, r in enumerate(response.recommendations, start=1):
            cuisines_list = r.cuisine if isinstance(r.cuisine, list) else [str(r.cuisine)]
            card_img = get_restaurant_card_image(cuisines_list, r.restaurant_name)
            cuisines_html = "".join([f'<span class="cuisine-badge-item">{c.title()}</span>' for c in cuisines_list[:4]])

            veg_html = '<span class="veg-indicator">🌱 Pure Veg</span>' if is_pure_veg or any("veg" in c.lower() for c in cuisines_list) else ''
            loc_str = r.locality.title() if r.locality else city_clean_name

            # Flat HTML with zero leading indentation to guarantee no markdown code block triggers
            card_html = (
                f'<div class="zomato-restaurant-card">'
                f'<div class="card-media-wrap">'
                f'<img src="{card_img}" alt="{r.restaurant_name}" />'
                f'<span class="overlay-rank-tag">#{rank} TOP MATCH</span>'
                f'<span class="overlay-discount-tag">🔥 FLAT 20% OFF</span>'
                f'<span class="overlay-time-tag">25-30 MINS</span>'
                f'</div>'
                f'<div class="card-content-area">'
                f'<div class="card-top-row">'
                f'<h4 class="restaurant-title">{r.restaurant_name}</h4>'
                f'<span class="rating-badge-pill">★ {r.rating:.1f}</span>'
                f'</div>'
                f'<div class="locality-subrow">'
                f'<span>📍 {loc_str}, {city_clean_name}</span>'
                f'<span>•</span>'
                f'<span class="cost-highlight">{r.estimated_cost}</span>'
                f'</div>'
                f'<div style="margin-bottom:0.4rem;">{veg_html}{cuisines_html}</div>'
                f'<div class="ai-insight-box">'
                f'<div class="ai-insight-title">✨ Why this matches your craving</div>'
                f'<div class="ai-insight-text">{r.ai_explanation}</div>'
                f'</div>'
                f'</div>'
                f'</div>'
            )
            render_html(card_html)

    else:
        st.warning(f"No restaurants matched your exact criteria in {city_clean_name}. Try broadening your rating threshold or area.")


# ── CURATED DINING COLLECTIONS SHOWCASE ────────────────────────────────────────
render_html('<div class="collection-section-header">🌟 Curated Dining Collections</div>')

thali_b64 = load_base64_image("thali_thumb.jpg")
chaat_b64 = load_base64_image("chaat_thumb.jpg")
cafe_b64 = load_base64_image("cafe_thumb.jpg")
sweets_b64 = load_base64_image("sweets_thumb.jpg")

col1, col2 = st.columns(2)
with col1:
    render_html(
        f"""
        <div class="collection-card">
            <img src="{thali_b64}" class="collection-img" alt="Pure Veg Thalis" />
            <div class="collection-body">
                <h4>🥗 Pure Veg Delights & Royal Thalis</h4>
                <p>Rich Dal Makhani, Paneer Butter Masala, Tandoori Rotis, and unlimited traditional Rajasthani and North Indian thalis for family dining.</p>
            </div>
        </div>
        """
    )

with col2:
    render_html(
        f"""
        <div class="collection-card">
            <img src="{chaat_b64}" class="collection-img" alt="Delhi NCR Street Food" />
            <div class="collection-body">
                <h4>🍲 Famous Delhi NCR Street Food & Chaat</h4>
                <p>Puffed golden Chole Bhature, spicy Gol Gappe, buttery Pav Bhaji, Aloo Tikki, and crispy evening snack joints.</p>
            </div>
        </div>
        """
    )

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    render_html(
        f"""
        <div class="collection-card">
            <img src="{cafe_b64}" class="collection-img" alt="Aesthetic Cafes" />
            <div class="collection-body">
                <h4>☕ Cozy Cafes & Artisan Bakeries</h4>
                <p>Freshly brewed espresso, artisan croissants, avocado toasts, and relaxed work-friendly cafes with peaceful ambiance.</p>
            </div>
        </div>
        """
    )

with col4:
    render_html(
        f"""
        <div class="collection-card">
            <img src="{sweets_b64}" class="collection-img" alt="Heritage Sweet Shops" />
            <div class="collection-body">
                <h4>🍨 Festive Family Sweets & Heritage Dining</h4>
                <p>Iconic mithai makers like Om Sweets, Bikaner, Haldiram's, and heritage family lunch spots for festive celebrations.</p>
            </div>
        </div>
        """
    )
