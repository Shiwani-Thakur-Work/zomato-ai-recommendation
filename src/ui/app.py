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


# ── Bulletproof Light-Themed Zomato CSS Styling (Stitch Design System) ────────
ZOMATO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

/* Force Light Background & High Contrast Across ALL Systems & OS Modes */
:root {
    color-scheme: light !important;
    --primary: #E23744;
    --primary-container: #DB313F;
    --secondary: #24963F;
    --tertiary: #7B2CBF;
    --on-surface: #1C1C1C;
    --on-surface-variant: #696969;
    --surface-bg: #F8F9FA;
    --surface-card: #FFFFFF;
    --border-color: #EAEAEA;
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
div[class*="viewerBadge"] { display: none !important; }
[data-testid="stViewerBadge"] { display: none !important; }
.viewerBadge_container__1QSob { display: none !important; }

/* Base App Container */
.main .block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1220px !important;
}

/* Typography Overrides */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif !important;
    color: var(--on-surface) !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em !important;
}

p, span, label, div {
    font-family: 'Inter', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--on-surface);
}

/* ── Zomato Top Navigation Bar (Stitch Elevated) ── */
.zomato-nav-bar {
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 18px !important;
    padding: 0.95rem 1.6rem !important;
    margin-bottom: 1.6rem !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04) !important;
}

.brand-section {
    display: flex !important;
    align-items: center !important;
    gap: 1.1rem !important;
}

.zomato-logo-text {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.25rem !important;
    font-weight: 800 !important;
    color: var(--primary) !important;
    letter-spacing: -0.04em !important;
    font-style: italic !important;
    line-height: 1 !important;
}

.ai-logo-tag {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-style: normal !important;
    font-weight: 800 !important;
    color: var(--tertiary) !important;
    font-size: 1.7rem !important;
    margin-left: -2px !important;
}

.nav-city-badge {
    background: #FFF5F5 !important;
    color: var(--primary) !important;
    border: 1px solid var(--border-subtle) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 0.35rem 0.85rem !important;
    border-radius: 9999px !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.35rem !important;
}

.nav-right-cluster {
    display: flex !important;
    align-items: center !important;
    gap: 0.75rem !important;
}

@keyframes pulse-ring {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.2); }
}

.pulse-dot-wrap {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 8px !important;
    height: 8px !important;
    border-radius: 50% !important;
    background-color: var(--secondary) !important;
    box-shadow: 0 0 0 3px rgba(36, 150, 63, 0.22) !important;
    animation: pulse-ring 2s infinite ease-in-out !important;
    margin-right: 6px !important;
}

.ai-concierge-pill {
    background: #F0EDED !important;
    color: var(--on-surface) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 0.35rem 0.85rem !important;
    border-radius: 9999px !important;
    display: inline-flex !important;
    align-items: center !important;
    border: 1px solid #E5E2E1 !important;
}

.cli-badge-pill {
    background: #FFF5F5 !important;
    color: var(--primary) !important;
    border: 1px solid var(--border-subtle) !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    padding: 0.32rem 0.75rem !important;
    border-radius: 9999px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
}

/* ── HIGHLIGHTED HERO SEARCH CONSOLE (Stitch Top Stripe) ── */
.search-console-hero {
    position: relative !important;
    overflow: hidden !important;
    background: var(--surface-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 20px !important;
    padding: 2.2rem 2.4rem 1.8rem 2.4rem !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05) !important;
}

.search-console-hero::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 4px !important;
    background: linear-gradient(90deg, #E23744 0%, #DB313F 45%, #7B2CBF 100%) !important;
}

.hero-top-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: flex-start !important;
    gap: 1.5rem !important;
    flex-wrap: wrap !important;
    margin-bottom: 1.6rem !important;
}

.hero-brand-tag {
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.4rem !important;
    background: #FFF5F5 !important;
    color: var(--primary) !important;
    border: 1px solid var(--border-subtle) !important;
    font-size: 0.74rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    padding: 0.28rem 0.8rem !important;
    border-radius: 9999px !important;
    margin-bottom: 0.65rem !important;
}

.hero-main-title {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: var(--on-surface) !important;
    margin: 0 0 0.4rem 0 !important;
    line-height: 1.2 !important;
    letter-spacing: -0.02em !important;
}

.hero-subtext {
    font-size: 0.98rem !important;
    color: var(--on-surface-variant) !important;
    margin: 0 !important;
    font-weight: 500 !important;
    max-width: 650px !important;
    line-height: 1.5 !important;
}

.hero-telemetry-badge {
    background: #F6F3F2 !important;
    border: 1px solid #EAE7E7 !important;
    border-radius: 14px !important;
    padding: 0.65rem 1.1rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.75rem !important;
}

.telemetry-icon {
    font-size: 1.25rem !important;
    color: var(--secondary) !important;
    font-weight: 800 !important;
}

.telemetry-label {
    font-size: 0.72rem !important;
    color: var(--on-surface-variant) !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.telemetry-val {
    font-size: 0.9rem !important;
    font-weight: 800 !important;
    color: var(--on-surface) !important;
}

/* Form Controls & Inputs */
.stTextInput > div > div > input {
    background-color: #FFFFFF !important;
    color: var(--on-surface) !important;
    border: 1.5px solid #D0D5DD !important;
    border-radius: 12px !important;
    padding: 0.65rem 0.95rem !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(226, 55, 68, 0.15) !important;
}

.stSelectbox > div > div {
    background-color: #FFFFFF !important;
    color: var(--on-surface) !important;
    border: 1.5px solid #D0D5DD !important;
    border-radius: 12px !important;
}

label[data-testid="stWidgetLabel"] p {
    font-weight: 700 !important;
    color: var(--on-surface) !important;
    font-size: 0.92rem !important;
    letter-spacing: -0.01em !important;
}

/* Big Red Submit CTA Button */
.stButton > button {
    background: linear-gradient(135deg, #E23744 0%, #D32330 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.15rem !important;
    padding: 0.85rem 2rem !important;
    border-radius: 14px !important;
    width: 100% !important;
    box-shadow: 0 6px 20px rgba(226, 55, 68, 0.32) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
    margin-top: 0.75rem !important;
    letter-spacing: 0.01em !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #D32330 0%, #B81D28 100%) !important;
    box-shadow: 0 8px 26px rgba(226, 55, 68, 0.45) !important;
    transform: translateY(-2px) !important;
}

/* Quick Suggestion Pills Bar */
.quick-chips-wrapper {
    margin-top: 1.2rem !important;
    padding-top: 1rem !important;
    border-top: 1px dashed var(--border-color) !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    flex-wrap: wrap !important;
}

.quick-chip-label {
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    color: var(--on-surface-variant) !important;
    margin-right: 0.3rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.quick-chip-item {
    background: #FFF5F5 !important;
    color: var(--primary) !important;
    border: 1px solid var(--border-subtle) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 0.32rem 0.85rem !important;
    border-radius: 9999px !important;
    display: inline-block !important;
    transition: all 0.2s ease !important;
}

.quick-chip-item:hover {
    background: #FFEBEB !important;
    border-color: var(--primary) !important;
}

/* ── AI CURATOR STRATEGY BANNER (Stitch Inspired) ── */
.ai-strategy-card {
    background: linear-gradient(135deg, rgba(226, 55, 68, 0.04) 0%, rgba(123, 44, 191, 0.05) 100%) !important;
    border: 1px solid rgba(226, 55, 68, 0.16) !important;
    border-radius: 18px !important;
    padding: 1.4rem 1.6rem !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.03) !important;
}

.strategy-header {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin-bottom: 0.6rem !important;
    flex-wrap: wrap !important;
    gap: 0.6rem !important;
}

.strategy-title {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 800 !important;
    color: var(--on-surface) !important;
    font-size: 1.05rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.4rem !important;
}

.strategy-badge {
    background: #FFFFFF !important;
    color: var(--on-surface-variant) !important;
    border: 1px solid var(--border-color) !important;
    font-size: 0.75rem !important;
    font-family: monospace !important;
    padding: 0.25rem 0.65rem !important;
    border-radius: 9999px !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04) !important;
}

.strategy-body {
    font-size: 0.95rem !important;
    color: var(--on-surface) !important;
    line-height: 1.6 !important;
    margin: 0 !important;
}

.strategy-relaxation-pill {
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.4rem !important;
    background: #F1DBFF !important;
    color: #4A1570 !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    padding: 0.35rem 0.8rem !important;
    border-radius: 10px !important;
    margin-top: 0.8rem !important;
}

/* ── RECOMMENDATION RESULTS CARDS (Stitch Luxury Layout) ── */
.results-header-banner {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin: 2.2rem 0 1.2rem 0 !important;
    padding-bottom: 0.8rem !important;
    border-bottom: 2px solid var(--border-color) !important;
}

.results-header-banner h3 {
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    color: var(--on-surface) !important;
    margin: 0 !important;
}

.zomato-restaurant-card {
    background: var(--surface-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 18px !important;
    padding: 1.5rem 1.7rem !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.05) !important;
    margin-bottom: 1.4rem !important;
    transition: all 0.25s ease !important;
}

.zomato-restaurant-card:hover {
    border-color: var(--primary) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 26px rgba(226, 55, 68, 0.12) !important;
}

.card-top-header {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 0.8rem !important;
    margin-bottom: 0.5rem !important;
}

.card-title-cluster {
    display: flex !important;
    align-items: center !important;
    gap: 0.75rem !important;
    flex-wrap: wrap !important;
}

.match-rank-badge {
    background: #FFF5F5 !important;
    color: var(--primary) !important;
    border: 1px solid var(--border-subtle) !important;
    font-size: 0.75rem !important;
    font-weight: 800 !important;
    padding: 0.28rem 0.7rem !important;
    border-radius: 9999px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.restaurant-title {
    font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    color: var(--on-surface) !important;
    margin: 0 !important;
    line-height: 1.25 !important;
}

.rating-badge-pill {
    background: var(--secondary) !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    padding: 0.35rem 0.75rem !important;
    border-radius: 8px !important;
    display: inline-flex !important;
    align-items: baseline !important;
    gap: 0.2rem !important;
    box-shadow: 0 2px 6px rgba(36, 150, 63, 0.25) !important;
}

.rating-sub {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    opacity: 0.85 !important;
}

.locality-subrow {
    font-size: 0.92rem !important;
    color: var(--on-surface-variant) !important;
    margin-bottom: 0.75rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    flex-wrap: wrap !important;
}

.meta-dot {
    color: #C0C0C0 !important;
}

.cost-highlight {
    font-weight: 700 !important;
    color: var(--on-surface) !important;
}

.cuisines-row {
    display: flex !important;
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 0.4rem !important;
    margin-bottom: 0.85rem !important;
}

.cuisine-badge-item {
    background: #FFF5F5 !important;
    color: var(--primary) !important;
    border: 1px solid var(--border-subtle) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 0.25rem 0.65rem !important;
    border-radius: 9999px !important;
    display: inline-block !important;
}

.veg-indicator {
    background: #E6F4EA !important;
    color: #137333 !important;
    border: 1px solid #CEEAD6 !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    padding: 0.25rem 0.65rem !important;
    border-radius: 9999px !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
}

/* AI Recommendation Insight Box (Stitch Signature) */
.ai-insight-box {
    background: #FFFDF9 !important;
    border-left: 4px solid var(--primary) !important;
    border: 1px solid #F5ECE0 !important;
    border-left-width: 4px !important;
    border-radius: 0 12px 12px 0 !important;
    padding: 0.95rem 1.25rem !important;
    margin-top: 0.9rem !important;
}

.ai-insight-title {
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    color: var(--primary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    margin-bottom: 0.3rem !important;
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

.card-footer-action-row {
    margin-top: 1rem !important;
    padding-top: 0.85rem !important;
    border-top: 1px solid #F0EDED !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 0.5rem !important;
}

.card-address-text {
    font-size: 0.84rem !important;
    color: var(--on-surface-variant) !important;
    display: flex !important;
    align-items: center !important;
    gap: 4px !important;
}

.card-action-btn-group {
    display: flex !important;
    gap: 0.5rem !important;
}

.card-btn-outline {
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    padding: 0.35rem 0.85rem !important;
    border-radius: 8px !important;
    background: #F8F9FA !important;
    border: 1px solid #EAEAEA !important;
    color: var(--on-surface) !important;
    text-decoration: none !important;
    display: inline-block !important;
}

/* ── DINING COLLECTIONS GRID (Stitch Elevated) ── */
.collection-section-header {
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    color: var(--on-surface) !important;
    margin: 2.8rem 0 1.2rem 0 !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}

.collection-card {
    background: var(--surface-card) !important;
    border: 1px solid var(--border-color) !important;
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
    box-shadow: 0 10px 26px rgba(226, 55, 68, 0.12) !important;
    border-color: var(--border-subtle) !important;
}

.collection-img {
    width: 100% !important;
    height: 165px !important;
    object-fit: cover !important;
    display: block !important;
    transition: transform 0.4s ease !important;
}

.collection-card:hover .collection-img {
    transform: scale(1.03) !important;
}

.collection-tag {
    display: inline-block !important;
    font-size: 0.7rem !important;
    font-weight: 800 !important;
    color: var(--primary) !important;
    background: #FFF5F5 !important;
    padding: 0.18rem 0.55rem !important;
    border-radius: 9999px !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 0.4rem !important;
}

.collection-body {
    padding: 1.2rem 1.4rem !important;
    flex: 1 !important;
    background: var(--surface-card) !important;
}

.collection-body h4 {
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    color: var(--on-surface) !important;
    margin: 0 0 0.4rem 0 !important;
}

.collection-body p {
    font-size: 0.88rem !important;
    color: var(--on-surface-variant) !important;
    line-height: 1.5 !important;
    margin: 0 !important;
    font-weight: 400 !important;
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


# ── ZOMATO TOP NAVIGATION BAR (Clean & User-Centric) ─────────────────────────
render_html(
    """
    <div class="zomato-nav-bar">
        <div class="brand-section">
            <span class="zomato-logo-text">zomato <span class="ai-logo-tag">AI</span></span>
            <span class="nav-city-badge">📍 Delhi NCR • Bangalore • Faridabad (33,476+ Places)</span>
        </div>
        <div class="nav-right-cluster">
            <div class="ai-concierge-pill">
                <span class="pulse-dot-wrap"></span>
                <span>AI Dining Concierge Active</span>
            </div>
        </div>
    </div>
    """
)


# ── MAIN SCREEN: HIGHLIGHTED HERO SEARCH & FILTER CONSOLE ─────────────────────
render_html(
    """
    <div class="search-console-hero">
        <div class="hero-top-row">
            <div>
                <div class="hero-brand-tag">✨ AI-POWERED RESTAURANT DISCOVERY</div>
                <h1 class="hero-main-title">What are you craving today?</h1>
                <p class="hero-subtext">Discover handpicked dining spots and hidden gems tailored dynamically to your taste, budget, and dining mood.</p>
            </div>
            <div class="hero-telemetry-badge">
                <span class="telemetry-icon">✓</span>
                <div>
                    <div class="telemetry-label">Curated Selection</div>
                    <div class="telemetry-val">33,476+ Verified Places</div>
                </div>
            </div>
        </div>
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
            help="Strictly filter for 100% pure vegetarian restaurants and dining",
        )

    with col_f2:
        budget_selection = st.selectbox(
            "💰 Budget for Two",
            options=["Any Budget (All Tiers)", "Low (< ₹400)", "Mid (₹400 – ₹1,000)", "High (> ₹1,000)"],
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

    # Row 3: Cuisine Filters & Quick Craving Presets
    col_c1, col_c2 = st.columns([1.5, 1.5])
    with col_c1:
        selected_cuisines = st.multiselect(
            "🍽️ Filter Cuisines (Optional)",
            options=[
                "North Indian",
                "South Indian",
                "Chinese",
                "Street Food & Chaat",
                "Sweets & Mithai",
                "Italian & Pizzas",
                "Fast Food & Burgers",
                "Bakery & Desserts",
                "Cafe & Beverages",
                "Mughlai & Biryani",
                "Continental",
                "Pan-Asian & Thai",
            ],
            default=[],
            help="Choose specific cuisines or leave empty to match across all cuisines",
        )
    with col_c2:
        quick_craving_preset = st.selectbox(
            "⚡ Quick Craving Shortcut (Optional)",
            options=[
                "Custom Craving (Use text above)",
                "🥗 Pure Veg Thali & North Indian",
                "🍲 Delhi Chole Bhature & Chaat",
                "☕ Cozy Cafe, Coffee & Bakery",
                "🍨 Heritage Mithai, Sweets & Kulfi",
                "🍛 Creamy Dal Makhani & Garlic Naan",
                "🥟 Crisp Street Chaat & Snacks",
                "🍕 Woodfired Pizza & Pasta",
                "🍔 Gourmet Burgers & Shakes",
            ],
            index=0,
            help="Pick a popular curated craving to instantly enrich your search",
        )

    # Prominent Submit Button (Stitch Elevated)
    submitted = st.form_submit_button("✨ Find My Perfect Table", use_container_width=True)

# Quick Cravings Suggestion Bar
render_html(
    """
    <div class="quick-chips-wrapper">
        <span class="quick-chip-label">Popular Cravings:</span>
        <span class="quick-chip-item">🥗 Pure Veg Thali</span>
        <span class="quick-chip-item">🍲 Delhi Chole Bhature</span>
        <span class="quick-chip-item">☕ Cozy Cafe & Sourdough</span>
        <span class="quick-chip-item">🍨 Heritage Mithai</span>
        <span class="quick-chip-item">🍛 Dal Makhani & Naan</span>
        <span class="quick-chip-item">🥟 Crisp Street Chaat</span>
    </div>
    """
)


# ── EXECUTION & RECOMMENDATION RESULTS ────────────────────────────────────────

budget_tier_map = {
    "Any Budget (All Tiers)": BudgetTier.ANY,
    "Low (< ₹400)": BudgetTier.LOW,
    "Mid (₹400 – ₹1,000)": BudgetTier.MEDIUM,
    "High (> ₹1,000)": BudgetTier.HIGH,
}
target_budget = budget_tier_map.get(budget_selection, BudgetTier.ANY)

# Resolve effective target location for engine
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

effective_query = craving_query.strip()
if quick_craving_preset != "Custom Craving (Use text above)":
    preset_clean = quick_craving_preset.split(" ", 1)[-1]
    effective_query = f"{effective_query} {preset_clean}".strip() if effective_query else preset_clean

if is_pure_veg and "veg" not in effective_query.lower():
    effective_query = f"pure vegetarian {effective_query}".strip()

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
                <h3>🍽️ Top Verified AI Matches in {city_clean_name}</h3>
                <span style="color:#24963F; font-size:0.95rem; font-weight:700;">✓ {len(response.recommendations)} Verified Matches</span>
            </div>
            """
        )

        if response.curator_summary:
            relaxation_html = ""
            if response.relaxation_applied:
                relaxation_html = f'<div class="strategy-relaxation-pill">💡 <b>Helpful Note:</b> {response.relaxation_applied}</div>'

            render_html(
                f"""
                <div class="ai-strategy-card">
                    <div class="strategy-header">
                        <div class="strategy-title">💡 AI Dining Curator Insights</div>
                    </div>
                    <p class="strategy-body">{response.curator_summary}</p>
                    {relaxation_html}
                </div>
                """
            )

        # Render Clean, Authentic Restaurant Cards (Image-Free, User-Friendly)
        for rank, r in enumerate(response.recommendations, start=1):
            cuisines_list = r.cuisine if isinstance(r.cuisine, list) else [str(r.cuisine)]
            cuisines_html = "".join([f'<span class="cuisine-badge-item">{c.title()}</span>' for c in cuisines_list[:5]])

            is_veg_card = bool(getattr(r, "is_pure_veg", False))
            veg_html = '<span class="veg-indicator">🌱 Pure Veg</span>' if is_veg_card else ''
            loc_str = r.locality.title() if r.locality else city_clean_name
            addr_str = r.address if r.address else f"{loc_str}, {city_clean_name}"

            # Flat HTML with zero leading indentation to guarantee no markdown code block triggers
            card_html = (
                f'<div class="zomato-restaurant-card">'
                f'<div class="card-top-header">'
                f'<div class="card-title-cluster">'
                f'<span class="match-rank-badge">#{rank} Top Match</span>'
                f'<h4 class="restaurant-title">{r.restaurant_name}</h4>'
                f'</div>'
                f'<div class="rating-badge-pill">'
                f'<span>★ {r.rating:.1f}</span>'
                f'<span class="rating-sub">/ 5.0</span>'
                f'</div>'
                f'</div>'
                f'<div class="locality-subrow">'
                f'<span>📍 {loc_str}, {city_clean_name}</span>'
                f'<span class="meta-dot">•</span>'
                f'<span class="cost-highlight">💰 {r.estimated_cost}</span>'
                f'</div>'
                f'<div class="cuisines-row">{veg_html}{cuisines_html}</div>'
                f'<div class="ai-insight-box">'
                f'<div class="ai-insight-title">✨ Why You\'ll Love This Place</div>'
                f'<div class="ai-insight-text">{r.ai_explanation}</div>'
                f'</div>'
                f'<div class="card-footer-action-row">'
                f'<span class="card-address-text">📌 {addr_str}</span>'
                f'<div class="card-action-btn-group">'
                f'<span class="card-btn-outline">Directions & Map</span>'
                f'</div>'
                f'</div>'
                f'</div>'
            )
            render_html(card_html)

    else:
        st.warning(f"No restaurants matched your exact criteria in {city_clean_name}. Try broadening your rating threshold or area.")


# ── CURATED DINING COLLECTIONS SHOWCASE (Stitch Elevated) ─────────────────────
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
                <span class="collection-tag">AUTHENTIC FEASTS</span>
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
                <span class="collection-tag">ICONIC STREET BITES</span>
                <h4>🍲 Famous Street Food & Crisp Chaat</h4>
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
                <span class="collection-tag">ARTISAN BREWS</span>
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
                <span class="collection-tag">HERITAGE MITHAI</span>
                <h4>🍨 Festive Sweets & Heritage Confectionery</h4>
                <p>Iconic mithai makers like Om Sweets, Bikaner, Haldiram's, and heritage family lunch spots for festive celebrations.</p>
            </div>
        </div>
        """
    )
