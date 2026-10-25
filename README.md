# 🍽️ Zomato AI Restaurant Recommender

[![Tests](https://img.shields.io/badge/pytest-93%20passed-brightgreen.svg)](#-testing)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.14-blue.svg)](https://www.python.org/)
[![LLM](https://img.shields.io/badge/Groq-Llama%203.3%2070B-orange.svg)](https://console.groq.com/)
[![UI](https://img.shields.io/badge/Streamlit-1.40+-red.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-33%2C476%2B%20Venues-E23744.svg)](#-dataset--city-coverage)

An AI-powered restaurant recommendation web application styled in Zomato's signature luxury design language. It pairs comprehensive real-world dining data across major Indian culinary hubs with **Groq Cloud's ultra-low latency LPU inference** (`llama-3.3-70b-versatile`) to deliver grounded, hyper-personalized dining suggestions.

---

## 🌟 Key Features & Highlights

1. **Multi-City Real Restaurant Dataset**: Preprocessed and normalized catalog of **33,476+ verified restaurants** across Delhi NCR, Haryana, and Bangalore with comprehensive sanitization (cost strings, rating formats, transliterations, and cuisine canonicalization).
2. **Signature Zomato Visual Experience**: Crafted with Zomato’s authentic visual identity (`#E23744`), clean card-based hierarchy, curated dining collection photography, and high-contrast typography.
3. **Curated Dining Collections**: Quick one-click discovery for popular dining categories:
   - 🍛 **Pure Veg & Thalis**: Authentic regional vegetarian feasts and thalis.
   - 🥟 **Street Chaat & Snacks**: Crispy golgappas, tikki, and legendary street bites.
   - ☕ **Cozy Cafes & Workspaces**: Artisan brews, continental snacks, and relaxed vibes.
   - 🍬 **Mithai & Sweet Spots**: Fresh jalebis, gulab jamuns, and traditional confectioneries.
4. **Deterministic Funnel Filtering**: Rapidly narrows down 33,000+ venues into top candidate pools using fuzzy locality/city matching (`rapidfuzz`), budget tiers, rating floors, and composite scoring ($\text{Rating} \times 0.5 + \log_{10}(\text{Votes} + 1) \times 0.3 + \text{CuisineMatch} \times 0.2$).
5. **Adaptive Constraint Relaxation**: Automatically widens rating or budget thresholds when strict criteria yield fewer than 3 candidates, ensuring users always receive relevant alternatives.
6. **Qualitative Groq LLM Reasoning**: Feeds candidate pools to `llama-3.3-70b-versatile` with strict JSON mode enforcement and anti-hallucination guards to synthesize personalized culinary justifications.
7. **Heuristic Fallback Engine**: If the Groq API key is omitted or hits rate limits, the system seamlessly transitions to rule-based ranking without crashing or degrading the user experience.

---

## 📍 Dataset & City Coverage

The system indexes **33,476+ dining venues** with rich metadata (cuisines, cost for two, aggregate rating, vote counts, locality, and highlights):

| Region | Primary Cities & Hubs Covered | Venue Count |
|---|---|---|
| **Delhi NCR** | New Delhi, Gurgaon, Noida, Faridabad | 19,737+ |
| **Karnataka** | Bangalore (Indiranagar, Koramangala, Whitefield, HSR, Jayanagar, etc.) | 12,151+ |
| **Haryana & North** | Chandigarh, Rohtak, Hisar, Panipat, Karnal, Ambala, Sonipat, Bahadurgarh | 1,588+ |
| **Total** | **All Supported Cities** | **33,476+** |

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Streamlit Web Interface (app.py)                  │
│     Zomato Red Theme • Hero Search • Dining Collections • Live Cards   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ User Preferences (City, Cuisine, Budget, Rating, Vibe)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     Candidate Funnel Filter (filter.py)                │
│  - Case-insensitive rapidfuzz locality & city matching                 │
│  - Weighted composite scoring & adaptive constraint relaxation         │
└───────────────────────┬───────────────────────────┬────────────────────┘
                        │ Top 10-15 Candidates      │ Parquet
                        ▼                           ▼
┌───────────────────────────────────────┐ ┌──────────────────────────────┐
│            Groq LLM Client            │ │   data/processed/            │
│  - Llama 3.3 70B (Native JSON mode)   │ │  zomato_cleaned.parquet      │
│  - Grounded Anti-Hallucination Guard  │ │  (33,476 canonical records)  │
│  - Seamless Heuristic Fallback Engine │ └──────────────────────────────┘
└───────────────────────┬───────────────┘
                        │ Structured Recommendation Payload
                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│             Interactive Recommendation Cards & Culinary Badges         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) (recommended for fast virtualenvs) **or** standard `pip`

### 1. Clone the repository
```bash
git clone https://github.com/Shiwani-Thakur-Work/zomato-ai-recommendation.git
cd zomato-ai-recommendation
```

### 2. Install dependencies
```bash
# Using uv (fastest):
uv sync

# Or using pip:
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and supply your Groq API key:
# GROQ_API_KEY="gsk_..."
```

> **Note**: If `GROQ_API_KEY` is not provided, the application will automatically run in **Heuristic Fallback Mode**, allowing you to test candidate search, filtering, and the UI immediately.

---

## 🏃 Running the Application

Launch the Streamlit web interface locally:

```bash
# Using uv:
uv run streamlit run app.py

# Or using standard python:
streamlit run app.py
```

Open your browser at **`http://localhost:8501`**.

---

## 🧪 Testing

The automated test suite covers data pipelines, edge-case sanitization, Pydantic schemas, fuzzy filtering, adaptive relaxation, anti-hallucination prompt compliance, and fallback logic:

```bash
# Run all 93 unit and integration tests
uv run pytest -v

# Run fast unit tests only
uv run pytest -v -m "not integration"
```

All 93 tests pass with 100% success rate.

---

## 📂 Project Structure

```text
zomato-ai-recommendation/
├── .streamlit/
│   └── config.toml                # Zomato custom theme settings
├── data/
│   └── processed/
│       └── zomato_cleaned.parquet # 33,476 clean, canonical records (2 MB)
├── scripts/
│   ├── export_viewer.py           # Standalone viewer export script
│   └── inspect_data.py            # Dataset diagnostic inspector
├── src/
│   ├── __init__.py
│   ├── config.py                  # Typed configuration, env loader & secrets fallback
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              # HuggingFace streaming dataset loader
│   │   └── preprocessor.py        # Normalization, cleaning & Parquet pipeline
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── filter.py              # Multi-criteria candidate filter & ranking
│   │   ├── llm_client.py          # Groq Cloud API client & heuristic fallback
│   │   ├── recommend.py           # End-to-end recommendation orchestrator
│   │   └── schemas.py             # Pydantic v2 data models & validation
│   └── ui/
│       ├── __init__.py
│       ├── app.py                 # Core Streamlit application & custom styling
│       └── assets/                # Curated culinary photography & badges
├── tests/
│   ├── test_data_pipeline.py      # 59 tests for ETL & data cleaning edge cases
│   └── test_engine.py             # 34 tests for engine, schemas, LLM & E2E
├── .env.example
├── app.py                         # Root deployment entrypoint
├── dataset_viewer.html            # Interactive HTML dataset viewer
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🛠️ Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| **Language** | Python 3.10 – 3.14 | Modern typing, pattern matching, async-ready |
| **Data Processing** | Pandas, PyArrow | High-speed columnar Parquet storage and filtering |
| **Data Ingestion** | Hugging Face `datasets` | Streaming ingestion and local caching |
| **Data Validation** | Pydantic v2 | Strict runtime type contracts and schema validation |
| **Fuzzy Matching** | RapidFuzz | C-optimized Levenshtein & token-set locality matching |
| **LLM Provider** | **Groq Cloud** | Sub-second LPU inference for `llama-3.3-70b-versatile` |
| **UI Framework** | Streamlit | Responsive web UI with custom Zomato aesthetic |
| **Test Framework**| pytest, pytest-mock | Automated test discovery & integration assertions |
