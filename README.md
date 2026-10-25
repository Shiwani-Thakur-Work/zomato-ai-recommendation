# 🍽️ Zomato AI Restaurant Recommender

[![Tests](https://img.shields.io/badge/pytest-93%20passed-brightgreen.svg)](#testing)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.14-blue.svg)](https://www.python.org/)
[![LLM](https://img.shields.io/badge/Groq-Llama%203.3%2070B-orange.svg)](https://console.groq.com/)
[![UI](https://img.shields.io/badge/Streamlit-1.66-red.svg)](https://streamlit.io/)

An AI-powered restaurant recommendation system combining real-world Zomato restaurant data with **Groq Cloud's ultra-low latency LPU inference** (`llama-3.3-70b-versatile`) to deliver grounded, hyper-personalized dining suggestions.

---

## 🌟 What It Does

1. **ETL & Data Pipeline**: Automatically ingests and cleans the Zomato Bangalore dataset (51,717 raw records → 12,151 high-quality deduplicated venues) with comprehensive edge-case sanitization (cost strings, rating formats, transliterations, cuisine normalizations).
2. **Deterministic Funnel Filtering**: Quickly narrows 12,000+ venues down to the top 10–15 candidates using fuzzy city/locality matching (`rapidfuzz`), rating floors, budget tiers, and composite scoring ($\text{Rating} \times 0.5 + \log_{10}(\text{Votes} + 1) \times 0.3 + \text{CuisineMatch} \times 0.2$).
3. **Adaptive Relaxation**: Automatically widens rating or budget constraints if initial hard filters yield fewer than 3 candidates.
4. **Qualitative Groq LLM Reasoning**: Dispatches candidate pools to `llama-3.3-70b-versatile` with strict JSON mode enforcement and anti-hallucination guards to synthesize tailored culinary justifications.
5. **Heuristic Fallback Engine**: If the Groq API key is missing or encounters a rate limit/network timeout, the system seamlessly transitions to rule-based composite ranking without crashing or degrading UI responsiveness.
6. **Glassmorphic Streamlit UI**: Presents recommendations as rich visual cards featuring ratings, cost, cuisines, neighborhood badges, and AI culinary match insights.

---

## 🏗️ System Architecture

For exhaustive architecture specifications, see [`docs/architecture.md`](docs/architecture.md).

```
┌────────────────────────────────────────────────────────┐
│                   Streamlit UI (app.py)                │
└──────────────────────────┬─────────────────────────────┘
                           │ User Preferences
                           ▼
┌────────────────────────────────────────────────────────┐
│             Candidate Filter (filter.py)               │
│  - Case-insensitive rapidfuzz locality/city match      │
│  - Composite Scoring & Adaptive Relaxation             │
└──────────────┬───────────────────────────┬─────────────┘
               │ 10-15 Candidates          │ Parquet
               ▼                           ▼
┌──────────────────────────────┐ ┌───────────────────────┐
│     Groq LLM Client          │ │   data/processed/     │
│  - Llama 3.3 70B (JSON mode) │ │  zomato_cleaned.      │
│  - Anti-hallucination guard  │ │  parquet (12,151 rows)│
│  - Heuristic Fallback engine │ └───────────────────────┘
└──────────────┬───────────────┘
               │ Structured Recommendations
               ▼
┌────────────────────────────────────────────────────────┐
│            UI Recommendation Cards & Badges            │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) **or** pip

### 1. Clone the repository
```bash
git clone <repo-url>
cd Milestone-1--Zomato-Recommendation
```

### 2. Install dependencies
```bash
uv sync          # using uv (recommended)
# or
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and provide your GROQ_API_KEY
# Get a free key at: https://console.groq.com/keys
```

> **Note**: If `GROQ_API_KEY` is not provided, the application will automatically run in **Heuristic Fallback Mode**, allowing you to explore the interface and candidate ranking engine immediately!

---

## 🏃 Running the Application

Launch the Streamlit web interface:
```bash
uv run streamlit run src/ui/app.py
# or
streamlit run src/ui/app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧪 Testing

The test suite covers data ETL, parsing edge cases, Pydantic data contracts, candidate ranking, adaptive relaxation, anti-hallucination guards, and end-to-end integration:

```bash
# Run all 93 unit and integration tests
uv run pytest -v

# Run unit tests only (fast)
uv run pytest -v -m "not integration"
```

All 93 tests pass with 100% test success rate.

---

## 📂 Project Structure

```text
Milestone-1--Zomato-Recommendation/
├── data/
│   ├── raw/                       # Cached Hugging Face dataset
│   └── processed/
│       └── zomato_cleaned.parquet # 12,151 cleaned, canonical records
├── docs/
│   ├── ProblemStatement.txt       # Project problem statement
│   ├── context.md                 # System requirements and business context
│   ├── architecture.md            # Complete architecture & design specs
│   ├── implementation-plan.md     # 7-phase detailed engineering plan
│   └── edge-cases.md              # Corner scenarios and mitigation matrix
├── src/
│   ├── __init__.py
│   ├── config.py                  # Typed configuration & env loader
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              # HuggingFace dataset downloader & cache
│   │   └── preprocessor.py        # Robust ETL pipeline & Parquet persistence
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── schemas.py             # Pydantic v2 data models
│   │   ├── filter.py              # Multi-criteria candidate filter & ranking
│   │   ├── llm_client.py          # Groq Cloud API client & heuristic fallback
│   │   └── recommend.py           # Top-level recommendation orchestrator
│   └── ui/
│       ├── __init__.py
│       └── app.py                 # Streamlit UI with glassmorphic cards
├── tests/
│   ├── test_data_pipeline.py      # 59 tests for ETL & data edge cases
│   └── test_engine.py             # 34 tests for engine, schemas, LLM & E2E
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🛠️ Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| **Language** | Python 3.10 – 3.14 | Modern typing, pattern matching, async-ready |
| **Data Processing** | Pandas, PyArrow | Fast columnar Parquet operations |
| **Data Ingestion** | Hugging Face `datasets` | Efficient streaming & local caching |
| **Data Validation** | Pydantic v2 | Strict runtime type contracts and serialization |
| **Fuzzy Matching** | RapidFuzz | C-optimized Levenshtein & token-set matching |
| **LLM Provider** | **Groq Cloud** | Sub-second LPU inference for `llama-3.3-70b-versatile` |
| **UI Framework** | Streamlit | Rapid interactive web interface with custom CSS |
| **Test Framework**| pytest, pytest-mock | Automated test discovery & integration assertions |

---

## 📖 Documentation Links

- [System Context & Requirements](docs/context.md)
- [System Architecture & Sequence Flows](docs/architecture.md)
- [Implementation Plan](docs/implementation-plan.md)
- [Edge Cases & Corner Scenarios](docs/edge-cases.md)
