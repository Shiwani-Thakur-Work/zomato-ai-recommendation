"""
app.py
------
Root entrypoint for the Streamlit application.
Forwards to src/ui/app.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute the main Streamlit application
from src.ui import app  # noqa: F401
