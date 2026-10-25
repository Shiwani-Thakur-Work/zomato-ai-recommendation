"""
app.py
------
Root entrypoint for the Streamlit application.
Forwards to src/ui/app.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import runpy

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute the main Streamlit application so every interaction re-runs properly
ui_path = ROOT_DIR / "src" / "ui" / "app.py"
runpy.run_path(str(ui_path), run_name="__main__")
