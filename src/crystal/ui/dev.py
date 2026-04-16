"""Dev launcher for Gradio hot-reload.

Usage:
    .venv/bin/gradio src/crystal/ui/dev.py --watch-dirs src/crystal
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from crystal.ui.app import build_ui  # noqa: E402
import gradio as gr  # noqa: E402

demo = build_ui()
demo.launch(theme=gr.themes.Soft())
