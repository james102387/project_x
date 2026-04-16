"""
Crystal Web UI — Gradio demo interface.

Launch:
    python -m crystal.ui

Features:
    - Legal KG loaded from SQLite by default (SCOTUS cases, citations)
    - Remulak demo KG available as fallback
    - Upload documents (CSV, JSON, TXT) to build a custom KG
    - Ask questions with side-by-side Crystal vs. naked LLM comparison
    - KG explorer showing entities and facts
    - Batch review: accept/reject generated questions, see ingestion context

Module layout:
    - ``state``       — compiled graph, KG registry, default KG
    - ``formatting``  — pure Markdown/DataFrame formatters (no Gradio)
    - ``tabs/ask``    — "Ask" tab
    - ``tabs/ingest`` — "Ingest Documents" tab
    - ``tabs/kg``     — "Knowledge Graph" tab (owns cross-tab KG switch/import)
    - ``tabs/review`` — "Review" tab
    - ``app``         — thin ``build_ui()`` composition root (this file)
"""

import gradio as gr

from crystal.ui.formatting import (
    _default_kg_info,
    _format_kg_facts,
    _kg_info,
)
from crystal.ui.state import KG_MODES, _DEFAULT_KG_MODE, _default_kg
from crystal.ui.tabs.ask import build_ask_tab
from crystal.ui.tabs.ingest import build_ingest_tab
from crystal.ui.tabs.kg import build_kg_tab, import_structured_data
from crystal.ui.tabs.review import build_review_tab


# Re-exports for backwards compatibility (tests import from ``crystal.ui.app``).
__all__ = [
    "KG_MODES",
    "_default_kg_info",
    "_format_kg_facts",
    "_format_kg_stats",
    "_kg_info",
    "build_ui",
    "import_structured_data",
    "main",
]


# Also re-export `_format_kg_stats` under its original name (imported in tests).
from crystal.ui.formatting import _format_kg_stats  # noqa: E402


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Crystal — Neuro-symbolic Prompt Compiler") as demo:
        gr.Markdown(
            "# Crystal\n"
            "*Neuro-symbolic prompt compiler for LLMs — grounded answers, fewer hallucinations.*"
        )

        kg_state = gr.State(_default_kg)

        kg_banner = gr.Markdown(
            value=(
                f"**Active KG:** {_DEFAULT_KG_MODE} · "
                f"{len(_default_kg)} triplets · {len(_default_kg.entities)} entities"
            ),
        )

        with gr.Tab("Ask"):
            build_ask_tab(kg_state)

        with gr.Tab("Ingest Documents"):
            ingest_tab = build_ingest_tab(kg_state)

        with gr.Tab("Knowledge Graph"):
            build_kg_tab(kg_state, kg_banner, ingest_tab)

        with gr.Tab("Review"):
            build_review_tab(kg_state)

    return demo


def main():
    demo = build_ui()
    demo.launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
