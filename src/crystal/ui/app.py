"""
Crystal Web UI — Gradio demo interface.

Launch:
    python -m crystal.ui

Features:
    - Pre-loaded Remulak KG (works out of the box)
    - Upload documents (CSV, JSON, TXT) to build a custom KG
    - Ask questions with side-by-side Crystal vs. naked LLM comparison
    - KG explorer showing entities and facts
"""

import time
from pathlib import Path

import gradio as gr

from crystal.graph import build_crystal_graph
from crystal.state import make_initial_state
from crystal.ingest import ingest, build_kg
from crystal.ingest.ner import ingest_text
from crystal.tools.kg import remulak_kg
from crystal.tools.kg.graph import KnowledgeGraph
import crystal.llm


# ── App state ────────────────────────────────────────────────────────────

_graph = build_crystal_graph()


def _default_kg_info() -> dict:
    """Return info dict for a KnowledgeGraph."""
    return _kg_info(remulak_kg, "Remulak (built-in)")


def _kg_info(kg: KnowledgeGraph, source: str) -> dict:
    return {
        "source": source,
        "triplets": len(kg),
        "entities": len(kg.entities),
        "subjects": len(kg.subjects),
    }


def _format_kg_stats(info: dict) -> str:
    return (
        f"**{info['source']}**\n\n"
        f"- Triplets: {info['triplets']}\n"
        f"- Entities: {info['entities']}\n"
        f"- Subjects: {info['subjects']}"
    )


def _format_kg_facts(kg: KnowledgeGraph, max_facts: int = 200) -> str:
    """Format KG triplets as a readable table."""
    if not kg.triplets:
        return "No facts loaded."
    lines = ["| Subject | Predicate | Object |", "| --- | --- | --- |"]
    for s, p, o in kg.triplets[:max_facts]:
        lines.append(f"| {s} | {p} | {o} |")
    if len(kg.triplets) > max_facts:
        lines.append(f"\n*...and {len(kg.triplets) - max_facts} more*")
    return "\n".join(lines)


# ── Core actions ─────────────────────────────────────────────────────────

def load_document(file, kg_state):
    """Ingest an uploaded document and build a new KG."""
    if file is None:
        return kg_state, "No file uploaded.", _format_kg_stats(_kg_info(kg_state, "Remulak (built-in)")), _format_kg_facts(kg_state)

    path = Path(file.name)
    try:
        result = ingest(path)
        if not result.triplets:
            return kg_state, f"No triplets extracted from {path.name}.", _format_kg_stats(_kg_info(kg_state, kg_state._source if hasattr(kg_state, '_source') else "current")), _format_kg_facts(kg_state)
        kg = build_kg(result)
        info = _kg_info(kg, path.name)
        return kg, f"Loaded {info['triplets']} triplets from {path.name}.", _format_kg_stats(info), _format_kg_facts(kg)
    except Exception as e:
        return kg_state, f"Error ingesting {path.name}: {e}", _format_kg_stats(_kg_info(kg_state, "current")), _format_kg_facts(kg_state)


def ingest_raw_text(text, kg_state):
    """Ingest raw pasted text via NER and build a KG."""
    if not text or not text.strip():
        return kg_state, "No text provided.", _format_kg_stats(_kg_info(kg_state, "current")), _format_kg_facts(kg_state)

    try:
        result = ingest_text(text.strip())
        if not result.triplets:
            return kg_state, "No triplets extracted from text.", _format_kg_stats(_kg_info(kg_state, "current")), _format_kg_facts(kg_state)
        kg = build_kg(result)
        info = _kg_info(kg, "pasted text")
        return kg, f"Extracted {info['triplets']} triplets from text.", _format_kg_stats(info), _format_kg_facts(kg)
    except Exception as e:
        return kg_state, f"Error: {e}", _format_kg_stats(_kg_info(kg_state, "current")), _format_kg_facts(kg_state)


def reset_to_remulak():
    """Reset KG to the built-in Remulak dataset."""
    info = _default_kg_info()
    return remulak_kg, "Reset to Remulak.", _format_kg_stats(info), _format_kg_facts(remulak_kg)


def ask_question(question, kg_state):
    """Run a question through Crystal and naked LLM, return side-by-side results."""
    if not question or not question.strip():
        return "", "", "", ""

    question = question.strip()

    # Crystal pipeline
    state = make_initial_state(question, kg=kg_state)
    try:
        final = _graph.invoke(state)
        crystal_response = final.get("final_response", "")
        prompt_type = final.get("prompt_type", "unknown")
        metrics = final.get("token_metrics", {})

        crystal_meta = f"**Route:** `{prompt_type}`"
        if metrics:
            if metrics.get("actual_prompt_tokens"):
                crystal_meta += f"\n**Tokens:** {metrics['actual_prompt_tokens']} prompt, {metrics.get('actual_output_tokens', '?')} output"
            if metrics.get("actual_reasoning_tokens"):
                crystal_meta += f", {metrics['actual_reasoning_tokens']} reasoning"
            if prompt_type in ("pure_math", "math_answerable", "kg_answerable"):
                crystal_meta += "\n**LLM called:** No (direct return)"
    except Exception as e:
        crystal_response = f"Error: {e}"
        crystal_meta = "**Route:** error"

    # Naked LLM — brief pause to avoid back-to-back rate limit hits
    time.sleep(1)
    try:
        llm_response, llm_usage = crystal.llm.call_llm(question)
        llm_meta = ""
        if llm_usage:
            llm_meta = f"**Tokens:** {llm_usage.get('prompt_tokens', '?')} prompt, {llm_usage.get('output_tokens', '?')} output"
            if llm_usage.get("reasoning_tokens"):
                llm_meta += f", {llm_usage['reasoning_tokens']} reasoning"
    except Exception as e:
        llm_response = f"Error: {e}"
        llm_meta = ""

    return crystal_response, crystal_meta, llm_response, llm_meta


# ── Gradio layout ────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Crystal — Neuro-symbolic Prompt Compiler",
    ) as demo:
        gr.Markdown("# Crystal\n*Neuro-symbolic prompt compiler for LLMs — grounded answers, fewer hallucinations.*")

        kg_state = gr.State(remulak_kg)

        with gr.Tab("Ask"):
            gr.Markdown("Ask a question and compare Crystal's grounded answer against the naked LLM.")

            with gr.Row():
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="e.g. What is the capital of Remulak?",
                    scale=4,
                )
                ask_btn = gr.Button("Ask", variant="primary", scale=1)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Crystal")
                    crystal_output = gr.Textbox(label="Response", lines=6, interactive=False)
                    crystal_meta_output = gr.Markdown("")
                with gr.Column():
                    gr.Markdown("### Naked LLM")
                    llm_output = gr.Textbox(label="Response", lines=6, interactive=False)
                    llm_meta_output = gr.Markdown("")

            ask_btn.click(
                fn=ask_question,
                inputs=[question_input, kg_state],
                outputs=[crystal_output, crystal_meta_output, llm_output, llm_meta_output],
            )
            question_input.submit(
                fn=ask_question,
                inputs=[question_input, kg_state],
                outputs=[crystal_output, crystal_meta_output, llm_output, llm_meta_output],
            )

        with gr.Tab("Knowledge Graph"):
            gr.Markdown("Manage the active knowledge graph. Upload a document or paste text to build a custom KG.")

            kg_stats = gr.Markdown(_format_kg_stats(_default_kg_info()))
            status_msg = gr.Textbox(label="Status", interactive=False, value="Remulak KG loaded.")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Upload Document")
                    file_input = gr.File(
                        label="CSV, JSON, or TXT",
                        file_types=[".csv", ".json", ".txt"],
                    )
                    upload_btn = gr.Button("Ingest File")
                with gr.Column():
                    gr.Markdown("#### Paste Text")
                    text_input = gr.Textbox(
                        label="Raw text for NER extraction",
                        lines=6,
                        placeholder="Paste prose here. Crystal will extract entity-predicate-object triplets.",
                    )
                    text_btn = gr.Button("Ingest Text")

            reset_btn = gr.Button("Reset to Remulak", variant="secondary")

            gr.Markdown("#### Facts")
            kg_facts = gr.Markdown(_format_kg_facts(remulak_kg))

            upload_btn.click(
                fn=load_document,
                inputs=[file_input, kg_state],
                outputs=[kg_state, status_msg, kg_stats, kg_facts],
            )
            text_btn.click(
                fn=ingest_raw_text,
                inputs=[text_input, kg_state],
                outputs=[kg_state, status_msg, kg_stats, kg_facts],
            )
            reset_btn.click(
                fn=reset_to_remulak,
                inputs=[],
                outputs=[kg_state, status_msg, kg_stats, kg_facts],
            )

    return demo


def main():
    demo = build_ui()
    demo.launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
