"""
Crystal Web UI — Gradio demo interface.

Launch:
    python -m crystal.ui

Features:
    - Pre-loaded Remulak KG (works out of the box)
    - Upload documents (CSV, JSON, TXT) to build a custom KG
    - Ask questions with side-by-side Crystal vs. naked LLM comparison
    - KG explorer showing entities and facts
    - Batch review: accept/reject generated questions, see ingestion context
"""

import time
from pathlib import Path

import gradio as gr
import pandas as pd

from crystal.graph import build_crystal_graph
from crystal.state import make_initial_state
from crystal.ingest import ingest, build_kg
from crystal.ingest.ner import ingest_text
from crystal.review import (
    format_review_dashboard,
    list_batches,
    load_batch_context,
    load_batch_questions,
    load_known_gaps,
    load_pending_questions,
    save_review_decisions,
)
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


# ── Review helpers ────────────────────────────────────────────────────────

def _get_batch_choices() -> list[str]:
    """Get dropdown choices for batch selector."""
    batches = list_batches()
    if not batches:
        return ["(no batches)"]
    choices = []
    for b in batches:
        label = f"{b['id']} — {b['total_cases']} questions ({b['pending']} pending, {b['accepted']} accepted)"
        choices.append(label)
    return choices


def _extract_batch_id(choice: str) -> str | None:
    """Extract batch ID from dropdown label."""
    if not choice or choice == "(no batches)":
        return None
    return choice.split(" — ")[0].strip()


_TABLE_HEADERS = ["#", "Question", "Golden Answer", "Tier", "Source", "Status"]


def _load_batch_table(choice: str) -> pd.DataFrame:
    """Load questions for a batch as a DataFrame for Gradio."""
    batch_id = _extract_batch_id(choice)
    if not batch_id:
        return pd.DataFrame(columns=_TABLE_HEADERS)
    questions = load_batch_questions(batch_id)
    rows = []
    for i, q in enumerate(questions):
        src = q.get("source_triplet", [])
        source_str = f"{src[0]} | {src[1]}" if src and len(src) >= 2 else ""
        rows.append([
            str(i),
            q.get("question", ""),
            q.get("golden_answer", ""),
            str(q.get("tier", "")),
            source_str,
            q.get("status", "pending_review"),
        ])
    return pd.DataFrame(rows, columns=_TABLE_HEADERS)


def _load_batch_context_table(choice: str) -> str:
    """Load source triplets for a batch as markdown."""
    batch_id = _extract_batch_id(choice)
    if not batch_id:
        return "Select a batch to see its source data."

    triplets = load_batch_context(batch_id)
    if not triplets:
        return "No source triplets recorded for this batch."

    seen_subjects: dict[str, list[tuple[str, str]]] = {}
    for t in triplets:
        if len(t) == 3:
            subj, pred, obj = t
            seen_subjects.setdefault(subj, []).append((pred, obj))

    lines = [f"### Source Data ({len(triplets)} facts, {len(seen_subjects)} entities)\n"]
    for subj in sorted(seen_subjects.keys()):
        lines.append(f"**{subj.title()}**")
        for pred, obj in seen_subjects[subj]:
            lines.append(f"  - {pred}: {obj}")
        lines.append("")

    return "\n".join(lines)


def _load_batch_metadata(choice: str) -> str:
    """Load batch metadata as markdown."""
    batch_id = _extract_batch_id(choice)
    if not batch_id:
        return ""
    batches = list_batches()
    for b in batches:
        if b["id"] == batch_id:
            return (
                f"**Batch:** {b['id']}  \n"
                f"**Source:** {b['source']}  \n"
                f"**Records ingested:** {b['records_ingested']}  \n"
                f"**Timestamp:** {b['timestamp']}  \n"
                f"**Questions:** {b['total_cases']} total, "
                f"{b['pending']} pending, {b['accepted']} accepted, {b['rejected']} rejected"
            )
    return ""


def _save_decisions(choice: str, table_data) -> tuple[str, pd.DataFrame]:
    """Save accept/reject decisions from the interactive table."""
    batch_id = _extract_batch_id(choice)
    if not batch_id:
        return "No batch selected.", pd.DataFrame(columns=_TABLE_HEADERS)

    decisions: dict[int, str] = {}
    if table_data is not None:
        if isinstance(table_data, pd.DataFrame):
            table_data = table_data.values.tolist()
        for row in table_data:
            try:
                idx = int(row[0])
                status = row[5]
                if status in ("accepted", "rejected"):
                    decisions[idx] = status
            except (ValueError, IndexError):
                continue

    if not decisions:
        return "No decisions to save — change the Status column to 'accepted' or 'rejected'.", _load_batch_table(choice)

    save_review_decisions(batch_id, decisions)
    return f"Saved {len(decisions)} decisions for batch {batch_id}.", _load_batch_table(choice)


_GAPS_HEADERS = ["Question", "Expected", "Reason"]


def _get_known_gaps_df() -> pd.DataFrame:
    """Get known gaps as a DataFrame for the Dataframe widget."""
    try:
        gaps = load_known_gaps()
    except Exception:
        return pd.DataFrame(columns=_GAPS_HEADERS)
    rows = [[g["question"], g["expected_answer"], g["reason"]] for g in gaps]
    return pd.DataFrame(rows, columns=_GAPS_HEADERS)


def _refresh_review():
    """Refresh all review tab data including the questions table."""
    choices = _get_batch_choices()
    first = choices[0] if choices and choices[0] != "(no batches)" else None
    return (
        format_review_dashboard(),
        gr.update(choices=choices, value=first),
        _load_batch_metadata(first) if first else "",
        _load_batch_table(first) if first else pd.DataFrame(columns=_TABLE_HEADERS),
        _load_batch_context_table(first) if first else "Select a batch to see its source data.",
        _get_known_gaps_df(),
    )


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

        with gr.Tab("Review"):
            gr.Markdown(
                "Review generated questions by batch. Accept or reject questions "
                "to build golden answer sets for the Ralph Wiggum self-improvement loop."
            )

            review_dashboard = gr.Markdown(format_review_dashboard())

            with gr.Row():
                refresh_btn = gr.Button("Refresh", variant="secondary", scale=1)

            gr.Markdown("---")

            gr.Markdown("### Ingestion Batches")
            initial_choices = _get_batch_choices()
            initial_choice = initial_choices[0] if initial_choices and initial_choices[0] != "(no batches)" else None
            batch_selector = gr.Dropdown(
                choices=initial_choices,
                value=initial_choice,
                label="Select Batch",
                interactive=True,
            )
            batch_meta = gr.Markdown(
                _load_batch_metadata(initial_choice) if initial_choice else ""
            )

            initial_table = _load_batch_table(initial_choice) if initial_choice else pd.DataFrame(columns=_TABLE_HEADERS)
            initial_context = _load_batch_context_table(initial_choice) if initial_choice else "Select a batch to see its source data."

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Questions")
                    gr.Markdown(
                        "*Edit the **Status** column: change `pending_review` to "
                        "`accepted` or `rejected`, then click Save.*"
                    )
                    questions_table = gr.Dataframe(
                        value=initial_table,
                        interactive=True,
                    )
                    with gr.Row():
                        save_btn = gr.Button("Save Decisions", variant="primary")
                        save_status = gr.Textbox(label="Save Status", interactive=False, scale=3)

                with gr.Column(scale=1):
                    gr.Markdown("### Batch Context (Source Data)")
                    batch_context = gr.Markdown(initial_context)

            gr.Markdown("---")
            gr.Markdown("### Detector Known Gaps")
            gaps_table = gr.Dataframe(
                value=_get_known_gaps_df(),
                interactive=False,
            )

            batch_selector.change(
                fn=lambda c: (_load_batch_metadata(c), _load_batch_table(c), _load_batch_context_table(c)),
                inputs=[batch_selector],
                outputs=[batch_meta, questions_table, batch_context],
            )

            save_btn.click(
                fn=_save_decisions,
                inputs=[batch_selector, questions_table],
                outputs=[save_status, questions_table],
            )

            refresh_btn.click(
                fn=_refresh_review,
                inputs=[],
                outputs=[review_dashboard, batch_selector, batch_meta, questions_table, batch_context, gaps_table],
            )

    return demo


def main():
    demo = build_ui()
    demo.launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
