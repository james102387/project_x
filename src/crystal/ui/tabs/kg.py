"""Knowledge Graph tab — KG selection, import, and exploration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import pandas as pd

from crystal.ingest import build_kg, ingest
from crystal.ingest.loader import load_csv_text
from crystal.tools.kg.graph import KnowledgeGraph
from crystal.tools.kg.store import SqliteKnowledgeGraph

from crystal.ui.formatting import (
    _ORIGIN_DISPLAY,
    _ORIGIN_FILTER_CHOICES,
    _default_kg_info,
    _format_ingest_target,
    _format_kg_banner,
    _format_kg_stats,
    _get_source_document_choices,
    _kg_facts_df,
    _kg_info,
    _label_for_kg,
)
from crystal.ui.state import KG_MODES, _DEFAULT_KG_MODE, _default_kg
from crystal.ui.tabs.ingest import IngestTab


# ── Explorer helpers ─────────────────────────────────────────────────────

def _search_kg_entity(query: str, kg_state) -> str:
    """Search for an entity in the active KG and return all facts."""
    if not query or not query.strip():
        return "Type a case name or entity to search."

    query = query.strip()
    resolved, tier = kg_state._resolve_entity(query)
    if tier == "none":
        return f"No entity found matching **{query}**."

    facts = kg_state.lookup(subject=resolved)
    if not facts:
        return f"Entity **{resolved}** found (via {tier} match) but has no facts as a subject."

    lines = [f"### {resolved.title()}\n*Matched via: {tier}*\n"]
    for f in facts:
        origin = f.get("origin", "unknown")
        label = _ORIGIN_DISPLAY.get(origin, origin)
        src_doc = f.get("source_document", "")
        tag_parts = [label]
        if src_doc:
            tag_parts.append(src_doc)
        tag = f"  `[{' | '.join(tag_parts)}]`"
        lines.append(f"- **{f['predicate']}:** {f['object']}{tag}")

    return "\n".join(lines)


def _get_kg_predicate_summary(kg_state) -> pd.DataFrame:
    """Predicate frequency summary with a sample value per predicate."""
    if not hasattr(kg_state, 'triplets'):
        return pd.DataFrame(columns=["Predicate", "Count", "Sample Value"])

    pred_counts: Counter = Counter()
    pred_samples: dict[str, str] = {}
    for s, p, o in kg_state.triplets:
        pred_counts[p] += 1
        if p not in pred_samples:
            pred_samples[p] = o[:100]

    rows = [
        [pred, count, pred_samples.get(pred, "")]
        for pred, count in pred_counts.most_common()
    ]
    return pd.DataFrame(rows, columns=["Predicate", "Count", "Sample Value"])


def _get_kg_entity_list(kg_state, page: int = 0) -> pd.DataFrame:
    """Paginated list of subjects with fact counts."""
    subjects = sorted(kg_state.subjects)
    page_size = 50
    start = page * page_size
    end = start + page_size
    page_subjects = subjects[start:end]

    rows = []
    for s in page_subjects:
        facts = kg_state.lookup(subject=s)
        preds = ", ".join(sorted(set(f["predicate"] for f in facts)))
        rows.append([s.title(), len(facts), preds])

    return pd.DataFrame(rows, columns=["Entity", "Facts", "Predicates"])


# ── Cross-tab actions (switch / import) ──────────────────────────────────

def _import_error_outputs(kg_state, msg):
    """Build the full output tuple for import errors (no-op, keep current state)."""
    current_label = _label_for_kg(kg_state)
    info = _kg_info(kg_state, current_label)
    return (
        kg_state, msg, _format_kg_stats(info), _kg_facts_df(kg_state),
        _format_kg_banner(kg_state, current_label),
        _get_kg_predicate_summary(kg_state), _get_kg_entity_list(kg_state),
        _format_kg_stats(info), _format_ingest_target(kg_state, current_label),
        gr.update(),
    )


def import_structured_data(file, text, mode, kg_name, target_kg_label, kg_state):
    """Import structured CSV/JSON or pasted text as triplets.

    mode: "Create new KG" or "Append to existing KG"
    """
    result = None
    source_name = "pasted text"
    if file is not None:
        path = Path(file.name)
        source_name = path.name
        try:
            result = ingest(path)
        except Exception as e:
            return _import_error_outputs(kg_state, f"Error reading {path.name}: {e}")
    elif text and text.strip():
        try:
            result = load_csv_text(text.strip())
        except Exception as e:
            return _import_error_outputs(kg_state, f"Error parsing text: {e}")
    else:
        return _import_error_outputs(kg_state, "No file or text provided.")

    if not result or not result.triplets:
        return _import_error_outputs(kg_state, f"No triplets extracted from {source_name}.")

    tuples = result.as_tuples()
    n_parsed = len(tuples)

    if mode == "Append to existing KG":
        target = KG_MODES.get(target_kg_label)
        if target is None:
            return _import_error_outputs(kg_state, f"Unknown target KG: {target_kg_label}")

        if isinstance(target, SqliteKnowledgeGraph):
            added = target.bulk_insert(tuples, source=source_name)
        else:
            added = target.extend(tuples)

        label = target_kg_label
        status = f"Appended **{added}** new triplets to **{label}** (parsed {n_parsed} from {source_name})."
        kg = target
    else:
        label = (kg_name or "").strip() or source_name
        if label in KG_MODES:
            label = f"{label} ({len(KG_MODES)})"
        kg = build_kg(result)
        KG_MODES[label] = kg
        status = f"Created **{label}** with {n_parsed} triplets from {source_name}."

    info = _kg_info(kg, label)
    return (
        kg, status, _format_kg_stats(info), _kg_facts_df(kg),
        _format_kg_banner(kg, label),
        _get_kg_predicate_summary(kg), _get_kg_entity_list(kg),
        _format_kg_stats(info), _format_ingest_target(kg, label),
        gr.update(choices=list(KG_MODES.keys()), value=label),
    )


def switch_kg_mode(mode: str):
    """Switch the active KG by mode name."""
    kg = KG_MODES.get(mode)
    if kg is None:
        fallback_stats = _format_kg_stats(_default_kg_info())
        return (_default_kg, f"Unknown mode: {mode}", fallback_stats,
                _kg_facts_df(_default_kg), _format_kg_banner(_default_kg, _DEFAULT_KG_MODE),
                _get_kg_predicate_summary(_default_kg), _get_kg_entity_list(_default_kg),
                fallback_stats, _format_ingest_target(_default_kg, _DEFAULT_KG_MODE))
    info = _kg_info(kg, mode)
    stats = _format_kg_stats(info)
    return (kg, f"Switched to {mode}.", stats, _kg_facts_df(kg),
            _format_kg_banner(kg, mode),
            _get_kg_predicate_summary(kg), _get_kg_entity_list(kg),
            stats, _format_ingest_target(kg, mode))


# ── Layout ───────────────────────────────────────────────────────────────

@dataclass
class KgTab:
    kg_mode_selector: gr.Dropdown
    kg_stats: gr.Markdown
    status_msg: gr.Textbox
    file_input: gr.File
    text_input: gr.Textbox
    import_mode: gr.Radio
    import_kg_name: gr.Textbox
    import_target: gr.Dropdown
    import_btn: gr.Button
    import_status: gr.Markdown
    entity_search: gr.Textbox
    search_btn: gr.Button
    entity_results: gr.Markdown
    predicate_summary: gr.Dataframe
    entity_list: gr.Dataframe
    facts_origin_filter: gr.Dropdown
    facts_doc_filter: gr.Dropdown
    facts_filter_btn: gr.Button
    kg_facts: gr.Dataframe


def build_kg_tab(
    kg_state: gr.State,
    kg_banner: gr.Markdown,
    ingest_tab: IngestTab,
) -> KgTab:
    # ── Section 1: Select KG ──
    with gr.Column(variant="panel"):
        gr.Markdown("## Select Knowledge Graph")
        with gr.Row():
            kg_mode_selector = gr.Dropdown(
                choices=list(KG_MODES.keys()),
                value=_DEFAULT_KG_MODE,
                label="Active Knowledge Graph",
                interactive=True,
                scale=3,
            )
        kg_stats = gr.Markdown(_format_kg_stats(_default_kg_info()))
        status_msg = gr.Textbox(
            label="Status", interactive=False,
            value=f"{_DEFAULT_KG_MODE} loaded.", visible=False,
        )

    gr.HTML("<div style='height: 24px'></div>")

    # ── Section 2: Import structured KG data ──
    with gr.Column(variant="panel"):
        gr.Markdown("## Import Structured Data")
        gr.Markdown(
            "*Import a pre-formatted knowledge graph (CSV/JSON with subject-predicate-object columns). "
            "For document extraction, use the **Ingest Documents** tab instead.*"
        )
        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(
                    label="Upload CSV or JSON (s,p,o format)",
                    file_types=[".csv", ".json", ".txt"],
                )
                text_input = gr.Textbox(
                    label="Or paste structured triplets",
                    lines=4,
                    placeholder="subject, predicate, object (one per line)",
                )
            with gr.Column(scale=1):
                import_mode = gr.Radio(
                    choices=["Create new KG", "Append to existing KG"],
                    value="Create new KG",
                    label="Import mode",
                )
                import_kg_name = gr.Textbox(
                    label="New KG name",
                    placeholder="e.g. My Legal Data",
                    visible=True,
                )
                import_target = gr.Dropdown(
                    choices=list(KG_MODES.keys()),
                    value=_DEFAULT_KG_MODE,
                    label="Append to",
                    interactive=True,
                    visible=False,
                )
                import_btn = gr.Button("Import", variant="primary")
        import_status = gr.Markdown("")

        import_mode.change(
            fn=lambda m: (
                gr.update(visible=m == "Create new KG"),
                gr.update(visible=m == "Append to existing KG"),
            ),
            inputs=[import_mode],
            outputs=[import_kg_name, import_target],
        )

    gr.HTML("<div style='height: 24px'></div>")

    # ── Section 3: Explore KG ──
    with gr.Column(variant="panel"):
        gr.Markdown("## Explore")
        with gr.Row():
            entity_search = gr.Textbox(
                label="Search Entity",
                placeholder="e.g. Miranda v. Arizona, Roe v. Wade",
                scale=4,
            )
            search_btn = gr.Button("Search", variant="primary", scale=1)
        entity_results = gr.Markdown("Type a case name or entity to search.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Predicate Summary")
                predicate_summary = gr.Dataframe(
                    value=_get_kg_predicate_summary(_default_kg),
                    interactive=False,
                )
            with gr.Column():
                gr.Markdown("### Entities (first 50)")
                entity_list = gr.Dataframe(
                    value=_get_kg_entity_list(_default_kg),
                    interactive=False,
                )

    gr.HTML("<div style='height: 24px'></div>")

    # ── Section 4: Facts table ──
    with gr.Column(variant="panel"):
        gr.Markdown("## All Facts")
        with gr.Row():
            facts_origin_filter = gr.Dropdown(
                choices=_ORIGIN_FILTER_CHOICES,
                value="All origins",
                label="Filter by origin",
                interactive=True,
                scale=2,
            )
            facts_doc_filter = gr.Dropdown(
                choices=_get_source_document_choices(_default_kg),
                value="All documents",
                label="Filter by source document",
                interactive=True,
                scale=3,
            )
            facts_filter_btn = gr.Button("Apply Filters", variant="secondary", scale=1)
        kg_facts = gr.Dataframe(
            value=_kg_facts_df(_default_kg),
            interactive=False,
            wrap=True,
            max_height=400,
        )

    # ── Wiring (cross-tab because switch/import touch ingest tab too) ──
    _import_outputs = [
        kg_state, import_status, kg_stats, kg_facts, kg_banner,
        predicate_summary, entity_list,
        ingest_tab.ingest_kg_stats, ingest_tab.ingest_target_label,
        kg_mode_selector,
    ]

    kg_mode_selector.change(
        fn=switch_kg_mode,
        inputs=[kg_mode_selector],
        outputs=[kg_state, status_msg, kg_stats, kg_facts, kg_banner,
                 predicate_summary, entity_list,
                 ingest_tab.ingest_kg_stats, ingest_tab.ingest_target_label],
    )
    import_btn.click(
        fn=import_structured_data,
        inputs=[file_input, text_input, import_mode,
                import_kg_name, import_target, kg_state],
        outputs=_import_outputs,
    )
    search_btn.click(
        fn=_search_kg_entity,
        inputs=[entity_search, kg_state],
        outputs=[entity_results],
    )
    entity_search.submit(
        fn=_search_kg_entity,
        inputs=[entity_search, kg_state],
        outputs=[entity_results],
    )

    def _apply_facts_filter(origin_val, doc_val, kg):
        doc_filter = "" if doc_val == "All documents" else doc_val
        return _kg_facts_df(kg, origin_filter=origin_val, document_filter=doc_filter)

    facts_filter_btn.click(
        fn=_apply_facts_filter,
        inputs=[facts_origin_filter, facts_doc_filter, kg_state],
        outputs=[kg_facts],
    )

    return KgTab(
        kg_mode_selector=kg_mode_selector,
        kg_stats=kg_stats,
        status_msg=status_msg,
        file_input=file_input,
        text_input=text_input,
        import_mode=import_mode,
        import_kg_name=import_kg_name,
        import_target=import_target,
        import_btn=import_btn,
        import_status=import_status,
        entity_search=entity_search,
        search_btn=search_btn,
        entity_results=entity_results,
        predicate_summary=predicate_summary,
        entity_list=entity_list,
        facts_origin_filter=facts_origin_filter,
        facts_doc_filter=facts_doc_filter,
        facts_filter_btn=facts_filter_btn,
        kg_facts=kg_facts,
    )
