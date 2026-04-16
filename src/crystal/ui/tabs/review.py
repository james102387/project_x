"""Review tab — batch review, known gaps, benchmark + Ralph Wiggum loop."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr
import pandas as pd

from crystal.review import (
    find_batch_document_slugs,
    find_document_for_entity,
    format_review_dashboard,
    list_batches,
    load_batch_questions,
    load_known_gaps,
    load_source_document_text,
    revalidate_pending_questions,
    save_single_review_decision,
)
from crystal.state import make_initial_state

from crystal.ui.formatting import _ORIGIN_DISPLAY
from crystal.ui.state import _default_kg, _graph


# ── Table headers ────────────────────────────────────────────────────────

_OVERVIEW_HEADERS = ["#", "Question", "Status", "Route", "Origin", "Source Doc"]
_GAPS_HEADERS = ["Question", "Expected", "Reason"]
_BENCH_HEADERS = ["Question", "Golden Answer", "Crystal Answer", "Route", "Correct"]


# ── Batch / question selection helpers ───────────────────────────────────

def _get_batch_choices() -> list[str]:
    """Dropdown choices for batch selector."""
    batches = list_batches()
    if not batches:
        return ["(no batches)"]
    choices = []
    for b in batches:
        pending_tag = f" ** {b['pending']} PENDING **" if b["pending"] > 0 else ""
        label = (
            f"{b['id']} — {b['total_cases']} questions "
            f"({b['pending']} pending, {b['accepted']} accepted){pending_tag}"
        )
        choices.append(label)
    return choices


def _extract_batch_id(choice: str) -> str | None:
    if not choice or choice == "(no batches)":
        return None
    return choice.split(" — ")[0].strip()


def _load_overview_table(batch_id: str | None) -> pd.DataFrame:
    """Compact questions overview."""
    if not batch_id:
        return pd.DataFrame(columns=_OVERVIEW_HEADERS)
    questions = load_batch_questions(batch_id)
    rows = []
    for i, q in enumerate(questions):
        status = q.get("status", "pending_review")
        icon = {"accepted": "accepted", "rejected": "rejected"}.get(status, "PENDING")
        origin = q.get("origin", "unknown")
        origin_label = _ORIGIN_DISPLAY.get(origin, origin)
        rows.append([
            str(i + 1),
            q.get("question", "")[:80],
            icon,
            q.get("crystal_route", ""),
            origin_label,
            q.get("source_document", ""),
        ])
    return pd.DataFrame(rows, columns=_OVERVIEW_HEADERS)


def _load_batch_metadata(choice: str) -> str:
    batch_id = _extract_batch_id(choice)
    if not batch_id:
        return ""
    batches = list_batches()
    for b in batches:
        if b["id"] == batch_id:
            return (
                f"**Batch:** {b['id']}  \n"
                f"**Source:** {b['source']}  \n"
                f"**Timestamp:** {b['timestamp']}  \n"
                f"**Questions:** {b['total_cases']} total — "
                f"**{b['pending']} pending**, {b['accepted']} accepted, {b['rejected']} rejected"
            )
    return ""


def _get_question_choices(batch_id: str | None) -> list[str]:
    if not batch_id:
        return ["(no questions)"]
    questions = load_batch_questions(batch_id)
    if not questions:
        return ["(no questions)"]
    choices = []
    for i, q in enumerate(questions):
        status = q.get("status", "pending_review")
        tag = {"accepted": "[OK]", "rejected": "[REJ]"}.get(status, "[PENDING]")
        choices.append(f"{i + 1}. {tag} {q.get('question', '')[:60]}")
    return choices


def _load_question_detail(batch_id: str | None, question_idx: int) -> tuple:
    """All display fields for a single question.

    Returns (question_md, crystal_proposed, route_info, source_triplet_md,
             golden_answer, status_label).
    """
    empty = ("*Select a question to review.*", "", "", "", "", "")
    if not batch_id:
        return empty

    questions = load_batch_questions(batch_id)
    if not questions or not (0 <= question_idx < len(questions)):
        return empty

    q = questions[question_idx]
    status = q.get("status", "pending_review")
    status_badge = {
        "accepted": "ACCEPTED",
        "rejected": "REJECTED",
    }.get(status, "PENDING REVIEW")

    origin = q.get("origin", "unknown")
    origin_label = _ORIGIN_DISPLAY.get(origin, origin)
    source_doc = q.get("source_document", "")

    question_md = (
        f"## Question {question_idx + 1} of {len(questions)}  \n"
        f"### {q.get('question', '')}\n\n"
        f"**Status:** {status_badge}  \n"
        f"**Origin:** {origin_label}"
    )
    if source_doc:
        question_md += f"  \n**Source doc:** `{source_doc}`"

    crystal_proposed = q.get("crystal_proposed", q.get("golden_answer", ""))

    route = q.get("crystal_route", "unknown")
    confidence = q.get("crystal_confidence", "unknown")
    tier = q.get("tier", "")
    route_info = f"**Route:** {route}  \n**Confidence:** {confidence}  \n**Tier:** {tier}"

    st = q.get("source_triplet", [])
    if st and len(st) == 3:
        source_md = (
            f"**Subject:** {st[0]}  \n"
            f"**Predicate:** {st[1]}  \n"
            f"**Object:** {st[2]}"
        )
    else:
        source_md = "*No source triplet recorded.*"

    golden = q.get("golden_answer", "")

    return (question_md, crystal_proposed, route_info, source_md, golden, status_badge)


def _kg_subgraph_for_question(batch_id: str | None, question_idx: int, kg_state) -> str:
    """Render the KG subgraph relevant to a question's source entity."""
    if not batch_id:
        return "*Select a question to see its KG subgraph.*"

    questions = load_batch_questions(batch_id)
    if not questions or not (0 <= question_idx < len(questions)):
        return "*No question selected.*"

    q = questions[question_idx]
    st = q.get("source_triplet", [])
    if not st or len(st) < 3:
        return "*No source triplet recorded for this question.*"

    entity = st[0]
    facts = kg_state.lookup(subject=entity)
    if not facts:
        return f"*No KG facts found for entity: **{entity}***"

    lines = [f"### KG facts for: {entity.title()}\n"]
    for f in facts:
        origin = f.get("origin", "unknown")
        label = _ORIGIN_DISPLAY.get(origin, origin)
        src_doc = f.get("source_document", "")
        tag = f"  `[{label}]`"
        if src_doc:
            tag = f"  `[{label} | {src_doc}]`"
        lines.append(f"- **{f['predicate']}:** {f['object']}{tag}")

    return "\n".join(lines)


def _doc_for_question(batch_id: str | None, question_idx: int) -> tuple[str, str]:
    """Find (doc_display_name, doc_text) for a question's source entity."""
    if not batch_id:
        return ("", "")
    questions = load_batch_questions(batch_id)
    if not questions or not (0 <= question_idx < len(questions)):
        return ("", "")
    st = questions[question_idx].get("source_triplet", [])
    if not st:
        return ("", "")
    entity = str(st[0])
    match = find_document_for_entity(entity)
    if match is None:
        return ("", f"No document found for entity: {entity}")
    slug = match.stem
    display = slug.replace("-", " ").title() + f" ({slug})"
    text = load_source_document_text(slug)
    return (display, text)


def _load_doc_text(doc_choice: str):
    if not doc_choice or doc_choice == "(no source documents found)":
        return "Select a source document to view the original opinion text."
    slug = doc_choice.split("(")[-1].rstrip(")").strip() if "(" in doc_choice else doc_choice
    return load_source_document_text(slug)


# ── Batch selection / navigation handlers ────────────────────────────────

def _on_batch_selected(choice: str, kg_state=None):
    """When a batch is selected, load its metadata, questions, docs, and first question."""
    batch_id = _extract_batch_id(choice)

    meta = _load_batch_metadata(choice)
    q_choices = _get_question_choices(batch_id)
    first_q = q_choices[0] if q_choices and q_choices[0] != "(no questions)" else None

    first_pending_idx = 0
    if batch_id:
        questions = load_batch_questions(batch_id)
        for i, q in enumerate(questions):
            if q.get("status") == "pending_review":
                first_pending_idx = i
                break

    first_q = q_choices[first_pending_idx] if first_pending_idx < len(q_choices) else q_choices[0]

    q_detail = _load_question_detail(batch_id, first_pending_idx)

    doc_slugs = find_batch_document_slugs(batch_id) if batch_id else []
    doc_choices = [s.replace("-", " ").title() + f" ({s})" for s in doc_slugs]
    if not doc_choices:
        doc_choices = ["(no source documents found)"]

    first_doc_choice = doc_choices[0] if doc_choices else None
    first_doc_text = _load_doc_text(first_doc_choice) if first_doc_choice else ""

    overview = _load_overview_table(batch_id)

    subgraph = _kg_subgraph_for_question(batch_id, first_pending_idx, kg_state) if kg_state else ""

    return (
        meta,
        gr.update(choices=q_choices, value=first_q),
        first_pending_idx,
        *q_detail,
        gr.update(choices=doc_choices, value=first_doc_choice),
        first_doc_text,
        overview,
        subgraph,
    )


def _on_question_selected(choice: str, batch_choice: str, kg_state):
    batch_id = _extract_batch_id(batch_choice)
    if not choice or choice == "(no questions)":
        return (0, *_load_question_detail(None, 0), "*Select a question.*",
                gr.update(), "")

    try:
        idx = int(choice.split(".")[0]) - 1
    except (ValueError, IndexError):
        idx = 0

    detail = _load_question_detail(batch_id, idx)
    subgraph = _kg_subgraph_for_question(batch_id, idx, kg_state)
    doc_display, doc_text = _doc_for_question(batch_id, idx)
    doc_update = gr.update(value=doc_display) if doc_display else gr.update()
    return (idx, *detail, subgraph, doc_update, doc_text)


def _navigate_question(current_idx: int, direction: int, batch_choice: str, kg_state):
    batch_id = _extract_batch_id(batch_choice)
    if not batch_id:
        return (0, "(no questions)", *_load_question_detail(None, 0), "",
                gr.update(), "")

    questions = load_batch_questions(batch_id)
    if not questions:
        return (0, "(no questions)", *_load_question_detail(None, 0), "",
                gr.update(), "")

    new_idx = max(0, min(len(questions) - 1, current_idx + direction))
    q_choices = _get_question_choices(batch_id)
    detail = _load_question_detail(batch_id, new_idx)
    subgraph = _kg_subgraph_for_question(batch_id, new_idx, kg_state)
    doc_display, doc_text = _doc_for_question(batch_id, new_idx)
    doc_update = gr.update(value=doc_display) if doc_display else gr.update()

    return (new_idx, q_choices[new_idx] if new_idx < len(q_choices) else q_choices[0],
            *detail, subgraph, doc_update, doc_text)


# ── Accept / reject ──────────────────────────────────────────────────────

def _after_decision_outputs(batch_id, next_idx, batch_choice, kg_state):
    q_choices = _get_question_choices(batch_id)
    detail = _load_question_detail(batch_id, next_idx)
    overview = _load_overview_table(batch_id)
    meta = _load_batch_metadata(batch_choice)
    subgraph = _kg_subgraph_for_question(batch_id, next_idx, kg_state)
    doc_display, doc_text = _doc_for_question(batch_id, next_idx)
    doc_update = gr.update(value=doc_display) if doc_display else gr.update()
    selector_value = q_choices[next_idx] if next_idx < len(q_choices) else (
        q_choices[0] if q_choices else "(no questions)")
    return (
        next_idx,
        gr.update(choices=q_choices, value=selector_value),
        *detail,
        overview,
        meta,
        subgraph,
        doc_update,
        doc_text,
    )


def _accept_question(batch_choice: str, current_idx: int, golden_answer: str, kg_state):
    batch_id = _extract_batch_id(batch_choice)
    if not batch_id:
        return ("No batch selected.",) + _after_decision_outputs(None, 0, batch_choice, kg_state)

    ok = save_single_review_decision(batch_id, current_idx, golden_answer, "accepted")
    if not ok:
        return (f"Failed to save decision for question {current_idx + 1}.",) + \
            _after_decision_outputs(batch_id, current_idx, batch_choice, kg_state)

    questions = load_batch_questions(batch_id)
    next_idx = current_idx
    for i in range(current_idx + 1, len(questions)):
        if questions[i].get("status") == "pending_review":
            next_idx = i
            break
    else:
        for i in range(0, current_idx):
            if questions[i].get("status") == "pending_review":
                next_idx = i
                break

    pending = sum(1 for q in questions if q.get("status") == "pending_review")
    msg = f"Question {current_idx + 1} accepted. {pending} pending remaining."

    return (msg,) + _after_decision_outputs(batch_id, next_idx, batch_choice, kg_state)


def _reject_question(batch_choice: str, current_idx: int, golden_answer: str, kg_state):
    batch_id = _extract_batch_id(batch_choice)
    if not batch_id:
        return ("No batch selected.",) + _after_decision_outputs(None, 0, batch_choice, kg_state)

    ok = save_single_review_decision(batch_id, current_idx, golden_answer, "rejected")
    if not ok:
        return (f"Failed to save decision for question {current_idx + 1}.",) + \
            _after_decision_outputs(batch_id, current_idx, batch_choice, kg_state)

    questions = load_batch_questions(batch_id)
    next_idx = current_idx
    for i in range(current_idx + 1, len(questions)):
        if questions[i].get("status") == "pending_review":
            next_idx = i
            break
    else:
        for i in range(0, current_idx):
            if questions[i].get("status") == "pending_review":
                next_idx = i
                break

    pending = sum(1 for q in questions if q.get("status") == "pending_review")
    msg = f"Question {current_idx + 1} rejected. {pending} pending remaining."

    return (msg,) + _after_decision_outputs(batch_id, next_idx, batch_choice, kg_state)


# ── Known gaps ───────────────────────────────────────────────────────────

def _get_known_gaps_df() -> pd.DataFrame:
    try:
        gaps = load_known_gaps()
    except Exception:
        return pd.DataFrame(columns=_GAPS_HEADERS)
    rows = [[g["question"], g["expected_answer"], g["reason"]] for g in gaps]
    return pd.DataFrame(rows, columns=_GAPS_HEADERS)


def _refresh_review():
    return format_review_dashboard()


# ── Benchmark & Ralph Wiggum ─────────────────────────────────────────────

def _revalidate_pending(kg_state, batch_choice):
    """Revalidate all pending questions against the current KG."""
    result = revalidate_pending_questions(kg_state)
    n = result["total_rejected"]
    checked = result["total_checked"]

    if n == 0:
        msg = f"**Revalidation complete** — checked {checked} pending questions, all still valid."
    else:
        details = "\n".join(
            f"- **{r['question']}** — {r['reason']}" for r in result["rejected"]
        )
        msg = (
            f"**Revalidated:** {n} of {checked} pending questions rejected.\n\n{details}"
        )

    dashboard = format_review_dashboard()
    batch_id = _extract_batch_id(batch_choice)
    overview = _load_overview_table(batch_id)
    batch_meta_text = _load_batch_metadata(batch_choice) if batch_choice else ""

    return msg, dashboard, overview, batch_meta_text


def run_benchmark_on_accepted(kg_state):
    """Run all accepted golden answers through Crystal and score them."""
    from crystal.review import collect_accepted_cases
    from benchmarks.scoring.fitness import binary_correct

    cases = collect_accepted_cases()
    if not cases:
        empty = pd.DataFrame(columns=_BENCH_HEADERS)
        return "No accepted golden answers found. Accept some questions in the Review tab first.", empty, ""

    rows = []
    correct_count = 0

    for question, golden_answer, match_strings, is_negative in cases:
        try:
            state = make_initial_state(question, kg=kg_state)
            result = _graph.invoke(state)
            crystal_answer = result.get("final_response", "")
            route = result.get("prompt_type", "unknown")
        except Exception as e:
            crystal_answer = f"Error: {e}"
            route = "error"

        is_correct = binary_correct(crystal_answer, match_strings, is_negative)
        if is_correct:
            correct_count += 1
        rows.append([question, golden_answer[:120], crystal_answer[:120], route,
                      "YES" if is_correct else "NO"])

    score = correct_count / len(cases) if cases else 0.0
    status = f"**Benchmark: {correct_count}/{len(cases)} correct ({score:.0%})**"
    score_md = (
        f"### Accuracy: {score:.1%}\n\n"
        f"- Total questions: {len(cases)}\n"
        f"- Correct: {correct_count}\n"
        f"- Incorrect: {len(cases) - correct_count}"
    )
    df = pd.DataFrame(rows, columns=_BENCH_HEADERS)
    return status, df, score_md


def run_rw_on_accepted(kg_state):
    """Run Ralph Wiggum Orchestrator on accepted golden answers."""
    from crystal.review import collect_accepted_cases
    from benchmarks.ralph_wiggum.orchestrator import Orchestrator

    cases = collect_accepted_cases()
    if not cases:
        return "No accepted golden answers found.", ""

    if len(cases) < 3:
        return f"Only {len(cases)} accepted cases — need at least 3 for meaningful RW loop.", ""

    try:
        orch = Orchestrator(
            kg=kg_state,
            cases=cases,
            use_git=False,
            use_full_pipeline=True,
        )
        result = orch.run(threshold=0.90, max_iterations_per_loop=5)
        status = f"**RW complete — final score: {result.overall_score:.1%}**"
        return status, result.unified_report
    except Exception as e:
        return f"RW loop error: {e}", ""


# ── Layout ───────────────────────────────────────────────────────────────

@dataclass
class ReviewTab:
    review_dashboard: gr.Markdown
    revalidate_btn: gr.Button
    revalidate_status: gr.Markdown
    batch_selector: gr.Dropdown
    batch_meta: gr.Markdown
    overview_table: gr.Dataframe
    review_q_idx: gr.State
    prev_q_btn: gr.Button
    next_q_btn: gr.Button
    question_selector: gr.Dropdown
    question_text_md: gr.Markdown
    crystal_proposed_box: gr.Textbox
    route_info_md: gr.Markdown
    source_triplet_md: gr.Markdown
    golden_answer_box: gr.Textbox
    accept_btn: gr.Button
    reject_btn: gr.Button
    review_action_status: gr.Markdown
    current_status_label: gr.Markdown
    kg_subgraph_md: gr.Markdown
    doc_selector: gr.Dropdown
    doc_text_box: gr.Textbox
    gaps_table: gr.Dataframe
    bench_btn: gr.Button
    rw_btn: gr.Button
    bench_status: gr.Markdown
    bench_score_md: gr.Markdown
    bench_table: gr.Dataframe
    rw_status: gr.Markdown
    rw_report: gr.Markdown


def build_review_tab(kg_state: gr.State) -> ReviewTab:
    initial_choices = _get_batch_choices()
    initial_choice = initial_choices[0] if initial_choices and initial_choices[0] != "(no batches)" else None

    # ── Section 1: Dashboard summary ──
    with gr.Column(variant="panel"):
        gr.Markdown("## Review Dashboard")
        review_dashboard = gr.Markdown(format_review_dashboard())
        with gr.Row():
            revalidate_btn = gr.Button(
                "Revalidate Pending Questions",
                variant="secondary",
                size="sm",
            )
        revalidate_status = gr.Markdown("")

    gr.HTML("<div style='height: 24px'></div>")

    # ── Section 2: Batch selection ──
    with gr.Column(variant="panel"):
        gr.Markdown("## Select Batch")
        batch_selector = gr.Dropdown(
            choices=initial_choices,
            value=initial_choice,
            label="Review Batch",
            interactive=True,
        )
        batch_meta = gr.Markdown(
            _load_batch_metadata(initial_choice) if initial_choice else ""
        )

        gr.Markdown("### All Questions in Batch")
        overview_table = gr.Dataframe(
            value=_load_overview_table(_extract_batch_id(initial_choice)),
            interactive=False,
            wrap=True,
        )

    review_q_idx = gr.State(0)

    gr.HTML("<div style='height: 24px'></div>")

    # ── Section 3: Question review ──
    with gr.Column(variant="panel"):
        gr.Markdown("## Question Review")
        gr.Markdown(
            "Review each question: read the source material, verify or correct "
            "Crystal's proposed answer, then accept or reject."
        )

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    prev_q_btn = gr.Button("< Prev", size="sm", scale=1)
                    question_selector = gr.Dropdown(
                        choices=_get_question_choices(_extract_batch_id(initial_choice)),
                        label="Question",
                        interactive=True,
                        scale=6,
                    )
                    next_q_btn = gr.Button("Next >", size="sm", scale=1)

                question_text_md = gr.Markdown("*Select a batch to begin reviewing.*")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Crystal's Proposed Answer**")
                        crystal_proposed_box = gr.Textbox(
                            interactive=False, lines=5,
                            show_label=False, container=False,
                        )
                    with gr.Column():
                        gr.Markdown("**Routing Info**")
                        route_info_md = gr.Markdown("")
                        gr.Markdown("**Source Triplet**")
                        source_triplet_md = gr.Markdown("")

                gr.Markdown(
                    "**Golden Answer** — *Edit below if Crystal's answer is wrong. "
                    "This becomes the verified ground truth.*"
                )
                golden_answer_box = gr.Textbox(
                    interactive=True, lines=5,
                    show_label=False, container=False,
                    placeholder="The correct answer goes here...",
                )

                with gr.Row():
                    accept_btn = gr.Button("Accept", variant="primary", scale=2)
                    reject_btn = gr.Button("Reject", variant="stop", scale=2)
                review_action_status = gr.Markdown("")
                current_status_label = gr.Markdown("")

            with gr.Column(scale=2):
                gr.Markdown("### Source Material")

                with gr.Accordion("KG Subgraph for Source Entity", open=True):
                    gr.Markdown(
                        "*Facts from the KG for the entity in this question's "
                        "source triplet. Use this to verify context.*"
                    )
                    kg_subgraph_md = gr.Markdown(
                        "*Select a question to see its KG subgraph.*"
                    )

                with gr.Accordion("Source Document Text", open=False):
                    gr.Markdown(
                        "*Select a document to read the original opinion text.*"
                    )
                    _init_doc_slugs = find_batch_document_slugs(
                        _extract_batch_id(initial_choice)
                    ) if initial_choice else []
                    _init_doc_choices = [
                        s.replace("-", " ").title() + f" ({s})" for s in _init_doc_slugs
                    ] or ["(no source documents found)"]
                    doc_selector = gr.Dropdown(
                        choices=_init_doc_choices,
                        label="Source Document",
                        interactive=True,
                    )
                    doc_text_box = gr.Textbox(
                        label="Document Text",
                        interactive=False,
                        lines=30,
                        max_lines=60,
                    )

    gr.HTML("<div style='height: 24px'></div>")

    # ── Section 4: Known detection failures ──
    with gr.Accordion("Known Detection Failures — questions Crystal cannot yet answer correctly", open=False):
        gr.Markdown(
            "*These are known cases where the extraction pipeline or routing logic "
            "fails. They require engineering fixes, not review decisions.*"
        )
        gaps_table = gr.Dataframe(
            value=_get_known_gaps_df(),
            interactive=False,
            wrap=True,
        )

    gr.HTML("<div style='height: 24px'></div>")

    # ── Section 5: Benchmark & RW loop ──
    with gr.Column(variant="panel"):
        gr.Markdown("## Benchmark & Improvement")
        gr.Markdown(
            "Run all **accepted** golden answers through Crystal to measure accuracy, "
            "then optionally run Ralph Wiggum loops to improve."
        )
        with gr.Row():
            bench_btn = gr.Button("Run Benchmark on Accepted", variant="primary", scale=2)
            rw_btn = gr.Button("Improve with Ralph Wiggum", variant="secondary", scale=2)
        bench_status = gr.Markdown("")
        bench_score_md = gr.Markdown("")
        bench_table = gr.Dataframe(
            value=pd.DataFrame(columns=_BENCH_HEADERS),
            interactive=False,
            wrap=True,
        )
        rw_status = gr.Markdown("")
        rw_report = gr.Markdown("")

    # ── Load initial question if a batch exists ──
    _init_batch_id = _extract_batch_id(initial_choice)
    if _init_batch_id:
        _init_detail = _load_question_detail(_init_batch_id, 0)
        kg_subgraph_md.value = _kg_subgraph_for_question(
            _init_batch_id, 0, _default_kg,
        )
        if _init_doc_choices and _init_doc_choices[0] != "(no source documents found)":
            doc_text_box.value = _load_doc_text(_init_doc_choices[0])
    else:
        _init_detail = ("*Select a batch to begin reviewing.*", "", "", "", "", "")
    question_text_md.value = _init_detail[0]
    crystal_proposed_box.value = _init_detail[1]
    route_info_md.value = _init_detail[2]
    source_triplet_md.value = _init_detail[3]
    golden_answer_box.value = _init_detail[4]
    current_status_label.value = _init_detail[5]

    # ── Wiring ──

    batch_selector.change(
        fn=_on_batch_selected,
        inputs=[batch_selector, kg_state],
        outputs=[
            batch_meta,
            question_selector, review_q_idx,
            question_text_md, crystal_proposed_box, route_info_md,
            source_triplet_md, golden_answer_box, current_status_label,
            doc_selector, doc_text_box,
            overview_table,
            kg_subgraph_md,
        ],
    )

    question_selector.change(
        fn=_on_question_selected,
        inputs=[question_selector, batch_selector, kg_state],
        outputs=[
            review_q_idx,
            question_text_md, crystal_proposed_box, route_info_md,
            source_triplet_md, golden_answer_box, current_status_label,
            kg_subgraph_md,
            doc_selector, doc_text_box,
        ],
    )

    prev_q_btn.click(
        fn=lambda idx, bc, kg: _navigate_question(idx, -1, bc, kg),
        inputs=[review_q_idx, batch_selector, kg_state],
        outputs=[
            review_q_idx, question_selector,
            question_text_md, crystal_proposed_box, route_info_md,
            source_triplet_md, golden_answer_box, current_status_label,
            kg_subgraph_md,
            doc_selector, doc_text_box,
        ],
    )

    next_q_btn.click(
        fn=lambda idx, bc, kg: _navigate_question(idx, +1, bc, kg),
        inputs=[review_q_idx, batch_selector, kg_state],
        outputs=[
            review_q_idx, question_selector,
            question_text_md, crystal_proposed_box, route_info_md,
            source_triplet_md, golden_answer_box, current_status_label,
            kg_subgraph_md,
            doc_selector, doc_text_box,
        ],
    )

    accept_btn.click(
        fn=_accept_question,
        inputs=[batch_selector, review_q_idx, golden_answer_box, kg_state],
        outputs=[
            review_action_status,
            review_q_idx, question_selector,
            question_text_md, crystal_proposed_box, route_info_md,
            source_triplet_md, golden_answer_box, current_status_label,
            overview_table, batch_meta,
            kg_subgraph_md,
            doc_selector, doc_text_box,
        ],
    )

    reject_btn.click(
        fn=_reject_question,
        inputs=[batch_selector, review_q_idx, golden_answer_box, kg_state],
        outputs=[
            review_action_status,
            review_q_idx, question_selector,
            question_text_md, crystal_proposed_box, route_info_md,
            source_triplet_md, golden_answer_box, current_status_label,
            overview_table, batch_meta,
            kg_subgraph_md,
            doc_selector, doc_text_box,
        ],
    )

    doc_selector.change(
        fn=_load_doc_text,
        inputs=[doc_selector],
        outputs=[doc_text_box],
    )

    revalidate_btn.click(
        fn=_revalidate_pending,
        inputs=[kg_state, batch_selector],
        outputs=[revalidate_status, review_dashboard, overview_table, batch_meta],
    )

    bench_btn.click(
        fn=run_benchmark_on_accepted,
        inputs=[kg_state],
        outputs=[bench_status, bench_table, bench_score_md],
    )

    rw_btn.click(
        fn=run_rw_on_accepted,
        inputs=[kg_state],
        outputs=[rw_status, rw_report],
    )

    return ReviewTab(
        review_dashboard=review_dashboard,
        revalidate_btn=revalidate_btn,
        revalidate_status=revalidate_status,
        batch_selector=batch_selector,
        batch_meta=batch_meta,
        overview_table=overview_table,
        review_q_idx=review_q_idx,
        prev_q_btn=prev_q_btn,
        next_q_btn=next_q_btn,
        question_selector=question_selector,
        question_text_md=question_text_md,
        crystal_proposed_box=crystal_proposed_box,
        route_info_md=route_info_md,
        source_triplet_md=source_triplet_md,
        golden_answer_box=golden_answer_box,
        accept_btn=accept_btn,
        reject_btn=reject_btn,
        review_action_status=review_action_status,
        current_status_label=current_status_label,
        kg_subgraph_md=kg_subgraph_md,
        doc_selector=doc_selector,
        doc_text_box=doc_text_box,
        gaps_table=gaps_table,
        bench_btn=bench_btn,
        rw_btn=rw_btn,
        bench_status=bench_status,
        bench_score_md=bench_score_md,
        bench_table=bench_table,
        rw_status=rw_status,
        rw_report=rw_report,
    )
