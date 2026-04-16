"""Ingest Documents tab — document extraction, question generation,
and before/after comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import pandas as pd

import crystal.llm
from crystal.compare import before_after_comparison, generate_questions_from_triplets
from crystal.ingest import DocumentIngestionResult, ingest_document
from crystal.review import save_proposed_as_batch
from crystal.state import make_initial_state

from crystal.ui.formatting import (
    _CONFIDENCE_BADGES,
    _ROUTE_LABELS,
    _default_kg_info,
    _format_ingest_target,
    _format_kg_stats,
    _kg_info,
)
from crystal.ui.state import _DEFAULT_KG_MODE, _default_kg, _graph


# ── Column headers ───────────────────────────────────────────────────────

_INGEST_AUTO_HEADERS = ["Subject", "Predicate", "Object", "Source", "Confidence", "Sentence"]
_INGEST_PENDING_HEADERS = ["#", "Subject", "Predicate", "Object", "Source", "Confidence", "Sentence"]
_PROPOSED_HEADERS = ["Question", "Crystal Answer", "Route", "Confidence", "Expected", "Golden Answer"]
_COMPARE_HEADERS = ["Question", "Crystal (KG-grounded)", "Route", "LLM + Docs", "Naked LLM"]
_GOLDEN_COMPARE_HEADERS = [
    "Question", "Golden Answer", "Crystal", "C?", "LLM+Docs", "D?", "Naked LLM", "N?",
]


# ── Row shaping ──────────────────────────────────────────────────────────

def _scored_to_auto_df(triplets):
    rows = [
        [st.subject, st.predicate, st.object, st.extraction_source,
         f"{st.ingestion_confidence:.2f}", st.source_sentence[:120]]
        for st in triplets
    ]
    return pd.DataFrame(rows, columns=_INGEST_AUTO_HEADERS) if rows else pd.DataFrame(columns=_INGEST_AUTO_HEADERS)


def _scored_to_pending_df(triplets):
    rows = [
        [str(i), st.subject, st.predicate, st.object, st.extraction_source,
         f"{st.ingestion_confidence:.2f}", st.source_sentence[:120]]
        for i, st in enumerate(triplets)
    ]
    return pd.DataFrame(rows, columns=_INGEST_PENDING_HEADERS) if rows else pd.DataFrame(columns=_INGEST_PENDING_HEADERS)


def _ingest_stats_text(result):
    s = result.stats
    return (
        f"**Extracted {s.get('total_extracted', 0)} facts** from `{s.get('source', '?')}` "
        f"in {s.get('elapsed_seconds', 0):.1f}s\n\n"
        f"- Auto-accepted (above threshold → inserted into KG): **{s.get('auto_accepted', 0)}**\n"
        f"- Pending review (below threshold → needs your approval): **{s.get('pending_review', 0)}**\n"
        f"- Rejected (failed validation gates): **{s.get('rejected', 0)}**\n"
        f"- NER extractions: {s.get('ner_triplets', 0)} | LLM extractions: {s.get('llm_triplets', 0)}"
    )


# ── Core actions ─────────────────────────────────────────────────────────

def run_ingestion(files, paste_text, kg_state, threshold):
    """Run document ingestion on uploaded files or pasted text."""
    texts = []
    if files:
        for f in files:
            try:
                texts.append(("file", Path(f.name)))
            except Exception:
                pass
    if paste_text and paste_text.strip():
        texts.append(("text", paste_text.strip()))

    if not texts:
        empty = DocumentIngestionResult()
        return (
            kg_state, empty,
            "No documents provided.",
            pd.DataFrame(columns=_INGEST_AUTO_HEADERS),
            pd.DataFrame(columns=_INGEST_PENDING_HEADERS),
            _format_kg_stats(_kg_info(kg_state, "current")),
        )

    all_auto = []
    all_pending = []
    all_rejected = []
    combined_stats = {
        "total_extracted": 0, "auto_accepted": 0, "pending_review": 0,
        "rejected": 0, "ner_triplets": 0, "llm_triplets": 0,
        "elapsed_seconds": 0.0, "source": "",
    }

    try:
        call_llm = crystal.llm.call_llm
    except Exception:
        call_llm = None

    for kind, content in texts:
        try:
            result = ingest_document(
                content, kg_state,
                call_llm_fn=call_llm,
                auto_accept_threshold=float(threshold),
                domain="legal",
            )
            all_auto.extend(result.auto_accepted)
            all_pending.extend(result.pending_review)
            all_rejected.extend(result.rejected)
            for k in ("total_extracted", "auto_accepted", "pending_review",
                       "rejected", "ner_triplets", "llm_triplets"):
                combined_stats[k] += result.stats.get(k, 0)
            combined_stats["elapsed_seconds"] += result.stats.get("elapsed_seconds", 0)
            src = result.stats.get("source", "")
            combined_stats["source"] = src if not combined_stats["source"] else combined_stats["source"] + f", {src}"
        except Exception as e:
            return (
                kg_state, DocumentIngestionResult(),
                f"Error during ingestion: {e}",
                pd.DataFrame(columns=_INGEST_AUTO_HEADERS),
                pd.DataFrame(columns=_INGEST_PENDING_HEADERS),
                _format_kg_stats(_kg_info(kg_state, "current")),
            )

    combined = DocumentIngestionResult(
        auto_accepted=all_auto,
        pending_review=all_pending,
        rejected=all_rejected,
        stats=combined_stats,
    )
    combined._kg = kg_state
    combined._source = combined_stats["source"]

    source_label = combined_stats.get("source", "Ingested")
    info = _kg_info(kg_state, source_label)

    return (
        kg_state, combined,
        _ingest_stats_text(combined),
        _scored_to_auto_df(all_auto),
        _scored_to_pending_df(all_pending),
        _format_kg_stats(info),
    )


def accept_all_pending_action(ingest_result, kg_state):
    if ingest_result is None or not isinstance(ingest_result, DocumentIngestionResult):
        return "No ingestion result.", pd.DataFrame(columns=_INGEST_PENDING_HEADERS), _format_kg_stats(_kg_info(kg_state, "current"))
    n = ingest_result.accept_all_pending()
    info = _kg_info(kg_state, "current")
    return f"Accepted {n} triplets into KG.", _scored_to_pending_df(ingest_result.pending_review), _format_kg_stats(info)


def reject_all_pending_action(ingest_result, kg_state):
    if ingest_result is None or not isinstance(ingest_result, DocumentIngestionResult):
        return "No ingestion result.", pd.DataFrame(columns=_INGEST_PENDING_HEADERS)
    n = ingest_result.reject_pending(list(range(len(ingest_result.pending_review))))
    return f"Rejected {n} triplets.", _scored_to_pending_df(ingest_result.pending_review)


def save_pending_decisions(ingest_result, table_data, kg_state):
    """Accept or reject specific pending items (currently accepts all remaining)."""
    if ingest_result is None or not isinstance(ingest_result, DocumentIngestionResult):
        return "No ingestion result.", pd.DataFrame(columns=_INGEST_PENDING_HEADERS), _format_kg_stats(_kg_info(kg_state, "current"))

    n = ingest_result.accept_all_pending()
    info = _kg_info(kg_state, "current")
    return (
        f"Accepted {n} remaining pending triplets into KG.",
        _scored_to_pending_df(ingest_result.pending_review),
        _format_kg_stats(info),
    )


# ── Proposed answers ─────────────────────────────────────────────────────

def run_proposed_answers(ingest_result, kg_state):
    """Generate questions from extracted facts and show Crystal's proposed answers.

    The "Golden Answer" column is pre-filled with Crystal's answer for the user
    to verify or correct.
    """
    if ingest_result is None or not isinstance(ingest_result, DocumentIngestionResult):
        return "Run ingestion first.", pd.DataFrame(columns=_PROPOSED_HEADERS)

    all_triplets = [st.as_tuple() for st in ingest_result.auto_accepted]
    if not all_triplets:
        return "No extracted facts to generate questions from.", pd.DataFrame(columns=_PROPOSED_HEADERS)

    questions = generate_questions_from_triplets(all_triplets, max_questions=15)
    if not questions:
        return "Could not generate questions from extracted facts.", pd.DataFrame(columns=_PROPOSED_HEADERS)

    rows = []
    for q_text in questions:
        try:
            state = make_initial_state(q_text, kg=kg_state)
            final = _graph.invoke(state)
            answer = final.get("final_response", "")
            prompt_type = final.get("prompt_type", "unknown")
            route_label = _ROUTE_LABELS.get(prompt_type, prompt_type)
            confidence = _CONFIDENCE_BADGES.get(prompt_type, "UNKNOWN").split(" — ")[0]
        except Exception as e:
            answer = f"[Error: {e}]"
            route_label = "error"
            confidence = "N/A"

        expected = ""
        for s, p, o in all_triplets:
            if s.lower() in q_text.lower() and p.lower() in q_text.lower():
                expected = f"{o}"
                break
            if s.lower() in q_text.lower():
                expected = f"{o}"
                break

        golden = answer[:200]

        rows.append([q_text, answer[:200], route_label, confidence, expected[:100], golden])

    status = (
        f"Generated {len(rows)} questions from extracted facts. "
        "**Review the 'Golden Answer' column** — edit to correct Crystal's answer, "
        "then click 'Save to Review' to create ground truth."
    )
    return status, pd.DataFrame(rows, columns=_PROPOSED_HEADERS)


def save_proposed_to_review(proposed_table, ingest_result):
    """Save the proposed answers table as a review batch for ground truth generation."""
    if proposed_table is None:
        return "No proposed answers to save."

    if isinstance(proposed_table, pd.DataFrame):
        rows_data = proposed_table.values.tolist()
    else:
        rows_data = proposed_table

    if not rows_data or len(rows_data) == 0:
        return "No proposed answers to save."

    source = "document_extraction"
    if ingest_result and isinstance(ingest_result, DocumentIngestionResult):
        source = ingest_result.stats.get("source", "document_extraction")

    all_scored = []
    if ingest_result and isinstance(ingest_result, DocumentIngestionResult):
        all_scored = ingest_result.auto_accepted

    proposed_rows = []
    for row in rows_data:
        try:
            question = str(row[0])
            crystal_answer = str(row[1])
            route = str(row[2])
            confidence = str(row[3])
            expected = str(row[4]) if len(row) > 4 else ""
            golden = str(row[5]) if len(row) > 5 else crystal_answer

            src_triplet = []
            src_sentence = ""
            row_origin = "unknown"
            row_source_doc = ""
            for st in all_scored:
                if st.subject.lower() in question.lower():
                    src_triplet = [st.subject, st.predicate, st.object]
                    src_sentence = st.source_sentence
                    row_origin = st.origin
                    row_source_doc = st.source_document
                    break

            proposed_rows.append({
                "question": question,
                "crystal_answer": crystal_answer,
                "route": route,
                "confidence": confidence,
                "expected": expected,
                "golden_answer": golden,
                "source_triplet": src_triplet,
                "source_sentence": src_sentence,
                "origin": row_origin,
                "source_document": row_source_doc,
            })
        except (IndexError, ValueError):
            continue

    if not proposed_rows:
        return "No valid rows to save."

    batch_path = save_proposed_as_batch(proposed_rows, source=source)
    if batch_path is None:
        return "Failed to save batch."

    return (
        f"Saved **{len(proposed_rows)} questions** to review batch: "
        f"`{batch_path.name}`. Go to the **Review** tab to accept/reject."
    )


# ── Before/after comparison ──────────────────────────────────────────────

def run_comparison(questions_text, ingest_result, kg_state):
    """Run before/after comparison on user-provided questions."""
    if not questions_text or not questions_text.strip():
        if ingest_result and isinstance(ingest_result, DocumentIngestionResult):
            all_triplets = [st.as_tuple() for st in ingest_result.auto_accepted]
            questions = generate_questions_from_triplets(all_triplets, max_questions=5)
        else:
            return "Enter questions or run ingestion first.", pd.DataFrame(columns=_COMPARE_HEADERS)
    else:
        questions = [q.strip() for q in questions_text.strip().split("\n") if q.strip()]

    if not questions:
        return "No questions to compare.", pd.DataFrame(columns=_COMPARE_HEADERS)

    doc_text = ""
    if ingest_result and isinstance(ingest_result, DocumentIngestionResult):
        sentences = [st.source_sentence for st in ingest_result.auto_accepted + ingest_result.pending_review if st.source_sentence]
        doc_text = " ".join(sentences[:20])

    try:
        result = before_after_comparison(questions, kg_state, document_text=doc_text)
    except Exception as e:
        return f"Comparison error: {e}", pd.DataFrame(columns=_COMPARE_HEADERS)

    rows = [
        [r.question, r.crystal_answer[:200], r.crystal_route,
         r.llm_with_docs_answer[:200], r.llm_naked_answer[:200]]
        for r in result.rows
    ]
    status = f"Compared {len(result.rows)} questions across 3 modes."
    return status, pd.DataFrame(rows, columns=_COMPARE_HEADERS)


def run_golden_comparison(kg_state):
    """Run three-arm comparison on all accepted golden answers with accuracy scoring."""
    from benchmarks.three_arm_comparison import run_three_arm

    try:
        report = run_three_arm(kg=kg_state)
    except Exception as e:
        empty = pd.DataFrame(columns=_GOLDEN_COMPARE_HEADERS)
        return f"Error: {e}", "", empty

    if not report.results:
        empty = pd.DataFrame(columns=_GOLDEN_COMPARE_HEADERS)
        return "No accepted golden answers found. Accept some questions in the Review tab first.", "", empty

    rows = []
    for r in report.results:
        rows.append([
            r.question[:60],
            r.golden_answer[:50],
            r.crystal.answer[:60],
            "YES" if r.crystal.correct else "NO",
            r.llm_docs.answer[:60],
            "YES" if r.llm_docs.correct else "NO",
            r.llm_naked.answer[:60],
            "YES" if r.llm_naked.correct else "NO",
        ])

    n = len(report.results)
    crystal_wins = sum(1 for r in report.results if r.crystal.correct and not r.llm_naked.correct)

    scores_md = (
        f"### Accuracy Scores\n\n"
        f"| Arm | Accuracy |\n"
        f"|-----|----------|\n"
        f"| **Crystal + KG** | **{report.crystal_accuracy:.0%}** |\n"
        f"| LLM + Docs | {report.llm_docs_accuracy:.0%} |\n"
        f"| Naked LLM | {report.llm_naked_accuracy:.0%} |\n\n"
        f"Crystal wins over naked LLM on **{crystal_wins}/{n}** questions."
    )

    status = f"Compared {n} golden answers across 3 arms."
    return status, scores_md, pd.DataFrame(rows, columns=_GOLDEN_COMPARE_HEADERS)


# ── Layout ───────────────────────────────────────────────────────────────

@dataclass
class IngestTab:
    ingest_result_state: gr.State
    ingest_target_label: gr.Markdown
    ingest_files: gr.File
    ingest_paste: gr.Textbox
    ingest_threshold: gr.Slider
    ingest_btn: gr.Button
    ingest_status: gr.Markdown
    ingest_kg_stats: gr.Markdown
    ingest_auto_table: gr.Dataframe
    ingest_pending_table: gr.Dataframe
    accept_all_btn: gr.Button
    reject_all_btn: gr.Button
    ingest_decision_status: gr.Markdown
    proposed_btn: gr.Button
    proposed_status: gr.Markdown
    proposed_table: gr.Dataframe
    save_proposed_btn: gr.Button
    proposed_save_status: gr.Markdown
    compare_questions: gr.Textbox
    compare_btn: gr.Button
    compare_status: gr.Markdown
    compare_table: gr.Dataframe
    golden_compare_btn: gr.Button
    golden_compare_status: gr.Markdown
    golden_compare_scores: gr.Markdown
    golden_compare_table: gr.Dataframe


def build_ingest_tab(kg_state: gr.State) -> IngestTab:
    gr.Markdown(
        "Upload legal documents to extract facts into the active knowledge graph. "
        "High-confidence extractions are auto-accepted; lower-confidence ones require review."
    )

    ingest_result_state = gr.State(None)

    ingest_target_label = gr.Markdown(
        _format_ingest_target(_default_kg, _DEFAULT_KG_MODE),
    )

    # ── Section 1: Extraction ──
    with gr.Column(variant="panel"):
        gr.Markdown("## 1. Extract Facts from Documents")
        with gr.Row():
            with gr.Column(scale=2):
                ingest_files = gr.File(
                    label="Upload Documents (.txt, .csv, .json)",
                    file_types=[".txt", ".csv", ".json"],
                    file_count="multiple",
                )
            with gr.Column(scale=2):
                ingest_paste = gr.Textbox(
                    label="Or paste text",
                    lines=5,
                    placeholder="Paste legal opinion text here...",
                )
            with gr.Column(scale=1):
                ingest_threshold = gr.Slider(
                    minimum=0.40, maximum=0.95, value=0.70, step=0.05,
                    label="Auto-accept threshold",
                )
                ingest_btn = gr.Button("Extract & Ingest", variant="primary")

        ingest_status = gr.Markdown("")
        ingest_kg_stats = gr.Markdown(_format_kg_stats(_default_kg_info()))

    gr.HTML("<div style='height: 24px'></div>")

    # ── Section 2: Extracted Triplets ──
    with gr.Column(variant="panel"):
        gr.Markdown("## 2. Review Extracted Triplets")

        with gr.Accordion("Auto-Accepted — above confidence threshold", open=False):
            gr.Markdown(
                "*These triplets scored above the auto-accept threshold and were "
                "inserted into the KG automatically.*"
            )
            ingest_auto_table = gr.Dataframe(
                value=pd.DataFrame(columns=_INGEST_AUTO_HEADERS),
                interactive=False,
                wrap=True,
            )

        gr.Markdown("### Pending Review — below threshold, need your approval")
        ingest_pending_table = gr.Dataframe(
            value=pd.DataFrame(columns=_INGEST_PENDING_HEADERS),
            interactive=False,
            wrap=True,
        )

        with gr.Row():
            accept_all_btn = gr.Button("Accept All Pending", variant="primary")
            reject_all_btn = gr.Button("Reject All Pending", variant="stop")
        ingest_decision_status = gr.Markdown("")

    gr.HTML("<div style='height: 24px'></div>")

    # ── Section 3: Question & Answer Generation ──
    with gr.Column(variant="panel"):
        gr.Markdown("## 3. Generate Questions & Golden Answers")
        gr.Markdown(
            "Crystal generates questions from extracted facts and proposes answers. "
            "Review the **Golden Answer** column — edit any wrong answers, then save."
        )
        proposed_btn = gr.Button("Generate Questions from Extracted Facts", variant="secondary")
        proposed_status = gr.Markdown("")
        proposed_table = gr.Dataframe(
            value=pd.DataFrame(columns=_PROPOSED_HEADERS),
            interactive=True,
            wrap=True,
        )
        with gr.Row():
            save_proposed_btn = gr.Button("Save to Review", variant="primary")
        proposed_save_status = gr.Markdown("")

    gr.HTML("<div style='height: 24px'></div>")

    # ── Section 4: Test Your Ingestion ──
    with gr.Column(variant="panel"):
        gr.Markdown("## 4. Test Your Ingestion")

        with gr.Accordion("Quick Comparison — Crystal vs LLM", open=False):
            gr.Markdown(
                "Compare Crystal (with ingested KG) vs naked LLM. "
                "Enter questions below, or leave blank to auto-generate from extracted facts."
            )
            compare_questions = gr.Textbox(
                label="Questions (one per line)",
                lines=3,
                placeholder="What court decided Miranda v. Arizona?\nWho wrote the opinion in Roe v. Wade?",
            )
            compare_btn = gr.Button("Run Comparison", variant="primary")
            compare_status = gr.Markdown("")
            compare_table = gr.Dataframe(
                value=pd.DataFrame(columns=_COMPARE_HEADERS),
                interactive=False,
                wrap=True,
            )

        with gr.Accordion("Three-Arm Comparison — Crystal vs LLM+Docs vs Naked LLM", open=False):
            gr.Markdown(
                "Run **all accepted golden answers** through Crystal+KG, LLM+Docs, and Naked LLM. "
                "Shows accuracy for each arm and highlights where Crystal wins."
            )
            golden_compare_btn = gr.Button("Run Three-Arm Comparison", variant="primary")
            golden_compare_status = gr.Markdown("")
            golden_compare_scores = gr.Markdown("")
            golden_compare_table = gr.Dataframe(
                value=pd.DataFrame(columns=_GOLDEN_COMPARE_HEADERS),
                interactive=False,
                wrap=True,
            )

    # ── Wiring (within-tab only) ──
    proposed_btn.click(
        fn=run_proposed_answers,
        inputs=[ingest_result_state, kg_state],
        outputs=[proposed_status, proposed_table],
    )

    save_proposed_btn.click(
        fn=save_proposed_to_review,
        inputs=[proposed_table, ingest_result_state],
        outputs=[proposed_save_status],
    )

    ingest_btn.click(
        fn=run_ingestion,
        inputs=[ingest_files, ingest_paste, kg_state, ingest_threshold],
        outputs=[kg_state, ingest_result_state, ingest_status,
                 ingest_auto_table, ingest_pending_table, ingest_kg_stats],
    )

    accept_all_btn.click(
        fn=accept_all_pending_action,
        inputs=[ingest_result_state, kg_state],
        outputs=[ingest_decision_status, ingest_pending_table, ingest_kg_stats],
    )

    reject_all_btn.click(
        fn=reject_all_pending_action,
        inputs=[ingest_result_state, kg_state],
        outputs=[ingest_decision_status, ingest_pending_table],
    )

    compare_btn.click(
        fn=run_comparison,
        inputs=[compare_questions, ingest_result_state, kg_state],
        outputs=[compare_status, compare_table],
    )

    golden_compare_btn.click(
        fn=run_golden_comparison,
        inputs=[kg_state],
        outputs=[golden_compare_status, golden_compare_scores, golden_compare_table],
    )

    return IngestTab(
        ingest_result_state=ingest_result_state,
        ingest_target_label=ingest_target_label,
        ingest_files=ingest_files,
        ingest_paste=ingest_paste,
        ingest_threshold=ingest_threshold,
        ingest_btn=ingest_btn,
        ingest_status=ingest_status,
        ingest_kg_stats=ingest_kg_stats,
        ingest_auto_table=ingest_auto_table,
        ingest_pending_table=ingest_pending_table,
        accept_all_btn=accept_all_btn,
        reject_all_btn=reject_all_btn,
        ingest_decision_status=ingest_decision_status,
        proposed_btn=proposed_btn,
        proposed_status=proposed_status,
        proposed_table=proposed_table,
        save_proposed_btn=save_proposed_btn,
        proposed_save_status=proposed_save_status,
        compare_questions=compare_questions,
        compare_btn=compare_btn,
        compare_status=compare_status,
        compare_table=compare_table,
        golden_compare_btn=golden_compare_btn,
        golden_compare_status=golden_compare_status,
        golden_compare_scores=golden_compare_scores,
        golden_compare_table=golden_compare_table,
    )
