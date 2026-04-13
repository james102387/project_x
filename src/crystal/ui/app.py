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
"""

import time
from pathlib import Path

import gradio as gr
import pandas as pd

from crystal.graph import build_crystal_graph
from crystal.state import make_initial_state
from crystal.ingest import ingest, build_kg, ingest_document, DocumentIngestionResult
from crystal.ingest.ner import ingest_text
from crystal.compare import before_after_comparison, generate_questions_from_triplets
from crystal.review import (
    find_batch_document_slugs,
    format_review_dashboard,
    list_batches,
    load_batch_context,
    load_batch_questions,
    load_known_gaps,
    load_pending_questions,
    load_source_document_text,
    save_proposed_as_batch,
    save_review_decisions,
    save_single_review_decision,
)
from crystal.tools.kg import remulak_kg
from crystal.tools.kg.graph import KnowledgeGraph
from crystal.tools.kg.legal import load_legal_kg
import crystal.llm


# ── App state ────────────────────────────────────────────────────────────

_graph = build_crystal_graph()

_ROOT = Path(__file__).parent.parent.parent
_LEGAL_DB_PATH = _ROOT / "data" / "legal.sqlite"
_legal_kg = load_legal_kg(_LEGAL_DB_PATH)

KG_MODES = {}
if _legal_kg is not None:
    KG_MODES["Legal (SCOTUS — SQLite)"] = _legal_kg
KG_MODES["Remulak (demo)"] = remulak_kg

_DEFAULT_KG_MODE = "Legal (SCOTUS — SQLite)" if _legal_kg is not None else "Remulak (demo)"
_default_kg = KG_MODES[_DEFAULT_KG_MODE]


def _default_kg_info() -> dict:
    """Return info dict for the default KG."""
    return _kg_info(_default_kg, _DEFAULT_KG_MODE)


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
    info = _kg_info(remulak_kg, "Remulak (demo)")
    return remulak_kg, "Reset to Remulak.", _format_kg_stats(info), _format_kg_facts(remulak_kg)


def switch_kg_mode(mode: str):
    """Switch the active KG by mode name."""
    kg = KG_MODES.get(mode)
    if kg is None:
        return _default_kg, f"Unknown mode: {mode}", _format_kg_stats(_default_kg_info()), _format_kg_facts(_default_kg)
    info = _kg_info(kg, mode)
    return kg, f"Switched to {mode}.", _format_kg_stats(info), _format_kg_facts(kg)


_ROUTE_LABELS = {
    "kg_answerable": "KG Grounded (direct)",
    "kg_augmented": "KG + LLM Reasoning",
    "pure_math": "Calculator (direct)",
    "math_answerable": "Calculator (direct)",
    "math_augmented": "Calculator + LLM Reasoning",
    "no_math": "LLM Fallback (ungrounded)",
}

_CONFIDENCE_BADGES = {
    "kg_answerable": "HIGH — answer sourced directly from verified knowledge graph",
    "kg_augmented": "MEDIUM — KG facts injected, LLM reasoning applied",
    "pure_math": "HIGH — computed mathematically",
    "math_answerable": "HIGH — computed mathematically",
    "math_augmented": "MEDIUM — math computed, LLM reasoning applied",
    "no_math": "LOW — no grounding, pure LLM generation",
}


def ask_question(question, kg_state):
    """Run a question through Crystal and naked LLM, return side-by-side results."""
    if not question or not question.strip():
        return "", "", "", "", ""

    question = question.strip()

    state = make_initial_state(question, kg=kg_state)
    grounding_info = ""
    try:
        final = _graph.invoke(state)
        crystal_response = final.get("final_response", "")
        prompt_type = final.get("prompt_type", "unknown")
        metrics = final.get("token_metrics", {})

        route_label = _ROUTE_LABELS.get(prompt_type, prompt_type)
        confidence = _CONFIDENCE_BADGES.get(prompt_type, "UNKNOWN")
        crystal_meta = f"**Route:** {route_label}\n**Confidence:** {confidence}"

        if metrics:
            if metrics.get("actual_prompt_tokens"):
                crystal_meta += f"\n**Tokens:** {metrics['actual_prompt_tokens']} prompt, {metrics.get('actual_output_tokens', '?')} output"
            if metrics.get("actual_reasoning_tokens"):
                crystal_meta += f", {metrics['actual_reasoning_tokens']} reasoning"
            if prompt_type in ("pure_math", "math_answerable", "kg_answerable"):
                crystal_meta += "\n**LLM called:** No (direct return)"

        tool_results = final.get("tool_results", [])
        kg_results = [r for r in tool_results if r.get("tool") == "kg" and r.get("success")]
        if kg_results:
            facts = []
            for r in kg_results:
                for triplet in r.get("results", []):
                    s, p, o = triplet.get("subject", ""), triplet.get("predicate", ""), triplet.get("object", "")
                    facts.append(f"- **{s.title()}** — {p}: {o}")
            if facts:
                grounding_info = "### Grounding Facts\n" + "\n".join(facts[:20])
                entity_spans = final.get("tool_detections", [])
                for det in entity_spans:
                    if det.get("tool") == "kg":
                        tier = det.get("match_tier", "?")
                        lookup = det.get("lookup_type", "?")
                        grounding_info += f"\n\n*Match: {tier} | Lookup: {lookup}*"
                        break
        elif prompt_type in ("no_math",):
            grounding_info = "*No KG facts found for this query — answer is ungrounded.*"

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

    return crystal_response, crystal_meta, llm_response, llm_meta, grounding_info


# ── Document ingestion helpers ────────────────────────────────────────────

_INGEST_AUTO_HEADERS = ["Subject", "Predicate", "Object", "Source", "Confidence", "Sentence"]
_INGEST_PENDING_HEADERS = ["#", "Subject", "Predicate", "Object", "Source", "Confidence", "Sentence"]


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
        f"- Auto-accepted: **{s.get('auto_accepted', 0)}** (already in KG)\n"
        f"- Pending review: **{s.get('pending_review', 0)}**\n"
        f"- Rejected (low confidence): **{s.get('rejected', 0)}**\n"
        f"- NER extractions: {s.get('ner_triplets', 0)} | LLM extractions: {s.get('llm_triplets', 0)}"
    )


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
    """Accept or reject specific pending items based on user edits (not implemented as row-level yet)."""
    if ingest_result is None or not isinstance(ingest_result, DocumentIngestionResult):
        return "No ingestion result.", pd.DataFrame(columns=_INGEST_PENDING_HEADERS), _format_kg_stats(_kg_info(kg_state, "current"))

    n = ingest_result.accept_all_pending()
    info = _kg_info(kg_state, "current")
    return (
        f"Accepted {n} remaining pending triplets into KG.",
        _scored_to_pending_df(ingest_result.pending_review),
        _format_kg_stats(info),
    )


# ── Proposed answers helpers ──────────────────────────────────────────────

_PROPOSED_HEADERS = ["Question", "Crystal Answer", "Route", "Confidence", "Expected", "Golden Answer"]


def run_proposed_answers(ingest_result, kg_state):
    """Generate questions from extracted facts and show Crystal's proposed answers.

    The "Golden Answer" column is pre-filled with Crystal's answer for the user
    to verify or correct. This is the ground truth the user provides.
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
            for st in all_scored:
                if st.subject.lower() in question.lower():
                    src_triplet = [st.subject, st.predicate, st.object]
                    src_sentence = st.source_sentence
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


# ── Before/after comparison helpers ───────────────────────────────────────

_COMPARE_HEADERS = ["Question", "Crystal (KG-grounded)", "Route", "LLM + Docs", "Naked LLM"]


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


_GOLDEN_COMPARE_HEADERS = [
    "Question", "Golden Answer", "Crystal", "C?", "LLM+Docs", "D?", "Naked LLM", "N?",
]


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


# ── KG Explorer helpers ───────────────────────────────────────────────────

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
        lines.append(f"- **{f['predicate']}:** {f['object']}")

    return "\n".join(lines)


def _get_kg_predicate_summary(kg_state) -> pd.DataFrame:
    """Get a summary of predicates and their counts in the active KG."""
    if not hasattr(kg_state, 'triplets'):
        return pd.DataFrame(columns=["Predicate", "Count", "Sample Value"])

    from collections import Counter
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
    """Get a paginated list of entities with fact counts."""
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


# ── Review helpers ────────────────────────────────────────────────────────

def _get_batch_choices() -> list[str]:
    """Get dropdown choices for batch selector."""
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
    """Extract batch ID from dropdown label."""
    if not choice or choice == "(no batches)":
        return None
    return choice.split(" — ")[0].strip()


_OVERVIEW_HEADERS = ["#", "Question", "Status", "Route"]


def _load_overview_table(batch_id: str | None) -> pd.DataFrame:
    """Load questions overview as a compact DataFrame."""
    if not batch_id:
        return pd.DataFrame(columns=_OVERVIEW_HEADERS)
    questions = load_batch_questions(batch_id)
    rows = []
    for i, q in enumerate(questions):
        status = q.get("status", "pending_review")
        icon = {"accepted": "accepted", "rejected": "rejected"}.get(status, "PENDING")
        rows.append([
            str(i + 1),
            q.get("question", "")[:80],
            icon,
            q.get("crystal_route", ""),
        ])
    return pd.DataFrame(rows, columns=_OVERVIEW_HEADERS)


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
                f"**Timestamp:** {b['timestamp']}  \n"
                f"**Questions:** {b['total_cases']} total — "
                f"**{b['pending']} pending**, {b['accepted']} accepted, {b['rejected']} rejected"
            )
    return ""


def _get_question_choices(batch_id: str | None) -> list[str]:
    """Get dropdown choices for question selector."""
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
    """Load all display fields for a single question.

    Returns: (question_md, crystal_proposed, route_info, source_triplet_md,
              golden_answer, status_label)
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

    question_md = (
        f"## Question {question_idx + 1} of {len(questions)}  \n"
        f"### {q.get('question', '')}\n\n"
        f"**Status:** {status_badge}"
    )

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


def _on_batch_selected(choice: str):
    """When a batch is selected, load its metadata, questions, docs, and first question."""
    batch_id = _extract_batch_id(choice)

    meta = _load_batch_metadata(choice)
    q_choices = _get_question_choices(batch_id)
    first_q = q_choices[0] if q_choices and q_choices[0] != "(no questions)" else None

    # Find first pending question
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

    overview = _load_overview_table(batch_id)

    return (
        meta,
        gr.update(choices=q_choices, value=first_q),
        first_pending_idx,
        *q_detail,
        gr.update(choices=doc_choices, value=doc_choices[0] if doc_choices else None),
        "",
        overview,
    )


def _on_question_selected(choice: str, batch_choice: str):
    """When a question is selected from the dropdown, load its details."""
    batch_id = _extract_batch_id(batch_choice)
    if not choice or choice == "(no questions)":
        return (0, *_load_question_detail(None, 0))

    try:
        idx = int(choice.split(".")[0]) - 1
    except (ValueError, IndexError):
        idx = 0

    detail = _load_question_detail(batch_id, idx)
    return (idx, *detail)


def _navigate_question(current_idx: int, direction: int, batch_choice: str):
    """Navigate to previous/next question."""
    batch_id = _extract_batch_id(batch_choice)
    if not batch_id:
        return (0, "(no questions)", *_load_question_detail(None, 0))

    questions = load_batch_questions(batch_id)
    if not questions:
        return (0, "(no questions)", *_load_question_detail(None, 0))

    new_idx = max(0, min(len(questions) - 1, current_idx + direction))
    q_choices = _get_question_choices(batch_id)
    detail = _load_question_detail(batch_id, new_idx)

    return (new_idx, q_choices[new_idx] if new_idx < len(q_choices) else q_choices[0], *detail)


def _accept_question(batch_choice: str, current_idx: int, golden_answer: str):
    """Accept the current question with the given golden answer."""
    batch_id = _extract_batch_id(batch_choice)
    if not batch_id:
        return ("No batch selected.",) + _after_decision_outputs(None, 0, batch_choice)

    ok = save_single_review_decision(batch_id, current_idx, golden_answer, "accepted")
    if not ok:
        return (f"Failed to save decision for question {current_idx + 1}.",) + \
            _after_decision_outputs(batch_id, current_idx, batch_choice)

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

    return (msg,) + _after_decision_outputs(batch_id, next_idx, batch_choice)


def _reject_question(batch_choice: str, current_idx: int, golden_answer: str):
    """Reject the current question."""
    batch_id = _extract_batch_id(batch_choice)
    if not batch_id:
        return ("No batch selected.",) + _after_decision_outputs(None, 0, batch_choice)

    ok = save_single_review_decision(batch_id, current_idx, golden_answer, "rejected")
    if not ok:
        return (f"Failed to save decision for question {current_idx + 1}.",) + \
            _after_decision_outputs(batch_id, current_idx, batch_choice)

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

    return (msg,) + _after_decision_outputs(batch_id, next_idx, batch_choice)


def _after_decision_outputs(batch_id, next_idx, batch_choice):
    """Compute updated UI state after an accept/reject decision."""
    q_choices = _get_question_choices(batch_id)
    detail = _load_question_detail(batch_id, next_idx)
    overview = _load_overview_table(batch_id)
    meta = _load_batch_metadata(batch_choice)
    selector_value = q_choices[next_idx] if next_idx < len(q_choices) else (
        q_choices[0] if q_choices else "(no questions)")
    return (
        next_idx,
        gr.update(choices=q_choices, value=selector_value),
        *detail,
        overview,
        meta,
    )


def _load_doc_text(doc_choice: str):
    """Load document text when the user selects a document."""
    if not doc_choice or doc_choice == "(no source documents found)":
        return "Select a source document to view the original opinion text."
    slug = doc_choice.split("(")[-1].rstrip(")").strip() if "(" in doc_choice else doc_choice
    return load_source_document_text(slug)


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
    """Refresh the review dashboard."""
    return format_review_dashboard()


# ── Benchmark & RW loop helpers ──────────────────────────────────────────

_BENCH_HEADERS = ["Question", "Golden Answer", "Crystal Answer", "Route", "Correct"]


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


# ── Gradio layout ────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Crystal — Neuro-symbolic Prompt Compiler",
    ) as demo:
        gr.Markdown("# Crystal\n*Neuro-symbolic prompt compiler for LLMs — grounded answers, fewer hallucinations.*")

        kg_state = gr.State(_default_kg)

        with gr.Tab("Ask"):
            gr.Markdown("Ask a question and compare Crystal's grounded answer against the naked LLM.")

            _placeholder = (
                "e.g. What court decided Miranda v. Arizona?"
                if _legal_kg is not None
                else "e.g. What is the capital of Remulak?"
            )
            with gr.Row():
                question_input = gr.Textbox(
                    label="Question",
                    placeholder=_placeholder,
                    scale=4,
                )
                ask_btn = gr.Button("Ask", variant="primary", scale=1)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Crystal (Grounded)")
                    crystal_output = gr.Textbox(label="Response", lines=6, interactive=False)
                    crystal_meta_output = gr.Markdown("")
                with gr.Column():
                    gr.Markdown("### Naked LLM (Baseline)")
                    llm_output = gr.Textbox(label="Response", lines=6, interactive=False)
                    llm_meta_output = gr.Markdown("")

            grounding_output = gr.Markdown("")

            ask_btn.click(
                fn=ask_question,
                inputs=[question_input, kg_state],
                outputs=[crystal_output, crystal_meta_output, llm_output, llm_meta_output, grounding_output],
            )
            question_input.submit(
                fn=ask_question,
                inputs=[question_input, kg_state],
                outputs=[crystal_output, crystal_meta_output, llm_output, llm_meta_output, grounding_output],
            )

        with gr.Tab("Ingest Documents"):
            gr.Markdown(
                "Upload legal documents to extract facts into the knowledge graph. "
                "High-confidence extractions are auto-accepted; others go to a review queue."
            )

            ingest_result_state = gr.State(None)

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

            gr.Markdown("### Auto-Accepted (already in KG)")
            ingest_auto_table = gr.Dataframe(
                value=pd.DataFrame(columns=_INGEST_AUTO_HEADERS),
                interactive=False,
            )

            gr.Markdown("### Pending Review")
            ingest_pending_table = gr.Dataframe(
                value=pd.DataFrame(columns=_INGEST_PENDING_HEADERS),
                interactive=False,
            )

            with gr.Row():
                accept_all_btn = gr.Button("Accept All Pending", variant="primary")
                reject_all_btn = gr.Button("Reject All Pending", variant="stop")
                save_pending_btn = gr.Button("Save & Accept Pending")

            ingest_decision_status = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("### Crystal's Proposed Answers → Ground Truth")
            gr.Markdown(
                "After ingestion, Crystal generates questions from extracted facts and proposes answers. "
                "**Your job:** review the **Golden Answer** column. If Crystal got it wrong, edit the cell "
                "with the correct answer. Then click **Save to Review** to create verified ground truth."
            )
            proposed_btn = gr.Button("Generate & Answer", variant="secondary")
            proposed_status = gr.Markdown("")
            proposed_table = gr.Dataframe(
                value=pd.DataFrame(columns=_PROPOSED_HEADERS),
                interactive=True,
            )
            with gr.Row():
                save_proposed_btn = gr.Button("Save to Review", variant="primary")
            proposed_save_status = gr.Markdown("")

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

            save_pending_btn.click(
                fn=save_pending_decisions,
                inputs=[ingest_result_state, ingest_pending_table, kg_state],
                outputs=[ingest_decision_status, ingest_pending_table, ingest_kg_stats],
            )

            gr.Markdown("---")
            gr.Markdown("### Test Your Ingestion")
            gr.Markdown(
                "Compare Crystal (with ingested KG) vs naked LLM. "
                "Enter questions below, or leave blank to auto-generate from extracted facts."
            )
            compare_questions = gr.Textbox(
                label="Questions (one per line)",
                lines=4,
                placeholder="What court decided Miranda v. Arizona?\nWho wrote the opinion in Roe v. Wade?",
            )
            compare_btn = gr.Button("Run Comparison", variant="primary")
            compare_status = gr.Markdown("")
            compare_table = gr.Dataframe(
                value=pd.DataFrame(columns=_COMPARE_HEADERS),
                interactive=False,
            )

            compare_btn.click(
                fn=run_comparison,
                inputs=[compare_questions, ingest_result_state, kg_state],
                outputs=[compare_status, compare_table],
            )

            gr.Markdown("---")
            gr.Markdown("### Three-Arm Comparison on Golden Answers")
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
            )

            golden_compare_btn.click(
                fn=run_golden_comparison,
                inputs=[kg_state],
                outputs=[golden_compare_status, golden_compare_scores, golden_compare_table],
            )

        with gr.Tab("Knowledge Graph"):
            gr.Markdown("Manage the active knowledge graph. Switch between datasets, or upload custom data.")

            with gr.Row():
                kg_mode_selector = gr.Dropdown(
                    choices=list(KG_MODES.keys()),
                    value=_DEFAULT_KG_MODE,
                    label="Active Knowledge Graph",
                    interactive=True,
                    scale=3,
                )

            kg_stats = gr.Markdown(_format_kg_stats(_default_kg_info()))
            status_msg = gr.Textbox(label="Status", interactive=False, value=f"{_DEFAULT_KG_MODE} loaded.")

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
            kg_facts = gr.Markdown(_format_kg_facts(_default_kg))

            kg_mode_selector.change(
                fn=switch_kg_mode,
                inputs=[kg_mode_selector],
                outputs=[kg_state, status_msg, kg_stats, kg_facts],
            )
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

        with gr.Tab("KG Explorer"):
            gr.Markdown("Browse and search the active knowledge graph.")

            with gr.Row():
                entity_search = gr.Textbox(
                    label="Search Entity",
                    placeholder="e.g. Miranda v. Arizona, Roe v. Wade",
                    scale=4,
                )
                search_btn = gr.Button("Search", variant="primary", scale=1)

            entity_results = gr.Markdown("Type a case name or entity to search.")

            gr.Markdown("---")
            gr.Markdown("### Predicate Summary")
            predicate_summary = gr.Dataframe(
                value=_get_kg_predicate_summary(_default_kg),
                interactive=False,
            )

            gr.Markdown("### Entities (first 50)")
            entity_list = gr.Dataframe(
                value=_get_kg_entity_list(_default_kg),
                interactive=False,
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

        with gr.Tab("Review"):
            gr.Markdown(
                "## Review Queue\n"
                "Review pending questions one at a time. Read the source material, "
                "verify or correct Crystal's proposed answer, then accept or reject."
            )

            review_dashboard = gr.Markdown(format_review_dashboard())
            refresh_btn = gr.Button("Refresh Dashboard", variant="secondary", size="sm")

            gr.Markdown("---")

            # ── Batch selection ──
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

            # ── State ──
            review_q_idx = gr.State(0)

            gr.Markdown("---")

            # ── Per-question review ──
            with gr.Row():
                # Left panel: question detail
                with gr.Column(scale=3):
                    gr.Markdown("### Question Review")
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

                    gr.Markdown("---")
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

                # Right panel: source document viewer
                with gr.Column(scale=2):
                    gr.Markdown("### Source Material")
                    gr.Markdown(
                        "*Select a document to read the original opinion text. "
                        "Use this to verify Crystal's answer.*"
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

            # ── Questions overview table ──
            gr.Markdown("---")
            gr.Markdown("### All Questions in Batch")
            overview_table = gr.Dataframe(
                value=_load_overview_table(_extract_batch_id(initial_choice)),
                interactive=False,
            )

            # ── Known gaps ──
            with gr.Accordion("Detector Known Gaps", open=False):
                gaps_table = gr.Dataframe(
                    value=_get_known_gaps_df(),
                    interactive=False,
                )

            # ── Benchmark & RW loop ──
            gr.Markdown("---")
            gr.Markdown("### Benchmark & Improvement")
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
            )
            rw_status = gr.Markdown("")
            rw_report = gr.Markdown("")

            # ── Load initial question if batch exists ──
            _init_batch_id = _extract_batch_id(initial_choice)
            if _init_batch_id:
                _init_detail = _load_question_detail(_init_batch_id, 0)
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
                inputs=[batch_selector],
                outputs=[
                    batch_meta,
                    question_selector, review_q_idx,
                    question_text_md, crystal_proposed_box, route_info_md,
                    source_triplet_md, golden_answer_box, current_status_label,
                    doc_selector, doc_text_box,
                    overview_table,
                ],
            )

            question_selector.change(
                fn=_on_question_selected,
                inputs=[question_selector, batch_selector],
                outputs=[
                    review_q_idx,
                    question_text_md, crystal_proposed_box, route_info_md,
                    source_triplet_md, golden_answer_box, current_status_label,
                ],
            )

            prev_q_btn.click(
                fn=lambda idx, bc: _navigate_question(idx, -1, bc),
                inputs=[review_q_idx, batch_selector],
                outputs=[
                    review_q_idx, question_selector,
                    question_text_md, crystal_proposed_box, route_info_md,
                    source_triplet_md, golden_answer_box, current_status_label,
                ],
            )

            next_q_btn.click(
                fn=lambda idx, bc: _navigate_question(idx, +1, bc),
                inputs=[review_q_idx, batch_selector],
                outputs=[
                    review_q_idx, question_selector,
                    question_text_md, crystal_proposed_box, route_info_md,
                    source_triplet_md, golden_answer_box, current_status_label,
                ],
            )

            accept_btn.click(
                fn=_accept_question,
                inputs=[batch_selector, review_q_idx, golden_answer_box],
                outputs=[
                    review_action_status,
                    review_q_idx, question_selector,
                    question_text_md, crystal_proposed_box, route_info_md,
                    source_triplet_md, golden_answer_box, current_status_label,
                    overview_table, batch_meta,
                ],
            )

            reject_btn.click(
                fn=_reject_question,
                inputs=[batch_selector, review_q_idx, golden_answer_box],
                outputs=[
                    review_action_status,
                    review_q_idx, question_selector,
                    question_text_md, crystal_proposed_box, route_info_md,
                    source_triplet_md, golden_answer_box, current_status_label,
                    overview_table, batch_meta,
                ],
            )

            doc_selector.change(
                fn=_load_doc_text,
                inputs=[doc_selector],
                outputs=[doc_text_box],
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

            refresh_btn.click(
                fn=_refresh_review,
                inputs=[],
                outputs=[review_dashboard],
            )

    return demo


def main():
    demo = build_ui()
    demo.launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
