"""
Document → KG → Questions → Crystal Proposes → Review Batch pipeline.

Ingests a document, generates questions from extracted facts, has Crystal
propose answers, and saves everything as a review batch for human verification.

Usage:
    LLM_PROVIDER=anthropic LLM_MODEL=claude-haiku-4-5 \
    python -m crystal.ingest.review_pipeline benchmarks/documents/miranda-v-arizona.json

    # Multiple documents:
    python -m crystal.ingest.review_pipeline benchmarks/documents/miranda-v-arizona.json \
        benchmarks/documents/brown-v-board-of-education.json

    # NER only (no LLM extraction):
    python -m crystal.ingest.review_pipeline --ner-only benchmarks/documents/miranda-v-arizona.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from crystal.compare import generate_questions_from_triplets, generate_questions_llm
from crystal.graph import build_crystal_graph
from crystal.ingest import ingest_document, DocumentIngestionResult
from crystal.ingest.confidence import ScoredTriplet
from crystal.review import save_proposed_as_batch
from crystal.state import make_initial_state
from crystal.tools.kg.store import SqliteKnowledgeGraph

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent.parent
_LEGAL_DB = _ROOT / "data" / "legal.sqlite"


def run_review_pipeline(
    document_paths: list[Path],
    *,
    ner_only: bool = False,
    max_questions: int = 20,
    auto_accept_threshold: float = 0.50,
    db_path: Path | None = None,
) -> dict:
    """Full pipeline: ingest → extract → generate questions → Crystal proposes → save batch.

    Returns a summary dict with stats and the path to the saved review batch.
    """
    from crystal.llm import call_llm
    call_llm_fn = None if ner_only else call_llm

    db_path = db_path or _LEGAL_DB
    if db_path.exists():
        kg = SqliteKnowledgeGraph(db_path)
    else:
        kg = SqliteKnowledgeGraph()

    graph = build_crystal_graph()

    all_auto: list[ScoredTriplet] = []
    all_pending: list[ScoredTriplet] = []
    doc_sources: list[str] = []

    for doc_path in document_paths:
        text = _load_document_text(doc_path)
        if not text:
            print(f"  SKIP: {doc_path} (empty or unreadable)")
            continue

        print(f"  Ingesting: {doc_path.name} ({len(text):,} chars)")
        result = ingest_document(
            text, kg=kg,
            call_llm_fn=call_llm_fn,
            auto_accept_threshold=auto_accept_threshold,
            domain="legal",
        )
        result.accept_all_pending()
        for st in result.auto_accepted:
            if not st.source_document:
                st.source_document = doc_path.stem
        all_auto.extend(result.auto_accepted)
        all_pending.extend(result.pending_review)
        doc_sources.append(doc_path.stem)

        ner_count = result.stats.get("ner_triplets", 0)
        llm_count = result.stats.get("llm_triplets", 0)
        print(f"    → {len(result.auto_accepted)} accepted ({ner_count} NER, {llm_count} LLM)")

    if not all_auto:
        print("  No facts extracted. Nothing to review.")
        return {"error": "no_facts", "documents": len(document_paths)}

    triplets = [st.as_tuple() for st in all_auto]

    triplet_to_st: dict[tuple[str, str, str], ScoredTriplet] = {}
    for st in all_auto:
        triplet_to_st[(st.subject.lower(), st.predicate.lower(), st.object.lower())] = st

    def _origin_for_triplet(src_triplet: list) -> tuple[str, str]:
        """Return (origin, source_document) for a source triplet."""
        if src_triplet and len(src_triplet) >= 3:
            key = (str(src_triplet[0]).lower(), str(src_triplet[1]).lower(),
                   str(src_triplet[2]).lower())
            st = triplet_to_st.get(key)
            if st:
                return st.origin, st.source_document
        return "unknown", ""

    proposed_rows = []
    if call_llm_fn is not None:
        print(f"\n  Using LLM question generation on {len(triplets)} facts...")
        llm_questions = generate_questions_llm(
            triplets, call_llm_fn, max_questions=max_questions,
        )
        print(f"  LLM generated {len(llm_questions)} questions")
        for i, qd in enumerate(llm_questions):
            q_text = qd["question"]
            print(f"  [{i+1}/{len(llm_questions)}] {q_text[:70]}")
            try:
                state = make_initial_state(q_text, kg=kg)
                final = graph.invoke(state)
                answer = final.get("final_response", "")
                prompt_type = final.get("prompt_type", "unknown")
            except Exception as e:
                answer = f"[Error: {e}]"
                prompt_type = "error"

            src_triplet = qd.get("source_triplet", [])
            q_origin, q_source_doc = _origin_for_triplet(src_triplet)

            proposed_rows.append({
                "question": q_text,
                "crystal_answer": answer[:500],
                "route": prompt_type,
                "confidence": prompt_type,
                "golden_answer": qd.get("golden_answer", answer[:500]),
                "source_triplet": src_triplet,
                "origin": q_origin,
                "source_document": q_source_doc,
            })
            time.sleep(1)

    if not proposed_rows:
        questions = generate_questions_from_triplets(triplets, max_questions=max_questions)
        print(f"\n  Template fallback: {len(questions)} questions from {len(triplets)} facts")
        for i, q_text in enumerate(questions):
            print(f"  [{i+1}/{len(questions)}] {q_text[:70]}")
            try:
                state = make_initial_state(q_text, kg=kg)
                final = graph.invoke(state)
                answer = final.get("final_response", "")
                prompt_type = final.get("prompt_type", "unknown")
            except Exception as e:
                answer = f"[Error: {e}]"
                prompt_type = "error"

            src_triplet = []
            for s, p, o in triplets:
                if s.lower() in q_text.lower():
                    src_triplet = [s, p, o]
                    break

            q_origin, q_source_doc = _origin_for_triplet(src_triplet)

            proposed_rows.append({
                "question": q_text,
                "crystal_answer": answer[:500],
                "route": prompt_type,
                "confidence": prompt_type,
                "golden_answer": answer[:500],
                "source_triplet": src_triplet,
                "origin": q_origin,
                "source_document": q_source_doc,
            })
            time.sleep(1)

    print(f"\n  Total questions generated: {len(proposed_rows)}")

    source_label = ", ".join(doc_sources[:3])
    if len(doc_sources) > 3:
        source_label += f" (+{len(doc_sources) - 3} more)"

    batch_path = save_proposed_as_batch(proposed_rows, source=source_label)

    print(f"\n  ✓ Saved {len(proposed_rows)} questions to: {batch_path}")
    print(f"    Edit golden_answer in the JSON, then accept/reject in the Review tab.")
    print(f"    Or: python -m crystal.ui → Review tab → select the new batch")

    return {
        "documents": len(document_paths),
        "facts_extracted": len(triplets),
        "questions_generated": len(proposed_rows),
        "batch_file": str(batch_path) if batch_path else None,
    }


def _load_document_text(path: Path) -> str:
    """Load document text from a cached opinion JSON or plain text file."""
    path = Path(path)
    if not path.exists():
        return ""

    text = path.read_text(encoding="utf-8")

    if path.suffix == ".json":
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                for key in ("plain_text", "text", "opinion_text", "content"):
                    if key in data and data[key]:
                        return str(data[key])
                if "opinions" in data and isinstance(data["opinions"], list):
                    parts = []
                    for op in data["opinions"]:
                        if isinstance(op, dict) and op.get("plain_text"):
                            parts.append(op["plain_text"])
                    if parts:
                        return "\n\n".join(parts)
            return text
        except json.JSONDecodeError:
            return text

    return text


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest documents → generate questions → Crystal proposes → review batch",
    )
    parser.add_argument("documents", nargs="+", help="Paths to document files")
    parser.add_argument("--ner-only", action="store_true", help="Skip LLM extraction")
    parser.add_argument("--max-questions", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Auto-accept confidence threshold")
    parser.add_argument("--db-path", default=None, help="SQLite KG path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    print(f"\n{'=' * 60}")
    print(f"  DOCUMENT → REVIEW PIPELINE")
    print(f"  Mode: {'NER-only' if args.ner_only else 'NER + LLM'}")
    print(f"  Documents: {len(args.documents)}")
    print(f"{'=' * 60}\n")

    result = run_review_pipeline(
        [Path(p) for p in args.documents],
        ner_only=args.ner_only,
        max_questions=args.max_questions,
        auto_accept_threshold=args.threshold,
        db_path=Path(args.db_path) if args.db_path else None,
    )

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"  Documents processed: {result.get('documents', 0)}")
    print(f"  Facts extracted:     {result.get('facts_extracted', 0)}")
    print(f"  Questions generated: {result.get('questions_generated', 0)}")
    if result.get("batch_file"):
        print(f"  Review batch:        {result['batch_file']}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
