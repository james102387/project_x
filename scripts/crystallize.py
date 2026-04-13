#!/usr/bin/env python3
"""Crystallization cycle CLI — orchestrates the full ingest-to-improvement pipeline.

Usage:
    # Ingest documents, generate questions, save review batch:
    python scripts/crystallize.py \\
        --documents benchmarks/documents/miranda-v-arizona.json \\
        --max-questions 15 \\
        --auto-accept-threshold 0.50

    # After reviewing golden answers in the UI, run benchmark + RW:
    python scripts/crystallize.py --benchmark

    # Run purification audit on the KG:
    python scripts/crystallize.py --purify
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / "data" / "legal.sqlite"


def _load_kg():
    from crystal.tools.kg.legal import load_legal_kg
    kg = load_legal_kg(DB_PATH)
    if kg is None:
        logger.error("No KG found at %s", DB_PATH)
        sys.exit(1)
    return kg


def cmd_ingest(args):
    """Ingest documents → extract → auto-accept → generate questions → save batch."""
    from crystal.ingest import ingest_document
    from crystal.compare import generate_questions_from_triplets
    from crystal.review import save_proposed_as_batch
    from crystal.graph import build_crystal_graph
    from crystal.state import make_initial_state

    kg = _load_kg()
    graph = build_crystal_graph()

    all_auto = []
    all_pending = []
    sources = []

    for doc_path_str in args.documents:
        doc_path = Path(doc_path_str)
        if not doc_path.exists():
            logger.warning("Document not found: %s — skipping", doc_path)
            continue

        logger.info("Ingesting %s ...", doc_path.name)
        result = ingest_document(
            doc_path,
            kg=kg,
            auto_accept_threshold=args.auto_accept_threshold,
        )
        logger.info(
            "  → %d auto-accepted, %d pending, %d rejected",
            len(result.auto_accepted), len(result.pending_review), len(result.rejected),
        )
        all_auto.extend(result.auto_accepted)
        all_pending.extend(result.pending_review)
        sources.append(doc_path.stem)

    if not all_auto and not all_pending:
        logger.error("No facts extracted from any document.")
        return

    if args.accept_all_pending and all_pending:
        logger.info("Auto-accepting all %d pending triplets (--accept-all)", len(all_pending))
        pending_tuples = [st.as_tuple_with_sentence() for st in all_pending]
        source_label = ", ".join(sources[:3])
        kg.bulk_insert(pending_tuples, source=source_label)
        all_auto.extend(all_pending)
        all_pending = []

    triplets = [st.as_tuple() for st in all_auto]
    logger.info("Generating questions from %d triplets...", len(triplets))
    questions = generate_questions_from_triplets(triplets, max_questions=args.max_questions)
    logger.info("Generated %d questions", len(questions))

    if not questions:
        logger.warning("No questions generated. Check triplet quality.")
        return

    proposed_rows = []
    for q in questions:
        try:
            state = make_initial_state(q, kg=kg)
            final = graph.invoke(state)
            crystal_answer = final.get("final_response", "")
            route = final.get("prompt_type", "unknown")
        except Exception as e:
            crystal_answer = f"Error: {e}"
            route = "error"

        src_triplet = []
        src_sentence = ""
        for st in all_auto:
            if st.subject.lower() in q.lower():
                src_triplet = [st.subject, st.predicate, st.object]
                src_sentence = st.source_sentence
                break

        proposed_rows.append({
            "question": q,
            "crystal_answer": crystal_answer,
            "route": route,
            "confidence": "",
            "expected": "",
            "golden_answer": crystal_answer,
            "source_triplet": src_triplet,
            "source_sentence": src_sentence,
        })

    source_label = ", ".join(sources[:3])
    if len(sources) > 3:
        source_label += f" (+{len(sources) - 3} more)"
    batch_path = save_proposed_as_batch(proposed_rows, source=source_label)

    if batch_path:
        print(f"\nSaved {len(proposed_rows)} questions to: {batch_path}")
        print("Next steps:")
        print("  1. Review golden answers in the UI: python -m crystal.ui")
        print("  2. Then run: python scripts/crystallize.py --benchmark")
    else:
        logger.error("Failed to save review batch.")


def cmd_benchmark(args):
    """Run benchmark on all accepted golden answers, optionally run RW."""
    from crystal.review import collect_accepted_cases
    from benchmarks.scoring.fitness import binary_correct
    from crystal.graph import build_crystal_graph
    from crystal.state import make_initial_state

    kg = _load_kg()
    cases = collect_accepted_cases()

    if not cases:
        logger.error("No accepted golden answers found. Review some questions first.")
        return

    graph = build_crystal_graph()
    correct_count = 0
    results_detail = []

    logger.info("Running benchmark on %d accepted cases...", len(cases))

    for question, golden_answer, match_strings, is_negative in cases:
        try:
            state = make_initial_state(question, kg=kg)
            final = graph.invoke(state)
            crystal_answer = final.get("final_response", "")
            route = final.get("prompt_type", "unknown")
        except Exception as e:
            crystal_answer = f"Error: {e}"
            route = "error"

        is_correct = binary_correct(crystal_answer, match_strings, is_negative)
        if is_correct:
            correct_count += 1
        results_detail.append({
            "question": question,
            "golden": golden_answer,
            "crystal": crystal_answer[:200],
            "route": route,
            "correct": is_correct,
        })

    score = correct_count / len(cases) if cases else 0.0
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS: {correct_count}/{len(cases)} correct ({score:.0%})")
    print(f"{'='*60}")

    for r in results_detail:
        mark = "✓" if r["correct"] else "✗"
        print(f"  {mark} [{r['route']}] {r['question']}")
        if not r["correct"]:
            print(f"    Golden:  {r['golden'][:80]}")
            print(f"    Crystal: {r['crystal'][:80]}")

    if args.rw and score < 1.0:
        print(f"\nRunning Ralph Wiggum loops to improve from {score:.0%}...")
        _run_rw(kg, cases)


def cmd_purify(args):
    """Run KG purification audit."""
    from crystal.tools.kg.audit import audit_kg

    logger.info("Running KG audit on %s...", DB_PATH)
    report = audit_kg(DB_PATH)
    print(report.to_markdown())

    if report.critical_count > 0:
        print(f"\nWARNING: {report.critical_count} critical issues found!")


def _run_rw(kg, cases):
    """Run Ralph Wiggum Orchestrator."""
    from benchmarks.ralph_wiggum.orchestrator import Orchestrator

    orch = Orchestrator(
        kg=kg,
        cases=cases,
        use_git=False,
        use_full_pipeline=True,
    )
    result = orch.run(threshold=0.90, max_iterations_per_loop=5)
    print(f"\nRalph Wiggum final score: {result.overall_score:.1%}")
    if result.unified_report:
        print(result.unified_report)


def main():
    parser = argparse.ArgumentParser(
        description="Crystal crystallization cycle CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    ingest_p = sub.add_parser("ingest", help="Ingest documents and generate review batch")
    ingest_p.add_argument("--documents", nargs="+", required=True, help="Document paths")
    ingest_p.add_argument("--max-questions", type=int, default=15)
    ingest_p.add_argument("--auto-accept-threshold", type=float, default=0.50)
    ingest_p.add_argument("--accept-all", dest="accept_all_pending", action="store_true",
                          help="Auto-accept all pending triplets (for demo)")

    bench_p = sub.add_parser("benchmark", help="Run benchmark on accepted golden answers")
    bench_p.add_argument("--rw", action="store_true", help="Also run Ralph Wiggum improvement loops")

    sub.add_parser("purify", help="Run KG purification audit")

    # Support --benchmark and --purify as top-level flags for convenience
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark (shortcut)")
    parser.add_argument("--purify", action="store_true", help="Run purification audit (shortcut)")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "purify":
        cmd_purify(args)
    elif args.benchmark:
        args.rw = False
        cmd_benchmark(args)
    elif args.purify:
        cmd_purify(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
