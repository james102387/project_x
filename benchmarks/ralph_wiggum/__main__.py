"""CLI entry point: python -m benchmarks.ralph_wiggum"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

from benchmarks.ralph_wiggum.base import call_ralph_llm, RALPH_MODEL, git_cmd
from benchmarks.ralph_wiggum.orchestrator import Orchestrator
from benchmarks.ralph_wiggum.predicate_loop import PredicateLoop
from benchmarks.ralph_wiggum.entity_loop import EntityLoop
from benchmarks.ralph_wiggum.threshold_loop import ThresholdLoop


def main():
    parser = argparse.ArgumentParser(
        description="Ralph Wiggum v3 — multi-loop self-improvement engine",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.90,
        help="Fitness score threshold for convergence (default: 0.90)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10,
        help="Max iterations per loop (default: 10)",
    )
    parser.add_argument(
        "--review-dir", default=None,
        help="Path to review directory with accepted cases",
    )
    parser.add_argument(
        "--db-path", default=None,
        help="Path to SQLite KG database",
    )
    parser.add_argument(
        "--use-git", action="store_true",
        help="Use git commits for each mutation",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Evaluation only, no LLM proposals",
    )
    parser.add_argument(
        "--loop", choices=["all", "predicate", "entity", "threshold"],
        default="all",
        help="Which loop to run (default: all)",
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="Use v1 detect_kg_query-only evaluation (not full pipeline)",
    )
    parser.add_argument(
        "--report-path", default=None,
        help="Path to write change report markdown",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    from crystal.review import collect_accepted_cases, REVIEW_DIR
    from crystal.tools.kg.store import SqliteKnowledgeGraph

    review_dir = Path(args.review_dir) if args.review_dir else REVIEW_DIR
    cases = collect_accepted_cases(review_dir)

    if not cases:
        from benchmarks.ground_truth.legal import LEGAL_BENCHMARK_CASES
        logger.warning(
            "No accepted cases found in %s. Falling back to LEGAL_BENCHMARK_CASES (%d cases).",
            review_dir, len(LEGAL_BENCHMARK_CASES),
        )
        cases = LEGAL_BENCHMARK_CASES

    logger.info("Loaded %d test cases", len(cases))

    if args.db_path:
        kg = SqliteKnowledgeGraph(args.db_path)
    else:
        from crystal.tools.kg.legal import build_legal_kg_memory
        from tests.fixtures.scotus_sample import SCOTUS_SAMPLE
        logger.info("No --db-path provided, building in-memory KG from SCOTUS sample fixtures")
        kg = build_legal_kg_memory(SCOTUS_SAMPLE)

    call_llm_fn = None
    if not args.dry_run:
        try:
            call_ralph_llm("ping")
            call_llm_fn = call_ralph_llm
            logger.info(
                "Anthropic API available (model: %s) — will propose code changes",
                RALPH_MODEL,
            )
        except Exception as e:
            logger.warning("Anthropic API not available (%s) — running in evaluation-only mode", e)

    common_kwargs = dict(
        kg=kg,
        cases=cases,
        call_llm_fn=call_llm_fn,
        use_git=args.use_git,
        use_full_pipeline=not args.legacy,
    )

    if args.loop == "all":
        orchestrator = Orchestrator(**common_kwargs)
        orch_result = orchestrator.run(
            threshold=args.threshold,
            max_iterations_per_loop=args.max_iterations,
        )
        report = orch_result.unified_report
        print(f"\n--- Ralph Wiggum v3 Orchestrator Summary ---")
        print(f"Overall score: {orch_result.overall_score:.1%}")
        for name, lr in orch_result.loop_results.items():
            status = "converged" if lr.converged else "did not converge"
            print(f"  {name}: {lr.final_score:.1%} ({lr.iterations_run} iter, {status})")
    else:
        loop_map = {
            "predicate": PredicateLoop,
            "entity": EntityLoop,
            "threshold": ThresholdLoop,
        }
        loop_cls = loop_map[args.loop]
        loop = loop_cls(**common_kwargs)
        result = loop.run(
            threshold=args.threshold,
            max_iterations=args.max_iterations,
        )
        report = result.change_report
        print(f"\n--- {loop.LOOP_NAME} Summary ---")
        print(f"Converged: {result.converged}")
        print(f"Final score: {result.final_score:.1%}")
        print(f"Best score: {result.best_score:.1%} (iteration {result.best_iteration})")
        print(f"Iterations: {result.iterations_run}")

    report_path = Path(args.report_path) if args.report_path else Path(__file__).parent.parent / "ralph_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Change report: {report_path}")

    if hasattr(kg, "close"):
        kg.close()


if __name__ == "__main__":
    main()
