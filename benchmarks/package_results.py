#!/usr/bin/env python3
"""Package comprehensive benchmark results for the MVP demo.

Runs the three-arm comparison on multiple corpora and produces a unified
report showing Crystal's advantage across all test sets.

Usage:
    python -m benchmarks.package_results
    python -m benchmarks.package_results --output results/demo_report.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.three_arm_comparison import run_three_arm, ComparisonReport

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSuite:
    name: str
    report: ComparisonReport


def run_all_benchmarks(kg=None) -> list[BenchmarkSuite]:
    """Run three-arm comparison on all available corpora."""
    suites = []

    from benchmarks.ground_truth.opinion_golden import OPINION_GOLDEN_CASES
    if OPINION_GOLDEN_CASES:
        logger.info("Running opinion golden benchmark (%d cases)...", len(OPINION_GOLDEN_CASES))
        report = run_three_arm(cases=OPINION_GOLDEN_CASES, kg=kg)
        suites.append(BenchmarkSuite("Adversarial Opinion Golden", report))

    from benchmarks.ground_truth.opinion_holdout import OPINION_HOLDOUT_CASES
    if OPINION_HOLDOUT_CASES:
        logger.info("Running opinion holdout benchmark (%d cases)...", len(OPINION_HOLDOUT_CASES))
        report = run_three_arm(cases=OPINION_HOLDOUT_CASES, kg=kg)
        suites.append(BenchmarkSuite("Holdout Validation", report))

    from crystal.review import collect_accepted_cases
    accepted = collect_accepted_cases()
    if accepted:
        logger.info("Running accepted review cases benchmark (%d cases)...", len(accepted))
        report = run_three_arm(cases=accepted, kg=kg)
        suites.append(BenchmarkSuite("Accepted Review Cases", report))

    return suites


def format_demo_report(suites: list[BenchmarkSuite]) -> str:
    """Format all benchmark results into a single demo report."""
    lines = [
        "# Crystal MVP Demo — Benchmark Results\n",
        "## Executive Summary\n",
        "| Corpus | Questions | Crystal | LLM+Docs | Naked LLM | Crystal Advantage |",
        "|--------|-----------|---------|----------|-----------|-------------------|",
    ]

    for suite in suites:
        r = suite.report
        n = len(r.results)
        if n == 0:
            continue
        advantage = r.crystal_accuracy - r.llm_naked_accuracy
        lines.append(
            f"| {suite.name} | {n} | **{r.crystal_accuracy:.0%}** | "
            f"{r.llm_docs_accuracy:.0%} | {r.llm_naked_accuracy:.0%} | "
            f"+{advantage:.0%} |"
        )

    lines.append("\n---\n")

    for suite in suites:
        if suite.report.results:
            lines.append(f"## {suite.name}\n")
            lines.append(suite.report.to_markdown())
            lines.append("\n---\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Package comprehensive benchmark results")
    parser.add_argument("--output", "-o", type=str, help="Save report to file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    suites = run_all_benchmarks()

    if not suites:
        print("No benchmark corpora found. Run opinion_golden.py or accept review cases first.")
        return

    md = format_demo_report(suites)
    print(md)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md, encoding="utf-8")
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
