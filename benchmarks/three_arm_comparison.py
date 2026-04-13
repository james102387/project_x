#!/usr/bin/env python3
"""Three-arm comparison benchmark — Crystal+KG vs LLM+docs vs Naked LLM.

Runs accepted golden answers through all three arms and produces a markdown
report with per-question results and aggregate accuracy scores.

Usage:
    python -m benchmarks.three_arm_comparison
    python -m benchmarks.three_arm_comparison --output report.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.scoring.fitness import binary_correct

logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / "data" / "legal.sqlite"


@dataclass
class ArmResult:
    answer: str
    correct: bool
    route: str = ""


@dataclass
class QuestionResult:
    question: str
    golden_answer: str
    match_strings: list[str]
    is_negative: bool
    crystal: ArmResult = field(default_factory=lambda: ArmResult("", False))
    llm_docs: ArmResult = field(default_factory=lambda: ArmResult("", False))
    llm_naked: ArmResult = field(default_factory=lambda: ArmResult("", False))


@dataclass
class ComparisonReport:
    results: list[QuestionResult] = field(default_factory=list)
    crystal_accuracy: float = 0.0
    llm_docs_accuracy: float = 0.0
    llm_naked_accuracy: float = 0.0

    def to_markdown(self) -> str:
        lines = ["# Three-Arm Comparison Report\n"]

        lines.append("## Accuracy Summary\n")
        lines.append("| Arm | Correct | Total | Accuracy |")
        lines.append("|-----|---------|-------|----------|")
        n = len(self.results)
        c_correct = sum(1 for r in self.results if r.crystal.correct)
        d_correct = sum(1 for r in self.results if r.llm_docs.correct)
        n_correct = sum(1 for r in self.results if r.llm_naked.correct)
        lines.append(f"| **Crystal + KG** | {c_correct} | {n} | **{self.crystal_accuracy:.0%}** |")
        lines.append(f"| LLM + Docs | {d_correct} | {n} | {self.llm_docs_accuracy:.0%} |")
        lines.append(f"| Naked LLM | {n_correct} | {n} | {self.llm_naked_accuracy:.0%} |")
        lines.append("")

        crystal_wins = sum(
            1 for r in self.results
            if r.crystal.correct and not r.llm_naked.correct
        )
        crystal_loses = sum(
            1 for r in self.results
            if not r.crystal.correct and r.llm_naked.correct
        )
        lines.append(f"Crystal beats naked LLM on **{crystal_wins}** questions")
        if crystal_loses:
            lines.append(f"Crystal loses to naked LLM on **{crystal_loses}** questions")
        lines.append("")

        lines.append("## Per-Question Results\n")
        lines.append("| # | Question | Crystal | LLM+Docs | Naked | Golden |")
        lines.append("|---|----------|---------|----------|-------|--------|")
        for i, r in enumerate(self.results, 1):
            c_mark = "YES" if r.crystal.correct else "NO"
            d_mark = "YES" if r.llm_docs.correct else "NO"
            n_mark = "YES" if r.llm_naked.correct else "NO"
            lines.append(
                f"| {i} | {r.question[:50]} | {c_mark} | {d_mark} | {n_mark} "
                f"| {r.golden_answer[:40]} |"
            )
        lines.append("")

        failures = [r for r in self.results if not r.crystal.correct]
        if failures:
            lines.append("## Crystal Failures\n")
            for r in failures:
                lines.append(f"**Q:** {r.question}")
                lines.append(f"- Golden: {r.golden_answer}")
                lines.append(f"- Crystal: {r.crystal.answer[:200]}")
                lines.append(f"- Route: {r.crystal.route}")
                lines.append("")

        return "\n".join(lines)


def _get_document_text_for_question(question: str, cases_with_docs: dict) -> str:
    """Try to find relevant document text for a question."""
    q_lower = question.lower()
    for case_name, doc_text in cases_with_docs.items():
        if case_name.lower() in q_lower:
            return doc_text[:4000]
    return ""


def run_three_arm(
    cases: list[tuple[str, str, list[str], bool]] | None = None,
    kg=None,
    document_texts: dict[str, str] | None = None,
) -> ComparisonReport:
    """Run the three-arm comparison on accepted golden answers.

    Args:
        cases: List of (question, golden_answer, match_strings, is_negative).
               If None, loads from review batches.
        kg: Knowledge graph. If None, loads from default DB.
        document_texts: Dict mapping case names to document text for LLM+docs arm.
    """
    from crystal.graph import build_crystal_graph
    from crystal.state import make_initial_state
    from crystal.llm import call_llm

    if cases is None:
        from crystal.review import collect_accepted_cases
        cases = collect_accepted_cases()

    if not cases:
        logger.error("No accepted cases to benchmark.")
        return ComparisonReport()

    if kg is None:
        from crystal.tools.kg.legal import load_legal_kg
        kg = load_legal_kg(DB_PATH)
        if kg is None:
            logger.error("No KG found at %s", DB_PATH)
            return ComparisonReport()

    if document_texts is None:
        document_texts = _load_available_documents()

    graph = build_crystal_graph()
    report = ComparisonReport()

    for question, golden_answer, match_strings, is_negative in cases:
        qr = QuestionResult(
            question=question,
            golden_answer=golden_answer,
            match_strings=match_strings,
            is_negative=is_negative,
        )

        # Arm 1: Crystal + KG
        try:
            state = make_initial_state(question, kg=kg)
            final = graph.invoke(state)
            crystal_answer = final.get("final_response", "")
            crystal_route = final.get("prompt_type", "unknown")
        except Exception as e:
            crystal_answer = f"Error: {e}"
            crystal_route = "error"
        qr.crystal = ArmResult(
            answer=crystal_answer,
            correct=binary_correct(crystal_answer, match_strings, is_negative),
            route=crystal_route,
        )

        # Arm 2: LLM + document context
        doc_text = _get_document_text_for_question(question, document_texts)
        if doc_text:
            try:
                prompt = (
                    f"Based on the following document excerpt, answer this question:\n\n"
                    f"--- DOCUMENT ---\n{doc_text}\n--- END ---\n\n"
                    f"Question: {question}"
                )
                llm_docs_answer, _ = call_llm(prompt)
            except Exception as e:
                llm_docs_answer = f"Error: {e}"
        else:
            llm_docs_answer = "(no document text available)"
        qr.llm_docs = ArmResult(
            answer=llm_docs_answer,
            correct=binary_correct(llm_docs_answer, match_strings, is_negative),
        )

        # Arm 3: Naked LLM
        try:
            llm_naked_answer, _ = call_llm(question)
        except Exception as e:
            llm_naked_answer = f"Error: {e}"
        qr.llm_naked = ArmResult(
            answer=llm_naked_answer,
            correct=binary_correct(llm_naked_answer, match_strings, is_negative),
        )

        report.results.append(qr)
        logger.info(
            "  %s — Crystal:%s  LLM+Docs:%s  Naked:%s",
            question[:40],
            "✓" if qr.crystal.correct else "✗",
            "✓" if qr.llm_docs.correct else "✗",
            "✓" if qr.llm_naked.correct else "✗",
        )

    n = len(report.results)
    if n:
        report.crystal_accuracy = sum(1 for r in report.results if r.crystal.correct) / n
        report.llm_docs_accuracy = sum(1 for r in report.results if r.llm_docs.correct) / n
        report.llm_naked_accuracy = sum(1 for r in report.results if r.llm_naked.correct) / n

    return report


def _load_available_documents() -> dict[str, str]:
    """Load all cached opinion texts, keyed by case name."""
    import json
    docs_dir = PROJECT_ROOT / "benchmarks" / "documents"
    if not docs_dir.exists():
        return {}

    texts = {}
    for path in docs_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            text = ""
            if isinstance(data, dict):
                for key in ("text", "plain_text", "opinion_text", "content"):
                    if key in data and data[key]:
                        text = str(data[key])
                        break
                if not text and "opinions" in data and isinstance(data["opinions"], list):
                    parts = [
                        op["plain_text"]
                        for op in data["opinions"]
                        if isinstance(op, dict) and op.get("plain_text")
                    ]
                    text = "\n\n".join(parts)
            if text:
                case_name = path.stem.replace("-", " ")
                texts[case_name] = text
        except Exception:
            continue
    return texts


def main():
    parser = argparse.ArgumentParser(description="Three-arm comparison benchmark")
    parser.add_argument("--output", "-o", type=str, help="Save report to file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    report = run_three_arm()

    md = report.to_markdown()
    print(md)

    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
