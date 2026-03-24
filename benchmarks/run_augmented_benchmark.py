"""
Augmented output quality benchmark — measures whether Crystal's augmentation
improves LLM output quality on reasoning questions.

Runs identical questions through:
  1. Naked LLM (baseline) — no grounding context
  2. Crystal pipeline with real LLM (treatment) — augmented with KG facts or math

Scores both with the D1 rubric (accuracy, specificity, no-hallucination)
to verify the augmentation helps rather than misleads.

Usage:
    python -m benchmarks.run_augmented_benchmark
    python -m benchmarks.run_augmented_benchmark --baseline-only
    python -m benchmarks.run_augmented_benchmark --treatment-only

Requires GOOGLE_API_KEY in environment.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from benchmarks.ground_truth import AUGMENTED_BENCHMARK_CASES
from benchmarks.rubric import RubricResult, score_rubric
from benchmarks.scoring import score_response

RESULTS_DIR = Path(__file__).parent / "results"


def run_baseline(cases: list[tuple]) -> list[dict]:
    """Send each question to the naked LLM (no pipeline augmentation)."""
    from crystal.llm import call_llm

    results = []
    for i, (question, ground_truth, match_strings, _) in enumerate(cases):
        print(f"  [{i+1}/{len(cases)}] {question}")
        try:
            response, usage = call_llm(question)
        except Exception as e:
            response = f"[ERROR: {e}]"
            usage = None

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "match_strings": match_strings,
            "response": response,
            "usage": usage,
        })
        time.sleep(0.5)

    return results


def run_treatment(cases: list[tuple]) -> list[dict]:
    """Run each question through the full Crystal pipeline (real LLM for augmented)."""
    from crystal.graph import build_crystal_graph
    from crystal.state import make_initial_state

    graph = build_crystal_graph()
    results = []

    for i, (question, ground_truth, match_strings, _) in enumerate(cases):
        print(f"  [{i+1}/{len(cases)}] {question}")
        try:
            state = make_initial_state(question)
            final = graph.invoke(state)
            response = final.get("final_response", "")
            prompt_type = final.get("prompt_type", "unknown")
            kg_results = final.get("kg_results", [])
            token_metrics = final.get("token_metrics", {})
        except Exception as e:
            response = f"[ERROR: {e}]"
            prompt_type = "error"
            kg_results = []
            token_metrics = {}

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "match_strings": match_strings,
            "response": response,
            "prompt_type": prompt_type,
            "kg_results": kg_results,
            "token_metrics": token_metrics,
        })
        time.sleep(0.5)

    return results


def score_results(
    results: list[dict],
    kg_results_from_treatment: dict[str, list[dict]] | None = None,
) -> list[dict]:
    """Score a result set with both binary accuracy and rubric dimensions.

    If kg_results_from_treatment is provided (keyed by question), use those
    KG results for rubric specificity/grounding — this lets us score the
    baseline against the same KG facts Crystal would inject.
    """
    scored = []
    for r in results:
        correct = score_response(r["response"], r["match_strings"])
        kg_res = r.get("kg_results") or []
        if kg_results_from_treatment and not kg_res:
            kg_res = kg_results_from_treatment.get(r["question"], [])
        rubric = score_rubric(
            response=r["response"],
            match_strings=r["match_strings"],
            kg_results=kg_res,
            is_negative=False,
        )
        scored.append({**r, "correct": correct, "rubric": asdict(rubric)})
    return scored


def summarize_scored(scored: list[dict]) -> dict:
    """Aggregate rubric scores and accuracy across scored results."""
    n = len(scored)
    if n == 0:
        return {"count": 0, "accuracy": 0.0, "rubric_averages": {}}

    correct = sum(1 for s in scored if s["correct"])
    totals = {"accuracy": 0.0, "specificity": 0.0, "no_hallucination": 0.0}
    for s in scored:
        for dim in totals:
            totals[dim] += s["rubric"][dim]

    return {
        "count": n,
        "correct": correct,
        "accuracy": correct / n,
        "rubric_averages": {k: v / n for k, v in totals.items()},
    }


def print_report(
    baseline_scored: list[dict] | None,
    treatment_scored: list[dict] | None,
) -> None:
    """Print a side-by-side quality comparison."""
    b_summary = summarize_scored(baseline_scored) if baseline_scored else None
    t_summary = summarize_scored(treatment_scored) if treatment_scored else None

    print(f"\n{'='*70}")
    print(f"  AUGMENTED OUTPUT QUALITY BENCHMARK")
    print(f"{'='*70}")

    if b_summary:
        _print_arm("Baseline (naked LLM)", b_summary)
    if t_summary:
        _print_arm("Treatment (Crystal pipeline)", t_summary)

    if b_summary and t_summary:
        print(f"  --- Comparison ---")
        ba, ta = b_summary["rubric_averages"], t_summary["rubric_averages"]
        print(f"  {'Dimension':24s} {'Baseline':>10s} {'Crystal':>10s} {'Delta':>10s}")
        print(f"  {'-'*54}")
        print(f"  {'Binary accuracy':24s} {b_summary['accuracy']:10.1%} "
              f"{t_summary['accuracy']:10.1%} "
              f"{t_summary['accuracy'] - b_summary['accuracy']:+10.1%}")
        for dim in ("accuracy", "specificity", "no_hallucination"):
            b, t = ba[dim], ta[dim]
            print(f"  {dim:24s} {b:10.2f} {t:10.2f} {t - b:+10.2f}")
        print()

    if treatment_scored:
        print(f"  --- Per-Case Detail (treatment) ---")
        for s in treatment_scored:
            r = s["rubric"]
            mark = "✓" if s["correct"] else "✗"
            route = s.get("prompt_type", "?")
            print(f"  {mark} [{route:15s}] {s['question'][:50]:50s}  "
                  f"acc={r['accuracy']:.1f} spec={r['specificity']:.2f} "
                  f"hal={r['no_hallucination']:.2f}")
        print()

    if baseline_scored:
        print(f"  --- Per-Case Detail (baseline) ---")
        for s in baseline_scored:
            r = s["rubric"]
            mark = "✓" if s["correct"] else "✗"
            resp_preview = s["response"][:60].replace("\n", " ")
            print(f"  {mark} {s['question'][:50]:50s}  "
                  f"acc={r['accuracy']:.1f} → {resp_preview}")
        print()


def _print_arm(name: str, summary: dict) -> None:
    print(f"\n  {name}:")
    print(f"    Cases:    {summary['count']}")
    print(f"    Correct:  {summary['correct']}/{summary['count']} "
          f"({summary['accuracy']:.0%})")
    ra = summary["rubric_averages"]
    print(f"    Rubric:   accuracy={ra['accuracy']:.2f}  "
          f"specificity={ra['specificity']:.2f}  "
          f"no_hallucination={ra['no_hallucination']:.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Crystal Augmented Output Quality Benchmark",
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Run baseline (naked LLM) only",
    )
    parser.add_argument(
        "--treatment-only", action="store_true",
        help="Run treatment (Crystal pipeline) only",
    )
    args = parser.parse_args()

    run_both = not args.baseline_only and not args.treatment_only
    cases = AUGMENTED_BENCHMARK_CASES

    baseline_scored = None
    treatment_scored = None
    kg_from_treatment = None

    if args.treatment_only or run_both:
        print("\n--- Treatment: Crystal pipeline (augmented LLM) ---")
        treatment_raw = run_treatment(cases)
        kg_from_treatment = {
            r["question"]: r.get("kg_results", []) for r in treatment_raw
        }
        treatment_scored = score_results(treatment_raw)

    if args.baseline_only or run_both:
        print("\n--- Baseline: Naked LLM (no augmentation) ---")
        baseline_raw = run_baseline(cases)
        baseline_scored = score_results(baseline_raw, kg_from_treatment)

    print_report(baseline_scored, treatment_scored)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": ts,
        "cases": len(cases),
    }
    if baseline_scored:
        output["baseline"] = {
            "summary": summarize_scored(baseline_scored),
            "details": baseline_scored,
        }
    if treatment_scored:
        output["treatment"] = {
            "summary": summarize_scored(treatment_scored),
            "details": treatment_scored,
        }

    path = RESULTS_DIR / f"augmented_{ts}.json"
    path.write_text(json.dumps(output, indent=2, default=str))
    print(f"  Results saved to {path}")


if __name__ == "__main__":
    main()
