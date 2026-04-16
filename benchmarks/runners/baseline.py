"""
Benchmark runner — baseline (naked LLM) vs treatment (Crystal + KG).

Usage:
    python -m benchmarks.runners.baseline              # both baseline + treatment
    python -m benchmarks.runners.baseline --baseline    # baseline only
    python -m benchmarks.runners.baseline --treatment   # treatment only

Requires GOOGLE_API_KEY in environment.
Results are written to benchmarks/results/ as JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from benchmarks.ground_truth.remulak import BENCHMARK_CASES
from benchmarks.scoring.binary import score_batch, score_batch_rubric

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _unpack_case(case: tuple) -> tuple[str, str, list[str], bool]:
    """Unpack a benchmark case, handling both 3-tuple and 4-tuple formats."""
    if len(case) == 4:
        return case[0], case[1], case[2], case[3]
    return case[0], case[1], case[2], False


def run_baseline(cases: list[tuple]) -> list[dict]:
    """Send each question to the naked LLM with no KG augmentation."""
    from crystal.llm import call_llm

    results = []
    for i, case in enumerate(cases):
        question, ground_truth, match_strings, is_negative = _unpack_case(case)
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
            "is_negative": is_negative,
            "response": response,
            "usage": usage,
        })
        time.sleep(0.5)

    return results


def run_treatment(cases: list[tuple]) -> list[dict]:
    """Run each question through the full Crystal pipeline with KG."""
    import spacy

    from crystal.detectors.kg import detect_kg_query
    from crystal.nodes.compiler import (
        _classify_prompt_type, _format_kg_results, _build_kg_augmented_prompt,
    )
    from crystal.tools.kg import remulak_kg

    nlp = spacy.load("en_core_web_sm")
    results = []

    for i, case in enumerate(cases):
        question, ground_truth, match_strings, is_negative = _unpack_case(case)
        print(f"  [{i+1}/{len(cases)}] {question}")

        doc = nlp(question)
        detection = detect_kg_query(doc, remulak_kg)

        kg_results = None
        if detection is not None:
            tool_results = [{
                "tool": "kg",
                "operation": "lookup",
                "entity": detection["entity"],
                "results": detection["results"],
                "lookup_type": detection.get("lookup_type", "subject_scan"),
                "success": True,
            }]
            prompt_type = _classify_prompt_type(question, doc, tool_results)
            kg_results = detection["results"]

            if prompt_type == "kg_augmented":
                response = _build_kg_augmented_prompt(question, tool_results)
            else:
                response = _format_kg_results(tool_results)
        else:
            prompt_type = "no_match"
            response = "[NO KG MATCH]"

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "match_strings": match_strings,
            "is_negative": is_negative,
            "kg_results": kg_results,
            "response": response,
            "prompt_type": prompt_type,
        })

    return results


def print_report(name: str, scored: dict) -> None:
    """Print a human-readable summary."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Total:             {scored['total']}")
    print(f"  Correct:           {scored['correct']}")
    print(f"  Accuracy:          {scored['accuracy']:.1%}")
    print(f"  Hallucination:     {scored['hallucination_rate']:.1%}")

    if "rubric_averages" in scored:
        print()
        print(f"  --- Rubric Scores (averages) ---")
        rubric = scored["rubric_averages"]
        print(f"  Factual Accuracy:  {rubric['accuracy']:.2f}")
        print(f"  Abstention:        {rubric['abstention']:.2f}")
        print()
        print(f"  Positive cases:    {scored['positive_cases']}")
        print(f"  Negative cases:    {scored['negative_cases']}")

    print()

    incorrect = [d for d in scored["details"] if not d["correct"]]
    if incorrect:
        print(f"  Incorrect answers ({len(incorrect)}):")
        for d in incorrect:
            resp_preview = d["response"][:80].replace("\n", " ")
            print(f"    Q: {d['question']}")
            print(f"    Expected: {d['ground_truth']}")
            print(f"    Got: {resp_preview}...")
            if "rubric" in d:
                r = d["rubric"]
                print(f"    Rubric: acc={r['accuracy']:.2f} "
                      f"abstention={r['abstention']:.2f}")
            print()


def save_results(name: str, scored: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{name}_{ts}.json"
    path.write_text(json.dumps(scored, indent=2, default=str))
    print(f"  Results saved to {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Crystal KG Benchmark")
    parser.add_argument("--baseline", action="store_true", help="Run baseline only")
    parser.add_argument("--treatment", action="store_true", help="Run treatment only")
    args = parser.parse_args()

    run_both = not args.baseline and not args.treatment

    baseline_scored = None
    treatment_scored = None

    if args.baseline or run_both:
        print("\n--- Baseline: Naked LLM (no KG) ---")
        baseline_results = run_baseline(BENCHMARK_CASES)
        baseline_scored = score_batch_rubric(baseline_results)
        print_report("BASELINE (naked LLM)", baseline_scored)
        save_results("baseline", baseline_scored)

    if args.treatment or run_both:
        print("\n--- Treatment: Crystal + KG ---")
        treatment_results = run_treatment(BENCHMARK_CASES)
        treatment_scored = score_batch_rubric(treatment_results)
        print_report("TREATMENT (Crystal + KG)", treatment_scored)
        save_results("treatment", treatment_scored)

    if baseline_scored and treatment_scored:
        print(f"\n{'='*60}")
        print(f"  COMPARISON")
        print(f"{'='*60}")
        delta = treatment_scored["accuracy"] - baseline_scored["accuracy"]
        print(f"  Baseline accuracy:   {baseline_scored['accuracy']:.1%}")
        print(f"  Treatment accuracy:  {treatment_scored['accuracy']:.1%}")
        print(f"  Improvement:         {delta:+.1%}")
        print(f"  Hallucination reduction: "
              f"{baseline_scored['hallucination_rate']:.1%} → "
              f"{treatment_scored['hallucination_rate']:.1%}")

        if "rubric_averages" in baseline_scored and "rubric_averages" in treatment_scored:
            print()
            print(f"  --- Rubric Comparison ---")
            for dim in ("accuracy", "abstention"):
                b = baseline_scored["rubric_averages"][dim]
                t = treatment_scored["rubric_averages"][dim]
                print(f"  {dim:20s}  {b:.2f} → {t:.2f}  ({t - b:+.2f})")

        print()


if __name__ == "__main__":
    main()
