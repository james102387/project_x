"""
Reasoning cost benchmark — measures K-reduction from grounding.

Tests the thesis that Crystal's grounding reduces reasoning loops (K),
not just prompt tokens (N). The cost model is K × (N + N²) where K is
reasoning steps. Even if grounding increases N, the reduction in K can
dominate because K multiplies the quadratic term.

Usage:
    python -m benchmarks.run_reasoning_benchmark
    python -m benchmarks.run_reasoning_benchmark --model gemini-2.5-flash
    python -m benchmarks.run_reasoning_benchmark --cases augmented_only

Requires GOOGLE_API_KEY in environment.
Uses a thinking-capable model (default: gemini-2.5-flash).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from benchmarks.ground_truth.remulak import BENCHMARK_CASES as REMULAK_CASES
from benchmarks.ground_truth.legal import LEGAL_BENCHMARK_CASES
from benchmarks.scoring.binary import score_response
from crystal.metrics import ReasoningComparison, summarize_reasoning_comparisons

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _unpack_case(case: tuple) -> tuple[str, str, list[str], bool]:
    if len(case) == 4:
        return case[0], case[1], case[2], case[3]
    return case[0], case[1], case[2], False


def _call_thinking_llm(
    prompt: str, model: str, max_retries: int = 3,
) -> tuple[str, dict | None]:
    """Call a thinking-capable model and return (response, usage)."""
    from crystal.llm import _get_client, _extract_usage

    client = _get_client()
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            usage = _extract_usage(response)
            return response.text.strip(), usage
        except Exception as e:
            if "429" in str(e) or "TooManyRequests" in str(e):
                wait = 2 ** attempt * 5
                print(
                    f"    Rate limited, waiting {wait}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait)
            else:
                raise
    return "[ERROR: Rate limit exceeded after retries]", None


def _build_grounded_prompt(question: str, kg=None) -> tuple[str | None, str]:
    """Run Crystal's KG pipeline on a question.

    Args:
        question: The question to ground.
        kg: KnowledgeGraph to use. Defaults to remulak_kg if None.

    Returns (grounded_prompt_or_None, prompt_type).
    """
    import spacy

    from crystal.detectors.kg import detect_kg_query
    from crystal.nodes.compiler import (
        _classify_prompt_type, _format_kg_results, _build_kg_augmented_prompt,
    )

    if kg is None:
        from crystal.tools.kg import remulak_kg
        kg = remulak_kg

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(question)
    detection = detect_kg_query(doc, kg)

    if detection is None:
        return None, "no_match"

    tool_results = [{
        "tool": "kg",
        "operation": "lookup",
        "entity": detection["entity"],
        "results": detection["results"],
        "lookup_type": detection.get("lookup_type", "subject_scan"),
        "success": True,
    }]
    prompt_type = _classify_prompt_type(question, doc, tool_results)

    if prompt_type == "kg_augmented":
        return _build_kg_augmented_prompt(question, tool_results), prompt_type
    elif prompt_type == "kg_answerable":
        return _format_kg_results(tool_results), prompt_type
    return None, prompt_type


def run_reasoning_benchmark(
    cases: list[tuple],
    model: str,
    filter_type: str = "all",
    kg=None,
) -> list[ReasoningComparison]:
    """Run both grounded and ungrounded queries, capturing token usage.

    Args:
        cases: Benchmark case tuples (question, answer, match_strings, is_negative).
        model: Thinking-capable model name.
        filter_type: "all", "augmented_only", or "answerable_only".
        kg: KnowledgeGraph to use. Defaults to remulak_kg if None.
    """
    comparisons = []

    for i, case in enumerate(cases):
        question, ground_truth, match_strings, is_negative = _unpack_case(case)

        if is_negative:
            continue

        grounded_prompt, prompt_type = _build_grounded_prompt(question, kg=kg)

        if filter_type == "augmented_only" and prompt_type != "kg_augmented":
            continue
        if filter_type == "answerable_only" and prompt_type != "kg_answerable":
            continue

        print(f"  [{i+1}/{len(cases)}] {question} ({prompt_type})")

        # Baseline: naked LLM
        print(f"    Baseline (ungrounded)...")
        baseline_resp, baseline_usage = _call_thinking_llm(question, model)
        baseline_correct = score_response(baseline_resp, match_strings)
        time.sleep(1.0)

        # Treatment: grounded prompt (or direct answer for kg_answerable)
        comp = ReasoningComparison(question=question)

        if baseline_usage:
            comp.baseline_prompt_tokens = baseline_usage.get("prompt_tokens")
            comp.baseline_output_tokens = baseline_usage.get("output_tokens")
            comp.baseline_reasoning_tokens = baseline_usage.get("reasoning_tokens")
            comp.baseline_total_tokens = baseline_usage.get("total_tokens")
        comp.baseline_correct = baseline_correct

        if prompt_type == "kg_answerable":
            # Crystal bypasses LLM entirely — zero tokens
            print(f"    Grounded (LLM bypassed, kg_answerable)")
            grounded_correct = score_response(grounded_prompt or "", match_strings)
            comp.grounded_prompt_tokens = 0
            comp.grounded_output_tokens = 0
            comp.grounded_reasoning_tokens = 0
            comp.grounded_total_tokens = 0
            comp.grounded_correct = grounded_correct
        elif grounded_prompt:
            print(f"    Grounded (augmented)...")
            grounded_resp, grounded_usage = _call_thinking_llm(grounded_prompt, model)
            grounded_correct = score_response(grounded_resp, match_strings)
            if grounded_usage:
                comp.grounded_prompt_tokens = grounded_usage.get("prompt_tokens")
                comp.grounded_output_tokens = grounded_usage.get("output_tokens")
                comp.grounded_reasoning_tokens = grounded_usage.get("reasoning_tokens")
                comp.grounded_total_tokens = grounded_usage.get("total_tokens")
            comp.grounded_correct = grounded_correct
            time.sleep(1.0)
        else:
            print(f"    No grounding available, skipping")
            continue

        comparisons.append(comp)

    return comparisons


def print_reasoning_report(
    comparisons: list[ReasoningComparison],
    summary: dict,
    model: str,
) -> None:
    print(f"\n{'='*70}")
    print(f"  REASONING COST BENCHMARK — {model}")
    print(f"{'='*70}")
    print(f"  Queries compared:    {summary['count']}")
    print(f"  Baseline accuracy:   {summary['baseline_accuracy']:.1%}")
    print(f"  Grounded accuracy:   {summary['grounded_accuracy']:.1%}")
    print(f"  Accuracy delta:      {summary['accuracy_delta']:+.1%}")
    print()

    if summary.get("avg_baseline_total_tokens") is not None:
        print(f"  --- Token Usage (averages per query) ---")
        print(f"  {'':24s} {'Baseline':>10s} {'Grounded':>10s} {'Delta':>10s}")

        for label, bkey, gkey in [
            ("Total tokens", "avg_baseline_total_tokens", "avg_grounded_total_tokens"),
            ("Reasoning tokens", "avg_baseline_reasoning_tokens", "avg_grounded_reasoning_tokens"),
        ]:
            b = summary.get(bkey)
            g = summary.get(gkey)
            if b is not None and g is not None:
                print(f"  {label:24s} {b:10.0f} {g:10.0f} {g - b:+10.0f}")

        print()
        if summary.get("total_token_savings_pct") is not None:
            pct = summary["total_token_savings_pct"]
            print(f"  Total token savings:     {pct:+.1%}")
        if summary.get("reasoning_token_savings_pct") is not None:
            pct = summary["reasoning_token_savings_pct"]
            print(f"  Reasoning token savings: {pct:+.1%}")

    print()
    print(f"  --- Per-Query Detail ---")
    for comp in comparisons:
        bt = comp.baseline_total_tokens or 0
        gt = comp.grounded_total_tokens or 0
        br = comp.baseline_reasoning_tokens
        gr = comp.grounded_reasoning_tokens
        delta = comp.total_token_delta

        reasoning_info = ""
        if br is not None and gr is not None:
            reasoning_info = f"  reasoning: {br} → {gr}"

        delta_str = f"{delta:+d}" if delta is not None else "?"
        print(
            f"  {'✓' if comp.grounded_correct else '✗'} "
            f"{comp.question[:50]:50s}  "
            f"total: {bt} → {gt} ({delta_str})"
            f"{reasoning_info}"
        )
    print()


_CASE_SETS = {
    "remulak": REMULAK_CASES,
    "legal": LEGAL_BENCHMARK_CASES,
}


def main():
    parser = argparse.ArgumentParser(description="Crystal Reasoning Cost Benchmark")
    parser.add_argument(
        "--model", default="gemini-2.5-flash",
        help="Thinking-capable model to use (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--cases", default="all",
        choices=["all", "augmented_only", "answerable_only"],
        help="Filter benchmark cases by type",
    )
    parser.add_argument(
        "--dataset", default="remulak",
        choices=list(_CASE_SETS.keys()),
        help="Which ground truth dataset to use (default: remulak)",
    )
    args = parser.parse_args()

    os.environ.setdefault("LLM_MODEL", args.model)

    selected_cases = _CASE_SETS[args.dataset]

    kg = None
    if args.dataset == "legal":
        from crystal.tools.kg.legal import build_legal_kg_memory
        from tests.fixtures.scotus_sample import SCOTUS_SAMPLE
        kg = build_legal_kg_memory(SCOTUS_SAMPLE)

    print(f"\n--- Reasoning Cost Benchmark (model: {args.model}, dataset: {args.dataset}) ---")
    comparisons = run_reasoning_benchmark(
        selected_cases, args.model, filter_type=args.cases, kg=kg,
    )

    if not comparisons:
        print("  No comparable cases found.")
        return

    summary = summarize_reasoning_comparisons(comparisons)
    print_reasoning_report(comparisons, summary, args.model)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"reasoning_{ts}.json"
    output = {
        "model": args.model,
        "filter": args.cases,
        "summary": summary,
        "comparisons": [c.to_dict() for c in comparisons],
    }
    path.write_text(json.dumps(output, indent=2, default=str))
    print(f"  Results saved to {path}")


if __name__ == "__main__":
    main()
