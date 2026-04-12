"""
B6: Three-arm comparison benchmark — orchestrator and report generator.

Runs three benchmark arms on document-answerable questions:
  Arm 1: Naked LLM (no context)
  Arm 2: LLM + real opinion text (lawyer-pasted document)
  Arm 3: Crystal pipeline (KG-grounded)

Separately runs Crystal on KG-only questions that can't be answered from
documents, and tests abstention on negatives.

Per-question results are cached to disk so rate limits only slow you down,
never lose progress. Questions are grouped by case name so Gemini's implicit
context caching kicks in for Arm 2 (same document prefix → cached tokens).

Usage:
    PYTHONPATH=src:. python -m benchmarks.runners.comparison
    PYTHONPATH=src:. python -m benchmarks.runners.comparison --from-cache
    PYTHONPATH=src:. python -m benchmarks.runners.comparison --batch
    PYTHONPATH=src:. python -m benchmarks.runners.comparison --clear-cache
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from benchmarks.answerability import partition_cases, partition_summary
from benchmarks.cache import get_cached, set_cached, cache_stats
from benchmarks.scoring.binary import score_batch_rubric

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _unpack_case(case: tuple) -> tuple[str, str, list[str], bool]:
    if len(case) == 4:
        return case[0], case[1], case[2], case[3]
    return case[0], case[1], case[2], False


def _get_model_tag() -> str:
    """Current model identifier for cache keys."""
    from crystal.llm import LLM_PROVIDER, LLM_MODEL
    return f"{LLM_PROVIDER}:{LLM_MODEL}"


def _sort_by_case_name(cases: list[tuple]) -> list[tuple]:
    """Sort cases by extracted case name so same-document questions are adjacent.

    This triggers Gemini's implicit context caching — when consecutive
    prompts share a long prefix (the opinion document), cached tokens
    are reused at ~90% discount.
    """
    from benchmarks.runners.document import _extract_case_name
    from crystal.data.legal_ontology import normalize_case_name

    def sort_key(case):
        name = _extract_case_name(case[0])
        return normalize_case_name(name).lower() if name else ""

    return sorted(cases, key=sort_key)


def run_arm_naked_llm(
    cases: list[tuple],
    *,
    call_llm_fn=None,
    sleep_between: float = 4.0,
) -> list[dict]:
    """Arm 1: Naked LLM — question only, no context. Uses per-question cache."""
    if call_llm_fn is None:
        from crystal.llm import call_llm
        call_llm_fn = call_llm

    model_tag = _get_model_tag()
    results = []
    cache_hits = 0

    for i, case in enumerate(cases):
        question, ground_truth, match_strings, is_negative = _unpack_case(case)

        cached = get_cached(question, "arm1_naked", model_tag)
        if cached is not None:
            cache_hits += 1
            print(f"  [Arm1 {i+1}/{len(cases)}] (cached) {question[:60]}")
            results.append(cached)
            continue

        print(f"  [Arm1 {i+1}/{len(cases)}] {question[:60]}")
        try:
            response, usage = call_llm_fn(question)
        except Exception as e:
            response = f"[ERROR: {e}]"
            usage = None

        result = {
            "question": question,
            "ground_truth": ground_truth,
            "match_strings": match_strings,
            "is_negative": is_negative,
            "response": response,
            "usage": usage,
            "prompt_tokens_estimate": len(question) // 4,
        }
        results.append(result)
        set_cached(question, "arm1_naked", model_tag, result)

        if sleep_between > 0:
            time.sleep(sleep_between)

    if cache_hits:
        print(f"  → {cache_hits}/{len(cases)} from cache")
    return results


def run_arm_document(
    cases: list[tuple],
    opinions: dict[str, str],
    *,
    call_llm_fn=None,
    sleep_between: float = 4.0,
) -> list[dict]:
    """Arm 2: LLM + real opinion text. Uses per-question cache."""
    from benchmarks.runners.document import run_document_baseline_cached
    model_tag = _get_model_tag()
    return run_document_baseline_cached(
        cases, opinions,
        arm_name="arm2_doc",
        model_tag=model_tag,
        call_llm_fn=call_llm_fn,
        sleep_between=sleep_between,
    )


def run_arm_crystal(
    cases: list[tuple],
    *,
    kg=None,
    sleep_between: float = 4.0,
) -> list[dict]:
    """Arm 3: Crystal pipeline with KG. Uses per-question cache."""
    from crystal.graph import build_crystal_graph
    from crystal.state import make_initial_state

    model_tag = _get_model_tag()
    graph = build_crystal_graph()
    results = []
    cache_hits = 0

    for i, case in enumerate(cases):
        question, ground_truth, match_strings, is_negative = _unpack_case(case)

        cached = get_cached(question, "arm3_crystal", model_tag)
        if cached is not None:
            cache_hits += 1
            print(f"  [Arm3 {i+1}/{len(cases)}] (cached) {question[:60]}")
            results.append(cached)
            continue

        print(f"  [Arm3 {i+1}/{len(cases)}] {question[:60]}")
        try:
            state = make_initial_state(question, kg=kg)
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

        actual_prompt = token_metrics.get("actual_prompt_tokens") or 0
        actual_output = token_metrics.get("actual_output_tokens") or 0
        actual_reasoning = token_metrics.get("actual_reasoning_tokens") or 0

        result = {
            "question": question,
            "ground_truth": ground_truth,
            "match_strings": match_strings,
            "is_negative": is_negative,
            "response": response,
            "prompt_type": prompt_type,
            "kg_results": kg_results,
            "token_metrics": token_metrics,
            "prompt_tokens_estimate": actual_prompt,
            "output_tokens": actual_output,
            "reasoning_tokens": actual_reasoning,
        }
        results.append(result)
        set_cached(question, "arm3_crystal", model_tag, result)

        if sleep_between > 0:
            time.sleep(sleep_between)

    if cache_hits:
        print(f"  → {cache_hits}/{len(cases)} from cache")
    return results


def run_arm1_batch(
    cases: list[tuple],
) -> list[dict]:
    """Arm 1 via Gemini Batch API — all uncached prompts in one batch job."""
    from crystal.llm import call_llm_batch

    model_tag = _get_model_tag()
    results: list[dict | None] = [None] * len(cases)
    pending_indices: list[int] = []
    pending_prompts: list[str] = []

    for i, case in enumerate(cases):
        question = _unpack_case(case)[0]
        cached = get_cached(question, "arm1_naked", model_tag)
        if cached is not None:
            results[i] = cached
        else:
            pending_indices.append(i)
            pending_prompts.append(question)

    print(f"  Arm1 batch: {len(cases)} total, {len(cases) - len(pending_indices)} cached, "
          f"{len(pending_indices)} to submit")

    if pending_prompts:
        batch_results = call_llm_batch(pending_prompts, display_name="arm1-naked-llm")

        for idx, (response, usage) in zip(pending_indices, batch_results):
            question, ground_truth, match_strings, is_negative = _unpack_case(cases[idx])
            result = {
                "question": question,
                "ground_truth": ground_truth,
                "match_strings": match_strings,
                "is_negative": is_negative,
                "response": response,
                "usage": usage,
                "prompt_tokens_estimate": len(question) // 4,
            }
            results[idx] = result
            set_cached(question, "arm1_naked", model_tag, result)

    return results


def run_arm2_batch(
    cases: list[tuple],
    opinions: dict[str, str],
) -> list[dict]:
    """Arm 2 via Gemini Batch API — all uncached doc prompts in one batch job."""
    from benchmarks.runners.document import _resolve_document
    from crystal.llm import call_llm_batch

    model_tag = _get_model_tag()
    results: list[dict | None] = [None] * len(cases)
    pending_indices: list[int] = []
    pending_prompts: list[str] = []
    pending_meta: list[tuple[str, str]] = []

    for i, case in enumerate(cases):
        question = _unpack_case(case)[0]
        cached = get_cached(question, "arm2_doc", model_tag)
        if cached is not None:
            results[i] = cached
        else:
            prompt, prompt_source = _resolve_document(question, opinions)
            pending_indices.append(i)
            pending_prompts.append(prompt)
            pending_meta.append((prompt_source, str(len(prompt))))

    print(f"  Arm2 batch: {len(cases)} total, {len(cases) - len(pending_indices)} cached, "
          f"{len(pending_indices)} to submit")

    if pending_prompts:
        batch_results = call_llm_batch(pending_prompts, display_name="arm2-llm-doc")

        for j, (response, usage) in enumerate(batch_results):
            idx = pending_indices[j]
            question, ground_truth, match_strings, is_negative = _unpack_case(cases[idx])
            prompt_source, prompt_len = pending_meta[j]
            result = {
                "question": question,
                "ground_truth": ground_truth,
                "match_strings": match_strings,
                "is_negative": is_negative,
                "response": response,
                "usage": usage,
                "prompt_source": prompt_source,
                "prompt_chars": int(prompt_len),
                "prompt_tokens_estimate": int(prompt_len) // 4,
            }
            results[idx] = result
            set_cached(question, "arm2_doc", model_tag, result)

    return results


def _save_arm_results(arm_name: str, results: list[dict]) -> Path:
    """Cache arm results to JSON (in addition to per-question cache)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"arm_{arm_name}_results.json"
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return path


def _load_arm_results(arm_name: str) -> list[dict] | None:
    """Load cached arm results if they exist."""
    path = RESULTS_DIR / f"arm_{arm_name}_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_comparison(
    cases: list[tuple],
    opinions: dict[str, str],
    *,
    kg=None,
    call_llm_fn=None,
    from_cache: bool = False,
    use_batch: bool = False,
    sleep_between: float = 4.0,
) -> dict:
    """Run the full three-arm comparison benchmark.

    Args:
        use_batch: If True, use Gemini Batch API for Arms 1 & 2
            (separate rate limits, 50% cost discount).

    Returns a structured report dict with per-section results.
    """
    doc_answerable, kg_only, negatives, subject_scan = partition_cases(cases)

    fair_ab_cases = _sort_by_case_name(doc_answerable + subject_scan)

    print(f"\n{'=' * 60}")
    print(f"  THREE-ARM COMPARISON BENCHMARK")
    print(f"{'=' * 60}")
    summary = partition_summary(cases)
    stats = cache_stats()
    print(f"  Total cases:          {summary['total']}")
    print(f"  Document-answerable:  {summary['document_answerable']}")
    print(f"  Subject scan:         {summary['subject_scan']}")
    print(f"  KG-only:              {summary['kg_only']}")
    print(f"  Negatives:            {summary['negative']}")
    print(f"  Question cache:       {stats['entries']} entries")
    print(f"  Mode:                 {'batch' if use_batch else 'sequential'}")
    print(f"  Predicate breakdown:  {summary['by_predicate']}")
    print()

    arm1_fair = None
    arm2_fair = None
    arm3_fair = None
    arm3_kg_only = None
    arm1_neg = None
    arm2_neg = None
    arm3_neg = None

    if from_cache:
        arm1_fair = _load_arm_results("1_fair")
        arm2_fair = _load_arm_results("2_fair")
        arm3_fair = _load_arm_results("3_fair")
        arm3_kg_only = _load_arm_results("3_kg_only")
        arm1_neg = _load_arm_results("1_neg")
        arm2_neg = _load_arm_results("2_neg")
        arm3_neg = _load_arm_results("3_neg")

    if fair_ab_cases:
        if arm1_fair is None:
            print("\n--- Arm 1: Naked LLM (fair A/B) ---")
            if use_batch:
                arm1_fair = run_arm1_batch(fair_ab_cases)
            else:
                arm1_fair = run_arm_naked_llm(
                    fair_ab_cases, call_llm_fn=call_llm_fn, sleep_between=sleep_between,
                )
            _save_arm_results("1_fair", arm1_fair)

        if arm2_fair is None:
            print("\n--- Arm 2: LLM + Document (fair A/B) ---")
            if use_batch:
                arm2_fair = run_arm2_batch(fair_ab_cases, opinions)
            else:
                arm2_fair = run_arm_document(
                    fair_ab_cases, opinions,
                    call_llm_fn=call_llm_fn, sleep_between=sleep_between,
                )
            _save_arm_results("2_fair", arm2_fair)

        if arm3_fair is None:
            print("\n--- Arm 3: Crystal Pipeline (fair A/B) ---")
            arm3_fair = run_arm_crystal(
                fair_ab_cases, kg=kg, sleep_between=sleep_between,
            )
            _save_arm_results("3_fair", arm3_fair)

    if kg_only and arm3_kg_only is None:
        print("\n--- Arm 3: Crystal Pipeline (KG-only questions) ---")
        arm3_kg_only = run_arm_crystal(
            kg_only, kg=kg, sleep_between=sleep_between,
        )
        _save_arm_results("3_kg_only", arm3_kg_only)

    if negatives:
        if arm1_neg is None:
            print("\n--- Arm 1: Naked LLM (negatives) ---")
            if use_batch:
                arm1_neg = run_arm1_batch(negatives)
            else:
                arm1_neg = run_arm_naked_llm(
                    negatives, call_llm_fn=call_llm_fn, sleep_between=sleep_between,
                )
            _save_arm_results("1_neg", arm1_neg)

        if arm2_neg is None:
            print("\n--- Arm 2: LLM + Document (negatives) ---")
            if use_batch:
                arm2_neg = run_arm2_batch(negatives, opinions)
            else:
                arm2_neg = run_arm_document(
                    negatives, opinions,
                    call_llm_fn=call_llm_fn, sleep_between=sleep_between,
                )
            _save_arm_results("2_neg", arm2_neg)

        if arm3_neg is None:
            print("\n--- Arm 3: Crystal Pipeline (negatives) ---")
            arm3_neg = run_arm_crystal(
                negatives, kg=kg, sleep_between=sleep_between,
            )
            _save_arm_results("3_neg", arm3_neg)

    scored_1_fair = score_batch_rubric(arm1_fair) if arm1_fair else None
    scored_2_fair = score_batch_rubric(arm2_fair) if arm2_fair else None
    scored_3_fair = score_batch_rubric(arm3_fair) if arm3_fair else None
    scored_3_kg = score_batch_rubric(arm3_kg_only) if arm3_kg_only else None
    scored_1_neg = score_batch_rubric(arm1_neg) if arm1_neg else None
    scored_2_neg = score_batch_rubric(arm2_neg) if arm2_neg else None
    scored_3_neg = score_batch_rubric(arm3_neg) if arm3_neg else None

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": _get_model_tag(),
        "partition": summary,
        "fair_ab": {
            "cases": len(fair_ab_cases),
            "arm1_naked_llm": scored_1_fair,
            "arm2_llm_document": scored_2_fair,
            "arm3_crystal": scored_3_fair,
        },
        "kg_only": {
            "cases": len(kg_only),
            "arm3_crystal": scored_3_kg,
        },
        "negatives": {
            "cases": len(negatives),
            "arm1_naked_llm": scored_1_neg,
            "arm2_llm_document": scored_2_neg,
            "arm3_crystal": scored_3_neg,
        },
    }

    return report


def _fmt_pct(val: float | None) -> str:
    return f"{val:.1%}" if val is not None else "N/A"


def _fmt_score(val: float | None) -> str:
    return f"{val:.2f}" if val is not None else "N/A"


def _avg_tokens(scored: dict | None, field: str = "prompt_tokens_estimate") -> str:
    if not scored or not scored.get("details"):
        return "N/A"
    tokens = [d.get(field, 0) or 0 for d in scored["details"]]
    if not tokens:
        return "N/A"
    avg = sum(tokens) / len(tokens)
    return f"~{int(avg):,}"


def print_report(report: dict) -> None:
    """Print a human-readable comparison report."""
    print(f"\n{'=' * 70}")
    print(f"  THREE-ARM COMPARISON REPORT  ({report.get('model', 'unknown')})")
    print(f"{'=' * 70}")

    fair = report.get("fair_ab", {})
    a1, a2, a3 = fair.get("arm1_naked_llm"), fair.get("arm2_llm_document"), fair.get("arm3_crystal")

    print(f"\n  === FAIR A/B COMPARISON ({fair.get('cases', 0)} document-answerable questions) ===\n")
    print(f"  {'Metric':24s} {'Naked LLM':>12s} {'LLM+Doc':>12s} {'Crystal':>12s}")
    print(f"  {'-' * 60}")
    print(f"  {'Accuracy':24s} {_fmt_pct(a1 and a1.get('accuracy')):>12s} "
          f"{_fmt_pct(a2 and a2.get('accuracy')):>12s} "
          f"{_fmt_pct(a3 and a3.get('accuracy')):>12s}")
    print(f"  {'Hallucination rate':24s} {_fmt_pct(a1 and a1.get('hallucination_rate')):>12s} "
          f"{_fmt_pct(a2 and a2.get('hallucination_rate')):>12s} "
          f"{_fmt_pct(a3 and a3.get('hallucination_rate')):>12s}")

    print(f"  {'Avg input tokens':24s} {_avg_tokens(a1):>12s} {_avg_tokens(a2):>12s} {_avg_tokens(a3):>12s}")
    print(f"  {'Avg output tokens':24s} {_avg_tokens(a1, 'output_tokens'):>12s} "
          f"{_avg_tokens(a2, 'output_tokens'):>12s} "
          f"{_avg_tokens(a3, 'output_tokens'):>12s}")
    print(f"  {'Avg reasoning tokens':24s} {_avg_tokens(a1, 'reasoning_tokens'):>12s} "
          f"{_avg_tokens(a2, 'reasoning_tokens'):>12s} "
          f"{_avg_tokens(a3, 'reasoning_tokens'):>12s}")

    kg = report.get("kg_only", {})
    kg3 = kg.get("arm3_crystal")
    print(f"\n  === CRYSTAL-ONLY ({kg.get('cases', 0)} KG metadata questions) ===\n")
    if kg3:
        print(f"  Crystal accuracy:     {_fmt_pct(kg3.get('accuracy'))}")
        print(f"  (Arms 1 & 2 cannot answer — data not in any document)")
    else:
        print(f"  No KG-only questions in this benchmark set.")

    neg = report.get("negatives", {})
    n1, n2, n3 = neg.get("arm1_naked_llm"), neg.get("arm2_llm_document"), neg.get("arm3_crystal")
    print(f"\n  === NEGATIVES ({neg.get('cases', 0)} abstention questions) ===\n")
    if n1 or n2 or n3:
        print(f"  {'Metric':24s} {'Naked LLM':>12s} {'LLM+Doc':>12s} {'Crystal':>12s}")
        print(f"  {'-' * 60}")
        print(f"  {'Abstention rate':24s} "
              f"{_fmt_pct(n1 and n1.get('accuracy')):>12s} "
              f"{_fmt_pct(n2 and n2.get('accuracy')):>12s} "
              f"{_fmt_pct(n3 and n3.get('accuracy')):>12s}")
    else:
        print(f"  No negative cases in this benchmark set.")

    print()


def save_report(report: dict) -> Path:
    """Save the full report to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"comparison_report_{ts}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="B6: Three-arm comparison benchmark")
    parser.add_argument("--from-cache", action="store_true", help="Load arm-level results from cache")
    parser.add_argument("--clear-cache", action="store_true", help="Clear per-question cache and exit")
    parser.add_argument("--batch", action="store_true", help="Use Gemini Batch API for Arms 1 & 2")
    parser.add_argument("--tier2", action="store_true", help="Use Tier 2 sampled cases")
    parser.add_argument("--sample-size", type=int, default=200, help="Tier 2 sample size")
    parser.add_argument("--db-path", default=None, help="Path to SQLite KG")
    args = parser.parse_args()

    if args.clear_cache:
        from benchmarks.cache import clear_cache
        n = clear_cache()
        print(f"Cleared {n} cached question results.")
        return

    from benchmarks.documents import load_all_opinions
    from benchmarks.ground_truth.legal import LEGAL_BENCHMARK_CASES

    if args.tier2:
        from crystal.review import collect_accepted_cases
        from benchmarks.sampling import sample_benchmark_cases
        all_cases = collect_accepted_cases()
        cases = sample_benchmark_cases(all_cases, n=args.sample_size)
        print(f"Tier 2: sampled {len(cases)} from {len(all_cases)} accepted cases")
    else:
        cases = LEGAL_BENCHMARK_CASES
        print(f"Tier 1: {len(cases)} hand-crafted benchmark cases")

    opinions = load_all_opinions()
    print(f"Loaded {len(opinions)} cached opinion documents")

    kg = None
    if args.db_path:
        from crystal.tools.kg.store import SqliteKnowledgeGraph
        kg = SqliteKnowledgeGraph(args.db_path)
    else:
        try:
            from crystal.tools.kg.legal import load_legal_kg
            kg = load_legal_kg()
        except Exception:
            pass

    use_batch = False
    if args.batch:
        from crystal.llm import LLM_PROVIDER
        if LLM_PROVIDER != "gemini":
            print("Warning: --batch only supported for Gemini. Falling back to sequential.")
        else:
            print("Batch mode enabled — Arms 1 & 2 will use Gemini Batch API")
            use_batch = True

    report = run_comparison(
        cases, opinions,
        kg=kg,
        from_cache=args.from_cache,
        use_batch=use_batch,
    )

    print_report(report)
    path = save_report(report)
    print(f"  Report saved to {path}")


if __name__ == "__main__":
    main()
