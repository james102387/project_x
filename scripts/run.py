#!/usr/bin/env python3
"""
Interactive runner for Crystal.

Usage:
    python scripts/run.py                    # Run all golden tests (local only)
    python scripts/run.py --prompt "5 + 3"   # Test a single prompt
    python scripts/run.py --parse "add 5 and 3"  # Show spaCy parse tree
    python scripts/run.py --prompt "5 + 3" --full  # Full graph including LLM
"""

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import spacy

from crystal.detectors.calculator import EXPLICIT_PATTERNS
from crystal.detectors.semantic import match_semantic_verb_pattern, evaluate_semantic_steps
from crystal.nodes.compiler import _classify_prompt_type, _build_simplified_prompt
from crystal.nodes.parser import show_parse
from crystal.metrics import estimate_metrics

nlp = spacy.load("en_core_web_sm")


def run_local(prompt: str) -> dict:
    """Run the full local pipeline and return results with timing."""
    tracemalloc.start()
    start = time.perf_counter()

    doc = nlp(prompt)

    detections = []
    for pattern_name, matcher in EXPLICIT_PATTERNS:
        args = matcher(doc)
        if args is not None:
            detections.append({
                "tool": "calculator", "operation": "add",
                "raw_args": args, "matched_pattern": pattern_name,
            })
            break

    if not detections:
        semantic_steps = match_semantic_verb_pattern(doc)
        if semantic_steps is not None:
            evaluation = evaluate_semantic_steps(semantic_steps)
            if evaluation is not None:
                detections.append({
                    "tool": "calculator", "operation": "semantic_math",
                    "raw_args": evaluation["args"], "steps": evaluation["steps"],
                    "result": evaluation["result"], "matched_pattern": "semantic_verb",
                })

    if not detections:
        elapsed = time.perf_counter() - start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics = estimate_metrics(prompt, prompt, "no_match")
        return {"prompt_type": "no_match", "result": None, "pattern": None,
                "time_ms": elapsed * 1000, "mem_kb": peak_mem / 1024,
                "metrics": metrics.to_dict()}

    detection = detections[0]
    if detection["operation"] == "add":
        result = int(np.sum(detection["raw_args"]))
    elif detection["operation"] == "semantic_math":
        result = detection["result"]
    else:
        result = None

    tool_results = [{"success": True, "operation": detection["operation"],
                     "result": result, "args": detection["raw_args"]}]
    if detection["operation"] == "semantic_math":
        tool_results[0]["steps"] = detection.get("steps", [])

    prompt_type = _classify_prompt_type(prompt, doc, tool_results)

    compiled_prompt = ""
    if prompt_type == "math_augmented":
        compiled_prompt = _build_simplified_prompt(prompt, tool_results)

    metrics = estimate_metrics(prompt, compiled_prompt, prompt_type)

    elapsed = time.perf_counter() - start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "prompt_type": prompt_type, "result": result,
        "pattern": detection.get("matched_pattern"),
        "steps": detection.get("steps"),
        "compiled_prompt": compiled_prompt if prompt_type == "math_augmented" else None,
        "time_ms": elapsed * 1000, "mem_kb": peak_mem / 1024,
        "metrics": metrics.to_dict(),
    }


def run_golden_tests():
    """Run all golden test cases and report results."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tests.golden.test_cases import (
        PURE_MATH_CASES, MATH_ANSWERABLE_CASES, MATH_AUGMENTED_CASES, NEGATIVE_CASES,
    )

    passed = failed = 0
    errors = []
    metrics_by_type: dict[str, list[float]] = {}

    all_cases = (
        [("pure_math", c) for c in PURE_MATH_CASES]
        + [("math_answerable", c) for c in MATH_ANSWERABLE_CASES]
        + [("math_augmented", c) for c in MATH_AUGMENTED_CASES]
        + [("negative", c) for c in NEGATIVE_CASES]
    )

    print()
    w = 150
    print("=" * w)
    print(f"  {'PROMPT':<55} {'EXPECTED':<18} {'GOT':<18} {'RESULT':<8} "
          f"{'TOKENS':<9} {'ISOLATED':<9} {'MARGINAL':<9} {'STATUS'}")
    print("=" * w)

    for category, (prompt, expected_type, expected_result) in all_cases:
        result = run_local(prompt)
        actual_type = result["prompt_type"]
        actual_result = result["result"]

        type_ok = actual_type == expected_type
        result_ok = actual_result == expected_result

        m = result.get("metrics", {})
        tok_sav = m.get("token_savings_pct", 0.0)
        iso_sav = m.get("savings_pct", 0.0)
        mar_sav = m.get("marginal_savings_pct", 0.0)

        metrics_by_type.setdefault(actual_type, []).append(
            {"token": tok_sav, "isolated": iso_sav, "marginal": mar_sav}
        )

        if type_ok and result_ok:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
            errors.append({"prompt": prompt,
                           "expected": (expected_type, expected_result),
                           "got": (actual_type, actual_result)})

        display_result = str(actual_result) if actual_result is not None else "—"
        print(f"  {prompt:<55} {expected_type:<18} {actual_type:<18} "
              f"{display_result:<8} {tok_sav:+.0%}     {iso_sav:+.0%}     "
              f"{mar_sav:+.0%}     {status}")

    print()
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")

    if metrics_by_type:
        print()
        print("  Savings by Type (averages):")
        print(f"    {'TYPE':<20} {'TOKENS':>8} {'ISOLATED':>10} {'MARGINAL':>10}  {'n':>3}")
        print(f"    {'─' * 55}")
        for ptype in ("pure_math", "math_answerable", "math_augmented", "no_match"):
            vals = metrics_by_type.get(ptype, [])
            if vals:
                avg_tok = sum(v["token"] for v in vals) / len(vals)
                avg_iso = sum(v["isolated"] for v in vals) / len(vals)
                avg_mar = sum(v["marginal"] for v in vals) / len(vals)
                print(f"    {ptype:<20} {avg_tok:>+7.0%} {avg_iso:>+9.0%} "
                      f"{avg_mar:>+9.0%}  {len(vals):>3}")

    if errors:
        print()
        print("  FAILURES:")
        for e in errors:
            print(f"    {e['prompt']}")
            print(f"      expected: type={e['expected'][0]}, result={e['expected'][1]}")
            print(f"      got:      type={e['got'][0]}, result={e['got'][1]}")

    return failed == 0


def run_full_graph(prompt: str):
    """Run the complete LangGraph pipeline including LLM calls."""
    from crystal.graph import build_crystal_graph
    from crystal.state import make_initial_state

    app = build_crystal_graph()
    state = make_initial_state(prompt)

    tracemalloc.start()
    start = time.perf_counter()
    final = app.invoke(state)
    elapsed = time.perf_counter() - start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    m = final.get("token_metrics", {})

    print(f"\n  Prompt:          {prompt}")
    print(f"  Prompt type:     {final.get('prompt_type', 'N/A')}")
    print(f"  Final response:  {final.get('final_response', 'N/A')}")
    compiled = final.get('compiled_prompt', '')
    if compiled:
        print(f"  Compiled prompt: {compiled[:100]}")
    print(f"  Time:            {elapsed * 1000:.1f}ms")
    print(f"  Peak memory:     {peak_mem / 1024:.1f}KB")

    if m:
        print(f"\n  Token Metrics:")
        print(f"    Raw tokens:      {m.get('raw_prompt_tokens', 'N/A')}")
        print(f"    Compiled tokens: {m.get('compiled_prompt_tokens', 'N/A')}")
        print(f"    Raw compute:     {m.get('raw_compute', 'N/A')}")
        print(f"    Compiled compute:{m.get('compiled_compute', 'N/A')}")
        print(f"    Token savings:   {m.get('token_savings_pct', 0):+.0%}")
        print(f"    Isolated N+N²:   {m.get('savings_pct', 0):+.0%}")
        print(f"    Marginal (B=2k): {m.get('marginal_savings_pct', 0):+.0%}")
        if m.get("actual_prompt_tokens") is not None:
            print(f"    API prompt tkns: {m['actual_prompt_tokens']}")
            print(f"    API output tkns: {m['actual_output_tokens']}")


def main():
    parser = argparse.ArgumentParser(description="Crystal interactive runner")
    parser.add_argument("--prompt", type=str, help="Test a single prompt (local pipeline)")
    parser.add_argument("--parse", type=str, help="Show spaCy parse tree for a prompt")
    parser.add_argument("--full", action="store_true", help="Run full graph with LLM")
    parser.add_argument("--golden", action="store_true", help="Run golden test suite")

    args = parser.parse_args()

    if args.parse:
        show_parse(args.parse)
    elif args.prompt and args.full:
        run_full_graph(args.prompt)
    elif args.prompt:
        result = run_local(args.prompt)
        m = result.get("metrics", {})
        print(f"\n  Prompt:     {args.prompt}")
        print(f"  Type:       {result['prompt_type']}")
        print(f"  Result:     {result['result']}")
        print(f"  Pattern:    {result.get('pattern', '—')}")
        print(f"  Time:       {result['time_ms']:.2f}ms")
        print(f"  Memory:     {result['mem_kb']:.1f}KB")
        if result.get("steps"):
            print(f"  Steps:")
            for s in result["steps"]:
                print(f"    {s['op']:>10}  {s['value']}  ({s['verb']})")
        if result.get("compiled_prompt"):
            print(f"\n  Compiled prompt:")
            print(f"  {result['compiled_prompt']}")
        if m:
            print(f"\n  Token Metrics:")
            print(f"    Raw tokens:      {m.get('raw_prompt_tokens', 'N/A')}")
            print(f"    Compiled tokens: {m.get('compiled_prompt_tokens', 'N/A')}")
            print(f"    Token savings:   {m.get('token_savings_pct', 0):+.0%}")
            print(f"    Isolated N+N²:   {m.get('savings_pct', 0):+.0%}")
            print(f"    Marginal (B=2k): {m.get('marginal_savings_pct', 0):+.0%}")
    else:
        success = run_golden_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
