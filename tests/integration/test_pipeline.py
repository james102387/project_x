"""
Integration tests — full local pipeline against golden test cases.
No LLM calls.
"""

import pytest
import numpy as np
import spacy

from crystal.detectors.math import EXPLICIT_PATTERNS
from crystal.detectors.math import match_semantic_verb_pattern, evaluate_semantic_steps
from crystal.detectors.kg import detect_kg_query
from crystal.nodes.compiler import _classify_prompt_type, _format_kg_results
from crystal.tools.kg import remulak_kg
from tests.golden.test_cases import (
    PURE_MATH_CASES, MATH_ANSWERABLE_CASES, MATH_AUGMENTED_CASES,
    KG_ANSWERABLE_CASES, NEGATIVE_CASES,
)

nlp = spacy.load("en_core_web_sm")


def run_local_pipeline(prompt: str) -> dict:
    """Run the full local pipeline (no LLM) and return results."""
    doc = nlp(prompt)

    # --- Calculator detection ---
    detections = []
    for pattern_name, matcher in EXPLICIT_PATTERNS:
        args = matcher(doc)
        if args is not None:
            detections.append({
                "tool": "calculator",
                "operation": "add",
                "raw_args": args,
                "matched_pattern": pattern_name,
            })
            break

    if not detections:
        semantic_steps = match_semantic_verb_pattern(doc)
        if semantic_steps is not None:
            evaluation = evaluate_semantic_steps(semantic_steps)
            if evaluation is not None:
                detections.append({
                    "tool": "calculator",
                    "operation": "semantic_math",
                    "raw_args": evaluation["args"],
                    "steps": evaluation["steps"],
                    "result": evaluation["result"],
                    "matched_pattern": "semantic_verb",
                })

    # --- KG detection (only if no calculator match) ---
    if not detections:
        kg_detection = detect_kg_query(doc, remulak_kg)
        if kg_detection is not None:
            tool_results = [{
                "tool": "kg",
                "operation": "lookup",
                "entity": kg_detection["entity"],
                "results": kg_detection["results"],
                "success": True,
            }]
            prompt_type = _classify_prompt_type(prompt, doc, tool_results)
            display = _format_kg_results(tool_results)
            return {"prompt_type": prompt_type, "result": display}

    if not detections:
        return {"prompt_type": "no_match", "result": None}

    detection = detections[0]
    if detection["operation"] == "add":
        result = int(np.sum(detection["raw_args"]))
    elif detection["operation"] == "semantic_math":
        result = detection["result"]
    else:
        result = None

    tool_results = [{
        "success": True,
        "operation": detection["operation"],
        "result": result,
        "args": detection["raw_args"],
    }]
    if detection["operation"] == "semantic_math":
        tool_results[0]["steps"] = detection.get("steps", [])

    prompt_type = _classify_prompt_type(prompt, doc, tool_results)

    return {"prompt_type": prompt_type, "result": result}


@pytest.mark.parametrize("prompt,expected_type,expected_result", PURE_MATH_CASES)
def test_pure_math(prompt, expected_type, expected_result):
    result = run_local_pipeline(prompt)
    assert result["prompt_type"] == expected_type
    assert result["result"] == expected_result


@pytest.mark.parametrize("prompt,expected_type,expected_result", MATH_ANSWERABLE_CASES)
def test_math_answerable(prompt, expected_type, expected_result):
    result = run_local_pipeline(prompt)
    assert result["prompt_type"] == expected_type
    assert result["result"] == expected_result


@pytest.mark.parametrize("prompt,expected_type,expected_result", MATH_AUGMENTED_CASES)
def test_math_augmented(prompt, expected_type, expected_result):
    result = run_local_pipeline(prompt)
    assert result["prompt_type"] == expected_type
    assert result["result"] == expected_result


@pytest.mark.parametrize("prompt,expected_type,expected_result", KG_ANSWERABLE_CASES)
def test_kg_answerable(prompt, expected_type, expected_result):
    result = run_local_pipeline(prompt)
    assert result["prompt_type"] == expected_type
    assert result["result"] == expected_result


@pytest.mark.parametrize("prompt,expected_type,expected_result", NEGATIVE_CASES)
def test_negative(prompt, expected_type, expected_result):
    result = run_local_pipeline(prompt)
    assert result["prompt_type"] == "no_match"
    assert result["result"] is None
