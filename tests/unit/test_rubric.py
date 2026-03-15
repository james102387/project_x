"""Unit tests for the quality rubric scorers and batch scoring integration."""

import pytest

from benchmarks.rubric import (
    RubricResult,
    accuracy_score,
    calibration_score,
    grounding_score,
    score_rubric,
    specificity_score,
)
from benchmarks.scoring import score_batch, score_batch_rubric


# ── Fixtures ───────────────────────────────────────────────────────────────

SAMPLE_KG_RESULTS = [
    {"subject": "Remulak", "predicate": "capital", "object": "Zelphos"},
]

MULTI_KG_RESULTS = [
    {"subject": "Remulak", "predicate": "capital", "object": "Zelphos"},
    {"subject": "Remulak", "predicate": "leader", "object": "Grand Vizier Korth"},
    {"subject": "Remulak", "predicate": "population", "object": "4.3 billion"},
]


# ── accuracy_score ─────────────────────────────────────────────────────────


class TestAccuracyScore:
    def test_all_present(self):
        assert accuracy_score("The capital is Zelphos", ["zelphos"]) == 1.0

    def test_none_present(self):
        assert accuracy_score("I have no idea", ["zelphos"]) == 0.0

    def test_partial(self):
        score = accuracy_score(
            "Remulak uses agriculture", ["agriculture", "bioengineering"]
        )
        assert score == pytest.approx(0.5)

    def test_case_insensitive(self):
        assert accuracy_score("ZELPHOS is the capital", ["zelphos"]) == 1.0

    def test_empty_match_strings(self):
        assert accuracy_score("anything", []) == 0.0

    def test_multiple_all_present(self):
        score = accuracy_score(
            "agriculture and bioengineering are key",
            ["agriculture", "bioengineering"],
        )
        assert score == 1.0


# ── specificity_score ──────────────────────────────────────────────────────


class TestSpecificityScore:
    def test_exact_value_present(self):
        assert specificity_score("The capital is Zelphos", SAMPLE_KG_RESULTS) == 1.0

    def test_vague_response(self):
        assert specificity_score("Remulak has a capital city", SAMPLE_KG_RESULTS) == 0.0

    def test_no_kg_results(self):
        assert specificity_score("anything", []) == 1.0

    def test_partial_specificity(self):
        response = "The capital is Zelphos and population is 4.3 billion"
        score = specificity_score(response, MULTI_KG_RESULTS)
        assert score == pytest.approx(2.0 / 3.0)

    def test_all_specific(self):
        response = "Zelphos is the capital, Grand Vizier Korth leads, population 4.3 billion"
        score = specificity_score(response, MULTI_KG_RESULTS)
        assert score == 1.0


# ── grounding_score ────────────────────────────────────────────────────────


class TestGroundingScore:
    def test_fully_grounded(self):
        response = "Remulak has capital Zelphos"
        assert grounding_score(response, SAMPLE_KG_RESULTS) == 1.0

    def test_partially_grounded(self):
        response = "Zelphos is great"
        score = grounding_score(response, SAMPLE_KG_RESULTS)
        assert 0.0 < score < 1.0

    def test_no_kg_results(self):
        assert grounding_score("anything", []) == 1.0

    def test_nothing_grounded(self):
        response = "I made this up entirely"
        assert grounding_score(response, SAMPLE_KG_RESULTS) == 0.0

    def test_multi_result_grounding(self):
        response = "Remulak: capital Zelphos, leader Grand Vizier Korth, population 4.3 billion"
        score = grounding_score(response, MULTI_KG_RESULTS)
        assert score == 1.0


# ── calibration_score ──────────────────────────────────────────────────────


class TestCalibrationScore:
    def test_positive_case_always_passes(self):
        assert calibration_score("fabricated nonsense", is_negative=False) == 1.0

    def test_negative_with_abstention(self):
        assert calibration_score("I don't have information about that", is_negative=True) == 1.0

    def test_negative_without_abstention(self):
        assert calibration_score("The GDP of Remulak is 500 trillion", is_negative=True) == 0.0

    def test_negative_with_no_match(self):
        assert calibration_score("[NO KG MATCH]", is_negative=True) == 1.0

    def test_negative_various_phrases(self):
        phrases = [
            "I'm not sure about that",
            "No information is available",
            "Cannot determine from the data",
            "No data on that topic",
            "Unable to find that information",
        ]
        for phrase in phrases:
            assert calibration_score(phrase, is_negative=True) == 1.0, f"Failed: {phrase}"


# ── score_rubric (integration) ─────────────────────────────────────────────


class TestScoreRubric:
    def test_perfect_positive_response(self):
        result = score_rubric(
            response="Remulak — capital: Zelphos",
            match_strings=["zelphos"],
            kg_results=SAMPLE_KG_RESULTS,
            is_negative=False,
        )
        assert isinstance(result, RubricResult)
        assert result.accuracy == 1.0
        assert result.specificity == 1.0
        assert result.no_hallucination == 1.0

    def test_negative_case_with_abstention(self):
        result = score_rubric(
            response="I don't have information about GDP",
            match_strings=[],
            kg_results=None,
            is_negative=True,
        )
        assert result.accuracy == 0.0
        assert result.specificity == 1.0
        assert result.no_hallucination == 1.0

    def test_negative_case_without_abstention(self):
        result = score_rubric(
            response="The GDP of Remulak is 500 trillion credits",
            match_strings=[],
            kg_results=None,
            is_negative=True,
        )
        assert result.accuracy == 0.0
        assert result.no_hallucination == 0.0

    def test_defaults(self):
        result = score_rubric(
            response="Zelphos",
            match_strings=["zelphos"],
        )
        assert result.accuracy == 1.0
        assert result.specificity == 1.0
        assert result.no_hallucination == 1.0


# ── score_batch backward compat ────────────────────────────────────────────


class TestScoreBatchBackwardCompat:
    """score_batch() still works with result dicts that have no rubric fields."""

    def test_basic_correct(self):
        results = [
            {
                "question": "What is the capital?",
                "response": "Zelphos",
                "match_strings": ["zelphos"],
                "ground_truth": "Zelphos",
            },
        ]
        scored = score_batch(results)
        assert scored["total"] == 1
        assert scored["correct"] == 1
        assert scored["accuracy"] == 1.0

    def test_basic_incorrect(self):
        results = [
            {
                "question": "What is the capital?",
                "response": "I don't know",
                "match_strings": ["zelphos"],
                "ground_truth": "Zelphos",
            },
        ]
        scored = score_batch(results)
        assert scored["correct"] == 0
        assert scored["accuracy"] == 0.0

    def test_empty_batch(self):
        scored = score_batch([])
        assert scored["total"] == 0
        assert scored["accuracy"] == 0.0


# ── score_batch_rubric ─────────────────────────────────────────────────────


class TestScoreBatchRubric:
    def test_positive_batch(self):
        results = [
            {
                "question": "What is the capital of Remulak?",
                "response": "Remulak — capital: Zelphos",
                "match_strings": ["zelphos"],
                "ground_truth": "Zelphos",
                "is_negative": False,
                "kg_results": [
                    {"subject": "Remulak", "predicate": "capital", "object": "Zelphos"},
                ],
            },
        ]
        scored = score_batch_rubric(results)
        assert scored["total"] == 1
        assert scored["correct"] == 1
        assert scored["positive_cases"] == 1
        assert scored["negative_cases"] == 0
        assert "rubric_averages" in scored
        assert scored["rubric_averages"]["accuracy"] == 1.0
        assert scored["rubric_averages"]["specificity"] == 1.0
        assert scored["rubric_averages"]["no_hallucination"] == 1.0
        assert "rubric" in scored["details"][0]

    def test_negative_batch_with_abstention(self):
        results = [
            {
                "question": "What is the GDP of Remulak?",
                "response": "[NO KG MATCH]",
                "match_strings": [],
                "ground_truth": "[ABSTAIN]",
                "is_negative": True,
                "kg_results": None,
            },
        ]
        scored = score_batch_rubric(results)
        assert scored["negative_cases"] == 1
        assert scored["rubric_averages"]["no_hallucination"] == 1.0

    def test_mixed_batch(self):
        results = [
            {
                "question": "What is the capital?",
                "response": "Remulak — capital: Zelphos",
                "match_strings": ["zelphos"],
                "ground_truth": "Zelphos",
                "is_negative": False,
                "kg_results": [
                    {"subject": "Remulak", "predicate": "capital", "object": "Zelphos"},
                ],
            },
            {
                "question": "What is the GDP?",
                "response": "[NO KG MATCH]",
                "match_strings": [],
                "ground_truth": "[ABSTAIN]",
                "is_negative": True,
                "kg_results": None,
            },
        ]
        scored = score_batch_rubric(results)
        assert scored["total"] == 2
        assert scored["positive_cases"] == 1
        assert scored["negative_cases"] == 1
        assert scored["rubric_averages"]["no_hallucination"] == 1.0

    def test_defaults_when_missing_optional_fields(self):
        """score_batch_rubric works even without kg_results and is_negative."""
        results = [
            {
                "question": "What is the capital?",
                "response": "Zelphos",
                "match_strings": ["zelphos"],
                "ground_truth": "Zelphos",
            },
        ]
        scored = score_batch_rubric(results)
        assert scored["total"] == 1
        assert scored["correct"] == 1
        assert scored["rubric_averages"]["accuracy"] == 1.0
