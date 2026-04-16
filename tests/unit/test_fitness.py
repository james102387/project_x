"""Tests for the Ralph Wiggum fitness function."""

import pytest

from benchmarks.scoring.fitness import binary_correct, fitness_score, evaluate_cases


class TestBinaryCorrect:
    def test_positive_all_match(self):
        assert binary_correct("Zelphos is the capital", ["zelphos"]) is True

    def test_positive_partial_match(self):
        assert binary_correct("The capital is unknown", ["zelphos"]) is False

    def test_positive_multiple_match_strings(self):
        assert binary_correct(
            "Warren and Blackmun decided",
            ["warren", "blackmun"],
        ) is True

    def test_positive_missing_one(self):
        assert binary_correct(
            "Warren decided the case",
            ["warren", "blackmun"],
        ) is False

    def test_positive_case_insensitive(self):
        assert binary_correct("ZELPHOS", ["zelphos"]) is True

    def test_positive_empty_match_strings(self):
        assert binary_correct("anything", []) is False

    def test_negative_abstains(self):
        assert binary_correct(
            "I don't have information about that",
            [],
            is_negative=True,
        ) is True

    def test_negative_fabricates(self):
        assert binary_correct(
            "The GDP of Remulak is 5 trillion",
            [],
            is_negative=True,
        ) is False

    def test_negative_with_no_match_phrase(self):
        assert binary_correct(
            "No match found in the knowledge base",
            [],
            is_negative=True,
        ) is True


class TestFitnessScore:
    def test_perfect_score(self):
        results = [
            {"response": "Zelphos", "match_strings": ["zelphos"]},
            {"response": "Veldra-7", "match_strings": ["veldra-7"]},
        ]
        assert fitness_score(results) == 1.0

    def test_zero_score(self):
        results = [
            {"response": "wrong", "match_strings": ["zelphos"]},
            {"response": "wrong", "match_strings": ["veldra-7"]},
        ]
        assert fitness_score(results) == 0.0

    def test_mixed_score(self):
        results = [
            {"response": "Zelphos", "match_strings": ["zelphos"]},
            {"response": "wrong", "match_strings": ["veldra-7"]},
        ]
        assert fitness_score(results) == 0.5

    def test_empty_batch(self):
        assert fitness_score([]) == 0.0

    def test_negative_cases_scored(self):
        results = [
            {"response": "I don't have that info", "match_strings": [], "is_negative": True},
            {"response": "Made up answer", "match_strings": [], "is_negative": True},
        ]
        assert fitness_score(results) == 0.5


class TestEvaluateCases:
    def test_runs_and_scores(self):
        cases = [
            ("What is the capital?", "Zelphos", ["zelphos"], False),
            ("What is the GDP?", "[ABSTAIN]", [], True),
        ]

        def mock_run(q):
            if "capital" in q:
                return "The capital is Zelphos"
            return "I don't have information about GDP"

        score, results = evaluate_cases(cases, mock_run)
        assert score == 1.0
        assert len(results) == 2
        assert all(r["correct"] for r in results)

    def test_partial_score(self):
        cases = [
            ("Q1", "A1", ["answer1"], False),
            ("Q2", "A2", ["answer2"], False),
        ]
        score, results = evaluate_cases(cases, lambda q: "answer1 is here")
        assert score == 0.5

    def test_results_include_metadata(self):
        cases = [("Q", "A", ["a"], False)]
        _, results = evaluate_cases(cases, lambda q: "a")
        r = results[0]
        assert "question" in r
        assert "golden_answer" in r
        assert "response" in r
        assert "correct" in r
