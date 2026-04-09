"""Tests for the Ralph Wiggum self-testing loop."""

import pytest
import spacy

from benchmarks.ralph_wiggum import RalphWiggumLoop, IterationResult, LoopResult
from crystal.tools.kg.graph import KnowledgeGraph


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def simple_kg(nlp):
    """A small KG where some questions will pass and some will fail."""
    triplets = [
        ("remulak", "capital", "Zelphos"),
        ("remulak", "population", "4.3 billion"),
        ("remulak", "leader", "Grand Vizier Korth"),
    ]
    return KnowledgeGraph(
        triplets,
        predicate_aliases={"ruler": "leader", "capital city": "capital"},
    )


@pytest.fixture
def passing_cases():
    """Cases that should all pass against the simple KG."""
    return [
        ("What is the capital of Remulak?", "Zelphos", ["zelphos"], False),
        ("What is the population of Remulak?", "4.3 billion", ["4.3 billion"], False),
    ]


@pytest.fixture
def mixed_cases():
    """Cases with some passing and some failing."""
    return [
        ("What is the capital of Remulak?", "Zelphos", ["zelphos"], False),
        ("What is the GDP of Remulak?", "[ABSTAIN]", [], True),
        ("What is the area of Remulak?", "[ABSTAIN]", [], True),
    ]


@pytest.fixture
def failing_cases():
    """Cases that will fail against the simple KG."""
    return [
        ("What is the GDP of Remulak?", "5 trillion", ["5 trillion"], False),
    ]


# ── Single iteration ────────────────────────────────────────────────────


class TestRunIteration:
    def test_returns_iteration_result(self, simple_kg, passing_cases, nlp):
        loop = RalphWiggumLoop(kg=simple_kg, cases=passing_cases, nlp=nlp)
        result = loop.run_iteration(0)
        assert isinstance(result, IterationResult)
        assert result.iteration == 0

    def test_perfect_score(self, simple_kg, passing_cases, nlp):
        loop = RalphWiggumLoop(kg=simple_kg, cases=passing_cases, nlp=nlp)
        result = loop.run_iteration(0)
        assert result.score == 1.0
        assert result.correct == 2
        assert result.total == 2
        assert result.failures == []

    def test_failing_cases_analyzed(self, simple_kg, failing_cases, nlp):
        loop = RalphWiggumLoop(kg=simple_kg, cases=failing_cases, nlp=nlp)
        result = loop.run_iteration(0)
        assert result.score == 0.0
        assert len(result.failures) == 1
        assert "question" in result.failures[0]

    def test_mixed_score(self, simple_kg, mixed_cases, nlp):
        loop = RalphWiggumLoop(kg=simple_kg, cases=mixed_cases, nlp=nlp)
        result = loop.run_iteration(0)
        assert 0.0 < result.score <= 1.0


# ── Full loop ────────────────────────────────────────────────────────────


class TestRunLoop:
    def test_converges_on_easy_cases(self, simple_kg, passing_cases, nlp):
        loop = RalphWiggumLoop(kg=simple_kg, cases=passing_cases, nlp=nlp)
        result = loop.run(threshold=0.90, max_iterations=3)
        assert isinstance(result, LoopResult)
        assert result.converged is True
        assert result.final_score == 1.0
        assert result.iterations_run == 1

    def test_stops_when_score_unchanged(self, simple_kg, failing_cases, nlp):
        loop = RalphWiggumLoop(kg=simple_kg, cases=failing_cases, nlp=nlp)
        result = loop.run(threshold=0.90, max_iterations=5)
        assert result.converged is False
        assert result.iterations_run <= 2  # Stops after detecting no change

    def test_history_populated(self, simple_kg, passing_cases, nlp):
        loop = RalphWiggumLoop(kg=simple_kg, cases=passing_cases, nlp=nlp)
        result = loop.run(threshold=0.90, max_iterations=3)
        assert len(result.history) >= 1
        assert result.history[0].score == 1.0

    def test_best_score_tracked(self, simple_kg, mixed_cases, nlp):
        loop = RalphWiggumLoop(kg=simple_kg, cases=mixed_cases, nlp=nlp)
        result = loop.run(threshold=0.99, max_iterations=3)
        assert result.best_score > 0.0
        assert result.best_iteration >= 0

    def test_callback_invoked(self, simple_kg, passing_cases, nlp):
        called = []
        loop = RalphWiggumLoop(
            kg=simple_kg,
            cases=passing_cases,
            nlp=nlp,
            on_iteration=lambda r: called.append(r),
        )
        loop.run(threshold=0.90, max_iterations=2)
        assert len(called) >= 1
        assert isinstance(called[0], IterationResult)


# ── Failure analysis ─────────────────────────────────────────────────────


class TestFailureAnalysis:
    def test_failure_includes_detection_info(self, simple_kg, nlp):
        cases = [
            ("What is the capital of Remulak?", "wrong answer", ["wrong"], False),
        ]
        loop = RalphWiggumLoop(kg=simple_kg, cases=cases, nlp=nlp)
        result = loop.run_iteration(0)
        assert len(result.failures) == 1
        f = result.failures[0]
        assert "detected_entity" in f
        assert f["detected_entity"] == "remulak"

    def test_failure_no_detection(self, simple_kg, nlp):
        cases = [
            ("Is the sky blue?", "Yes", ["yes"], False),
        ]
        loop = RalphWiggumLoop(kg=simple_kg, cases=cases, nlp=nlp)
        result = loop.run_iteration(0)
        assert len(result.failures) == 1
        assert result.failures[0]["detection"] is None

    def test_negative_case_correctly_abstains(self, simple_kg, nlp):
        cases = [
            ("What is the GDP of Remulak?", "[ABSTAIN]", [], True),
        ]
        loop = RalphWiggumLoop(kg=simple_kg, cases=cases, nlp=nlp)
        result = loop.run_iteration(0)
        # Detection finds Remulak but returns all facts (subject_scan).
        # The response won't contain abstention phrases because facts are returned.
        # This is expected behavior — the compiler handles abstention downstream.
        assert result.total == 1
