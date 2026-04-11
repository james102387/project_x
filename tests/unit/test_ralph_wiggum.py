"""Tests for the Ralph Wiggum v3 multi-loop self-improvement engine."""

import pytest
import spacy

from benchmarks.ralph_wiggum import (
    IterationResult, LoopResult, FailureCategory,
    diagnose_failure, build_change_report,
)
from benchmarks.ralph_wiggum.base import BaseLoop
from benchmarks.ralph_wiggum.predicate_loop import PredicateLoop
from benchmarks.ralph_wiggum.entity_loop import EntityLoop
from benchmarks.ralph_wiggum.threshold_loop import ThresholdLoop
from benchmarks.ralph_wiggum.orchestrator import Orchestrator, OrchestratorResult
from crystal.tools.kg.graph import KnowledgeGraph


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def simple_kg(nlp):
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
    return [
        ("What is the capital of Remulak?", "Zelphos", ["zelphos"], False),
        ("What is the population of Remulak?", "4.3 billion", ["4.3 billion"], False),
    ]


@pytest.fixture
def mixed_cases():
    return [
        ("What is the capital of Remulak?", "Zelphos", ["zelphos"], False),
        ("What is the GDP of Remulak?", "[ABSTAIN]", [], True),
        ("What is the area of Remulak?", "[ABSTAIN]", [], True),
    ]


@pytest.fixture
def failing_cases():
    return [
        ("What is the GDP of Remulak?", "5 trillion", ["5 trillion"], False),
    ]


# ── BaseLoop contract (tested via PredicateLoop) ────────────────────


class TestBaseLoopLegacy:
    """Tests using detect_kg_query-only mode (use_full_pipeline=False)."""

    def test_returns_iteration_result(self, simple_kg, passing_cases, nlp):
        loop = PredicateLoop(
            kg=simple_kg, cases=passing_cases, nlp=nlp, use_full_pipeline=False,
        )
        result = loop.run_iteration(0)
        assert isinstance(result, IterationResult)
        assert result.iteration == 0

    def test_perfect_score(self, simple_kg, passing_cases, nlp):
        loop = PredicateLoop(
            kg=simple_kg, cases=passing_cases, nlp=nlp, use_full_pipeline=False,
        )
        result = loop.run_iteration(0)
        assert result.score == 1.0
        assert result.correct == 2
        assert result.total == 2
        assert result.failures == []

    def test_failing_cases_analyzed(self, simple_kg, failing_cases, nlp):
        loop = PredicateLoop(
            kg=simple_kg, cases=failing_cases, nlp=nlp, use_full_pipeline=False,
        )
        result = loop.run_iteration(0)
        assert result.score == 0.0
        assert len(result.failures) == 1
        assert "question" in result.failures[0]


# ── Full pipeline mode ──────────────────────────────────────────────


class TestBaseLoopPipeline:
    """Tests using full graph.invoke() mode."""

    def test_returns_iteration_result(self, simple_kg, passing_cases):
        loop = PredicateLoop(kg=simple_kg, cases=passing_cases)
        result = loop.run_iteration(0)
        assert isinstance(result, IterationResult)

    def test_perfect_score(self, simple_kg, passing_cases):
        loop = PredicateLoop(kg=simple_kg, cases=passing_cases)
        result = loop.run_iteration(0)
        assert result.score == 1.0
        assert result.correct == 2

    def test_diagnosis_populated(self, simple_kg, failing_cases):
        loop = PredicateLoop(kg=simple_kg, cases=failing_cases)
        result = loop.run_iteration(0)
        assert result.diagnosis_summary
        assert sum(result.diagnosis_summary.values()) == len(result.failures)

    def test_loop_name_set(self, simple_kg, passing_cases):
        loop = PredicateLoop(kg=simple_kg, cases=passing_cases)
        result = loop.run_iteration(0)
        assert result.loop_name == "PredicateLoop"


# ── Individual loop run() ───────────────────────────────────────────


class TestLoopRun:
    def test_converges_on_easy_cases(self, simple_kg, passing_cases):
        loop = PredicateLoop(kg=simple_kg, cases=passing_cases)
        result = loop.run(threshold=0.90, max_iterations=3)
        assert isinstance(result, LoopResult)
        assert result.converged is True
        assert result.final_score == 1.0
        assert result.iterations_run == 1

    def test_stops_when_score_unchanged(self, simple_kg, failing_cases):
        loop = PredicateLoop(kg=simple_kg, cases=failing_cases)
        result = loop.run(threshold=0.90, max_iterations=5)
        assert result.converged is False
        assert result.iterations_run <= 2

    def test_history_populated(self, simple_kg, passing_cases):
        loop = PredicateLoop(kg=simple_kg, cases=passing_cases)
        result = loop.run(threshold=0.90, max_iterations=3)
        assert len(result.history) >= 1
        assert result.history[0].score == 1.0

    def test_best_score_tracked(self, simple_kg, mixed_cases):
        loop = PredicateLoop(kg=simple_kg, cases=mixed_cases)
        result = loop.run(threshold=0.99, max_iterations=3)
        assert result.best_score > 0.0
        assert result.best_iteration >= 0

    def test_callback_invoked(self, simple_kg, passing_cases):
        called = []
        loop = PredicateLoop(
            kg=simple_kg,
            cases=passing_cases,
            on_iteration=lambda r: called.append(r),
        )
        loop.run(threshold=0.90, max_iterations=2)
        assert len(called) >= 1
        assert isinstance(called[0], IterationResult)

    def test_change_report_generated(self, simple_kg, passing_cases):
        loop = PredicateLoop(kg=simple_kg, cases=passing_cases)
        result = loop.run(threshold=0.90, max_iterations=2)
        assert result.change_report
        assert "PredicateLoop" in result.change_report

    def test_loop_name_in_result(self, simple_kg, passing_cases):
        loop = EntityLoop(kg=simple_kg, cases=passing_cases)
        result = loop.run(threshold=0.90, max_iterations=2)
        assert result.loop_name == "EntityLoop"


# ── Each specialized loop has correct metadata ──────────────────────


class TestLoopMetadata:
    def test_predicate_loop_categories(self):
        assert FailureCategory.PREDICATE_MISMATCH in PredicateLoop.FAILURE_CATEGORIES

    def test_entity_loop_categories(self):
        assert FailureCategory.ENTITY_MISMATCH in EntityLoop.FAILURE_CATEGORIES

    def test_threshold_loop_categories(self):
        assert FailureCategory.ROUTING_ERROR in ThresholdLoop.FAILURE_CATEGORIES

    def test_loops_have_distinct_categories(self):
        all_cats = (
            PredicateLoop.FAILURE_CATEGORIES
            | EntityLoop.FAILURE_CATEGORIES
            | ThresholdLoop.FAILURE_CATEGORIES
        )
        assert len(all_cats) == 3

    def test_loops_have_distinct_names(self):
        names = {PredicateLoop.LOOP_NAME, EntityLoop.LOOP_NAME, ThresholdLoop.LOOP_NAME}
        assert len(names) == 3


# ── Failure diagnosis ────────────────────────────────────────────────


class TestDiagnoseFailure:
    def test_correct_case_returns_correct(self):
        result = diagnose_failure(
            "What is X?", "Y", ["y"], False,
            {"final_response": "Y", "prompt_type": "kg_answerable", "kg_results": []},
            {"tool": "kg"},
        )
        assert result == FailureCategory.CORRECT

    def test_no_detection_diagnosed(self):
        result = diagnose_failure(
            "What is the GDP?", "5T", ["5t"], False,
            {"final_response": "nope", "prompt_type": "no_math", "kg_entities_found": []},
            None,
        )
        assert result == FailureCategory.NO_DETECTION

    def test_routing_error_negative_with_kg(self):
        result = diagnose_failure(
            "What was the vote?", "[ABSTAIN]", [], True,
            {"final_response": "some answer", "prompt_type": "kg_answerable"},
            {"tool": "kg"},
        )
        assert result == FailureCategory.ROUTING_ERROR


# ── Change report ────────────────────────────────────────────────────


class TestChangeReport:
    def test_empty_history(self):
        report = build_change_report([])
        assert "No iterations" in report

    def test_with_history(self):
        history = [
            IterationResult(
                iteration=0, score=0.5, total=10, correct=5,
                failures=[{"question": "q1", "diagnosis": "entity_mismatch"}],
                diagnosis_summary={"entity_mismatch": 5},
            ),
        ]
        report = build_change_report(history, "TestLoop")
        assert "50.0%" in report
        assert "entity_mismatch" in report
        assert "TestLoop" in report


# ── Failure analysis ────────────────────────────────────────────────


class TestFailureAnalysis:
    def test_failure_includes_detection_info(self, simple_kg, nlp):
        cases = [
            ("What is the capital of Remulak?", "wrong answer", ["wrong"], False),
        ]
        loop = PredicateLoop(
            kg=simple_kg, cases=cases, nlp=nlp, use_full_pipeline=False,
        )
        result = loop.run_iteration(0)
        assert len(result.failures) == 1
        f = result.failures[0]
        assert "detected_entity" in f
        assert f["detected_entity"] == "remulak"

    def test_failure_no_detection(self, simple_kg, nlp):
        cases = [
            ("Is the sky blue?", "Yes", ["yes"], False),
        ]
        loop = PredicateLoop(
            kg=simple_kg, cases=cases, nlp=nlp, use_full_pipeline=False,
        )
        result = loop.run_iteration(0)
        assert len(result.failures) == 1
        assert result.failures[0]["detection"] is None

    def test_negative_case_correctly_abstains(self, simple_kg, nlp):
        cases = [
            ("What is the GDP of Remulak?", "[ABSTAIN]", [], True),
        ]
        loop = PredicateLoop(
            kg=simple_kg, cases=cases, nlp=nlp, use_full_pipeline=False,
        )
        result = loop.run_iteration(0)
        assert result.total == 1

    def test_diagnosis_in_failure_records(self, simple_kg, failing_cases):
        loop = PredicateLoop(kg=simple_kg, cases=failing_cases)
        result = loop.run_iteration(0)
        for f in result.failures:
            assert "diagnosis" in f


# ── Orchestrator ─────────────────────────────────────────────────────


class TestOrchestrator:
    def test_runs_all_loops(self, simple_kg, passing_cases):
        orch = Orchestrator(kg=simple_kg, cases=passing_cases)
        result = orch.run(threshold=0.90, max_iterations_per_loop=2)
        assert isinstance(result, OrchestratorResult)
        assert "PredicateLoop" in result.loop_results
        assert "EntityLoop" in result.loop_results
        assert "ThresholdLoop" in result.loop_results

    def test_unified_report_generated(self, simple_kg, passing_cases):
        orch = Orchestrator(kg=simple_kg, cases=passing_cases)
        result = orch.run(threshold=0.90, max_iterations_per_loop=2)
        assert result.unified_report
        assert "Ralph Wiggum v3" in result.unified_report

    def test_overall_score_from_last_loop(self, simple_kg, passing_cases):
        orch = Orchestrator(kg=simple_kg, cases=passing_cases)
        result = orch.run(threshold=0.90, max_iterations_per_loop=2)
        assert result.overall_score == 1.0


# ── _my_failures filter ─────────────────────────────────────────────


class TestMyFailuresFilter:
    def test_predicate_loop_only_sees_predicate_failures(self, simple_kg, failing_cases):
        loop = PredicateLoop(kg=simple_kg, cases=failing_cases)
        result = loop.run_iteration(0)
        my = loop._my_failures(result.failures)
        for f in my:
            assert f["diagnosis"] == FailureCategory.PREDICATE_MISMATCH

    def test_entity_loop_only_sees_entity_failures(self, simple_kg, failing_cases):
        loop = EntityLoop(kg=simple_kg, cases=failing_cases)
        result = loop.run_iteration(0)
        my = loop._my_failures(result.failures)
        for f in my:
            assert f["diagnosis"] == FailureCategory.ENTITY_MISMATCH
