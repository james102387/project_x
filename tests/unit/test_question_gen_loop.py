"""Tests for LLM-based question generation and the QuestionGenLoop."""

from __future__ import annotations

import json
import textwrap

import pytest

from crystal.compare import (
    QUESTION_GEN_PROMPT,
    _parse_question_response,
    generate_questions_from_triplets,
    generate_questions_llm,
)
from crystal.ingest.question_gen import (
    PREDICATE_QUESTION_FORMS,
    canonical_template,
)


# ── Doctrinal template tests ────────────────────────────────────────────


class TestDoctrinalTemplates:
    """Verify that holding/doctrine/reasoning resolve to templates via the
    single source of truth (`PREDICATE_QUESTION_FORMS`)."""

    def test_canonical_template_resolves_doctrinal_preds(self):
        for pred in ("holding", "doctrine", "reasoning"):
            assert canonical_template(pred) is not None, f"Missing template for {pred}"

    def test_predicate_forms_have_variants(self):
        for pred in ("holding", "doctrine", "reasoning"):
            assert pred in PREDICATE_QUESTION_FORMS, f"Missing template for {pred}"
            assert len(PREDICATE_QUESTION_FORMS[pred]) >= 2

    def test_template_generates_holding_question(self):
        triplets = [
            ("miranda v. arizona", "holding",
             "Prosecution may not use statements from custodial interrogation"),
        ]
        qs = generate_questions_from_triplets(triplets, max_questions=5)
        assert len(qs) == 1
        assert "hold" in qs[0].lower() or "Miranda" in qs[0]

    def test_template_generates_doctrine_question(self):
        triplets = [
            ("marbury v. madison", "doctrine",
             "Judicial review: courts can strike down unconstitutional laws"),
        ]
        qs = generate_questions_from_triplets(triplets, max_questions=5)
        assert len(qs) == 1
        assert "Marbury" in qs[0]

    def test_template_generates_reasoning_question(self):
        triplets = [
            ("roe v. wade", "reasoning",
             "The right to privacy extends to a woman's decision"),
        ]
        qs = generate_questions_from_triplets(triplets, max_questions=5)
        assert len(qs) == 1
        assert "Roe" in qs[0]


class TestDoctrinalGoldenFacts:
    """Ensure GOLDEN_DOCTRINAL_FACTS all pass validation and generate questions."""

    def test_all_doctrinal_facts_pass_validation(self):
        from crystal.ingest.validation import validate_triplet
        from tests.golden.test_cases import GOLDEN_DOCTRINAL_FACTS

        for subj, pred, obj in GOLDEN_DOCTRINAL_FACTS:
            vr = validate_triplet(subj, pred, obj)
            assert vr.valid, f"Failed: ({subj}, {pred}, {obj[:50]}...) — {vr.reasons()}"

    def test_all_doctrinal_facts_generate_questions(self):
        from tests.golden.test_cases import GOLDEN_DOCTRINAL_FACTS

        qs = generate_questions_from_triplets(
            GOLDEN_DOCTRINAL_FACTS, max_questions=20,
        )
        assert len(qs) >= len(GOLDEN_DOCTRINAL_FACTS)


# ── Object length cap tests ─────────────────────────────────────────────


class TestObjectLengthCap:
    """Verify that question_gen.py raises the object length cap for doctrinal predicates."""

    def test_long_holding_not_skipped(self):
        from crystal.ingest.question_gen import _LONG_PREDICATES

        assert "holding" in _LONG_PREDICATES
        assert "doctrine" in _LONG_PREDICATES
        assert "reasoning" in _LONG_PREDICATES

    def test_tier1_generates_long_holding_question(self):
        from crystal.tools.kg.store import SqliteKnowledgeGraph
        from crystal.ingest.question_gen import generate_tier1

        kg = SqliteKnowledgeGraph(":memory:")
        long_holding = "A" * 500
        kg.bulk_insert([
            ("miranda v. arizona", "holding", long_holding),
            ("miranda v. arizona", "court", "Supreme Court of the United States"),
        ])
        cases = generate_tier1(kg)
        preds = {c.source_triplet[1] for c in cases if c.source_triplet}
        assert "holding" in preds


# ── LLM question generation tests ───────────────────────────────────────


def _mock_llm_for_questions(prompt: str):
    """Mock LLM that returns plausible question JSON."""
    return json.dumps([
        {
            "question": "What did the court hold in Miranda V. Arizona?",
            "golden_answer": "Prosecution may not use statements from custodial interrogation",
        },
        {
            "question": "What court decided Miranda V. Arizona?",
            "golden_answer": "Supreme Court of the United States",
        },
    ]), {"prompt_tokens": 100, "output_tokens": 50}


def _mock_llm_garbage(prompt: str):
    """Mock LLM that returns garbage."""
    return "I don't understand the question.", None


class TestParseQuestionResponse:
    def test_parses_valid_json(self):
        text = json.dumps([
            {"question": "Q1?", "golden_answer": "A1"},
            {"question": "Q2?", "golden_answer": "A2"},
        ])
        result = _parse_question_response(text)
        assert len(result) == 2
        assert result[0]["question"] == "Q1?"
        assert result[1]["golden_answer"] == "A2"

    def test_parses_json_with_markdown_fences(self):
        text = '```json\n[{"question": "Q1?", "golden_answer": "A1"}]\n```'
        result = _parse_question_response(text)
        assert len(result) == 1

    def test_returns_empty_for_garbage(self):
        assert _parse_question_response("not json at all") == []

    def test_filters_incomplete_items(self):
        text = json.dumps([
            {"question": "Q1?", "golden_answer": "A1"},
            {"question": "", "golden_answer": "A2"},
            {"question": "Q3?"},
        ])
        result = _parse_question_response(text)
        assert len(result) == 1

    def test_handles_wrapped_json(self):
        text = 'Here are the questions:\n[{"question": "Q?", "golden_answer": "A"}]\nDone.'
        result = _parse_question_response(text)
        assert len(result) == 1


class TestGenerateQuestionsLlm:
    def test_generates_questions_with_mock(self):
        triplets = [
            ("miranda v. arizona", "holding",
             "Prosecution may not use statements from custodial interrogation"),
            ("miranda v. arizona", "court",
             "Supreme Court of the United States"),
        ]
        results = generate_questions_llm(
            triplets, _mock_llm_for_questions, max_questions=5,
        )
        assert len(results) >= 1
        assert all("question" in r for r in results)
        assert all("golden_answer" in r for r in results)
        assert all("source_triplet" in r for r in results)

    def test_falls_back_to_templates_on_garbage(self):
        triplets = [
            ("miranda v. arizona", "court",
             "Supreme Court of the United States"),
        ]
        results = generate_questions_llm(
            triplets, _mock_llm_garbage, max_questions=5,
        )
        assert len(results) >= 1
        assert "court" in results[0]["question"].lower()

    def test_respects_max_questions(self):
        triplets = [
            (f"case{i} v. state", "court", "Court") for i in range(10)
        ]
        results = generate_questions_llm(
            triplets, _mock_llm_for_questions, max_questions=2,
        )
        assert len(results) <= 2

    def test_filters_bad_subjects(self):
        triplets = [
            ("this case", "court", "Supreme Court"),
            ("defendant", "holding", "Some holding"),
        ]
        results = generate_questions_llm(
            triplets, _mock_llm_for_questions, max_questions=5,
        )
        assert len(results) == 0

    def test_empty_triplets(self):
        results = generate_questions_llm([], _mock_llm_for_questions)
        assert results == []

    def test_source_triplet_is_list(self):
        triplets = [
            ("miranda v. arizona", "court",
             "Supreme Court of the United States"),
        ]
        results = generate_questions_llm(
            triplets, _mock_llm_for_questions, max_questions=5,
        )
        for r in results:
            assert isinstance(r["source_triplet"], list)


# ── Template fallback in question_gen.py ─────────────────────────────────


class TestQuestionGenTemplateFallback:
    """generate_all() uses templates when no LLM is provided."""

    def test_generate_all_without_llm(self):
        from crystal.tools.kg.store import SqliteKnowledgeGraph
        from crystal.ingest.question_gen import generate_all

        kg = SqliteKnowledgeGraph(":memory:")
        kg.bulk_insert([
            ("miranda v. arizona", "court", "Supreme Court of the United States"),
            ("miranda v. arizona", "date_filed", "1966-06-13"),
        ])
        cases = generate_all(kg)
        tier1 = [c for c in cases if c.tier == 1 and not c.is_negative]
        assert len(tier1) >= 1

    def test_generate_all_with_mock_llm(self):
        from crystal.tools.kg.store import SqliteKnowledgeGraph
        from crystal.ingest.question_gen import generate_all

        kg = SqliteKnowledgeGraph(":memory:")
        kg.bulk_insert([
            ("miranda v. arizona", "court", "Supreme Court of the United States"),
            ("miranda v. arizona", "holding", "Procedural safeguards required"),
        ])
        cases = generate_all(kg, call_llm_fn=_mock_llm_for_questions)
        tier1 = [c for c in cases if c.tier == 1 and not c.is_negative]
        assert len(tier1) >= 1


# ── QuestionGenLoop tests ───────────────────────────────────────────────


class TestQuestionGenLoop:
    @pytest.fixture
    def loop_setup(self):
        from crystal.tools.kg.store import SqliteKnowledgeGraph
        from benchmarks.ralph_wiggum.question_gen_loop import QuestionGenLoop

        kg = SqliteKnowledgeGraph(":memory:")
        kg.bulk_insert([
            ("miranda v. arizona", "court", "Supreme Court of the United States"),
            ("miranda v. arizona", "date_filed", "1966-06-13"),
            ("miranda v. arizona", "holding", "Procedural safeguards required"),
        ])
        triplets = [
            ("miranda v. arizona", "court", "Supreme Court of the United States"),
            ("miranda v. arizona", "date_filed", "1966-06-13"),
        ]
        loop = QuestionGenLoop(
            kg=kg,
            cases=[],
            sample_triplets=triplets,
            call_llm_fn=_mock_llm_for_questions,
            use_full_pipeline=True,
        )
        return loop

    def test_validate_proposal_valid(self, loop_setup):
        proposal = {
            "question_gen_prompt": "Generate {max_questions} questions about {subject}.\n{facts}\nJSON:",
        }
        assert loop_setup._validate_proposal(proposal) is True

    def test_validate_proposal_missing_placeholders(self, loop_setup):
        proposal = {
            "question_gen_prompt": "Generate some questions please.",
        }
        assert loop_setup._validate_proposal(proposal) is False

    def test_validate_proposal_empty(self, loop_setup):
        assert loop_setup._validate_proposal({}) is False
        assert loop_setup._validate_proposal(None) is False

    def test_validate_proposal_too_short(self, loop_setup):
        proposal = {"question_gen_prompt": "{subject} {facts}"}
        assert loop_setup._validate_proposal(proposal) is False

    def test_build_proposal_prompt(self, loop_setup):
        failures = [{
            "question": "What did Miranda hold?",
            "golden_answer": "Procedural safeguards",
            "crystal_response": "I don't know",
            "response": "I don't know",
            "source_triplet": ["miranda v. arizona", "holding", "Procedural safeguards"],
            "diagnosis": "question_quality",
        }]
        prompt = loop_setup._build_proposal_prompt(failures)
        assert "QUESTION_GEN_PROMPT" in prompt
        assert "Miranda" in prompt
        assert "question_gen_prompt" in prompt

    def test_apply_and_revert_proposal(self, loop_setup, tmp_path):
        from benchmarks.ralph_wiggum.question_gen_loop import (
            _COMPARE_FILE,
            _read_question_gen_prompt,
            _write_question_gen_prompt,
        )
        import shutil

        original = _read_question_gen_prompt()
        assert len(original) > 50

        new_prompt = (
            "New improved prompt for {subject} with {facts}. "
            "Generate {max_questions} questions. JSON:"
        )
        proposal = {"question_gen_prompt": new_prompt}

        loop_setup._apply_proposal(proposal)
        current = _read_question_gen_prompt()
        assert "New improved prompt" in current

        loop_setup._revert_proposal(proposal)
        reverted = _read_question_gen_prompt()
        assert original == reverted

    def test_registered_in_orchestrator(self):
        from benchmarks.ralph_wiggum.orchestrator import Orchestrator
        from benchmarks.ralph_wiggum.question_gen_loop import QuestionGenLoop

        loop_names = [cls.LOOP_NAME for cls in Orchestrator.LOOP_CLASSES]
        assert "QuestionGenLoop" in loop_names

    def test_no_questions_generated_reports_failure_not_vacuous_success(self):
        """When 0 questions are generated the loop must fail loudly.

        Regression: previously returned score=1.0/total=0 which the
        orchestrator treated as a perfect pass and never triggered mutation.
        """
        from crystal.tools.kg.store import SqliteKnowledgeGraph
        from benchmarks.ralph_wiggum.question_gen_loop import (
            QuestionGenLoop,
            NO_QUESTIONS_GENERATED,
        )

        kg = SqliteKnowledgeGraph(":memory:")

        def _llm_returns_nothing(_prompt):
            return "[]", None

        loop = QuestionGenLoop(
            kg=kg,
            cases=[],
            sample_triplets=[("this case", "court", "scotus")],
            call_llm_fn=_llm_returns_nothing,
            use_full_pipeline=False,
        )
        result = loop.run_iteration(iteration=0)
        assert result.score == 0.0
        assert result.total >= 1
        assert result.failures
        assert result.failures[0]["diagnosis"] == NO_QUESTIONS_GENERATED


# ── QUESTION_GEN_PROMPT format tests ────────────────────────────────────


class TestQuestionGenPrompt:
    def test_prompt_has_required_placeholders(self):
        assert "{subject}" in QUESTION_GEN_PROMPT
        assert "{facts}" in QUESTION_GEN_PROMPT
        assert "{max_questions}" in QUESTION_GEN_PROMPT

    def test_prompt_can_be_formatted(self):
        result = QUESTION_GEN_PROMPT.format(
            subject="Test Case",
            facts="- court: Supreme Court",
            max_questions=3,
        )
        assert "Test Case" in result
        assert "Supreme Court" in result
