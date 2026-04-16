"""Tests for the augmented output quality benchmark (D4)."""

import pytest

from benchmarks.ground_truth.remulak import AUGMENTED_BENCHMARK_CASES
from benchmarks.runners.augmented import score_results, summarize_scored


# ── Ground truth sanity ───────────────────────────────────────────────────


class TestAugmentedCasesFormat:
    """Verify AUGMENTED_BENCHMARK_CASES are well-formed."""

    def test_non_empty(self):
        assert len(AUGMENTED_BENCHMARK_CASES) > 0

    def test_tuple_format(self):
        for case in AUGMENTED_BENCHMARK_CASES:
            assert len(case) == 4, f"Expected 4-tuple, got {len(case)}: {case[0][:40]}"
            question, ground_truth, match_strings, is_negative = case
            assert isinstance(question, str) and question
            assert isinstance(ground_truth, str) and ground_truth
            assert isinstance(match_strings, list)
            assert len(match_strings) > 0, f"Augmented case needs match_strings: {question}"
            assert is_negative is False, f"Augmented cases should not be negative: {question}"

    def test_has_kg_augmented_cases(self):
        kg_keywords = ["remulak", "korth", "draveth", "sulari", "festival", "vohn"]
        kg_count = sum(
            1 for q, *_ in AUGMENTED_BENCHMARK_CASES
            if any(k in q.lower() for k in kg_keywords)
        )
        assert kg_count >= 3, "Need at least 3 KG augmented cases"

    def test_has_math_augmented_cases(self):
        math_keywords = ["earned", "spent", "shares", "sold", "started", "lost"]
        math_count = sum(
            1 for q, *_ in AUGMENTED_BENCHMARK_CASES
            if any(k in q.lower() for k in math_keywords)
        )
        assert math_count >= 3, "Need at least 3 math augmented cases"


# ── Scoring ───────────────────────────────────────────────────────────────


class TestScoreResults:
    """Test score_results with synthetic responses."""

    def _make_result(self, question, match_strings, response, kg_results=None):
        return {
            "question": question,
            "ground_truth": "expected",
            "match_strings": match_strings,
            "response": response,
            "kg_results": kg_results or [],
        }

    def test_correct_response_scores_well(self):
        results = [self._make_result(
            "What?", ["zelphos"],
            "The capital is Zelphos",
            [{"subject": "Remulak", "predicate": "capital", "object": "Zelphos"}],
        )]
        scored = score_results(results)
        assert scored[0]["correct"] is True
        assert scored[0]["rubric"]["accuracy"] == 1.0

    def test_wrong_response_scores_poorly(self):
        results = [self._make_result(
            "What?", ["zelphos"],
            "I have no idea about that planet",
            [{"subject": "Remulak", "predicate": "capital", "object": "Zelphos"}],
        )]
        scored = score_results(results)
        assert scored[0]["correct"] is False
        assert scored[0]["rubric"]["accuracy"] == 0.0

    def test_partial_accuracy(self):
        results = [self._make_result(
            "What?", ["agriculture", "bioengineering"],
            "Khotane is known for agriculture",
            [{"subject": "Khotane", "predicate": "known for",
              "object": "agriculture and bioengineering"}],
        )]
        scored = score_results(results)
        assert scored[0]["rubric"]["accuracy"] == pytest.approx(0.5)

    def test_kg_results_from_treatment_fallback(self):
        """Baseline results use treatment's KG results for rubric scoring."""
        results = [self._make_result(
            "What is the capital?", ["zelphos"],
            "Zelphos is the capital of Remulak",
        )]
        treatment_kg = {
            "What is the capital?": [
                {"subject": "Remulak", "predicate": "capital", "object": "Zelphos"},
            ],
        }
        scored = score_results(results, kg_results_from_treatment=treatment_kg)
        assert scored[0]["rubric"]["accuracy"] == 1.0

    def test_empty_results(self):
        scored = score_results([])
        assert scored == []


# ── Summarize ─────────────────────────────────────────────────────────────


class TestSummarizeScored:
    def test_perfect_summary(self):
        scored = [
            {"correct": True, "rubric": {"accuracy": 1.0, "abstention": 1.0}},
            {"correct": True, "rubric": {"accuracy": 1.0, "abstention": 1.0}},
        ]
        s = summarize_scored(scored)
        assert s["count"] == 2
        assert s["correct"] == 2
        assert s["accuracy"] == 1.0
        assert s["rubric_averages"]["abstention"] == pytest.approx(1.0)

    def test_mixed_summary(self):
        scored = [
            {"correct": True, "rubric": {"accuracy": 1.0, "abstention": 1.0}},
            {"correct": False, "rubric": {"accuracy": 0.0, "abstention": 0.5}},
        ]
        s = summarize_scored(scored)
        assert s["correct"] == 1
        assert s["accuracy"] == 0.5
        assert s["rubric_averages"]["accuracy"] == pytest.approx(0.5)
        assert s["rubric_averages"]["abstention"] == pytest.approx(0.75)

    def test_empty_summary(self):
        s = summarize_scored([])
        assert s["count"] == 0
        assert s["accuracy"] == 0.0


# ── Pipeline routing (integration, no LLM) ───────────────────────────────


class TestAugmentedRouting:
    """Verify augmented cases route to augmented paths (no LLM call needed)."""

    @pytest.fixture(autouse=True, scope="class")
    def _setup_pipeline(self, request):
        """Build pipeline and spaCy model once for all tests in this class."""
        import spacy
        from crystal.detectors.kg import detect_kg_query
        from crystal.nodes.compiler import _classify_prompt_type, _has_reasoning_signals
        from crystal.tools.kg import remulak_kg

        request.cls.nlp = spacy.load("en_core_web_sm")
        request.cls.remulak_kg = remulak_kg
        request.cls.detect_kg_query = staticmethod(detect_kg_query)
        request.cls._classify_prompt_type = staticmethod(_classify_prompt_type)
        request.cls._has_reasoning_signals = staticmethod(_has_reasoning_signals)

    @pytest.mark.parametrize("question,ground_truth,match_strings,is_negative", [
        c for c in AUGMENTED_BENCHMARK_CASES
        if any(k in c[0].lower() for k in
               ["remulak", "korth", "draveth", "sulari", "festival", "vohn"])
    ], ids=lambda c: c[:40] if isinstance(c, str) else None)
    def test_kg_cases_route_to_augmented(self, question, ground_truth, match_strings, is_negative):
        doc = self.nlp(question)
        detection = self.detect_kg_query(doc, self.remulak_kg)
        assert detection is not None, f"No KG detection for: {question}"
        tool_results = [{
            "tool": "kg", "operation": "lookup",
            "entity": detection["entity"],
            "results": detection["results"],
            "lookup_type": detection.get("lookup_type", "subject_scan"),
            "success": True,
        }]
        prompt_type = self._classify_prompt_type(question, doc, tool_results)
        assert prompt_type == "kg_augmented", f"Expected kg_augmented, got {prompt_type}"

    @pytest.mark.parametrize("question,ground_truth,match_strings,is_negative", [
        c for c in AUGMENTED_BENCHMARK_CASES
        if any(k in c[0].lower() for k in ["earned", "spent", "shares", "sold", "started", "lost"])
    ], ids=lambda c: c[:40] if isinstance(c, str) else None)
    def test_math_cases_have_reasoning_signals(self, question, ground_truth, match_strings, is_negative):
        doc = self.nlp(question)
        assert self._has_reasoning_signals(doc), f"No reasoning signal in: {question}"
