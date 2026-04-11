"""Tests for B3: document-context baseline runner."""

from __future__ import annotations

import pytest

from benchmarks.runners.document import (
    _build_document_prompt,
    _extract_case_name,
    _MAX_DOC_CHARS,
    run_document_baseline,
)


class TestExtractCaseName:
    def test_standard_v_dot(self):
        assert _extract_case_name("What court decided Miranda v. Arizona?") == "Miranda v. Arizona"

    def test_inc_suffix(self):
        name = _extract_case_name(
            "What court decided Coventry Health Care of Mo., Inc. v. Nevils?"
        )
        assert name is not None
        assert "Coventry" in name
        assert "Nevils" in name

    def test_citation_format(self):
        name = _extract_case_name("What court decided 384 U.S. 436?")
        assert name == "384 U.S. 436"

    def test_no_case_name(self):
        assert _extract_case_name("What is the weather today?") is None

    def test_complex_name(self):
        name = _extract_case_name(
            "When was Brown v. Board of Education decided?"
        )
        assert name is not None
        assert "Brown" in name


class TestBuildDocumentPrompt:
    def test_includes_opinion_text(self):
        prompt = _build_document_prompt("The court held that...", "What was decided?")
        assert "The court held that..." in prompt
        assert "What was decided?" in prompt

    def test_truncation(self):
        long_text = "x" * (_MAX_DOC_CHARS + 1000)
        prompt = _build_document_prompt(long_text, "Question?")
        assert "[Document truncated" in prompt
        assert len(prompt) < _MAX_DOC_CHARS + 500

    def test_prompt_structure(self):
        prompt = _build_document_prompt("opinion text", "the question")
        assert prompt.startswith("Here is the full text")
        assert "---" in prompt
        assert "opinion text" in prompt
        assert "the question" in prompt


class TestRunDocumentBaseline:
    def _make_mock_llm(self, responses: dict[str, str] | None = None):
        """Return a mock call_llm that returns canned responses."""
        default = "The court was the Supreme Court of the United States"

        def mock_llm(prompt: str, **kwargs):
            for key, val in (responses or {}).items():
                if key in prompt:
                    return val, {"prompt_tokens": len(prompt) // 4}
            return default, {"prompt_tokens": len(prompt) // 4}
        return mock_llm

    def test_basic_run(self):
        cases = [
            ("What court decided Miranda v. Arizona?", "SCOTUS", ["supreme court"], False),
        ]
        opinions = {
            "miranda v. arizona": "This is the full opinion text for Miranda...",
        }
        results = run_document_baseline(
            cases, opinions,
            call_llm_fn=self._make_mock_llm(),
            sleep_between=0,
        )
        assert len(results) == 1
        assert results[0]["prompt_source"] == "document"
        assert results[0]["prompt_chars"] > 100

    def test_missing_document_falls_back(self):
        cases = [
            ("What court decided Unknown v. Case?", "SCOTUS", ["supreme court"], False),
        ]
        results = run_document_baseline(
            cases, {},
            call_llm_fn=self._make_mock_llm(),
            sleep_between=0,
        )
        assert results[0]["prompt_source"] == "no_document"

    def test_result_dict_shape(self):
        cases = [
            ("When was Miranda v. Arizona decided?", "1966", ["1966"], False),
        ]
        opinions = {"miranda v. arizona": "Opinion filed June 13, 1966..."}
        results = run_document_baseline(
            cases, opinions,
            call_llm_fn=self._make_mock_llm({"1966": "June 13, 1966"}),
            sleep_between=0,
        )
        r = results[0]
        assert "question" in r
        assert "ground_truth" in r
        assert "match_strings" in r
        assert "is_negative" in r
        assert "response" in r
        assert "prompt_tokens_estimate" in r

    def test_handles_llm_error(self):
        def failing_llm(prompt, **kw):
            raise RuntimeError("API down")

        cases = [
            ("What court decided Miranda v. Arizona?", "SCOTUS", ["supreme court"], False),
        ]
        results = run_document_baseline(
            cases, {},
            call_llm_fn=failing_llm,
            sleep_between=0,
        )
        assert "[ERROR" in results[0]["response"]

    def test_multiple_cases(self):
        cases = [
            ("What court decided Miranda v. Arizona?", "SCOTUS", ["supreme court"], False),
            ("When was Roe v. Wade decided?", "1973", ["1973"], False),
        ]
        opinions = {
            "miranda v. arizona": "Miranda opinion...",
            "roe v. wade": "Roe opinion, January 22, 1973...",
        }
        results = run_document_baseline(
            cases, opinions,
            call_llm_fn=self._make_mock_llm(),
            sleep_between=0,
        )
        assert len(results) == 2
        assert all(r["prompt_source"] == "document" for r in results)
