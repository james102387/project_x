"""Unit tests for the LLM-assisted triplet extractor (D2 Phase 2)."""

import json
import pytest

from crystal.ingest.llm_extract import (
    _format_sentences,
    _get_prompt,
    _parse_llm_response,
    extract_triplets_llm,
    normalize_predicate,
    EXTRACTION_PROMPT,
    LEGAL_EXTRACTION_PROMPT,
)
from crystal.ingest.schema import ReviewableTriplet, LLMExtractionResult


# ── Prompt formatting ────────────────────────────────────────────────


class TestFormatSentences:
    def test_single_sentence(self):
        result = _format_sentences([("Remulak borders Draveth.", ["Remulak", "Draveth"])])
        assert '"Remulak borders Draveth."' in result
        assert "entities: Remulak, Draveth" in result
        assert result.startswith("1.")

    def test_multiple_sentences(self):
        sents = [
            ("A borders B.", ["A", "B"]),
            ("C exports D.", ["C", "D"]),
        ]
        result = _format_sentences(sents)
        assert "1." in result
        assert "2." in result

    def test_empty(self):
        result = _format_sentences([])
        assert result == ""

    def test_prompt_template_formats(self):
        sents = [("Test sentence.", ["Entity"])]
        formatted = _format_sentences(sents)
        prompt = EXTRACTION_PROMPT.format(sentences=formatted)
        assert "Test sentence." in prompt
        assert "JSON array" in prompt


# ── Response parsing ─────────────────────────────────────────────────


class TestParseLlmResponse:
    def _sentences(self):
        return [
            ("Remulak borders Draveth.", ["Remulak", "Draveth"]),
            ("Sulari exports minerals.", ["Sulari"]),
        ]

    def test_valid_json_array(self):
        response = json.dumps([
            {"subject": "Remulak", "predicate": "borders", "object": "Draveth",
             "confidence": "high", "sentence_index": 1},
        ])
        result = _parse_llm_response(response, self._sentences())
        assert len(result) == 1
        assert result[0].subject == "Remulak"
        assert result[0].predicate == "borders"
        assert result[0].object == "Draveth"
        assert result[0].confidence == "high"
        assert result[0].source_sentence == "Remulak borders Draveth."
        assert result[0].status == "pending_review"

    def test_markdown_code_fence(self):
        response = '```json\n[{"subject": "A", "predicate": "b", "object": "C", "confidence": "medium", "sentence_index": 1}]\n```'
        result = _parse_llm_response(response, self._sentences())
        assert len(result) == 1
        assert result[0].subject == "A"

    def test_multiple_triplets(self):
        response = json.dumps([
            {"subject": "Remulak", "predicate": "borders", "object": "Draveth",
             "confidence": "high", "sentence_index": 1},
            {"subject": "Sulari", "predicate": "exports", "object": "minerals",
             "confidence": "medium", "sentence_index": 2},
        ])
        result = _parse_llm_response(response, self._sentences())
        assert len(result) == 2
        assert result[1].source_sentence == "Sulari exports minerals."

    def test_missing_fields_skipped(self):
        response = json.dumps([
            {"subject": "A"},
            {"subject": "B", "predicate": "c", "object": "D", "confidence": "high", "sentence_index": 1},
        ])
        result = _parse_llm_response(response, self._sentences())
        assert len(result) == 1
        assert result[0].subject == "B"

    def test_invalid_confidence_defaults_medium(self):
        response = json.dumps([
            {"subject": "A", "predicate": "b", "object": "C",
             "confidence": "ultra", "sentence_index": 1},
        ])
        result = _parse_llm_response(response, self._sentences())
        assert result[0].confidence == "medium"

    def test_invalid_sentence_index(self):
        response = json.dumps([
            {"subject": "A", "predicate": "b", "object": "C",
             "confidence": "high", "sentence_index": 99},
        ])
        result = _parse_llm_response(response, self._sentences())
        assert result[0].source_sentence == ""

    def test_missing_sentence_index(self):
        response = json.dumps([
            {"subject": "A", "predicate": "b", "object": "C", "confidence": "high"},
        ])
        result = _parse_llm_response(response, self._sentences())
        assert result[0].source_sentence == ""

    def test_empty_response(self):
        result = _parse_llm_response("", self._sentences())
        assert result == []

    def test_non_json_response(self):
        result = _parse_llm_response("I cannot extract triplets.", self._sentences())
        assert result == []

    def test_json_with_surrounding_text(self):
        response = 'Here are the results:\n[{"subject": "X", "predicate": "y", "object": "Z", "confidence": "low", "sentence_index": 1}]\nDone.'
        result = _parse_llm_response(response, self._sentences())
        assert len(result) == 1
        assert result[0].confidence == "low"

    def test_non_dict_items_skipped(self):
        response = json.dumps(["not a dict", {"subject": "A", "predicate": "b", "object": "C", "confidence": "high", "sentence_index": 1}])
        result = _parse_llm_response(response, self._sentences())
        assert len(result) == 1

    def test_whitespace_stripped(self):
        response = json.dumps([
            {"subject": "  Remulak  ", "predicate": "  borders  ", "object": "  Draveth  ",
             "confidence": "high", "sentence_index": 1},
        ])
        result = _parse_llm_response(response, self._sentences())
        assert result[0].subject == "Remulak"
        assert result[0].predicate == "borders"
        assert result[0].object == "Draveth"


# ── End-to-end extraction with mock LLM ──────────────────────────────


class TestExtractTripletsLlm:
    def _mock_llm(self, response_text):
        """Return a mock call_llm_fn that returns the given text."""
        def _call(prompt):
            return response_text, {"prompt_tokens": 100, "output_tokens": 50}
        return _call

    def test_basic_extraction(self):
        sentences = [("Remulak borders Draveth.", ["Remulak", "Draveth"])]
        response = json.dumps([
            {"subject": "Remulak", "predicate": "borders", "object": "Draveth",
             "confidence": "high", "sentence_index": 1},
        ])
        result = extract_triplets_llm(sentences, call_llm_fn=self._mock_llm(response))
        assert isinstance(result, LLMExtractionResult)
        assert len(result.reviewable) == 1
        assert result.reviewable[0].subject == "Remulak"
        assert result.skipped_sentences == []

    def test_empty_sentences(self):
        result = extract_triplets_llm([], call_llm_fn=self._mock_llm("[]"))
        assert len(result.reviewable) == 0
        assert result.skipped_sentences == []

    def test_llm_returns_no_triplets(self):
        sentences = [("Something unclear happened.", ["Something"])]
        result = extract_triplets_llm(sentences, call_llm_fn=self._mock_llm("[]"))
        assert len(result.reviewable) == 0
        assert len(result.skipped_sentences) == 1
        assert "Something unclear happened." in result.skipped_sentences

    def test_llm_error_adds_to_skipped(self):
        def _failing_llm(prompt):
            raise RuntimeError("API error")
        sentences = [("Remulak borders Draveth.", ["Remulak", "Draveth"])]
        result = extract_triplets_llm(sentences, call_llm_fn=_failing_llm)
        assert len(result.reviewable) == 0
        assert len(result.skipped_sentences) == 1

    def test_batching(self):
        sentences = [(f"Sentence {i}.", [f"Entity{i}"]) for i in range(5)]
        call_count = 0

        def _counting_llm(prompt):
            nonlocal call_count
            call_count += 1
            return "[]", None

        extract_triplets_llm(sentences, call_llm_fn=_counting_llm, batch_size=2)
        assert call_count == 3  # ceil(5/2) = 3 batches

    def test_all_reviewable_are_pending(self):
        sentences = [("A borders B.", ["A", "B"])]
        response = json.dumps([
            {"subject": "A", "predicate": "borders", "object": "B",
             "confidence": "high", "sentence_index": 1},
            {"subject": "A", "predicate": "near", "object": "C",
             "confidence": "low", "sentence_index": 1},
        ])
        result = extract_triplets_llm(sentences, call_llm_fn=self._mock_llm(response))
        for rt in result.reviewable:
            assert rt.status == "pending_review"

    def test_prompt_includes_entities(self):
        captured = {}

        def _capture_llm(prompt):
            captured["prompt"] = prompt
            return "[]", None

        sentences = [("X relates to Y.", ["Alpha", "Beta"])]
        extract_triplets_llm(sentences, call_llm_fn=_capture_llm)
        assert "Alpha" in captured["prompt"]
        assert "Beta" in captured["prompt"]
        assert "X relates to Y." in captured["prompt"]

    def test_legal_domain_uses_legal_prompt(self):
        captured = {}

        def _capture_llm(prompt):
            captured["prompt"] = prompt
            return "[]", None

        sentences = [("The Court held in Miranda v. Arizona.", ["Miranda v. Arizona"])]
        extract_triplets_llm(sentences, call_llm_fn=_capture_llm, domain="legal")
        assert "PREFERRED PREDICATES" in captured["prompt"]
        assert "court" in captured["prompt"]
        assert "opinion_author" in captured["prompt"]

    def test_general_domain_uses_generic_prompt(self):
        captured = {}

        def _capture_llm(prompt):
            captured["prompt"] = prompt
            return "[]", None

        sentences = [("X borders Y.", ["X", "Y"])]
        extract_triplets_llm(sentences, call_llm_fn=_capture_llm, domain="general")
        assert "PREFERRED PREDICATES" not in captured["prompt"]


# ── Prompt selection ─────────────────────────────────────────────────


class TestGetPrompt:
    def test_legal_returns_legal_prompt(self):
        prompt = _get_prompt("legal")
        assert "PREFERRED PREDICATES" in prompt

    def test_general_returns_generic_prompt(self):
        prompt = _get_prompt("general")
        assert prompt is EXTRACTION_PROMPT

    def test_unknown_returns_generic(self):
        prompt = _get_prompt("science")
        assert prompt is EXTRACTION_PROMPT


# ── Predicate normalization ──────────────────────────────────────────


LEGAL_PREDS = {"court", "date_filed", "judges", "cites", "opinion_author", "attorneys"}
LEGAL_ALIASES = {"decided by": "court", "filed on": "date_filed", "who argued": "attorneys"}


class TestNormalizePredicate:
    def test_exact_match(self):
        assert normalize_predicate("court", LEGAL_PREDS) == "court"

    def test_alias_match(self):
        assert normalize_predicate(
            "decided by", ontology_predicates=LEGAL_PREDS,
            predicate_aliases=LEGAL_ALIASES,
        ) == "court"

    def test_substring_containment(self):
        assert normalize_predicate("date_filed", LEGAL_PREDS) == "date_filed"

    def test_case_insensitive(self):
        assert normalize_predicate("Court", LEGAL_PREDS) == "court"

    def test_no_match_returns_raw(self):
        assert normalize_predicate("holding", LEGAL_PREDS) == "holding"

    def test_empty_returns_empty(self):
        assert normalize_predicate("", LEGAL_PREDS) == ""

    def test_alias_preferred_over_substring(self):
        assert normalize_predicate(
            "who argued", ontology_predicates=LEGAL_PREDS,
            predicate_aliases=LEGAL_ALIASES,
        ) == "attorneys"

    def test_no_ontology_returns_lowered(self):
        assert normalize_predicate("CUSTOM") == "custom"
