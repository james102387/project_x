"""Tests for Ralph Wiggum v3 — mutation validation, application, and thresholds."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest


class TestValidateProposal:
    """Test that LLM proposals are validated before application."""

    def test_accepts_valid_predicate_map_addition(self):
        from benchmarks.ralph_wiggum import _validate_proposal

        proposal = {"predicate_map": {"argued": "attorneys"}}
        assert _validate_proposal(proposal) is True

    def test_accepts_valid_alias_addition(self):
        from benchmarks.ralph_wiggum import _validate_proposal

        proposal = {"predicate_aliases": {"who argued": "attorneys"}}
        assert _validate_proposal(proposal) is True

    def test_accepts_entity_alias_addition(self):
        from benchmarks.ralph_wiggum import _validate_proposal

        proposal = {"entity_aliases": {"miranda": "miranda v. arizona"}}
        assert _validate_proposal(proposal) is True

    def test_accepts_valid_threshold_change(self):
        from benchmarks.ralph_wiggum import _validate_proposal

        proposal = {"confidence_threshold": 0.75}
        assert _validate_proposal(proposal) is True

    def test_rejects_threshold_out_of_bounds(self):
        from benchmarks.ralph_wiggum import _validate_proposal

        assert _validate_proposal({"confidence_threshold": 0.1}) is False
        assert _validate_proposal({"confidence_threshold": 0.99}) is False

    def test_accepts_null_threshold(self):
        from benchmarks.ralph_wiggum import _validate_proposal

        proposal = {
            "predicate_map": {"test": "court"},
            "confidence_threshold": None,
        }
        assert _validate_proposal(proposal) is True

    def test_rejects_empty_proposal(self):
        from benchmarks.ralph_wiggum import _validate_proposal

        assert _validate_proposal({}) is False
        assert _validate_proposal(None) is False

    def test_rejects_non_string_keys(self):
        from benchmarks.ralph_wiggum import _validate_proposal

        proposal = {"predicate_map": {123: "attorneys"}}
        assert _validate_proposal(proposal) is False

    def test_rejects_non_string_values(self):
        from benchmarks.ralph_wiggum import _validate_proposal

        proposal = {"predicate_map": {"argued": 123}}
        assert _validate_proposal(proposal) is False

    def test_rejects_unknown_sections(self):
        from benchmarks.ralph_wiggum import _validate_proposal

        proposal = {"delete_everything": True}
        assert _validate_proposal(proposal) is False


class TestApplyProposal:
    """Test that proposals are written correctly to the target files."""

    def test_apply_predicate_map_entries(self, tmp_path):
        from benchmarks.ralph_wiggum import _apply_proposal

        kg_file = tmp_path / "kg.py"
        kg_file.write_text(textwrap.dedent('''\
            QUESTION_PREDICATE_MAP = {
                "old": "age",
                "filed": "date_filed",
            }
        '''))

        proposal = {"predicate_map": {"argued": "attorneys"}}
        _apply_proposal(proposal, predicate_map_file=kg_file)

        content = kg_file.read_text()
        assert '"argued": "attorneys"' in content
        assert '"old": "age"' in content

    def test_apply_alias_entries(self, tmp_path):
        from benchmarks.ralph_wiggum import _apply_proposal

        ontology_file = tmp_path / "ontology.py"
        ontology_file.write_text(textwrap.dedent('''\
            LEGAL_PREDICATE_ALIASES: dict[str, str] = {
                "references": "cites",
            }
        '''))

        proposal = {"predicate_aliases": {"who argued": "attorneys"}}
        _apply_proposal(proposal, alias_file=ontology_file)

        content = ontology_file.read_text()
        assert '"who argued": "attorneys"' in content
        assert '"references": "cites"' in content

    def test_does_not_duplicate_existing_entries(self, tmp_path):
        from benchmarks.ralph_wiggum import _apply_proposal

        kg_file = tmp_path / "kg.py"
        kg_file.write_text(textwrap.dedent('''\
            QUESTION_PREDICATE_MAP = {
                "old": "age",
                "filed": "date_filed",
            }
        '''))

        proposal = {"predicate_map": {"old": "age"}}
        _apply_proposal(proposal, predicate_map_file=kg_file)

        content = kg_file.read_text()
        assert content.count('"old"') == 1


class TestThresholdUpdate:
    def test_update_threshold_in_file(self, tmp_path):
        from benchmarks.ralph_wiggum import _update_threshold, _PLANNER_FILE

        backup = _PLANNER_FILE.read_text()
        try:
            _update_threshold(0.75)
            content = _PLANNER_FILE.read_text()
            assert "CONFIDENCE_LOW = 0.75" in content
        finally:
            _PLANNER_FILE.write_text(backup)


class TestParseLLMProposal:
    """Test parsing of LLM response into structured proposal."""

    def test_parses_json_from_llm(self):
        from benchmarks.ralph_wiggum import _parse_llm_proposal

        raw = '```json\n{"predicate_map": {"argued": "attorneys"}}\n```'
        proposal = _parse_llm_proposal(raw)
        assert proposal == {"predicate_map": {"argued": "attorneys"}}

    def test_parses_bare_json(self):
        from benchmarks.ralph_wiggum import _parse_llm_proposal

        raw = '{"predicate_aliases": {"who argued": "attorneys"}}'
        proposal = _parse_llm_proposal(raw)
        assert proposal == {"predicate_aliases": {"who argued": "attorneys"}}

    def test_returns_none_on_garbage(self):
        from benchmarks.ralph_wiggum import _parse_llm_proposal

        assert _parse_llm_proposal("I don't know what to suggest") is None
        assert _parse_llm_proposal("") is None

    def test_parses_with_confidence_threshold(self):
        from benchmarks.ralph_wiggum import _parse_llm_proposal

        raw = '{"confidence_threshold": 0.75, "predicate_map": {"test": "court"}}'
        proposal = _parse_llm_proposal(raw)
        assert proposal["confidence_threshold"] == 0.75


class TestRalphWiggumMutationLoop:
    """Integration tests for loop evaluation."""

    def test_loop_with_cases(self):
        from benchmarks.ralph_wiggum import PredicateLoop
        from crystal.tools.kg.legal import build_legal_kg_memory
        from tests.fixtures.scotus_sample import SCOTUS_SAMPLE

        kg = build_legal_kg_memory(SCOTUS_SAMPLE[:10])

        cases = [
            ("What court decided Miranda v. Arizona?", "Supreme Court of the United States", ["supreme court"], False),
            ("When was Brown v. Board of Education decided?", "1954-05-17", ["1954"], False),
        ]

        loop = PredicateLoop(kg=kg, cases=cases)
        result = loop.run_iteration(0)
        assert result.total == 2

    def test_loop_records_history(self):
        from benchmarks.ralph_wiggum import PredicateLoop
        from crystal.tools.kg.legal import build_legal_kg_memory
        from tests.fixtures.scotus_sample import SCOTUS_SAMPLE

        kg = build_legal_kg_memory(SCOTUS_SAMPLE[:10])
        cases = [
            ("What court decided Miranda v. Arizona?", "Supreme Court of the United States", ["supreme court"], False),
        ]

        loop = PredicateLoop(kg=kg, cases=cases)
        result = loop.run(threshold=0.90, max_iterations=2)
        assert len(result.history) >= 1
        assert result.best_score >= 0.0
        assert result.change_report != ""


class TestPerLoopValidation:
    """Test that each specialized loop validates its own proposal format."""

    def test_predicate_loop_accepts_predicate_map(self):
        from benchmarks.ralph_wiggum import PredicateLoop
        loop = PredicateLoop.__new__(PredicateLoop)
        assert loop._validate_proposal({"predicate_map": {"x": "y"}}) is True

    def test_predicate_loop_rejects_entity_aliases(self):
        from benchmarks.ralph_wiggum import PredicateLoop
        loop = PredicateLoop.__new__(PredicateLoop)
        assert loop._validate_proposal({"entity_aliases": {"x": "y"}}) is False

    def test_entity_loop_accepts_entity_aliases(self):
        from benchmarks.ralph_wiggum import EntityLoop
        loop = EntityLoop.__new__(EntityLoop)
        assert loop._validate_proposal({"entity_aliases": {"x": "y"}}) is True

    def test_entity_loop_rejects_predicate_map(self):
        from benchmarks.ralph_wiggum import EntityLoop
        loop = EntityLoop.__new__(EntityLoop)
        assert loop._validate_proposal({"predicate_map": {"x": "y"}}) is False

    def test_threshold_loop_accepts_threshold(self):
        from benchmarks.ralph_wiggum import ThresholdLoop
        loop = ThresholdLoop.__new__(ThresholdLoop)
        assert loop._validate_proposal({"confidence_threshold": 0.75}) is True

    def test_threshold_loop_rejects_out_of_bounds(self):
        from benchmarks.ralph_wiggum import ThresholdLoop
        loop = ThresholdLoop.__new__(ThresholdLoop)
        assert loop._validate_proposal({"confidence_threshold": 0.99}) is False

    def test_threshold_loop_rejects_predicate_map(self):
        from benchmarks.ralph_wiggum import ThresholdLoop
        loop = ThresholdLoop.__new__(ThresholdLoop)
        assert loop._validate_proposal({"predicate_map": {"x": "y"}}) is False
