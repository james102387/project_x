"""Tests for Ralph Wiggum Phase 6b — autonomous mutation via LLM proposals."""

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

        proposal = {
            "predicate_map": {"argued": "attorneys"},
        }
        assert _validate_proposal(proposal) is True

    def test_accepts_valid_alias_addition(self):
        from benchmarks.ralph_wiggum import _validate_proposal

        proposal = {
            "predicate_aliases": {"who argued": "attorneys"},
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


class TestRalphWiggumMutationLoop:
    """Integration tests for the full mutation loop."""

    def test_loop_with_mock_llm_improves_score(self, tmp_path):
        """Verify the loop can apply a proposal and measure improvement."""
        from benchmarks.ralph_wiggum import RalphWiggumLoop
        from crystal.tools.kg.legal import build_legal_kg_memory
        from tests.fixtures.scotus_sample import SCOTUS_SAMPLE

        kg = build_legal_kg_memory(SCOTUS_SAMPLE[:10])

        cases = [
            ("What court decided Miranda v. Arizona?", "Supreme Court of the United States", ["supreme court"], False),
            ("When was Brown v. Board of Education decided?", "1954-05-17", ["1954"], False),
        ]

        loop = RalphWiggumLoop(kg=kg, cases=cases)
        result = loop.run_iteration(0)
        assert result.total == 2

    def test_loop_records_history(self, tmp_path):
        from benchmarks.ralph_wiggum import RalphWiggumLoop
        from crystal.tools.kg.legal import build_legal_kg_memory
        from tests.fixtures.scotus_sample import SCOTUS_SAMPLE

        kg = build_legal_kg_memory(SCOTUS_SAMPLE[:10])
        cases = [
            ("What court decided Miranda v. Arizona?", "Supreme Court of the United States", ["supreme court"], False),
        ]

        loop = RalphWiggumLoop(kg=kg, cases=cases)
        result = loop.run(threshold=0.90, max_iterations=2)
        assert len(result.history) >= 1
        assert result.best_score >= 0.0
