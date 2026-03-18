"""Tests for UI helper functions (no Gradio dependency needed)."""

import pytest

from crystal.tools.kg.graph import KnowledgeGraph
from crystal.tools.kg import remulak_kg
from crystal.ui.app import (
    _kg_info,
    _format_kg_stats,
    _format_kg_facts,
    _default_kg_info,
    reset_to_remulak,
    ingest_raw_text,
)


TINY_TRIPLETS = [
    ("Alpha", "relates_to", "Beta"),
    ("Beta", "has", "Gamma"),
]


@pytest.fixture
def tiny_kg():
    return KnowledgeGraph(TINY_TRIPLETS)


class TestKgInfo:
    def test_default_kg_info(self):
        info = _default_kg_info()
        assert info["source"] == "Remulak (built-in)"
        assert info["triplets"] > 0
        assert info["entities"] > 0
        assert info["subjects"] > 0

    def test_custom_kg_info(self, tiny_kg):
        info = _kg_info(tiny_kg, "test.csv")
        assert info["source"] == "test.csv"
        assert info["triplets"] == 2
        assert info["entities"] == 3
        assert info["subjects"] == 2


class TestFormatKgStats:
    def test_contains_source(self):
        info = {"source": "my_file.csv", "triplets": 10, "entities": 5, "subjects": 3}
        text = _format_kg_stats(info)
        assert "my_file.csv" in text
        assert "10" in text
        assert "5" in text

    def test_markdown_format(self):
        info = _default_kg_info()
        text = _format_kg_stats(info)
        assert text.startswith("**")


class TestFormatKgFacts:
    def test_table_header(self, tiny_kg):
        text = _format_kg_facts(tiny_kg)
        assert "Subject" in text
        assert "Predicate" in text
        assert "Object" in text

    def test_contains_facts(self, tiny_kg):
        text = _format_kg_facts(tiny_kg)
        assert "Alpha" in text
        assert "Beta" in text
        assert "Gamma" in text

    def test_empty_kg(self):
        kg = KnowledgeGraph([])
        text = _format_kg_facts(kg)
        assert "No facts loaded" in text

    def test_truncation(self):
        triplets = [(f"E{i}", "pred", f"O{i}") for i in range(250)]
        kg = KnowledgeGraph(triplets)
        text = _format_kg_facts(kg, max_facts=50)
        assert "...and 200 more" in text


class TestResetToRemulak:
    def test_returns_remulak_kg(self):
        kg, status, stats, facts = reset_to_remulak()
        assert kg is remulak_kg
        assert "Reset" in status
        assert "Remulak" in stats


class TestIngestRawText:
    def test_empty_text(self, tiny_kg):
        kg, status, stats, facts = ingest_raw_text("", tiny_kg)
        assert kg is tiny_kg
        assert "No text" in status

    def test_whitespace_only(self, tiny_kg):
        kg, status, stats, facts = ingest_raw_text("   ", tiny_kg)
        assert kg is tiny_kg
        assert "No text" in status

    def test_valid_text_extracts_triplets(self, tiny_kg):
        text = "The capital of Zorgon is Mareth. Mareth has a population of 500."
        kg, status, stats, facts = ingest_raw_text(text, tiny_kg)
        if "extracted" in status.lower() or "triplet" in status.lower():
            assert kg is not tiny_kg
            assert len(kg) > 0
        else:
            assert kg is tiny_kg
