"""Tests for UI helper functions (no Gradio dependency needed)."""

import pytest

from crystal.tools.kg.graph import KnowledgeGraph
from crystal.tools.kg import remulak_kg
from crystal.ingest.loader import load_csv_text
from crystal.ui.app import (
    _kg_info,
    _format_kg_stats,
    _format_kg_facts,
    _default_kg_info,
    import_structured_data,
    KG_MODES,
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
        assert info["source"] in ("Remulak (demo)", "Legal (SCOTUS — SQLite)")
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


class TestLoadCsvText:
    def test_parses_simple_csv(self):
        result = load_csv_text("Alpha, relates_to, Beta\nBeta, has, Gamma")
        assert len(result.triplets) == 2
        assert result.triplets[0].subject == "Alpha"

    def test_skips_header(self):
        result = load_csv_text("subject,predicate,object\nA, p, B")
        assert len(result.triplets) == 1

    def test_empty_text(self):
        result = load_csv_text("")
        assert len(result.triplets) == 0

    def test_skips_short_rows(self):
        result = load_csv_text("Alpha, relates_to\nBeta, has, Gamma")
        assert len(result.triplets) == 1


class TestImportStructuredData:
    def test_no_input_returns_error(self, tiny_kg):
        result = import_structured_data(None, "", "Create new KG", "", "", tiny_kg)
        assert "No file or text" in result[1]
        assert result[0] is tiny_kg

    def test_create_new_from_text(self, tiny_kg):
        name = "_test_import_create"
        try:
            result = import_structured_data(
                None, "X, p, Y\nX, q, Z",
                "Create new KG", name, "", tiny_kg,
            )
            assert "Created" in result[1]
            assert name in KG_MODES
            assert len(result[0]) == 2
        finally:
            KG_MODES.pop(name, None)

    def test_append_to_existing(self, tiny_kg):
        name = "_test_import_append"
        KG_MODES[name] = tiny_kg
        try:
            original_len = len(tiny_kg)
            result = import_structured_data(
                None, "NewSubj, pred, NewObj",
                "Append to existing KG", "", name, tiny_kg,
            )
            assert "Appended" in result[1]
            assert len(result[0]) == original_len + 1
        finally:
            KG_MODES.pop(name, None)
