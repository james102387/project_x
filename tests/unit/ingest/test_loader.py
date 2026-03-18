"""Unit tests for the hand-curated triplet loaders (CSV/JSON)."""

import json
import pytest
from pathlib import Path

from crystal.ingest.loader import load_csv, load_json, load_file, _looks_like_header

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


class TestLooksLikeHeader:
    def test_recognizes_header(self):
        assert _looks_like_header(["subject", "predicate", "object"])

    def test_case_insensitive(self):
        assert _looks_like_header(["Subject", "Predicate", "Object"])

    def test_with_extra_whitespace(self):
        assert _looks_like_header(["  subject ", " predicate", "object "])

    def test_data_row_not_header(self):
        assert not _looks_like_header(["Remulak", "capital", "Zelphos"])

    def test_partial_header_not_header(self):
        assert not _looks_like_header(["subject", "predicate", "value"])


class TestLoadCsv:
    def test_with_header(self):
        result = load_csv(FIXTURES / "sample_triplets.csv")
        assert len(result.triplets) == 3
        assert result.triplets[0].subject == "Remulak"
        assert result.triplets[0].predicate == "capital"
        assert result.triplets[0].object == "Zelphos"

    def test_without_header(self):
        result = load_csv(FIXTURES / "sample_triplets_no_header.csv")
        assert len(result.triplets) == 2
        assert result.triplets[0].subject == "Remulak"

    def test_source_set(self):
        result = load_csv(FIXTURES / "sample_triplets.csv")
        assert "sample_triplets.csv" in result.source

    def test_empty_csv(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        result = load_csv(p)
        assert len(result.triplets) == 0

    def test_skips_short_rows(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("subject,predicate,object\nRemulak,capital\nRemulak,capital,Zelphos\n")
        result = load_csv(p)
        assert len(result.triplets) == 1

    def test_strips_whitespace(self, tmp_path):
        p = tmp_path / "spaces.csv"
        p.write_text("  Remulak  , capital , Zelphos \n")
        result = load_csv(p)
        assert result.triplets[0].subject == "Remulak"
        assert result.triplets[0].predicate == "capital"
        assert result.triplets[0].object == "Zelphos"


class TestLoadJson:
    def test_object_format_with_aliases(self):
        result = load_json(FIXTURES / "sample_triplets.json")
        assert len(result.triplets) == 2
        assert result.entity_aliases == {"korth": "grand vizier korth"}
        assert result.predicate_aliases == {"capital city": "capital"}

    def test_flat_array_format(self):
        result = load_json(FIXTURES / "sample_triplets_flat.json")
        assert len(result.triplets) == 2
        assert result.triplets[0].subject == "Remulak"
        assert result.triplets[1].subject == "Draveth"

    def test_source_set(self):
        result = load_json(FIXTURES / "sample_triplets.json")
        assert "sample_triplets.json" in result.source

    def test_object_with_array_triplets(self, tmp_path):
        p = tmp_path / "mixed.json"
        p.write_text(json.dumps({
            "triplets": [["A", "b", "C"], {"subject": "D", "predicate": "e", "object": "F"}],
        }))
        result = load_json(p)
        assert len(result.triplets) == 2
        assert result.triplets[0].as_tuple() == ("A", "b", "C")
        assert result.triplets[1].as_tuple() == ("D", "e", "F")

    def test_skips_incomplete_entries(self, tmp_path):
        p = tmp_path / "incomplete.json"
        p.write_text(json.dumps([
            ["A", "b"],
            ["A", "b", "C"],
            {"subject": "D"},
        ]))
        result = load_json(p)
        assert len(result.triplets) == 1

    def test_flat_list_of_dicts(self, tmp_path):
        p = tmp_path / "dicts.json"
        p.write_text(json.dumps([
            {"subject": "X", "predicate": "y", "object": "Z"},
        ]))
        result = load_json(p)
        assert len(result.triplets) == 1
        assert result.triplets[0].as_tuple() == ("X", "y", "Z")


class TestLoadFile:
    def test_auto_csv(self):
        result = load_file(FIXTURES / "sample_triplets.csv")
        assert len(result.triplets) == 3

    def test_auto_json(self):
        result = load_file(FIXTURES / "sample_triplets.json")
        assert len(result.triplets) == 2

    def test_unsupported_format(self, tmp_path):
        p = tmp_path / "bad.xml"
        p.write_text("<nope/>")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_file(p)
