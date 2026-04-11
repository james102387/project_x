"""Tests for B1: opinion document loader and downloader utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.download_opinions import _strip_html, _case_slug


class TestStripHtml:
    def test_removes_tags(self):
        assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_br_becomes_newline(self):
        result = _strip_html("line one<br/>line two")
        assert "line one\nline two" in result

    def test_p_becomes_double_newline(self):
        result = _strip_html("<p>Para one</p><p>Para two</p>")
        assert "Para one" in result
        assert "Para two" in result

    def test_decodes_entities(self):
        assert _strip_html("&amp; &lt; &gt;") == "& < >"

    def test_collapses_excessive_newlines(self):
        result = _strip_html("<p></p><p></p><p>Content</p>")
        assert "\n\n\n" not in result

    def test_empty_string(self):
        assert _strip_html("") == ""

    def test_plain_text_passthrough(self):
        assert _strip_html("No HTML here") == "No HTML here"


class TestCaseSlug:
    def test_basic(self):
        assert _case_slug("Miranda v. Arizona") == "miranda-v-arizona"

    def test_strips_special_chars(self):
        slug = _case_slug("Chevron U.S.A., Inc. v. Natural Resources Defense Council, Inc.")
        assert slug.startswith("chevron-u-s-a")
        assert "," not in slug

    def test_truncates_long_names(self):
        long_name = "A" * 200 + " v. " + "B" * 200
        assert len(_case_slug(long_name)) <= 80

    def test_no_leading_trailing_hyphens(self):
        slug = _case_slug("  Miranda v. Arizona  ")
        assert not slug.startswith("-")
        assert not slug.endswith("-")


class TestDocumentsLoader:
    def test_load_opinion_missing_returns_none(self, tmp_path):
        from benchmarks.documents import load_opinion
        import benchmarks.documents as mod

        orig = mod.DOCUMENTS_DIR
        mod.DOCUMENTS_DIR = tmp_path
        try:
            assert load_opinion("nonexistent-case") is None
        finally:
            mod.DOCUMENTS_DIR = orig

    def test_load_opinion_reads_cached(self, tmp_path):
        from benchmarks.documents import load_opinion
        import benchmarks.documents as mod

        doc = {"text": "Opinion text here", "case_name": "Test v. Case"}
        (tmp_path / "test-v-case.json").write_text(json.dumps(doc))

        orig = mod.DOCUMENTS_DIR
        mod.DOCUMENTS_DIR = tmp_path
        try:
            assert load_opinion("test-v-case") == "Opinion text here"
        finally:
            mod.DOCUMENTS_DIR = orig

    def test_load_all_opinions(self, tmp_path):
        from benchmarks.documents import load_all_opinions
        import benchmarks.documents as mod

        for i, name in enumerate(["Alpha v. Beta", "Gamma v. Delta"]):
            doc = {"text": f"Opinion {i}", "case_name": name}
            slug = name.lower().replace(" ", "-").replace(".", "")
            (tmp_path / f"{slug}.json").write_text(json.dumps(doc))

        orig = mod.DOCUMENTS_DIR
        mod.DOCUMENTS_DIR = tmp_path
        try:
            opinions = load_all_opinions()
            assert len(opinions) == 2
            assert all(isinstance(v, str) for v in opinions.values())
        finally:
            mod.DOCUMENTS_DIR = orig

    def test_opinion_token_estimate(self):
        from benchmarks.documents import opinion_token_estimate
        assert opinion_token_estimate("a" * 400) == 100
        assert opinion_token_estimate("") == 0

    def test_list_cached_opinions(self, tmp_path):
        from benchmarks.documents import list_cached_opinions
        import benchmarks.documents as mod

        doc = {
            "text": "x" * 1000,
            "case_name": "Foo v. Bar",
            "slug": "foo-v-bar",
            "char_count": 1000,
            "token_estimate": 250,
            "opinion_id": 12345,
            "cluster_id": 67890,
        }
        (tmp_path / "foo-v-bar.json").write_text(json.dumps(doc))

        orig = mod.DOCUMENTS_DIR
        mod.DOCUMENTS_DIR = tmp_path
        try:
            items = list_cached_opinions()
            assert len(items) == 1
            assert items[0]["slug"] == "foo-v-bar"
            assert items[0]["opinion_id"] == 12345
        finally:
            mod.DOCUMENTS_DIR = orig
