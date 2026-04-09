"""Tests for CourtListener API client — uses mock HTTP, no real API calls."""

import pytest
import httpx

from crystal.ingest.sources.courtlistener import (
    CourtListenerClient,
    CitationResult,
    citations_to_triplets,
)


# ── Mock transport ───────────────────────────────────────────────────────


class MockTransport(httpx.BaseTransport):
    """Returns canned responses for CourtListener API endpoints."""

    def __init__(self, responses: dict[str, dict] | None = None):
        self._responses = responses or {}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for pattern, data in self._responses.items():
            if pattern in url:
                return httpx.Response(200, json=data)
        return httpx.Response(404, json={"detail": "Not found"})


def _make_client(responses: dict[str, dict]) -> CourtListenerClient:
    transport = MockTransport(responses)
    http_client = httpx.Client(transport=transport)
    return CourtListenerClient(
        api_token="test-token",
        rate_limit=0,
        http_client=http_client,
    )


# ── CourtListenerClient ─────────────────────────────────────────────────


class TestFetchCitations:
    def test_returns_citation_result(self):
        client = _make_client({
            "opinions-cited": {
                "count": 2,
                "next": None,
                "results": [
                    {"id": 1, "cited_opinion": 200, "depth": 1},
                    {"id": 2, "cited_opinion": 300, "depth": 1},
                ],
            }
        })
        result = client.fetch_citations(opinion_id=100, case_name="Test Case")
        assert result.opinion_id == 100
        assert result.case_name == "Test Case"
        assert len(result.cited_opinions) == 2
        assert result.error is None

    def test_empty_citations(self):
        client = _make_client({
            "opinions-cited": {"count": 0, "next": None, "results": []},
        })
        result = client.fetch_citations(opinion_id=100)
        assert len(result.cited_opinions) == 0

    def test_handles_api_error(self):
        transport = MockTransport({})  # Returns 404 for everything
        http_client = httpx.Client(transport=transport)
        client = CourtListenerClient(
            api_token="test",
            rate_limit=0,
            http_client=http_client,
        )
        result = client.fetch_citations(opinion_id=999)
        assert result.error is not None

    def test_pagination(self):
        page1_url = "opinions-cited/?citing_opinion=100"
        page2_url = "page2"

        class PaginatingTransport(httpx.BaseTransport):
            def __init__(self):
                self.call_count = 0

            def handle_request(self, request):
                self.call_count += 1
                if self.call_count == 1:
                    return httpx.Response(200, json={
                        "count": 3,
                        "next": "https://courtlistener.com/api/rest/v4/page2",
                        "results": [{"id": 1, "cited_opinion": 200}],
                    })
                else:
                    return httpx.Response(200, json={
                        "count": 3,
                        "next": None,
                        "results": [
                            {"id": 2, "cited_opinion": 300},
                            {"id": 3, "cited_opinion": 400},
                        ],
                    })

        transport = PaginatingTransport()
        http_client = httpx.Client(transport=transport)
        client = CourtListenerClient(
            api_token="test",
            rate_limit=0,
            http_client=http_client,
        )
        result = client.fetch_citations(opinion_id=100)
        assert len(result.cited_opinions) == 3
        assert transport.call_count == 2


class TestFetchReverseCitations:
    def test_returns_reverse_citations(self):
        client = _make_client({
            "opinions-cited": {
                "count": 1,
                "next": None,
                "results": [{"id": 1, "cited_opinion": 100, "depth": 1}],
            }
        })
        result = client.fetch_reverse_citations(opinion_id=100)
        assert len(result.cited_opinions) == 1


# ── citations_to_triplets ───────────────────────────────────────────────


class TestCitationsToTriplets:
    def test_basic_conversion(self):
        cr = CitationResult(
            opinion_id=100,
            case_name="Miranda v. Arizona",
            cited_opinions=[
                {"id": 1, "cited_opinion": 200},
                {"id": 2, "cited_opinion": 300},
            ],
        )
        result = citations_to_triplets("Miranda v. Arizona", cr)
        assert len(result.triplets) == 2
        assert all(t.predicate == "cites" for t in result.triplets)
        assert all(t.subject == "Miranda v. Arizona" for t in result.triplets)

    def test_with_name_resolver(self):
        cr = CitationResult(
            opinion_id=100,
            case_name="Miranda v. Arizona",
            cited_opinions=[{"id": 1, "cited_opinion": 200}],
        )
        resolver = lambda oid: "Mapp v. Ohio" if oid == 200 else None
        result = citations_to_triplets("Miranda v. Arizona", cr, resolve_opinion_name=resolver)
        assert result.triplets[0].object == "Mapp v. Ohio"

    def test_without_name_resolver_uses_id(self):
        cr = CitationResult(
            opinion_id=100,
            case_name="Test",
            cited_opinions=[{"id": 1, "cited_opinion": 200}],
        )
        result = citations_to_triplets("Test", cr)
        assert result.triplets[0].object == "opinion:200"

    def test_empty_citations(self):
        cr = CitationResult(opinion_id=100, case_name="Test", cited_opinions=[])
        result = citations_to_triplets("Test", cr)
        assert len(result.triplets) == 0

    def test_source_includes_opinion_id(self):
        cr = CitationResult(
            opinion_id=42,
            case_name="Test",
            cited_opinions=[{"id": 1, "cited_opinion": 200}],
        )
        result = citations_to_triplets("Test", cr)
        assert "42" in result.source


# ── Client construction ──────────────────────────────────────────────────


class TestClientConstruction:
    def test_auth_headers_with_token(self):
        client = CourtListenerClient(api_token="test-token", rate_limit=0)
        headers = client._auth_headers()
        assert headers["Authorization"] == "Token test-token"
        client.close()

    def test_auth_headers_without_token(self, monkeypatch):
        monkeypatch.delenv("COURTLISTENER_API_TOKEN", raising=False)
        client = CourtListenerClient(api_token="", rate_limit=0)
        headers = client._auth_headers()
        assert "Authorization" not in headers
        client.close()
