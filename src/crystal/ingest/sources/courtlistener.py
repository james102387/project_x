"""
CourtListener API client — citation graph extraction.

Fetches forward citations for cases and maps them to (Case A, cites, Case B) triplets.
Designed for incremental sync: tracks last-fetched state in SQLite sync_state table.

Requires COURTLISTENER_API_TOKEN env var (free signup at courtlistener.com).

Usage:
    from crystal.ingest.sources.courtlistener import CourtListenerClient
    client = CourtListenerClient()
    triplets = client.fetch_citations(opinion_id=108713)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

from crystal.ingest.schema import IngestResult, Triplet


_BASE_URL = "https://www.courtlistener.com/api/rest/v4"
_DEFAULT_RATE_LIMIT = 2.0  # requests per second


@dataclass
class CitationResult:
    """Result of a citation fetch for one opinion."""
    opinion_id: int
    case_name: str
    cited_opinions: list[dict] = field(default_factory=list)
    error: str | None = None


class CourtListenerClient:
    """REST client for the CourtListener API with rate limiting and retry."""

    def __init__(
        self,
        api_token: str | None = None,
        rate_limit: float = _DEFAULT_RATE_LIMIT,
        base_url: str = _BASE_URL,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_token = api_token or os.environ.get("COURTLISTENER_API_TOKEN", "")
        self.rate_limit = rate_limit
        self.base_url = base_url.rstrip("/")
        self._last_request_time = 0.0
        self._client = http_client or httpx.Client(
            headers=self._auth_headers(),
            timeout=30.0,
        )

    def _auth_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Token {self.api_token}"
        return headers

    def _rate_limit_wait(self) -> None:
        if self.rate_limit <= 0:
            return
        min_interval = 1.0 / self.rate_limit
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def _get(self, url: str, params: dict | None = None, retries: int = 3) -> dict:
        """GET with rate limiting, retry, and backoff."""
        for attempt in range(retries):
            self._rate_limit_wait()
            try:
                resp = self._client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                if attempt == retries - 1:
                    raise
                backoff = 2 ** attempt
                time.sleep(backoff)
        return {}

    def _paginate(self, url: str, params: dict | None = None) -> list[dict]:
        """Follow cursor-based pagination, collecting all results."""
        all_results: list[dict] = []
        current_url = url
        current_params = params

        while current_url:
            data = self._get(current_url, params=current_params)
            results = data.get("results", [])
            all_results.extend(results)
            current_url = data.get("next")
            current_params = None  # next URL includes params
            if not current_url:
                break

        return all_results

    def fetch_citations(
        self,
        opinion_id: int,
        case_name: str = "",
    ) -> CitationResult:
        """Fetch forward citations for an opinion (cases this opinion cites).

        Returns a CitationResult with cited opinion metadata.
        """
        try:
            url = f"{self.base_url}/opinions-cited/"
            results = self._paginate(url, params={"citing_opinion": opinion_id})
            return CitationResult(
                opinion_id=opinion_id,
                case_name=case_name,
                cited_opinions=results,
            )
        except Exception as e:
            return CitationResult(
                opinion_id=opinion_id,
                case_name=case_name,
                error=str(e),
            )

    def fetch_reverse_citations(
        self,
        opinion_id: int,
        case_name: str = "",
    ) -> CitationResult:
        """Fetch reverse citations (cases that cite this opinion)."""
        try:
            url = f"{self.base_url}/opinions-cited/"
            results = self._paginate(url, params={"cited_opinion": opinion_id})
            return CitationResult(
                opinion_id=opinion_id,
                case_name=case_name,
                cited_opinions=results,
            )
        except Exception as e:
            return CitationResult(
                opinion_id=opinion_id,
                case_name=case_name,
                error=str(e),
            )

    def close(self) -> None:
        self._client.close()


def citations_to_triplets(
    citing_case: str,
    citation_result: CitationResult,
    resolve_opinion_name: Callable[[int], str | None] | None = None,
) -> IngestResult:
    """Convert a CitationResult to Crystal triplets.

    Each cited opinion becomes a (citing_case, cites, cited_case) triplet.

    resolve_opinion_name: optional function that maps an opinion ID to a case
    name. If not provided, uses the opinion ID as the object.
    """
    triplets: list[Triplet] = []

    for cited in citation_result.cited_opinions:
        cited_id = cited.get("cited_opinion") or cited.get("id")
        if not cited_id:
            continue

        if resolve_opinion_name:
            cited_name = resolve_opinion_name(cited_id)
        else:
            cited_name = None

        object_val = cited_name or f"opinion:{cited_id}"
        triplets.append(Triplet(
            subject=citing_case,
            predicate="cites",
            object=object_val,
        ))

    return IngestResult(
        triplets=triplets,
        source=f"courtlistener:opinion:{citation_result.opinion_id}",
    )
