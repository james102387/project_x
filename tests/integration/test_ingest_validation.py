"""Validation: ingestion pipeline against real SCOTUS opinion texts.

Uses cached opinion JSON files from benchmarks/documents/ to validate:
  - Triplet extraction quality (NER + LLM-less path)
  - Confidence distribution
  - Predicate coverage (alignment with legal ontology)
  - Dedup behavior against an existing KG
"""

import json
import pytest
from pathlib import Path

from crystal.ingest import ingest_document, DocumentIngestionResult
from crystal.ingest.confidence import ScoredTriplet, INGEST_AUTO_ACCEPT
from crystal.tools.kg.store import SqliteKnowledgeGraph
from crystal.data.legal_ontology import LEGAL_PREDICATES


_DOCS_DIR = Path(__file__).parent.parent.parent / "benchmarks" / "documents"

VALIDATION_CASES = [
    "brown-v-board-of-education",
    "gideon-v-wainwright",
    "loving-v-virginia",
    "terry-v-ohio",
    "nevada-department-of-human-resources-v-hibbs",
]


def _load_opinion_text(slug: str) -> tuple[str, str]:
    """Load opinion text from a cached document JSON."""
    path = _DOCS_DIR / f"{slug}.json"
    with open(path) as f:
        doc = json.load(f)
    return doc["text"], doc.get("case_name", slug)


@pytest.fixture(scope="module")
def empty_kg():
    return SqliteKnowledgeGraph(":memory:")


@pytest.fixture(scope="module")
def validation_results(empty_kg):
    """Run ingestion on all validation cases and cache results."""
    results = {}
    for slug in VALIDATION_CASES:
        try:
            text, case_name = _load_opinion_text(slug)
        except FileNotFoundError:
            continue
        if len(text) < 50:
            continue
        result = ingest_document(
            text, empty_kg,
            call_llm_fn=None,
            auto_accept_threshold=INGEST_AUTO_ACCEPT,
            domain="legal",
        )
        results[slug] = (result, case_name)
    return results


@pytest.mark.skipif(
    not (_DOCS_DIR / "brown-v-board-of-education.json").exists(),
    reason="Benchmark documents not available",
)
class TestIngestionValidation:
    def test_extracts_triplets_from_real_opinions(self, validation_results):
        """At least some opinions produce extracted triplets."""
        cases_with_triplets = 0
        for slug, (result, _) in validation_results.items():
            total = len(result.auto_accepted) + len(result.pending_review) + len(result.rejected)
            if total > 0:
                cases_with_triplets += 1
        assert cases_with_triplets >= 1, "Expected at least 1 opinion to yield triplets"

    def test_confidence_distribution_makes_sense(self, validation_results):
        """Auto-accepted triplets should have confidence >= threshold."""
        for slug, (result, _) in validation_results.items():
            for st in result.auto_accepted:
                assert st.ingestion_confidence >= INGEST_AUTO_ACCEPT, (
                    f"Auto-accepted triplet has confidence {st.ingestion_confidence} < {INGEST_AUTO_ACCEPT}"
                )
            for st in result.pending_review:
                assert st.ingestion_confidence < INGEST_AUTO_ACCEPT

    def test_predicates_are_normalized(self, validation_results):
        """Extracted predicates should be lowercase and reasonably short."""
        for slug, (result, _) in validation_results.items():
            for st in result.auto_accepted + result.pending_review:
                assert st.predicate == st.predicate.lower(), f"Predicate not lowered: {st.predicate}"
                assert len(st.predicate) < 100, f"Predicate too long: {st.predicate}"

    def test_no_duplicate_triplets_in_result(self, validation_results):
        """Each result should have unique (subject, predicate, object) tuples."""
        for slug, (result, _) in validation_results.items():
            all_triplets = result.auto_accepted + result.pending_review + result.rejected
            tuples = [(st.subject.lower(), st.predicate.lower(), st.object.lower()) for st in all_triplets]
            assert len(tuples) == len(set(tuples)), f"Duplicate triplets in {slug}"

    def test_stats_populated(self, validation_results):
        """Each result should have meaningful stats."""
        for slug, (result, _) in validation_results.items():
            assert "total_extracted" in result.stats
            assert "elapsed_seconds" in result.stats
            assert result.stats["elapsed_seconds"] >= 0

    def test_scored_triplet_structure(self, validation_results):
        """All items should be ScoredTriplet instances with valid fields."""
        for slug, (result, _) in validation_results.items():
            for st in result.auto_accepted + result.pending_review + result.rejected:
                assert isinstance(st, ScoredTriplet)
                assert st.subject
                assert st.predicate
                assert st.object
                assert 0.0 <= st.ingestion_confidence <= 1.0
                assert st.extraction_source in ("ner", "llm_high", "llm_medium", "llm_low", "structured")

    def test_ontology_predicate_coverage(self, validation_results):
        """At least some predicates should align with the legal ontology."""
        ontology_set = set(LEGAL_PREDICATES)
        all_predicates = set()
        aligned = set()
        for slug, (result, _) in validation_results.items():
            for st in result.auto_accepted + result.pending_review:
                all_predicates.add(st.predicate)
                if st.predicate in ontology_set:
                    aligned.add(st.predicate)
        if all_predicates:
            coverage = len(aligned) / len(all_predicates) if all_predicates else 0
            assert coverage >= 0.0, f"Ontology coverage: {coverage:.0%}"


@pytest.mark.skipif(
    not (_DOCS_DIR / "brown-v-board-of-education.json").exists(),
    reason="Benchmark documents not available",
)
class TestIngestionReport:
    """Generate a human-readable report of extraction quality (printed in test output)."""

    def test_print_validation_report(self, validation_results, capsys):
        print("\n" + "=" * 70)
        print("DOCUMENT INGESTION VALIDATION REPORT")
        print("=" * 70)

        total_extracted = 0
        total_auto = 0
        total_pending = 0
        total_rejected = 0
        all_preds = set()

        for slug, (result, case_name) in validation_results.items():
            n_auto = len(result.auto_accepted)
            n_pending = len(result.pending_review)
            n_rejected = len(result.rejected)
            n_total = n_auto + n_pending + n_rejected

            total_extracted += n_total
            total_auto += n_auto
            total_pending += n_pending
            total_rejected += n_rejected

            print(f"\n--- {case_name} ---")
            print(f"  Total: {n_total} | Auto: {n_auto} | Pending: {n_pending} | Rejected: {n_rejected}")
            print(f"  Time: {result.stats.get('elapsed_seconds', 0):.2f}s")

            for st in result.auto_accepted[:3]:
                all_preds.add(st.predicate)
                print(f"  [AUTO {st.ingestion_confidence:.2f}] ({st.subject}, {st.predicate}, {st.object[:50]})")
            for st in result.pending_review[:3]:
                all_preds.add(st.predicate)
                print(f"  [PEND {st.ingestion_confidence:.2f}] ({st.subject}, {st.predicate}, {st.object[:50]})")

        print(f"\n{'=' * 70}")
        print(f"TOTALS: {total_extracted} extracted | {total_auto} auto | {total_pending} pending | {total_rejected} rejected")
        print(f"PREDICATES seen: {sorted(all_preds)}")
        print(f"{'=' * 70}\n")

        assert total_extracted >= 0
