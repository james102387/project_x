"""Tests for COLD Cases adapter — record-to-triplet mapping and batch ingestion."""

import pytest

from crystal.ingest.sources.cold_cases import (
    _record_to_triplets,
    ingest_cold_cases,
    scotus_filter,
)


# ── Sample records (mirrors real COLD Cases schema) ──────────────────────

MIRANDA_RECORD = {
    "id": 108713,
    "case_name": "Miranda v. Arizona",
    "case_name_short": "Miranda",
    "case_name_full": "Ernesto A. MIRANDA, Petitioner, v. State of ARIZONA",
    "slug": "miranda-v-arizona",
    "citations": ["384 U.S. 436", "86 S. Ct. 1602"],
    "citation_count": 9832,
    "court_short_name": "Supreme Court of the United States",
    "court_full_name": "Supreme Court of the United States",
    "court_type": "F",
    "court_jurisdiction": "United States, US",
    "date_filed": "1966-06-13",
    "judges": "Warren, Black, Douglas, Clark, Harlan, Brennan, Stewart, White, Fortas",
    "disposition": "Reversed and remanded",
    "nature_of_suit": None,
}

HOWERTON_RECORD = {
    "id": 2636981,
    "case_name": "State v. Howerton",
    "case_name_short": "Howerton",
    "case_name_full": "The STATE of Oklahoma, Appellant, v. Frankie HOWERTON, Appellee",
    "slug": "state-v-howerton",
    "citations": ["2002 OK CR 17", "46 P.3d 154"],
    "citation_count": 32,
    "court_short_name": "Court of Criminal Appeals of Oklahoma",
    "court_full_name": "Court of Criminal Appeals of Oklahoma",
    "court_type": "SA",
    "court_jurisdiction": "Oklahoma, OK",
    "date_filed": "2002-04-11",
    "judges": "Chapel, Johnson, Lile, Lumpkin, Strubhar",
    "disposition": None,
    "nature_of_suit": None,
}

EMPTY_RECORD = {
    "id": 999,
    "case_name": "",
    "case_name_short": "",
    "citations": [],
    "citation_count": 0,
    "court_type": "F",
}

MINIMAL_RECORD = {
    "id": 1000,
    "case_name": "Roe v. Wade",
    "case_name_short": "Roe",
    "citations": ["410 U.S. 113"],
    "citation_count": 0,
    "court_full_name": "Supreme Court of the United States",
    "court_type": "F",
    "date_filed": "1973-01-22",
    "judges": None,
    "disposition": None,
}


# ── _record_to_triplets ─────────────────────────────────────────────────


class TestRecordToTriplets:
    def test_miranda_produces_triplets(self):
        triplets, aliases = _record_to_triplets(MIRANDA_RECORD)
        subjects = {t.subject for t in triplets}
        assert "Miranda v. Arizona" in subjects

    def test_miranda_court_triplet(self):
        triplets, _ = _record_to_triplets(MIRANDA_RECORD)
        court_facts = [t for t in triplets if t.predicate == "court"]
        assert len(court_facts) == 1
        assert court_facts[0].object == "Supreme Court of the United States"

    def test_miranda_date_triplet(self):
        triplets, _ = _record_to_triplets(MIRANDA_RECORD)
        date_facts = [t for t in triplets if t.predicate == "date_filed"]
        assert len(date_facts) == 1
        assert "1966" in date_facts[0].object

    def test_miranda_citation_count(self):
        triplets, _ = _record_to_triplets(MIRANDA_RECORD)
        cite_facts = [t for t in triplets if t.predicate == "cited_by_count"]
        assert len(cite_facts) == 1
        assert cite_facts[0].object == "9832"

    def test_miranda_judges(self):
        triplets, _ = _record_to_triplets(MIRANDA_RECORD)
        judge_facts = [t for t in triplets if t.predicate == "judges"]
        assert len(judge_facts) == 1
        assert "Warren" in judge_facts[0].object

    def test_miranda_disposition(self):
        triplets, _ = _record_to_triplets(MIRANDA_RECORD)
        disp_facts = [t for t in triplets if t.predicate == "disposition"]
        assert len(disp_facts) == 1
        assert "remanded" in disp_facts[0].object.lower()

    def test_miranda_aliases_include_citation(self):
        _, aliases = _record_to_triplets(MIRANDA_RECORD)
        canonical = "miranda v. arizona"
        assert aliases.get("384 u.s. 436") == canonical

    def test_miranda_aliases_include_short_name(self):
        _, aliases = _record_to_triplets(MIRANDA_RECORD)
        canonical = "miranda v. arizona"
        assert aliases.get("miranda") == canonical

    def test_empty_record_returns_nothing(self):
        triplets, aliases = _record_to_triplets(EMPTY_RECORD)
        assert triplets == []
        assert aliases == {}

    def test_minimal_record_skips_zero_citation_count(self):
        triplets, _ = _record_to_triplets(MINIMAL_RECORD)
        cite_facts = [t for t in triplets if t.predicate == "cited_by_count"]
        assert len(cite_facts) == 0

    def test_minimal_record_skips_none_fields(self):
        triplets, _ = _record_to_triplets(MINIMAL_RECORD)
        predicates = {t.predicate for t in triplets}
        assert "judges" not in predicates
        assert "disposition" not in predicates

    def test_nature_of_suit_included(self):
        record = {**MIRANDA_RECORD, "nature_of_suit": "Criminal"}
        triplets, _ = _record_to_triplets(record)
        nature_facts = [t for t in triplets if t.predicate == "nature_of_suit"]
        assert len(nature_facts) == 1
        assert nature_facts[0].object == "Criminal"


# ── ingest_cold_cases (batch iteration) ──────────────────────────────────


class TestIngestColdCases:
    def test_yields_ingest_result(self):
        batches = list(ingest_cold_cases(
            records=[MIRANDA_RECORD, HOWERTON_RECORD],
            batch_size=10,
        ))
        assert len(batches) == 1
        assert batches[0].source == "cold-cases"

    def test_batch_contains_both_cases(self):
        batches = list(ingest_cold_cases(
            records=[MIRANDA_RECORD, HOWERTON_RECORD],
            batch_size=10,
        ))
        subjects = {t.subject for t in batches[0].triplets}
        assert "Miranda v. Arizona" in subjects
        assert "State v. Howerton" in subjects

    def test_batch_splitting(self):
        batches = list(ingest_cold_cases(
            records=[MIRANDA_RECORD, HOWERTON_RECORD],
            batch_size=1,
        ))
        assert len(batches) == 2

    def test_limit_parameter(self):
        records = [MIRANDA_RECORD, HOWERTON_RECORD, MINIMAL_RECORD]
        batches = list(ingest_cold_cases(records=records, limit=1, batch_size=10))
        total_cases = set()
        for b in batches:
            for t in b.triplets:
                total_cases.add(t.subject)
        assert len(total_cases) == 1

    def test_court_type_filter(self):
        records = [MIRANDA_RECORD, HOWERTON_RECORD]
        batches = list(ingest_cold_cases(
            records=records, court_type="SA", batch_size=10,
        ))
        subjects = set()
        for b in batches:
            for t in b.triplets:
                subjects.add(t.subject)
        assert "State v. Howerton" in subjects
        assert "Miranda v. Arizona" not in subjects

    def test_jurisdiction_filter(self):
        records = [MIRANDA_RECORD, HOWERTON_RECORD]
        batches = list(ingest_cold_cases(
            records=records, jurisdiction="Oklahoma", batch_size=10,
        ))
        subjects = set()
        for b in batches:
            for t in b.triplets:
                subjects.add(t.subject)
        assert "State v. Howerton" in subjects
        assert "Miranda v. Arizona" not in subjects

    def test_empty_records_skipped(self):
        batches = list(ingest_cold_cases(
            records=[EMPTY_RECORD], batch_size=10,
        ))
        assert len(batches) == 0

    def test_aliases_aggregated_in_batch(self):
        batches = list(ingest_cold_cases(
            records=[MIRANDA_RECORD, HOWERTON_RECORD],
            batch_size=10,
        ))
        aliases = batches[0].entity_aliases
        assert "miranda" in aliases
        assert "howerton" in aliases


# ── scotus_filter ────────────────────────────────────────────────────────


class TestScotusFilter:
    def test_scotus_case(self):
        assert scotus_filter(MIRANDA_RECORD) is True

    def test_state_court_not_scotus(self):
        assert scotus_filter(HOWERTON_RECORD) is False
