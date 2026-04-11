"""Tests for B2: document-answerability audit."""

from __future__ import annotations

import pytest

from benchmarks.answerability import (
    DOCUMENT_ANSWERABLE_PREDICATES,
    KG_ONLY_PREDICATES,
    classify_question,
    infer_predicate,
    partition_cases,
    partition_summary,
)


class TestInferPredicate:
    def test_court_questions(self):
        assert infer_predicate("What court decided Miranda v. Arizona?") == "court"
        assert infer_predicate("Which court decided Marbury v. Madison?") == "court"

    def test_date_questions(self):
        assert infer_predicate("When was Brown v. Board of Education decided?") == "date_filed"
        assert infer_predicate("What date was Loving v. Virginia filed?") == "date_filed"

    def test_judges_questions(self):
        assert infer_predicate("Who were the judges in Roe v. Wade?") == "judges"
        assert infer_predicate("Who decided Citizens United v. FEC?") == "judges"
        assert infer_predicate("List the judges in Gideon v. Wainwright") == "judges"

    def test_opinion_author_questions(self):
        assert infer_predicate("Who wrote the opinion in Dred Scott v. Sandford?") == "opinion_author"
        assert infer_predicate("Who authored the opinion in McCulloch v. Maryland?") == "opinion_author"

    def test_cited_by_count_questions(self):
        assert infer_predicate("How many times has Miranda v. Arizona been cited?") == "cited_by_count"
        assert infer_predicate("What is the citation count for Brown v. Board?") == "cited_by_count"

    def test_precedential_status_questions(self):
        assert infer_predicate("What is the precedential status of Miranda v. Arizona?") == "precedential_status"
        assert infer_predicate("Is Brown v. Board of Education a published opinion?") == "precedential_status"

    def test_attorneys_questions(self):
        assert infer_predicate("Who were the attorneys in Gideon v. Wainwright?") == "attorneys"

    def test_subject_scan_returns_none(self):
        assert infer_predicate("Tell me about Miranda v. Arizona") is None
        assert infer_predicate("What do we know about Roe v. Wade?") is None


class TestClassifyQuestion:
    def test_document_answerable(self):
        assert classify_question("What court decided Miranda v. Arizona?", ["supreme court"]) == "document_answerable"
        assert classify_question("When was Brown v. Board decided?", ["1954"]) == "document_answerable"
        assert classify_question("Who were the judges in Roe v. Wade?", ["burger"]) == "document_answerable"

    def test_kg_only(self):
        assert classify_question("How many times has Miranda been cited?", ["9832"]) == "kg_only"
        assert classify_question("What is the precedential status of Roe v. Wade?", ["published"]) == "kg_only"

    def test_negative(self):
        assert classify_question("What was the majority opinion in Miranda?", [], is_negative=True) == "negative"

    def test_subject_scan(self):
        assert classify_question("Tell me about Miranda v. Arizona", ["supreme court"]) == "subject_scan"


class TestPartitionCases:
    SAMPLE_CASES = [
        ("What court decided Miranda v. Arizona?", "SCOTUS", ["supreme court"], False),
        ("When was Brown v. Board decided?", "1954-05-17", ["1954"], False),
        ("Who were the judges in Roe v. Wade?", "Burger...", ["burger"], False),
        ("How many times has Miranda been cited?", "9832", ["9832"], False),
        ("What is the precedential status of Roe?", "Published", ["published"], False),
        ("Tell me about Miranda v. Arizona", "SCOTUS", ["supreme court"], False),
        ("What was the majority opinion?", "[ABSTAIN]", [], True),
    ]

    def test_partition_counts(self):
        doc, kg, neg, scan = partition_cases(self.SAMPLE_CASES)
        assert len(doc) == 3
        assert len(kg) == 2
        assert len(neg) == 1
        assert len(scan) == 1

    def test_partition_preserves_tuples(self):
        doc, kg, neg, scan = partition_cases(self.SAMPLE_CASES)
        for case in doc + kg + neg + scan:
            assert len(case) == 4

    def test_partition_exhaustive(self):
        doc, kg, neg, scan = partition_cases(self.SAMPLE_CASES)
        assert len(doc) + len(kg) + len(neg) + len(scan) == len(self.SAMPLE_CASES)


class TestPartitionSummary:
    def test_summary_totals(self):
        cases = [
            ("What court decided Miranda v. Arizona?", "SCOTUS", ["supreme court"], False),
            ("How many times has Miranda been cited?", "9832", ["9832"], False),
            ("What majority opinion?", "[ABSTAIN]", [], True),
        ]
        summary = partition_summary(cases)
        assert summary["total"] == 3
        assert summary["document_answerable"] == 1
        assert summary["kg_only"] == 1
        assert summary["negative"] == 1
        assert "court" in summary["by_predicate"]


class TestAgainstRealBenchmark:
    """Verify answerability classification on the actual legal benchmark cases."""

    def test_real_benchmark_partition(self):
        from benchmarks.ground_truth.legal import LEGAL_BENCHMARK_CASES
        doc, kg, neg, scan = partition_cases(LEGAL_BENCHMARK_CASES)

        assert len(neg) == 10
        assert len(kg) >= 4
        assert len(doc) >= 15
        assert len(doc) + len(kg) + len(neg) + len(scan) == len(LEGAL_BENCHMARK_CASES)

    def test_no_negative_in_positive_buckets(self):
        from benchmarks.ground_truth.legal import LEGAL_BENCHMARK_CASES
        doc, kg, _neg, scan = partition_cases(LEGAL_BENCHMARK_CASES)
        for case in doc + kg + scan:
            assert case[3] is False, f"Negative case misclassified: {case[0]}"
