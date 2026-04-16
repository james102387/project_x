"""Ingestion confidence scoring — progressive trust for extracted triplets.

Scores each extracted triplet on a 0.0-1.0 scale based on:
  - Extraction source (NER pattern vs LLM, and LLM self-reported confidence)
  - Entity recognition (subject already known in the KG)
  - Predicate alignment (predicate maps to a known ontology predicate)

The auto-accept threshold controls how much human review is needed.
Starts conservative (0.70 = only NER + high-confidence LLM), loosens
as the system proves itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from crystal.ingest.schema import ReviewableTriplet, Triplet


INGEST_AUTO_ACCEPT = 0.70

_SOURCE_BASE = {
    "ner": 0.85,
    "llm_high": 0.80,
    "llm_medium": 0.55,
    "llm_low": 0.30,
    "structured": 1.0,
}

_ENTITY_KNOWN_BONUS = 0.10
_PREDICATE_ALIGNED_BONUS = 0.10


@dataclass
class ScoredTriplet:
    """A triplet with ingestion confidence metadata."""

    subject: str
    predicate: str
    object: str
    source_sentence: str
    extraction_source: str
    ingestion_confidence: float
    status: str = "pending_review"
    source_document: str = ""
    origin: str = "opinion_doc"

    def to_triplet(self) -> Triplet:
        return Triplet(subject=self.subject, predicate=self.predicate, object=self.object)

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def as_tuple_with_sentence(self) -> tuple[str, str, str, str]:
        return (self.subject, self.predicate, self.object, self.source_sentence)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "source_sentence": self.source_sentence,
            "extraction_source": self.extraction_source,
            "ingestion_confidence": round(self.ingestion_confidence, 3),
            "status": self.status,
            "source_document": self.source_document,
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScoredTriplet:
        return cls(
            subject=d["subject"],
            predicate=d["predicate"],
            object=d["object"],
            source_sentence=d.get("source_sentence", ""),
            extraction_source=d.get("extraction_source", "llm_medium"),
            ingestion_confidence=d.get("ingestion_confidence", 0.5),
            status=d.get("status", "pending_review"),
            source_document=d.get("source_document", ""),
            origin=d.get("origin", "opinion_doc"),
        )

    @classmethod
    def from_reviewable(
        cls, rt: ReviewableTriplet, extraction_source: str, confidence: float,
        source_document: str = "",
    ) -> ScoredTriplet:
        return cls(
            subject=rt.subject,
            predicate=rt.predicate,
            object=rt.object,
            source_sentence=rt.source_sentence,
            extraction_source=extraction_source,
            ingestion_confidence=confidence,
            status=rt.status,
            source_document=source_document,
        )


def score_ingestion_confidence(
    subject: str,
    predicate: str,
    extraction_source: str,
    *,
    kg=None,
    ontology_predicates: set[str] | None = None,
    predicate_aliases: dict[str, str] | None = None,
) -> float:
    """Score a triplet's ingestion confidence on a 0.0-1.0 scale.

    Args:
        subject: The triplet's subject.
        predicate: The triplet's predicate.
        extraction_source: One of "ner", "llm_high", "llm_medium", "llm_low", "structured".
        kg: Optional KG to check if entity is already known.
        ontology_predicates: Set of canonical predicate names.
        predicate_aliases: Mapping of surface forms to canonical predicates.
    """
    base = _SOURCE_BASE.get(extraction_source, 0.50)

    bonus = 0.0

    if kg is not None:
        try:
            if kg.has_entity(subject):
                bonus += _ENTITY_KNOWN_BONUS
        except Exception:
            pass

    if predicate and (ontology_predicates or predicate_aliases):
        pred_lower = predicate.lower()
        aligned = False
        if ontology_predicates and pred_lower in ontology_predicates:
            aligned = True
        if not aligned and predicate_aliases and pred_lower in predicate_aliases:
            aligned = True
        if aligned:
            bonus += _PREDICATE_ALIGNED_BONUS

    return min(1.0, base + bonus)


def classify_extraction_source(confidence_label: str) -> str:
    """Map a ReviewableTriplet's confidence label to an extraction source key."""
    mapping = {"high": "llm_high", "medium": "llm_medium", "low": "llm_low"}
    return mapping.get(confidence_label, "llm_medium")
