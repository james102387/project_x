"""Shared types for the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Triplet:
    """A single (subject, predicate, object) fact."""
    subject: str
    predicate: str
    object: str
    source_sentence: str = ""

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


@dataclass
class IngestResult:
    """Output of an ingestion run — feeds directly into KnowledgeGraph."""
    triplets: list[Triplet] = field(default_factory=list)
    entity_aliases: dict[str, str] = field(default_factory=dict)
    predicate_aliases: dict[str, str] = field(default_factory=dict)
    source: str = ""

    def as_tuples(self) -> list[tuple[str, str, str]]:
        return [t.as_tuple() for t in self.triplets]

    def merge(self, other: IngestResult) -> IngestResult:
        """Combine two IngestResults, deduplicating triplets."""
        seen = {t.as_tuple() for t in self.triplets}
        merged_triplets = list(self.triplets)
        for t in other.triplets:
            if t.as_tuple() not in seen:
                merged_triplets.append(t)
                seen.add(t.as_tuple())
        return IngestResult(
            triplets=merged_triplets,
            entity_aliases={**self.entity_aliases, **other.entity_aliases},
            predicate_aliases={**self.predicate_aliases, **other.predicate_aliases},
            source=f"{self.source}+{other.source}" if self.source and other.source else self.source or other.source,
        )


# ── D2 Phase 2: LLM-assisted extraction ──────────────────────────────


@dataclass
class ReviewableTriplet:
    """A triplet extracted by LLM, pending human review."""
    subject: str
    predicate: str
    object: str
    source_sentence: str
    confidence: str  # "high", "medium", "low"
    status: str = "pending_review"  # "pending_review", "accepted", "rejected"

    def to_triplet(self) -> Triplet:
        return Triplet(subject=self.subject, predicate=self.predicate, object=self.object)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "source_sentence": self.source_sentence,
            "confidence": self.confidence,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ReviewableTriplet:
        return cls(
            subject=d["subject"],
            predicate=d["predicate"],
            object=d["object"],
            source_sentence=d.get("source_sentence", ""),
            confidence=d.get("confidence", "medium"),
            status=d.get("status", "pending_review"),
        )


@dataclass
class LLMExtractionResult:
    """Output of LLM-assisted extraction — kept separate for human review."""
    reviewable: list[ReviewableTriplet] = field(default_factory=list)
    skipped_sentences: list[str] = field(default_factory=list)
    source: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def accepted_triplets(self) -> list[Triplet]:
        return [r.to_triplet() for r in self.reviewable if r.status == "accepted"]

    def pending_triplets(self) -> list[ReviewableTriplet]:
        return [r for r in self.reviewable if r.status == "pending_review"]

    def to_ingest_result(self) -> IngestResult:
        """Convert accepted triplets into an IngestResult for KG building."""
        return IngestResult(
            triplets=self.accepted_triplets(),
            source=f"{self.source}(llm-reviewed)",
        )

    def to_review_dict(self) -> dict:
        """Serialize to JSON-friendly dict for human review."""
        return {
            "source": self.source,
            "generated_at": self.generated_at,
            "total_reviewable": len(self.reviewable),
            "total_skipped": len(self.skipped_sentences),
            "reviewable": [r.to_dict() for r in self.reviewable],
            "skipped_sentences": self.skipped_sentences,
        }

    @classmethod
    def from_review_dict(cls, d: dict) -> LLMExtractionResult:
        return cls(
            reviewable=[ReviewableTriplet.from_dict(r) for r in d.get("reviewable", [])],
            skipped_sentences=d.get("skipped_sentences", []),
            source=d.get("source", ""),
            generated_at=d.get("generated_at", ""),
        )
