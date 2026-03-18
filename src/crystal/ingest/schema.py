"""Shared types for the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Triplet:
    """A single (subject, predicate, object) fact."""
    subject: str
    predicate: str
    object: str

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
