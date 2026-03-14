"""Knowledge Graph tool — hash-table lookup over (subject, predicate, object) triplets.

Dataset-agnostic: accepts any list of triplets at construction time.
Optionally accepts a predicate alias map for synonym resolution.
"""

from __future__ import annotations
from typing import TypeAlias

Triplet: TypeAlias = tuple[str, str, str]


class KnowledgeGraph:
    """O(1) hash-table KG backed by forward and reverse indexes."""

    def __init__(
        self,
        triplets: list[Triplet],
        predicate_aliases: dict[str, str] | None = None,
    ) -> None:
        self.triplets = triplets
        self._predicate_aliases: dict[str, str] = {
            k.lower(): v.lower() for k, v in (predicate_aliases or {}).items()
        }
        self._forward: dict[str, dict] = {}
        self._reverse: dict[str, dict] = {}
        self._entity_index: set[str] = set()
        self._build(triplets)

    def _build(self, triplets: list[Triplet]) -> None:
        for subject, predicate, obj in triplets:
            entry = {"subject": subject, "predicate": predicate, "object": obj}
            self._forward[f"{subject}|{predicate}".lower()] = entry
            self._reverse[f"{predicate}|{obj}".lower()] = entry
            self._entity_index.add(subject.lower())
            self._entity_index.add(obj.lower())

    def _resolve_predicate(self, predicate: str) -> str:
        """Map an alias to its canonical predicate, or return as-is."""
        return self._predicate_aliases.get(predicate.lower(), predicate)

    @property
    def entities(self) -> set[str]:
        """All known entity strings (lowercased) from subjects and objects."""
        return self._entity_index

    def has_entity(self, text: str) -> bool:
        """Check whether a string matches a known entity."""
        return text.lower() in self._entity_index

    def lookup(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
    ) -> list[dict]:
        """Query the KG.

        Predicates are resolved through the alias map before lookup.

        Supports:
            lookup(subject="X", predicate="Y")  -> forward lookup
            lookup(predicate="Y", obj="Z")       -> reverse lookup
            lookup(subject="X")                   -> all facts about subject
        """
        if predicate:
            predicate = self._resolve_predicate(predicate)

        results: list[dict] = []

        if subject and predicate:
            key = f"{subject}|{predicate}".lower()
            if key in self._forward:
                results.append(self._forward[key])

        elif predicate and obj:
            key = f"{predicate}|{obj}".lower()
            if key in self._reverse:
                results.append(self._reverse[key])

        elif subject:
            prefix = f"{subject}|".lower()
            for key, value in self._forward.items():
                if key.startswith(prefix):
                    results.append(value)

        return results

    def __len__(self) -> int:
        return len(self.triplets)

    def __repr__(self) -> str:
        return f"KnowledgeGraph({len(self)} triplets)"
