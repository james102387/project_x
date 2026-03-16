"""Remulak KG — convenience instance wiring Remulak data to the generic KG tool.

Usage:
    from crystal.tools.kg import remulak_kg
    remulak_kg.lookup(subject="Remulak", predicate="capital")
"""

from crystal.data.remulak import ENTITY_ALIASES, PREDICATE_ALIASES, TRIPLETS
from .graph import KnowledgeGraph

kg = KnowledgeGraph(
    TRIPLETS,
    predicate_aliases=PREDICATE_ALIASES,
    entity_aliases=ENTITY_ALIASES,
)

if __name__ == "__main__":
    print(f"Remulak KG: {kg}")
    print()

    demos = [
        ("Forward (exact)", {"subject": "Remulak", "predicate": "capital"}),
        ("Forward (alias)", {"subject": "Remulak", "predicate": "capital city"}),
        ("Forward (alias)", {"subject": "Remulak", "predicate": "head of state"}),
        ("Forward (exact)", {"subject": "Grand Vizier Korth", "predicate": "real name"}),
        ("Reverse", {"predicate": "capital", "obj": "Zelphos"}),
        ("All facts", {"subject": "Draveth"}),
    ]

    for label, kwargs in demos:
        results = kg.lookup(**kwargs)
        print(f"  {label}: {kwargs}")
        for r in results:
            print(f"    → {r['subject']} | {r['predicate']} | {r['object']}")
        print()
