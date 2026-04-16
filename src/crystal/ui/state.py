"""Shared app-level state for the Crystal UI.

Owns the module-level singletons (compiled graph, available KGs, default mode).
Tab modules import from here to avoid re-initializing these objects.
"""

from pathlib import Path

from crystal.graph import build_crystal_graph
from crystal.tools.kg import remulak_kg
from crystal.tools.kg.legal import load_legal_kg


_graph = build_crystal_graph()

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LEGAL_DB_PATH = _ROOT / "data" / "legal.sqlite"
_legal_kg = load_legal_kg(_LEGAL_DB_PATH)

KG_MODES: dict = {}
if _legal_kg is not None:
    KG_MODES["Legal (SCOTUS — SQLite)"] = _legal_kg
KG_MODES["Remulak (demo)"] = remulak_kg

_DEFAULT_KG_MODE = "Legal (SCOTUS — SQLite)" if _legal_kg is not None else "Remulak (demo)"
_default_kg = KG_MODES[_DEFAULT_KG_MODE]
