"""Per-question result caching for benchmark runs.

Caches individual question results to disk so:
- Rate limits just slow us down, they don't lose progress
- Re-runs skip already-answered questions
- Works across sessions and providers

Cache key = SHA256(question + arm_name + model_name).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "results" / "question_cache"


def cache_key(question: str, arm: str, model: str) -> str:
    """Deterministic cache key for a (question, arm, model) triple."""
    raw = f"{question}\x00{arm}\x00{model}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cached(question: str, arm: str, model: str) -> dict | None:
    """Return cached result dict, or None if not cached."""
    path = CACHE_DIR / f"{cache_key(question, arm, model)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def set_cached(question: str, arm: str, model: str, result: dict) -> Path:
    """Write a result dict to the question cache. Returns the cache path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{cache_key(question, arm, model)}.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return path


def cache_stats() -> dict:
    """Return cache statistics."""
    if not CACHE_DIR.exists():
        return {"entries": 0, "size_bytes": 0}
    files = list(CACHE_DIR.glob("*.json"))
    total_bytes = sum(f.stat().st_size for f in files)
    return {"entries": len(files), "size_bytes": total_bytes}


def clear_cache() -> int:
    """Delete all cached results. Returns number of entries deleted."""
    if not CACHE_DIR.exists():
        return 0
    files = list(CACHE_DIR.glob("*.json"))
    for f in files:
        f.unlink()
    return len(files)
