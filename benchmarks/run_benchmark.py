"""Backward-compatible re-export — use benchmarks.runners.baseline instead."""

from benchmarks.runners.baseline import *  # noqa: F401, F403
from benchmarks.runners.baseline import main

if __name__ == "__main__":
    main()
