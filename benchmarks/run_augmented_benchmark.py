"""Backward-compatible re-export — use benchmarks.runners.augmented instead."""

from benchmarks.runners.augmented import *  # noqa: F401, F403
from benchmarks.runners.augmented import main

if __name__ == "__main__":
    main()
