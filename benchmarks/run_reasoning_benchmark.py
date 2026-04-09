"""Backward-compatible re-export — use benchmarks.runners.reasoning instead."""

from benchmarks.runners.reasoning import *  # noqa: F401, F403
from benchmarks.runners.reasoning import main

if __name__ == "__main__":
    main()
