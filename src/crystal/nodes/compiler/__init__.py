"""Prompt compiler — split into domain-specific modules.

- core.py: classification logic and the main prompt_compiler_node
- kg.py: KG-specific formatting and augmented prompt building
- math.py: math-specific formatting and simplified prompt building
"""

from crystal.nodes.compiler.core import (
    QUESTION_FILLER,
    REASONING_SIGNALS,
    _classify_prompt_type,
    _has_reasoning_signals,
    prompt_compiler_node,
)
from crystal.nodes.compiler.kg import (
    _build_kg_augmented_prompt,
    _format_kg_results,
)
from crystal.nodes.compiler.math import (
    _build_simplified_prompt,
    _format_result,
)

__all__ = [
    "QUESTION_FILLER",
    "REASONING_SIGNALS",
    "_classify_prompt_type",
    "_has_reasoning_signals",
    "prompt_compiler_node",
    "_build_kg_augmented_prompt",
    "_format_kg_results",
    "_build_simplified_prompt",
    "_format_result",
]
