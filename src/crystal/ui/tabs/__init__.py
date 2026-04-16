"""Per-tab modules for the Crystal UI.

Each submodule exposes a `build_<tab>_tab(shared, ...)` function that
creates the Gradio components for that tab and returns a dataclass of
component handles that other tabs (and `build_ui`) can wire up.
"""
