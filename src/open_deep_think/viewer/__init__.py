"""Streamlit-based viewer for tournament-merge-improve trace logs.

The viewer ingests the per-task artefacts produced by
:mod:`open_deep_think.scripts.tournament_merge_improve` (``progress.json``,
``llm_outputs.jsonl`` and ``solution.txt``) and surfaces them as a
MathArena-style markdown + KaTeX layout with a radio-selectable candidate
list.
"""

from open_deep_think.viewer import data, render

__all__ = ["data", "render"]
