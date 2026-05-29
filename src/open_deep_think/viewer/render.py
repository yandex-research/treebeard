r"""Helpers that prepare LLM-generated text for client-side LaTeX rendering.

The tournament logs use mixed LaTeX delimiters: solver and verifier outputs
typically wrap inline math in ``\(...\)`` and display math in ``\[...\]``
(LaTeX standard delimiters), while Streamlit's built-in markdown renderer
expects ``$...$`` / ``$$...$$``.  Converting between the two is purely a
syntactic substitution but has to be careful around code fences and escaped
backslashes — this module centralises that logic.
"""

from __future__ import annotations

import re

_CODE_FENCE = re.compile(r"```.*?```|`[^`]*`", re.DOTALL)
"""Matches fenced or inline code blocks so we do not rewrite their contents."""

_INLINE_OPEN = re.compile(r"(?<!\\)\\\(")
_INLINE_CLOSE = re.compile(r"(?<!\\)\\\)")
_DISPLAY_OPEN = re.compile(r"(?<!\\)\\\[")
_DISPLAY_CLOSE = re.compile(r"(?<!\\)\\\]")


def _convert_segment(segment: str) -> str:
    """Rewrite the LaTeX delimiters in a non-code segment."""
    segment = _DISPLAY_OPEN.sub("$$", segment)
    segment = _DISPLAY_CLOSE.sub("$$", segment)
    segment = _INLINE_OPEN.sub("$", segment)
    return _INLINE_CLOSE.sub("$", segment)


def normalise_math(text: str) -> str:
    r"""Replace ``\(...\)`` and ``\[...\]`` with ``$...$`` / ``$$...$$``.

    Code spans (fenced or backtick-quoted) are left untouched so that any
    literal ``\(`` inside, e.g., a Python snippet survives unchanged.

    Args:
        text: Raw markdown coming from a model response.

    Returns:
        Markdown with LaTeX delimiters compatible with ``st.markdown``.

    """
    if not text:
        return text
    out: list[str] = []
    last = 0
    for match in _CODE_FENCE.finditer(text):
        out.append(_convert_segment(text[last : match.start()]))
        out.append(match.group(0))
        last = match.end()
    out.append(_convert_segment(text[last:]))
    return "".join(out)


def collapse_blank_lines(text: str, *, max_consecutive: int = 2) -> str:
    """Collapse runs of >= ``max_consecutive + 1`` blank lines, preserving paragraphs.

    Some verifier outputs contain very long stretches of empty lines; this
    keeps things visually compact without altering content.
    """
    if not text:
        return text
    pattern = re.compile(r"\n{" + str(max_consecutive + 1) + r",}")
    return pattern.sub("\n" * max_consecutive, text)


def short_phase_label(phase: str) -> str:
    """Human-friendly label for a pipeline phase string."""
    mapping = {
        "initial_solution": "Initial solve",
        "verification": "Verification",
        "verification_check": "Classifier",
        "self_improvement": "Self-improvement",
        "tournament_merge": "Merge",
    }
    return mapping.get(phase, phase.replace("_", " ").capitalize())


def format_usage(usage: dict[str, object] | None) -> str:
    """Render an OpenAI-style usage dict as a compact one-liner."""
    if not usage:
        return ""
    pieces: list[str] = []
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if prompt_tokens is not None:
        pieces.append(f"prompt {prompt_tokens}")
    if completion_tokens is not None:
        pieces.append(f"completion {completion_tokens}")
    if total_tokens is not None:
        pieces.append(f"total {total_tokens}")
    return " · ".join(pieces)
