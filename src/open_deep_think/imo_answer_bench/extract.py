"""API utilities for single-turn model calls and response extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion


def _get_message_content(completion: ChatCompletion) -> str:
    """Get the full message content from a ChatCompletion, or empty string."""
    try:
        if not completion.choices:
            return ""
        content = completion.choices[0].message.content
    except (KeyError, IndexError, TypeError):
        return ""
    else:
        return content if content is not None else ""


def _get_message_reasoning(completion: ChatCompletion) -> str:
    """Get reasoning text from a ChatCompletion, or empty string.

    Prefers the explicit `message.reasoning` field when available.
    Returns empty string for missing or unsupported reasoning payloads.
    """
    try:
        if not completion.choices:
            return ""
        reasoning = getattr(completion.choices[0].message, "reasoning", None)
    except (KeyError, IndexError, TypeError):
        return ""

    if isinstance(reasoning, str):
        return reasoning.strip()

    return ""


# Pattern to match <think>...</think>
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _normalize_missing_opening_think(text: str) -> str:
    """Normalize text when the first ``<think>`` token is missing.

    Some LLM servers may omit the very first ``<think>`` token while still
    emitting the closing ``</think>`` token. When a closing token appears
    before any opening token, prepend ``<think>`` so standard extraction
    logic can work unchanged.
    """
    close_idx = text.find("</think>")
    if close_idx < 0:
        return text

    first_open_idx = text.find("<think>")
    if first_open_idx != -1 and first_open_idx < close_idx:
        return text

    return f"<think>{text}"


def extract_reasoning(completion: ChatCompletion) -> str:
    """Extract reasoning text from a ChatCompletion.

    Prefers `message.reasoning` when present, and otherwise falls back to
    everything inside <think>...</think> (all such blocks concatenated).
    Empty string if none.

    Args:
        completion: OpenAI SDK ChatCompletion response

    Returns:
        Reasoning text or empty string

    """
    direct_reasoning = _get_message_reasoning(completion)
    if direct_reasoning:
        return direct_reasoning

    text = _normalize_missing_opening_think(_get_message_content(completion))
    parts = _THINK_PATTERN.findall(text)
    return "\n\n".join(p.strip() for p in parts).strip() if parts else ""


def extract_solution(completion: ChatCompletion) -> str:
    """Extract solution text from a ChatCompletion.

    Returns the rest of the message content after removing <think>...</think>
    blocks.

    Args:
        completion: OpenAI SDK ChatCompletion response

    Returns:
        Solution text (no think blocks) or empty string

    """
    text = _normalize_missing_opening_think(_get_message_content(completion))
    rest = _THINK_PATTERN.sub("", text)
    return rest.strip() if rest else ""


def extract_boxed_answer(text: str) -> str | None:
    r"""Extract the answer from \boxed{} notation.

    Args:
        text: The text containing the boxed answer

    Returns:
        The extracted answer or None if not found

    """
    # Pattern to match \boxed{...} with nested braces support
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)

    if matches:
        # Return the last boxed answer if multiple exist
        return matches[-1].strip()

    return None
