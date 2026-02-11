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


# Pattern to match <think>...</think>
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_reasoning(completion: ChatCompletion) -> str:
    """Extract reasoning text from a ChatCompletion.

    Returns everything inside <think>...</think>
    (all such blocks concatenated). Empty string if none.

    Args:
        completion: OpenAI SDK ChatCompletion response

    Returns:
        Reasoning text or empty string

    """
    text = _get_message_content(completion)
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
    text = _get_message_content(completion)
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
