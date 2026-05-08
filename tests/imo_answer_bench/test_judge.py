"""Tests for open_deep_think.imo_answer_bench.judge."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai.types.chat import ChatCompletion

from open_deep_think.imo_answer_bench import judge

if TYPE_CHECKING:
    import pytest


def _completion_with_content(content: str | None) -> ChatCompletion:
    """Build a minimal ChatCompletion with the given message content."""
    return ChatCompletion.model_validate(
        {
            "id": "test-id",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                }
            ],
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
        }
    )


def test_judge_answer_strips_thinking_before_parsing_boxed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thinking blocks are stripped before boxed answer extraction."""
    completion = _completion_with_content(r"<thinking>analysis</thinking> final \\boxed{Correct}")
    monkeypatch.setattr(judge, "single_turn_api_call", lambda **_: completion)

    is_correct, response_text = judge.judge_answer("p", "s", "g", "m")
    assert is_correct is True
    assert "Correct" in response_text


def test_judge_answer_uses_last_boxed(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The last \\boxed{} in the response is used as the verdict."""
    completion = _completion_with_content(r"\\boxed{Incorrect} then \\boxed{Correct}")
    monkeypatch.setattr(judge, "single_turn_api_call", lambda **_: completion)

    is_correct, response_text = judge.judge_answer("p", "s", "g", "m")
    assert is_correct is True
    assert "Correct" in response_text


def test_judge_answer_ignores_boxed_inside_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Boxed answers inside thinking blocks are ignored; only the final answer counts."""
    completion = _completion_with_content(r"<thinking>\\boxed{Correct}</thinking> final \\boxed{Incorrect}")
    monkeypatch.setattr(judge, "single_turn_api_call", lambda **_: completion)

    is_correct, response_text = judge.judge_answer("p", "s", "g", "m")
    assert is_correct is False
    assert "Incorrect" in response_text


def test_judge_answer_returns_false_on_empty_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns (False, '') when the completion has no choices."""
    empty_completion = ChatCompletion.model_validate(
        {
            "id": "test-id",
            "choices": [],
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
        }
    )
    monkeypatch.setattr(judge, "single_turn_api_call", lambda **_: empty_completion)

    is_correct, response_text = judge.judge_answer("p", "s", "g", "m")
    assert is_correct is False
    assert response_text == ""


def test_judge_answer_fallback_when_no_boxed(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Falls back to keyword search when no \\boxed{} is present."""
    completion = _completion_with_content("The answer is correct based on my analysis.")
    monkeypatch.setattr(judge, "single_turn_api_call", lambda **_: completion)

    is_correct, response_text = judge.judge_answer("p", "s", "g", "m")
    assert is_correct is True
    assert "correct" in response_text.lower()


def test_judge_answer_fallback_incorrect_when_no_boxed(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Falls back to False when response contains 'incorrect' but no \\boxed{}."""
    completion = _completion_with_content("The answer is incorrect.")
    monkeypatch.setattr(judge, "single_turn_api_call", lambda **_: completion)

    is_correct, response_text = judge.judge_answer("p", "s", "g", "m")
    assert is_correct is False
    assert "incorrect" in response_text.lower()
