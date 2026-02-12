"""Tests for open_deep_think.imo_answer_bench.judge."""

from __future__ import annotations

from openai.types.chat import ChatCompletion

from open_deep_think.imo_answer_bench import judge


def _completion_with_content(content: str | None) -> ChatCompletion:
    """Build a minimal ChatCompletion with the given message content."""
    return ChatCompletion.model_validate({
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
    })


def test_strip_thinking_blocks_removes_tagged_content() -> None:
    text = "before<thinking>hidden</thinking>after"
    assert judge._strip_thinking_blocks(text) == "beforeafter"


def test_judge_answer_uses_last_boxed(monkeypatch) -> None:
    completion = _completion_with_content(r"\\boxed{Incorrect} then \\boxed{Correct}")
    monkeypatch.setattr(judge, "single_turn_api_call", lambda **_: completion)

    assert judge.judge_answer("p", "s", "g", "m") is True


def test_judge_answer_ignores_boxed_inside_thinking(monkeypatch) -> None:
    completion = _completion_with_content(
        r"<thinking>\\boxed{Correct}</thinking> final \\boxed{Incorrect}"
    )
    monkeypatch.setattr(judge, "single_turn_api_call", lambda **_: completion)

    assert judge.judge_answer("p", "s", "g", "m") is False
