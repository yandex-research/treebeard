"""Tests for open_deep_think.imo_answer_bench.extract."""

from __future__ import annotations

from openai.types.chat import ChatCompletion

from open_deep_think.imo_answer_bench.extract import (
    _get_message_content,
    extract_boxed_answer,
    extract_reasoning,
    extract_solution,
)


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


def _completion_empty_choices() -> ChatCompletion:
    """Build a ChatCompletion with no choices."""
    return ChatCompletion.model_validate(
        {
            "id": "test-id",
            "choices": [],
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
        }
    )


# --- _get_message_content ---


class TestGetMessageContent:
    def test_returns_content_when_present(self) -> None:
        completion = _completion_with_content("Hello, world.")
        assert _get_message_content(completion) == "Hello, world."

    def test_returns_empty_when_content_none(self) -> None:
        completion = _completion_with_content(None)
        assert _get_message_content(completion) == ""

    def test_returns_empty_when_no_choices(self) -> None:
        completion = _completion_empty_choices()
        assert _get_message_content(completion) == ""


# --- extract_reasoning ---


class TestExtractReasoning:
    def test_returns_empty_when_no_think_block(self) -> None:
        completion = _completion_with_content("Just the answer: 42.")
        assert extract_reasoning(completion) == ""

    def test_returns_think_content_single_block(self) -> None:
        completion = _completion_with_content("<think>Let me add 2 and 2.</think>\nSo the answer is 4.")
        assert extract_reasoning(completion) == "Let me add 2 and 2."

    def test_returns_think_content_multiline(self) -> None:
        completion = _completion_with_content("<think>Step 1: consider x.\nStep 2: therefore y.</think>\nDone.")
        assert extract_reasoning(completion) == "Step 1: consider x.\nStep 2: therefore y."

    def test_returns_concatenated_multiple_think_blocks(self) -> None:
        completion = _completion_with_content(
            "<think>First thought.</think>\nMiddle.\n<think>Second thought.</think>\nEnd."
        )
        assert extract_reasoning(completion) == "First thought.\n\nSecond thought."

    def test_returns_empty_when_no_choices(self) -> None:
        completion = _completion_empty_choices()
        assert extract_reasoning(completion) == ""

    def test_strips_whitespace_inside_think(self) -> None:
        completion = _completion_with_content("<think>  inner text  \n </think>\nAnswer.")
        assert extract_reasoning(completion) == "inner text"


# --- extract_solution ---


class TestExtractSolution:
    def test_returns_full_content_when_no_think_block(self) -> None:
        completion = _completion_with_content("The answer is 42.")
        assert extract_solution(completion) == "The answer is 42."

    def test_returns_rest_after_removing_single_think_block(self) -> None:
        completion = _completion_with_content("<think>Reasoning here.</think>\nSo the answer is 4.")
        assert extract_solution(completion) == "So the answer is 4."

    def test_returns_empty_when_only_think_blocks(self) -> None:
        completion = _completion_with_content("<think>Only reasoning.</think>")
        assert extract_solution(completion) == ""

    def test_returns_empty_when_no_choices(self) -> None:
        completion = _completion_empty_choices()
        assert extract_solution(completion) == ""

    def test_strips_outer_whitespace(self) -> None:
        completion = _completion_with_content("<think>think</think>\n  Answer line.  ")
        assert extract_solution(completion) == "Answer line."


# --- extract_boxed_answer ---


class TestExtractBoxedAnswer:
    def test_returns_none_when_no_boxed(self) -> None:
        assert extract_boxed_answer("No boxed here.") is None
        assert extract_boxed_answer("") is None

    def test_returns_content_of_single_boxed(self) -> None:
        assert extract_boxed_answer(r"The answer is \boxed{42}.") == "42"

    def test_returns_last_boxed_when_multiple(self) -> None:
        text = r"First \boxed{1} then \boxed{2} and finally \boxed{3}."
        assert extract_boxed_answer(text) == "3"

    def test_strips_inner_whitespace(self) -> None:
        assert extract_boxed_answer(r"\boxed{  x  }") == "x"

    def test_handles_nested_braces(self) -> None:
        assert extract_boxed_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_handles_empty_boxed(self) -> None:
        assert extract_boxed_answer(r"\boxed{}") == ""
