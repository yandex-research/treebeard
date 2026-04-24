"""Tests for the simple_tournament script helpers."""

from __future__ import annotations

import pytest

from open_deep_think.scripts.simple_tournament import (
    build_solver_messages,
    build_verification_prompt,
    extract_section,
    is_power_of_two,
    is_yes_response,
    parse_pick,
    sanitize_model_name,
)

_PICK_TWO = 2  # named constant to satisfy PLR2004

# ── extract_section ───────────────────────────────────────────────────────────


def test_extract_section_after_returns_text_after_marker() -> None:
    text = "Intro\nDetailed Solution\nProof body."
    assert extract_section(text, "Detailed Solution", after=True) == "Proof body."


def test_extract_section_before_returns_text_before_marker() -> None:
    text = "Summary\nDetailed Verification\nLog"
    assert extract_section(text, "Detailed Verification", after=False) == "Summary"


def test_extract_section_returns_empty_when_marker_missing() -> None:
    assert extract_section("No marker here", "Detailed Solution", after=True) == ""


# ── is_yes_response ───────────────────────────────────────────────────────────


def test_is_yes_response_detects_standalone_yes() -> None:
    assert is_yes_response("yes") is True


def test_is_yes_response_case_insensitive() -> None:
    assert is_yes_response("YES, the solution is correct") is True


def test_is_yes_response_rejects_substring() -> None:
    assert is_yes_response("yesterday") is False


# ── parse_pick ────────────────────────────────────────────────────────────────


def test_parse_pick_returns_1() -> None:
    assert parse_pick("1") == 1


def test_parse_pick_returns_2() -> None:
    assert parse_pick("2") == _PICK_TWO


def test_parse_pick_extracts_digit_from_prose() -> None:
    assert parse_pick("I choose solution 2 because it is more rigorous.") == _PICK_TWO


def test_parse_pick_defaults_to_1_when_no_digit_found() -> None:
    assert parse_pick("neither is good") == 1


def test_parse_pick_prefers_first_digit() -> None:
    # First valid digit wins.
    assert parse_pick("1 or 2") == 1


# ── is_power_of_two ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64])
def test_is_power_of_two_valid(n: int) -> None:
    assert is_power_of_two(n) is True


@pytest.mark.parametrize("n", [0, 3, 5, 6, 7, 9, 10, -1, -4])
def test_is_power_of_two_invalid(n: int) -> None:
    assert is_power_of_two(n) is False


# ── sanitize_model_name ───────────────────────────────────────────────────────


def test_sanitize_model_name_replaces_slash_and_colon() -> None:
    assert sanitize_model_name("provider/model:latest") == "provider__model_latest"


def test_sanitize_model_name_no_special_chars() -> None:
    assert sanitize_model_name("mymodel") == "mymodel"


# ── build_solver_messages ─────────────────────────────────────────────────────


def test_build_solver_messages_single_problem() -> None:
    messages = build_solver_messages("Find x.", ())
    assert messages == [{"role": "user", "content": "Find x."}]


def test_build_solver_messages_with_extra_prompts() -> None:
    messages = build_solver_messages("Problem", ("Hint A", "Hint B"))
    assert messages == [
        {"role": "user", "content": "Problem"},
        {"role": "user", "content": "Hint A"},
        {"role": "user", "content": "Hint B"},
    ]


# ── build_verification_prompt ─────────────────────────────────────────────────


def test_build_verification_prompt_includes_problem() -> None:
    prompt = build_verification_prompt(
        problem_statement="Find all integers n.",
        solution_text="Summary\nDetailed Solution\nHence n=0.",
    )
    assert "Find all integers n." in prompt


def test_build_verification_prompt_includes_detailed_solution_body() -> None:
    prompt = build_verification_prompt(
        problem_statement="Find all integers n.",
        solution_text="Summary\nDetailed Solution\nHence n=0.",
    )
    assert "Hence n=0." in prompt


def test_build_verification_prompt_missing_marker_uses_empty_solution() -> None:
    prompt = build_verification_prompt(
        problem_statement="Find x.",
        solution_text="No marker here.",
    )
    # Problem still present; solution body is empty but prompt is still built.
    assert "Find x." in prompt
