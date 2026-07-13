"""Tests for the tournament_merge script helpers."""

from __future__ import annotations

import pytest

from open_deep_think.imo_answer_bench.templates import (
    TOURNAMENT_MERGE_SYSTEM_PROMPT,
    build_tournament_merge_prompt,
)
from open_deep_think.scripts.tournament_merge import (
    build_solver_messages,
    build_verification_prompt,
    extract_section,
    is_power_of_two,
    is_yes_response,
    sanitize_model_name,
)

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


# ── build_tournament_merge_prompt ─────────────────────────────────────────────


def test_build_tournament_merge_prompt_contains_all_sections() -> None:
    """Merge prompt must contain the problem and both solutions."""
    prompt = build_tournament_merge_prompt(
        problem="Prove that 1+1=2.",
        solution_1="Solution A text.",
        solution_2="Solution B text.",
    )
    assert "Prove that 1+1=2." in prompt
    assert "Solution A text." in prompt
    assert "Solution B text." in prompt


def test_build_tournament_merge_prompt_labels_solutions() -> None:
    """The prompt must clearly label Solution 1 and Solution 2."""
    prompt = build_tournament_merge_prompt(
        problem="P",
        solution_1="S1",
        solution_2="S2",
    )
    assert "Solution 1" in prompt
    assert "Solution 2" in prompt


def test_build_tournament_merge_prompt_no_verification_sections() -> None:
    """The merge prompt must not contain verification report sections."""
    prompt = build_tournament_merge_prompt(
        problem="P",
        solution_1="S1",
        solution_2="S2",
    )
    assert "Verification Report" not in prompt


# ── TOURNAMENT_MERGE_SYSTEM_PROMPT ────────────────────────────────────────────


def test_tournament_merge_system_prompt_is_non_empty_string() -> None:
    assert isinstance(TOURNAMENT_MERGE_SYSTEM_PROMPT, str)
    assert len(TOURNAMENT_MERGE_SYSTEM_PROMPT) > 0


def test_tournament_merge_system_prompt_mentions_merge_goal() -> None:
    """The system prompt should instruct the model to aggregate / merge solutions."""
    lower = TOURNAMENT_MERGE_SYSTEM_PROMPT.lower()
    assert "merge" in lower or "synthesis" in lower or "combin" in lower or "aggregate" in lower
