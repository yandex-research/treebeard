"""Tests for the IMO25 solver script helpers."""

from __future__ import annotations

from open_deep_think.scripts.imo25_solve import (
    build_solver_messages,
    build_verification_prompt,
    extract_section,
    is_yes_response,
    sanitize_model_name,
)


def test_extract_section_after_returns_text_after_marker() -> None:
    text = "Intro\nDetailed Solution\nProof body."
    assert extract_section(text, "Detailed Solution", after=True) == "Proof body."


def test_extract_section_before_returns_text_before_marker() -> None:
    text = "Summary\nDetailed Verification\nLog"
    assert extract_section(text, "Detailed Verification", after=False) == "Summary"


def test_extract_section_returns_empty_when_marker_missing() -> None:
    assert extract_section("No marker", "Detailed Solution", after=True) == ""


def test_build_solver_messages_preserves_prompt_order() -> None:
    messages = build_solver_messages("Problem", ("Prompt A", "Prompt B"))
    assert messages == [
        {"role": "user", "content": "Problem"},
        {"role": "user", "content": "Prompt A"},
        {"role": "user", "content": "Prompt B"},
    ]


def test_build_verification_prompt_includes_problem_and_solution() -> None:
    prompt = build_verification_prompt(
        problem_statement="Find x.",
        solution_text="Summary\nDetailed Solution\nHence x=3.",
    )
    assert "Find x." in prompt
    assert "Hence x=3." in prompt


def test_is_yes_response_requires_word_boundary() -> None:
    assert is_yes_response("yes") is True
    assert is_yes_response("YES, correct") is True
    assert is_yes_response("yesterday") is False


def test_sanitize_model_name_replaces_path_and_colon() -> None:
    assert sanitize_model_name("provider/model:latest") == "provider__model_latest"
