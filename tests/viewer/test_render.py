"""Tests for the rendering helpers."""

from __future__ import annotations

from open_deep_think.viewer.render import (
    collapse_blank_lines,
    format_usage,
    normalise_math,
    short_phase_label,
)


def test_normalise_math_converts_inline_delimiters() -> None:
    assert normalise_math(r"Let \(x\) be even.") == "Let $x$ be even."


def test_normalise_math_converts_display_delimiters() -> None:
    assert normalise_math(r"\[a^2 + b^2 = c^2\]") == "$$a^2 + b^2 = c^2$$"


def test_normalise_math_preserves_code_spans() -> None:
    text = r"see `\(x\)` then \(y\)"
    converted = normalise_math(text)
    assert "`\\(x\\)`" in converted, "code span should remain untouched"
    assert converted.endswith("then $y$")


def test_normalise_math_preserves_fenced_code() -> None:
    text = "```python\n# uses \\(x\\) inside\nprint(1)\n```\nand \\(y\\)"
    converted = normalise_math(text)
    assert "\\(x\\)" in converted, "fenced code block should be preserved"
    assert converted.endswith("and $y$")


def test_normalise_math_handles_empty_string() -> None:
    assert normalise_math("") == ""


def test_collapse_blank_lines_caps_consecutive_blanks() -> None:
    text = "a\n\n\n\n\nb"
    assert collapse_blank_lines(text, max_consecutive=2) == "a\n\nb"


def test_collapse_blank_lines_keeps_short_runs() -> None:
    text = "a\n\nb\nc"
    assert collapse_blank_lines(text) == "a\n\nb\nc"


def test_short_phase_label_known_values() -> None:
    assert short_phase_label("initial_solution") == "Initial solve"
    assert short_phase_label("verification") == "Verification"
    assert short_phase_label("self_improvement") == "Self-improvement"


def test_short_phase_label_unknown_falls_back_to_human_form() -> None:
    assert short_phase_label("weird_phase") == "Weird phase"


def test_format_usage_renders_all_fields() -> None:
    assert format_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}) == (
        "prompt 10 · completion 5 · total 15"
    )


def test_format_usage_handles_partial_input() -> None:
    assert format_usage({"prompt_tokens": 10}) == "prompt 10"


def test_format_usage_handles_none() -> None:
    assert format_usage(None) == ""
