"""Tests for open_deep_think.imo_answer_bench.templates."""

from __future__ import annotations

from open_deep_think.imo_answer_bench.templates import (
    PROBLEM_PROMPT_PREFIX,
    build_problem_prompt,
)


def test_build_problem_prompt_uses_default_prefix() -> None:
    """Default prompt builder output should be deterministic."""
    problem = "Compute 1+1."

    prompt = build_problem_prompt(problem)

    assert prompt == f"{PROBLEM_PROMPT_PREFIX}\n\n{problem}"


def test_build_problem_prompt_allows_custom_prefix() -> None:
    """Custom prefixes should support reproducible prompt variants."""
    problem = "Find x."
    custom_prefix = "Answer only with an integer."

    prompt = build_problem_prompt(problem, prefix=custom_prefix)

    assert prompt == "Answer only with an integer.\n\nFind x."
