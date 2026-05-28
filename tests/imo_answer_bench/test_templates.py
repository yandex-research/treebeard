"""Tests for open_deep_think.imo_answer_bench.templates."""

from __future__ import annotations

from open_deep_think.imo_answer_bench.templates import (
    PROBLEM_PROMPT_PREFIX,
    JudgeType,
    build_judge_prompt,
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


def test_build_judge_prompt_replaces_all_placeholders() -> None:
    """Judge prompt builder should fill all data fields in one pass."""
    prompt = build_judge_prompt(
        judge_type=JudgeType.ANSWER,
        problem_statement="Find x.",
        model_solution="\\boxed{3}",
        golden_answer="3",
    )

    assert "Find x." in prompt
    assert "\\boxed{3}" in prompt
    assert "3" in prompt
    assert "{{Problem_Statement}}" not in prompt
    assert "{{Model_Solution}}" not in prompt
    assert "{{Golden_Answer}}" not in prompt
