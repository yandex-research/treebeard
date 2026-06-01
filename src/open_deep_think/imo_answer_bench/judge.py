"""Judge model answers against ground truth using an API-based judge."""

import re
from typing import Optional, Union

from open_deep_think.api import single_turn_api_call
from open_deep_think.imo_answer_bench.templates import JudgeType, build_judge_prompt


def _strip_thinking_blocks(text: str) -> str:
    """Remove all <thinking>...</thinking> blocks from text, including tags."""
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)


def judge_answer(  # noqa: PLR0913
    problem_statement: str,
    model_solution: str,
    ground_truth: str,
    judge_model_name: str,
    judge_type: JudgeType,
    guidelines: Optional[str] = None,
    max_tokens: int = 2048,
) -> tuple[Union[bool | int], str]:
    """Use the API-based judge model to compare model solution with ground truth.

    Args:
        problem_statement: The problem statement from the dataset.
        model_solution: The full solution text from the model.
        ground_truth: The ground truth answer from dataset.
        judge_model_name: Name of the judge model (e.g., "gemini-3-flash").
        judge_type: Whether to judge an answer or a proof.
        guidelines: Grading guidelines for proof judging.
        max_tokens: Maximum tokens for the judge response.

    Returns:
        A tuple of (is_correct, judge_response_text) where is_correct is True
        if the model answer matches the ground truth, and judge_response_text
        is the raw text returned by the judge model.

    """
    # Format the prompt with the problem, solution, and answer
    prompt = build_judge_prompt(
        problem_statement=problem_statement,
        model_solution=model_solution,
        golden_answer=ground_truth,
        guidelines=guidelines,
        judge_type=judge_type,
    )

    # Make API request
    try:
        completion = single_turn_api_call(
            model=judge_model_name,
            prompt=prompt,
            max_tokens=max_tokens,
        )

        # Extract response content from ChatCompletion
        if not completion.choices:
            return False, ""
        message = completion.choices[0].message
        response_text = message.content if message.content is not None else ""

        response_text = _strip_thinking_blocks(response_text)

        # Parse the judge's response - expecting \boxed{Correct} or \boxed{Incorrect}
        if judge_type == JudgeType.ANSWER:
            boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", response_text)

            if boxed_matches:
                verdict = boxed_matches[-1].strip().lower()
                return verdict == "correct", response_text
            return "correct" in response_text.lower() and "incorrect" not in response_text.lower(), response_text
        if judge_type == JudgeType.PROOF:
            if "<points>7 out of 7</points>" in response_text.lower():
                return 7, response_text
            if "<points>6 out of 7</points>" in response_text.lower():
                return 6, response_text
            if "<points>1 out of 7</points>" in response_text.lower():
                return 1, response_text
            return 0, response_text
        # Fallback: check if "correct" appears in the response

    except (KeyError, TypeError, ValueError, OSError):
        return False, ""
