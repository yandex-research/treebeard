"""Judge model answers against ground truth using an API-based judge."""

import re

from open_deep_think.api import single_turn_api_call
from open_deep_think.imo_answer_bench.templates import JUDGE_PROMPT_TEMPLATE


def judge_answer(  # noqa: PLR0913
    problem_statement: str,
    model_solution: str,
    ground_truth: str,
    judge_model_name: str,
    judge_prompt_template: str = JUDGE_PROMPT_TEMPLATE,
    max_tokens: int = 2048,
) -> bool:
    """Use the API-based judge model to compare model solution with ground truth.

    Args:
        problem_statement: The problem statement from the dataset
        model_solution: The full solution text from the model
        ground_truth: The ground truth answer from dataset
        judge_model_name: Name of the judge model (e.g., "gemini-3-flash")
        judge_prompt_template: Template for the judge prompt with placeholders
        max_tokens: Maximum tokens for the judge response

    Returns:
        True if answers match, False otherwise

    """
    # Format the prompt with the problem, solution, and answer
    prompt = judge_prompt_template.replace("{{Problem_Statement}}", problem_statement)
    prompt = prompt.replace("{{Model_Solution}}", model_solution)
    prompt = prompt.replace("{{Golden_Answer}}", ground_truth)

    # Make API request
    try:
        completion = single_turn_api_call(
            model=judge_model_name,
            prompt=prompt,
            max_tokens=max_tokens,
        )

        # Extract response content from ChatCompletion
        if not completion.choices:
            return False
        message = completion.choices[0].message
        response_text = message.content if message.content is not None else ""

        # Parse the judge's response - expecting \boxed{Correct} or \boxed{Incorrect}
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", response_text)

        if boxed_match:
            verdict = boxed_match.group(1).strip().lower()
            return verdict == "correct"

        # Fallback: check if "correct" appears in the response
        return "correct" in response_text.lower() and "incorrect" not in response_text.lower()

    except (KeyError, TypeError, ValueError, OSError):
        return False
