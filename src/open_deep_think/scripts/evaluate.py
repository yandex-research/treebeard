"""IMO Local Evaluation Script.

This script evaluates model solutions against the IMO AnswerBench dataset.
It uses an API-based judge model (Gemini 3 Flash) to compare model answers with ground truth.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import urllib3
from datasets import load_dataset
from tqdm import tqdm

from open_deep_think.imo_answer_bench.extract import extract_boxed_answer
from open_deep_think.imo_answer_bench.judge import judge_answer
from open_deep_think.imo_answer_bench.templates import JUDGE_PROMPT_TEMPLATE

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


def load_solutions(solutions_dir: Path) -> dict[str, str]:
    """Load all solution files from the directory.

    Args:
        solutions_dir: Path to directory containing solution files

    Returns:
        Dictionary mapping task_id to solution text

    """
    solutions = {}

    # Pattern: Task_{task_id}_seed_42.txt
    pattern = re.compile(r"Task_(\d+)_solution.txt")

    for file_path in solutions_dir.rglob("Task_*_solution.txt"):
        match = pattern.match(file_path.name)
        if match:
            task_id = match.group(1)
            with file_path.open(encoding="utf-8") as f:
                solutions[task_id] = f.read()

    return solutions


def evaluate_solutions(
    solutions_dir: str,
    judge_prompt_template: str = JUDGE_PROMPT_TEMPLATE,
    judge_model_name: str = "gemini-3-flash",
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Evaluate all solutions against the IMO AnswerBench dataset.

    Args:
        solutions_dir: Path to directory containing solution files
        judge_prompt_template: Template for the judge prompt
        judge_model_name: Name of the judge model (e.g., "gemini-3-flash")
        max_tokens: Maximum tokens for judge responses

    Returns:
        Dictionary containing evaluation results

    """
    solutions_path = Path(solutions_dir)

    if not solutions_path.exists():
        msg = f"Solutions directory does not exist: {solutions_dir}"
        raise ValueError(msg)

    # Load solutions
    logger.info("Loading solutions from: %s", solutions_dir)
    solutions = load_solutions(solutions_path)
    logger.info("Found %s solution files", len(solutions))

    # Load dataset
    logger.info("Loading IMO AnswerBench dataset...")
    dataset = load_dataset("Hwilner/imo-answerbench", split="train")
    logger.info("Loaded %s problems from dataset", len(dataset))
    logger.info("Using judge model: %s", judge_model_name)

    # Evaluation metrics
    results = {"total": 0, "correct": 0, "incorrect": 0, "no_boxed_answer": 0, "missing_solution": 0, "details": []}

    # Iterate over dataset problems
    logger.info("Evaluating solutions...")
    pbar = tqdm(dataset, desc="Evaluating (Acc: N/A)")

    for idx, problem in enumerate(pbar):
        task_id = str(idx)  # Assuming task_id corresponds to dataset index
        ground_truth = problem["Short Answer"]
        problem_statement = problem.get("Problem", "")  # Get problem statement

        result_entry = {
            "task_id": task_id,
            "ground_truth": ground_truth,
            "problem_statement": problem_statement,
            "status": None,
            "model_answer": None,
            "is_correct": None,
            "judge_response": None,
        }

        # Check if solution exists
        if task_id not in solutions:
            results["missing_solution"] += 1
            result_entry["status"] = "missing_solution"
            results["details"].append(result_entry)
            continue

        solution_text = solutions[task_id]

        # Extract boxed answer for tracking
        model_answer = extract_boxed_answer(solution_text)
        result_entry["model_answer"] = model_answer

        # if model_answer is None:
        #     results["no_boxed_answer"] += 1  # noqa: ERA001
        #     result_entry["status"] = "no_boxed_answer"  # noqa: ERA001
        #     results["details"].append(result_entry)  # noqa: ERA001
        #     continue  # noqa: ERA001

        # Judge the answer using the full solution text
        results["total"] += 1
        is_correct, judge_response = judge_answer(
            problem_statement, solution_text, ground_truth, judge_model_name, judge_prompt_template, max_tokens
        )

        result_entry["is_correct"] = is_correct
        result_entry["judge_response"] = judge_response

        if is_correct:
            results["correct"] += 1
            result_entry["status"] = "correct"
        else:
            results["incorrect"] += 1
            result_entry["status"] = "incorrect"

        results["details"].append(result_entry)

        # Update progress bar with running accuracy
        if results["total"] > 0:
            current_acc = results["correct"] / results["total"]
            pbar.set_description(f"Evaluating (Acc: {current_acc:.2%}, {results['correct']}/{results['total']})")

    # Calculate accuracy
    if results["total"] > 0:
        results["accuracy"] = results["correct"] / results["total"]
    else:
        results["accuracy"] = 0.0

    return results


def main() -> None:
    """Run the evaluation CLI: parse args, run evaluation, and log results."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Evaluate IMO solutions using API judge")
    parser.add_argument(
        "--solutions_dir",
        type=str,
        required=True,
        help="Path to directory containing solution files (Task_{task_id}_seed_42.txt)",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gemini-3-flash",
        help="Judge model name (e.g., 'gemini-3-flash', 'gemini-3-pro')",
    )
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens for judge responses")

    args = parser.parse_args()

    # Run evaluation
    logger.info("=" * 80)
    logger.info("IMO API Evaluation")
    logger.info("=" * 80)
    logger.info("Solutions directory: %s", args.solutions_dir)
    logger.info("Judge model: %s", args.judge_model)
    logger.info("Max tokens: %s", args.max_tokens)
    logger.info("=" * 80)

    results = evaluate_solutions(
        solutions_dir=args.solutions_dir,
        judge_prompt_template=JUDGE_PROMPT_TEMPLATE,
        judge_model_name=args.judge_model,
        max_tokens=args.max_tokens,
    )

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info("Total evaluated: %s", results["total"])
    logger.info("Correct: %s", results["correct"])
    logger.info("Incorrect: %s", results["incorrect"])
    logger.info("No boxed answer: %s", results["no_boxed_answer"])
    logger.info("Missing solution: %s", results["missing_solution"])
    logger.info("Accuracy: %.2f%%", results["accuracy"] * 100)
    logger.info("=" * 80)

    # Save results
    output = Path(args.solutions_dir) / "evaluation.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Detailed results saved to: %s", output)


if __name__ == "__main__":
    main()
