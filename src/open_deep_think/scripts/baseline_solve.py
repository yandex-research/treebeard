"""Baseline solver script for IMO Answer Bench.

This script processes problems from a dataset, sends them to an API for solving,
and stores the results in separate files for reasoning, solution, and full response.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import urllib3
from datasets import load_dataset

from open_deep_think.api import single_turn_api_call
from open_deep_think.imo_answer_bench.extract import (
    extract_reasoning,
    extract_solution,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


def save_results(output_path: str, task_id: int, reasoning: str, solution: str, response_data: dict[str, Any]) -> None:
    """Save reasoning, solution, and full response to separate files.

    Args:
        output_path: Base directory for output files
        task_id: Task identifier
        reasoning: Reasoning text
        solution: Solution text
        response_data: Full API response as a JSON-serializable dict

    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save reasoning
    reasoning_file = output_dir / f"Task_{task_id}_reasoning.txt"
    with reasoning_file.open("w", encoding="utf-8") as f:
        f.write(reasoning)

    # Save solution
    solution_file = output_dir / f"Task_{task_id}_solution.txt"
    with solution_file.open("w", encoding="utf-8") as f:
        f.write(solution)

    # Save full response
    response_file = output_dir / f"Task_{task_id}_response.json"
    with response_file.open("w", encoding="utf-8") as f:
        json.dump(response_data, f, indent=2, ensure_ascii=False)


def solve_problem(
    problem: str,
    model: str,
    task_id: int,
    output_path: str,
    max_tokens: int,
) -> None:
    """Solve a single problem and save results.

    Args:
        problem: Problem text
        model: Model identifier
        task_id: Task identifier
        output_path: Base directory for output files
        max_tokens: Maximum tokens for API response

    """
    logger.info("Processing Task %s...", task_id)

    try:
        prompt = "Please reason step by step, and put your final answer within \\boxed{}.\n\n" + problem

        completion = single_turn_api_call(model=model, prompt=prompt, max_tokens=max_tokens)

        reasoning = extract_reasoning(completion)
        solution = extract_solution(completion)

        save_results(output_path, task_id, reasoning, solution, completion.model_dump())

        logger.info("Task %s completed.", task_id)

    except Exception as e:
        logger.exception("Error processing Task %s", task_id)
        # Save error information
        error_response = {"error": str(e), "task_id": task_id, "prompt": prompt}
        save_results(output_path, task_id, "", "", error_response)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the baseline solver."""
    parser = argparse.ArgumentParser(description="Solve problems from a dataset using an API model")
    parser.add_argument("--start", type=int, help="Starting task index (inclusive)")
    parser.add_argument("--end", type=int, help="Ending task index (exclusive)")
    parser.add_argument("--model", type=str, help="Model identifier for API calls (e.g., 'deepseek/deepseek-v3.2')")
    parser.add_argument("--max_tokens", type=int, help="Max tokens for API call")
    parser.add_argument("--output_path", type=str, help="Output directory path")
    return parser.parse_args()


def main() -> None:
    """Run the baseline solver entry point."""
    args = parse_args()

    model_name = args.model.split("/")[-1]
    output_path = f"{args.output_path}/baseline/{model_name}"

    # Configure logging
    log_dir = Path(f"../logs/baseline/{model_name}")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"script_{args.start}_{args.end}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),  # Optional: also print to console
        ],
    )

    # Load dataset
    logger.info("Loading dataset")
    dataset = load_dataset("Hwilner/imo-answerbench", split="train")

    # Process tasks in range
    logger.info("Processing tasks %s to %s", args.start, args.end - 1)
    for task_id in range(args.start, args.end):
        problem = dataset[task_id]["Problem"]
        solve_problem(
            problem=problem, model=args.model, task_id=task_id, output_path=output_path, max_tokens=args.max_tokens
        )

    logger.info("All tasks completed.")


if __name__ == "__main__":
    main()
