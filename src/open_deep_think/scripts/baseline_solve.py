"""Baseline solver script for IMO Answer Bench.

This script processes problems from a dataset, sends them to an API for solving,
and stores the results in separate files for reasoning, solution, and full response.

Outputs per task:
- ``Task_{id}_solution.txt``
- ``Task_{id}_reasoning.txt``
- ``Task_{id}_response.json``
"""

from __future__ import annotations

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
from open_deep_think.imo_answer_bench.templates import build_problem_prompt

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


def is_task_done(output_path: str, task_id: int) -> bool:
    """Return True if the task solution file exists and is non-empty.

    A non-empty ``Task_{task_id}_solution.txt`` indicates the task was
    successfully completed in a previous run and can be skipped.

    Args:
        output_path: Base directory for output files.
        task_id: Task identifier.

    Returns:
        True if the solution file exists and contains at least one character.

    """
    solution_file = Path(output_path) / f"Task_{task_id}_solution.txt"
    return solution_file.exists() and solution_file.stat().st_size > 0


def save_results(output_path: str, task_id: int, reasoning: str, solution: str, response_data: dict[str, Any]) -> None:
    """Save reasoning, solution, and full response to separate files.

    The solution file is written last so its presence reliably signals that
    the task completed successfully.

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

    # Save full response
    response_file = output_dir / f"Task_{task_id}_response.json"
    with response_file.open("w", encoding="utf-8") as f:
        json.dump(response_data, f, indent=2, ensure_ascii=False)

    # Save final solution last — its presence signals successful completion
    solution_file = output_dir / f"Task_{task_id}_solution.txt"
    with solution_file.open("w", encoding="utf-8") as f:
        f.write(solution)


def solve_problem(  # noqa: PLR0913
    problem: str,
    model: str,
    task_id: int,
    output_path: str,
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
) -> None:
    """Solve a single problem and save results.

    Args:
        problem: Problem text
        model: Model identifier
        task_id: Task identifier
        output_path: Base directory for output files
        max_tokens: Maximum tokens for API response
        temperature: Sampling temperature for API response
        top_p: Nucleus sampling parameter for API response

    """
    logger.info("Processing Task %s...", task_id)

    try:
        prompt = build_problem_prompt(problem)

        completion = single_turn_api_call(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

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
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, help="Nucleus sampling top_p")
    parser.add_argument("--output_path", type=str, help="Output directory path")
    parser.add_argument("--run_name", type=str, default=None, help="Run name subdirectory")
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Zero-based shard index; only shard 0 writes config.json",
    )
    parser.add_argument("--dataset_name", type=str, help="Hugging face dataset name")
    parser.add_argument("--dataset_split", type=str, help="Hugging face dataset split")
    return parser.parse_args()


def main() -> None:
    """Run the baseline solver entry point."""
    args = parse_args()

    model_name = args.model.split("/")[-1]
    base = Path(args.output_path) / "baseline" / model_name
    output_path = str(base / args.run_name) if args.run_name else str(base)

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

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.shard_index == 0:
        config = {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "run_name": args.run_name,
        }
        config_path = output_dir / "config.json"
        normalised = json.loads(json.dumps(config))
        if config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing != normalised:
                msg = f"Config mismatch in {config_path}.\nExisting: {existing}\nCurrent:  {normalised}"
                raise ValueError(msg)
            logger.info("Config matches existing %s — no rewrite needed.", config_path)
        else:
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(normalised, f, indent=2, ensure_ascii=False)
            logger.info("Config written to %s", config_path)

    logger.info("Loading dataset")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    # Process tasks in range
    logger.info("Processing tasks %s to %s", args.start, args.end - 1)
    for task_id in range(args.start, args.end):
        if is_task_done(output_path, task_id):
            logger.info("Task %s already solved — skipping.", task_id)
            continue
        problem = dataset[task_id]["Problem"]
        solve_problem(
            problem=problem,
            model=args.model,
            task_id=task_id,
            output_path=output_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    logger.info("All tasks completed.")


if __name__ == "__main__":
    main()
