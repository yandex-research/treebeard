"""Evaluate LLM calls with specific phases from ablation output directories.

Reads per-task JSONL files produced by ablation scripts (e.g.
:mod:`self_improve`), filters LLM call records by a hardcoded list of
phases, and evaluates each matching response against the IMO AnswerBench
ground truth using the same judge logic as :mod:`evaluate`.

Each task is processed concurrently in its own thread.  Results are
written as one JSON file per task inside an ``evaluation/`` subdirectory
of the input data path.

Usage::

    uv run python -m open_deep_think.scripts.ablations.evaluate_phases \
        --data_dir data/ablations/ablation_self_improve_no_verification_gpt_oss \
        --judge_model google/gemini-3.1-pro-preview \
        --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import urllib3
from datasets import load_dataset

from open_deep_think.imo_answer_bench.extract import extract_boxed_answer
from open_deep_think.imo_answer_bench.judge import judge_answer
from open_deep_think.imo_answer_bench.templates import JudgeType

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded list of phases whose LLM responses should be evaluated.
# Add or remove phase names here to control what gets evaluated.
# ---------------------------------------------------------------------------
PHASES_TO_EVALUATE: list[str] = [
    "merge",
]

# Regex to extract task_id from filenames like ``Task_42_self_improve.jsonl``.
_TASK_FILE_PATTERN = re.compile(r"Task_(\d+)_.*\.jsonl$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def discover_task_files(data_dir: Path) -> dict[int, Path]:
    """Find all per-task JSONL files in *data_dir* and map task_id → path.

    Args:
        data_dir: Directory containing ``Task_*_*.jsonl`` files.

    Returns:
        Dict mapping integer task IDs to their JSONL file paths.

    """
    task_files: dict[int, Path] = {}
    for path in sorted(data_dir.glob("Task_*_*.jsonl")):
        match = _TASK_FILE_PATTERN.search(path.name)
        if match:
            task_id = int(match.group(1))
            task_files[task_id] = path
    return task_files


def load_phase_records(jsonl_path: Path, phases: list[str]) -> list[dict[str, Any]]:
    """Load JSONL records whose ``phase`` field is in *phases*.

    Args:
        jsonl_path: Path to a per-task JSONL log file.
        phases: Allowed phase names to keep.

    Returns:
        List of matching record dicts, preserving file order.

    """
    phase_set = set(phases)
    records: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").strip().splitlines():
        record = json.loads(line)
        if record.get("phase") in phase_set:
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------


def evaluate_task(
    *,
    task_id: int,
    jsonl_path: Path,
    problem_statement: str,
    ground_truth: str,
    judge_model: str,
    judge_type: JudgeType,
    max_tokens: int,
) -> dict[str, Any]:
    """Evaluate all phase-matching LLM calls for a single task.

    For each matching record the ``response_text`` is judged against the
    ground truth.  The result dict includes per-record verdicts as well as
    aggregate counts.

    Args:
        task_id: Dataset task identifier.
        jsonl_path: Path to the per-task JSONL file.
        problem_statement: Raw problem text from the dataset.
        ground_truth: Short ground-truth answer.
        judge_model: Model name for the judge.
        judge_type: Answer or proof judging mode.
        max_tokens: Max tokens for judge responses.

    Returns:
        A dict with ``task_id``, ``total``, ``correct``, ``incorrect``,
        and ``details`` (list of per-record evaluation entries).

    """
    records = load_phase_records(jsonl_path, PHASES_TO_EVALUATE)
    LOGGER.info("Task %s: found %s records to evaluate", task_id, len(records))

    result: dict[str, Any] = {
        "task_id": task_id,
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "details": [],
    }

    for record in records:
        solution_text = record.get("response_text", "")
        candidate_index = record.get("candidate_index")
        round_index = record.get("round_index")
        call_id = record.get("call_id")
        phase = record.get("phase")

        model_answer = extract_boxed_answer(solution_text)

        result["total"] += 1
        verdict, judge_response = judge_answer(
            problem_statement,
            solution_text,
            ground_truth,
            judge_model,
            judge_type,
            None,  # no grading guidelines for answer judging
            max_tokens,
        )

        is_correct = bool(verdict)

        if is_correct:
            result["correct"] += 1
        else:
            result["incorrect"] += 1

        entry: dict[str, Any] = {
            "task_id": task_id,
            "candidate_index": candidate_index,
            "round_index": round_index,
            "call_id": call_id,
            "phase": phase,
            "model_answer": model_answer,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "verdict": verdict,
            "judge_response": judge_response,
        }
        result["details"].append(entry)

        LOGGER.info(
            "Task %s cand=%s round=%s call=%s → %s",
            task_id,
            candidate_index,
            round_index,
            call_id,
            "correct" if is_correct else "incorrect",
        )

    return result


def is_task_done(output_dir: Path, task_id: int) -> bool:
    """Return ``True`` if the evaluation output for *task_id* already exists.

    A non-empty ``Task_{task_id}_evaluation.json`` indicates the task was
    already evaluated and can be skipped on re-runs.
    """
    output_file = output_dir / f"Task_{task_id}_evaluation.json"
    return output_file.exists() and output_file.stat().st_size > 0


def _evaluate_task_wrapper(
    task_id: int,
    jsonl_path: Path,
    problem_statement: str,
    ground_truth: str,
    judge_model: str,
    judge_type: JudgeType,
    max_tokens: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Thread-pool worker: skip-if-done guard, evaluate, and write output."""
    if is_task_done(output_dir, task_id):
        LOGGER.info("Task %s already evaluated — skipping.", task_id)
        return {"task_id": task_id, "status": "skipped"}

    try:
        result = evaluate_task(
            task_id=task_id,
            jsonl_path=jsonl_path,
            problem_statement=problem_statement,
            ground_truth=ground_truth,
            judge_model=judge_model,
            judge_type=judge_type,
            max_tokens=max_tokens,
        )
    except Exception:
        LOGGER.exception("Task %s evaluation failed", task_id)
        return {"task_id": task_id, "status": "error"}

    # Write per-task evaluation JSON.
    output_path = output_dir / f"Task_{task_id}_evaluation.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    LOGGER.info(
        "Task %s evaluation complete: %s/%s correct. Saved to %s",
        task_id,
        result["correct"],
        result["total"],
        output_path,
    )
    return {"task_id": task_id, "status": "success", "output_path": str(output_path)}


# ---------------------------------------------------------------------------
# Concurrent runner
# ---------------------------------------------------------------------------


def run_evaluations_concurrent(  # noqa: PLR0913
    *,
    task_files: dict[int, Path],
    dataset: Any,
    judge_model: str,
    judge_type: JudgeType,
    max_tokens: int,
    output_dir: Path,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Evaluate all tasks concurrently using a thread pool.

    Each thread picks up one task file and produces a separate JSON
    output file.

    Args:
        task_files: Mapping of task_id → JSONL path.
        dataset: Loaded HuggingFace dataset (indexed by task_id).
        judge_model: Judge model name.
        judge_type: Answer or proof judging mode.
        max_tokens: Max tokens for judge responses.
        output_dir: Directory to write evaluation JSONs.
        concurrency: Maximum concurrent threads.

    Returns:
        List of per-task status dicts.

    """
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_task = {}
        for task_id, jsonl_path in task_files.items():
            problem_statement = dataset[task_id]["Problem"]
            ground_truth = dataset[task_id]["Short Answer"]
            future = executor.submit(
                _evaluate_task_wrapper,
                task_id,
                jsonl_path,
                problem_statement,
                ground_truth,
                judge_model,
                judge_type,
                max_tokens,
                output_dir,
            )
            future_to_task[future] = task_id

        for future in as_completed(future_to_task):
            tid = future_to_task[future]
            try:
                result = future.result()
            except Exception:
                LOGGER.exception("Task %s failed with unrecoverable error", tid)
                result = {"task_id": tid, "status": "error"}
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the phase evaluation script."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LLM call responses for specific phases from ablation "
            "output directories against IMO AnswerBench ground truth."
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the ablation output directory containing Task_*_*.jsonl files.",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="google/gemini-3.1-pro-preview",
        help="Judge model name (default: google/gemini-3.1-pro-preview).",
    )
    parser.add_argument(
        "--judge_type",
        type=JudgeType,
        default="answer",
        help="Judge type: 'answer' or 'proof' (default: answer).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens for judge responses (default: 2048).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=15,
        help="Number of tasks to evaluate in parallel (default: 4).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Hwilner/imo-answerbench",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Set up stdout-only logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def main() -> None:
    """Run phase-based evaluation over ablation output files."""
    args = parse_args()
    configure_logging()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        LOGGER.error("Data directory does not exist: %s", data_dir)
        sys.exit(1)

    # Create evaluation output directory inside the data directory.
    output_dir = data_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover task files.
    task_files = discover_task_files(data_dir)
    if not task_files:
        LOGGER.error("No Task_*_*.jsonl files found in %s", data_dir)
        sys.exit(1)
    LOGGER.info("Found %s task files in %s", len(task_files), data_dir)
    LOGGER.info("Phases to evaluate: %s", PHASES_TO_EVALUATE)

    # Load dataset.
    LOGGER.info("Loading dataset: %s (%s)", args.dataset_name, args.dataset_split)
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    LOGGER.info("Loaded %s problems from dataset", len(dataset))

    # Run evaluations.
    LOGGER.info(
        "Starting evaluation with judge=%s, concurrency=%s, max_tokens=%s",
        args.judge_model,
        args.concurrency,
        args.max_tokens,
    )
    results = run_evaluations_concurrent(
        task_files=task_files,
        dataset=dataset,
        judge_model=args.judge_model,
        judge_type=args.judge_type,
        max_tokens=args.max_tokens,
        output_dir=output_dir,
        concurrency=args.concurrency,
    )

    # Print summary.
    successful = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errored = sum(1 for r in results if r["status"] == "error")
    LOGGER.info("=" * 60)
    LOGGER.info("EVALUATION SUMMARY")
    LOGGER.info("=" * 60)
    LOGGER.info("Total tasks:  %s", len(results))
    LOGGER.info("Successful:   %s", successful)
    LOGGER.info("Skipped:      %s", skipped)
    LOGGER.info("Errors:       %s", errored)
    LOGGER.info("Output dir:   %s", output_dir)
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
