"""Reproduce the IMO25 verification-and-refinement pipeline on IMO AnswerBench.

This script mirrors the public `agent.py` loop from:
`Winning Gold at IMO 2025 with a Model-Agnostic Verification-and-Refinement Pipeline`.

Key behavior:
- Initial solve + self-improvement pass.
- Verifier-generated bug reports.
- Iterative correction and re-verification.
- Success criterion based on repeated positive verification.

Outputs are saved per task with files compatible with `scripts/evaluate.py`:
- `Task_{id}_solution.txt`
- `Task_{id}_reasoning.txt`
- `Task_{id}_response.json`
- `Task_{id}_progress.json`
- `Task_{id}_llm_outputs.jsonl`
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import urllib3
from datasets import load_dataset

from open_deep_think.api import chat_api_call
from open_deep_think.imo_answer_bench.extract import extract_reasoning, extract_solution
from open_deep_think.imo_answer_bench.templates import (
    IMO25_BINARY_CORRECTNESS_PROMPT,
    IMO25_CORRECTION_PROMPT,
    IMO25_SELF_IMPROVEMENT_PROMPT,
    IMO25_STEP1_SYSTEM_PROMPT,
    IMO25_VERIFICATION_REMINDER,
    IMO25_VERIFICATION_SYSTEM_PROMPT,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

LOGGER = logging.getLogger(__name__)
_YES_PATTERN = re.compile(r"\byes\b", re.IGNORECASE)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the IMO25 verification-and-refinement pipeline."""

    solver_model: str
    verifier_model: str
    classifier_model: str
    solver_max_tokens: int
    verifier_max_tokens: int
    classifier_max_tokens: int
    max_runs: int
    max_iterations: int
    required_consecutive_passes: int
    max_consecutive_failures: int
    temperature: float | None
    top_p: float | None
    other_prompts: tuple[str, ...]


@dataclass(frozen=True)
class TaskFiles:
    """Filesystem paths for per-task outputs."""

    solution: Path
    reasoning: Path
    response: Path
    progress: Path
    llm_outputs: Path


@dataclass(frozen=True)
class VerificationResult:
    """Result of a verification pass."""

    is_pass: bool
    bug_report: str
    verifier_output: str
    classifier_output: str
    verifier_call_id: int
    classifier_call_id: int


@dataclass(frozen=True)
class CallResult:
    """Result of a single model call."""

    completion: ChatCompletion | None
    text: str
    call_id: int


class TaskCallLogger:
    """Persist every per-task LLM call to JSONL files."""

    def __init__(self, task_id: int, task_log_path: Path, global_log_path: Path) -> None:
        self._task_id = task_id
        self._task_log_path = task_log_path
        self._global_log_path = global_log_path
        self._next_call_id = 1
        self._task_log_path.write_text("", encoding="utf-8")

    def record(  # noqa: PLR0913
        self,
        *,
        phase: str,
        run_index: int,
        iteration: int | None,
        model: str,
        messages: list[dict[str, str]],
        completion: ChatCompletion | None,
        response_text: str,
        error: str | None = None,
    ) -> int:
        """Record a model interaction to task and global JSONL logs."""
        call_id = self._next_call_id
        self._next_call_id += 1
        payload = {
            "timestamp": utc_now_iso(),
            "task_id": self._task_id,
            "call_id": call_id,
            "phase": phase,
            "run_index": run_index,
            "iteration": iteration,
            "model": model,
            "messages": messages,
            "response_text": response_text,
            "completion": completion.model_dump() if completion is not None else None,
            "error": error,
        }
        append_jsonl(self._task_log_path, payload)
        append_jsonl(self._global_log_path, payload)
        return call_id


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def sanitize_model_name(model: str) -> str:
    """Convert a model identifier to a filesystem-safe slug."""
    return model.replace("/", "__").replace(":", "_")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object line to a JSONL file."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON data with UTF-8 encoding and indentation."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def completion_text(completion: ChatCompletion) -> str:
    """Extract the first assistant message content from a completion."""
    if not completion.choices:
        return ""
    message = completion.choices[0].message
    return message.content if message.content is not None else ""


def extract_section(text: str, marker: str, *, after: bool) -> str:
    """Extract a section before or after a marker.

    This follows the same simple marker slicing used in the official script.

    Args:
        text: Source text.
        marker: Marker to look up.
        after: If True, return text after the marker; else return text before it.

    Returns:
        Trimmed slice or empty string if marker is absent.

    """
    marker_index = text.find(marker)
    if marker_index == -1:
        return ""
    if after:
        return text[marker_index + len(marker) :].strip()
    return text[:marker_index].strip()


def build_solver_messages(problem_statement: str, other_prompts: tuple[str, ...]) -> list[dict[str, str]]:
    """Build the base user message sequence for solver calls."""
    messages = [{"role": "user", "content": problem_statement}]
    messages.extend({"role": "user", "content": prompt} for prompt in other_prompts)
    return messages


def build_verification_prompt(problem_statement: str, solution_text: str) -> str:
    """Build the verifier user prompt exactly as in the official pipeline."""
    detailed_solution = extract_section(solution_text, marker="Detailed Solution", after=True)
    return f"""
======================================================================
### Problem ###

{problem_statement}

======================================================================
### Solution ###

{detailed_solution}

{IMO25_VERIFICATION_REMINDER}
"""


def is_yes_response(text: str) -> bool:
    """Return True if the text contains a standalone 'yes' token."""
    return _YES_PATTERN.search(text) is not None


def call_model(  # noqa: PLR0913
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    phase: str,
    run_index: int,
    iteration: int | None,
    call_logger: TaskCallLogger,
) -> CallResult:
    """Call a chat model and persist the full request/response to JSONL logs."""
    completion: ChatCompletion | None = None
    response_text = ""
    try:
        completion = chat_api_call(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        response_text = completion_text(completion)
        call_id = call_logger.record(
            phase=phase,
            run_index=run_index,
            iteration=iteration,
            model=model,
            messages=messages,
            completion=completion,
            response_text=response_text,
        )
    except Exception as error:
        call_id = call_logger.record(
            phase=phase,
            run_index=run_index,
            iteration=iteration,
            model=model,
            messages=messages,
            completion=completion,
            response_text=response_text,
            error=str(error),
        )
        raise
    return CallResult(completion=completion, text=response_text, call_id=call_id)


def run_verification(  # noqa: PLR0913
    *,
    task_id: int,
    run_index: int,
    iteration: int | None,
    problem_statement: str,
    solution_text: str,
    config: PipelineConfig,
    call_logger: TaskCallLogger,
) -> VerificationResult:
    """Run the verifier + binary correctness checks and derive a bug report."""
    verification_prompt = build_verification_prompt(problem_statement, solution_text)
    verifier_messages = [
        {"role": "system", "content": IMO25_VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": verification_prompt},
    ]
    verifier_result = call_model(
        model=config.verifier_model,
        messages=verifier_messages,
        max_tokens=config.verifier_max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        phase="verification",
        run_index=run_index,
        iteration=iteration,
        call_logger=call_logger,
    )

    classifier_prompt = f"{IMO25_BINARY_CORRECTNESS_PROMPT}\n\n{verifier_result.text}"
    classifier_messages = [{"role": "user", "content": classifier_prompt}]
    classifier_result = call_model(
        model=config.classifier_model,
        messages=classifier_messages,
        max_tokens=config.classifier_max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        phase="verification_check",
        run_index=run_index,
        iteration=iteration,
        call_logger=call_logger,
    )

    passed = is_yes_response(classifier_result.text)
    bug_report = "" if passed else extract_section(verifier_result.text, marker="Detailed Verification", after=False)
    LOGGER.info(
        "Task %s run %s iter %s verification=%s",
        task_id,
        run_index,
        "-" if iteration is None else iteration,
        "pass" if passed else "fail",
    )
    return VerificationResult(
        is_pass=passed,
        bug_report=bug_report,
        verifier_output=verifier_result.text,
        classifier_output=classifier_result.text,
        verifier_call_id=verifier_result.call_id,
        classifier_call_id=classifier_result.call_id,
    )


def task_files(output_dir: Path, task_id: int) -> TaskFiles:
    """Build per-task output file paths."""
    prefix = output_dir / f"Task_{task_id}"
    return TaskFiles(
        solution=prefix.with_name(f"Task_{task_id}_solution.txt"),
        reasoning=prefix.with_name(f"Task_{task_id}_reasoning.txt"),
        response=prefix.with_name(f"Task_{task_id}_response.json"),
        progress=prefix.with_name(f"Task_{task_id}_progress.json"),
        llm_outputs=prefix.with_name(f"Task_{task_id}_llm_outputs.jsonl"),
    )


def save_task_outputs(
    *,
    files: TaskFiles,
    solution_text: str,
    latest_solver_completion: ChatCompletion | None,
    progress_payload: dict[str, Any],
) -> None:
    """Persist final task outputs to disk."""
    files.solution.write_text(solution_text, encoding="utf-8")

    reasoning_text = ""
    response_payload: dict[str, Any] = {}
    if latest_solver_completion is not None:
        reasoning_text = extract_reasoning(latest_solver_completion)
        response_payload = latest_solver_completion.model_dump()
    files.reasoning.write_text(reasoning_text, encoding="utf-8")
    write_json(files.response, response_payload)
    write_json(files.progress, progress_payload)


def solve_task(  # noqa: C901, PLR0915
    *,
    task_id: int,
    problem_statement: str,
    config: PipelineConfig,
    output_dir: Path,
    global_llm_log_path: Path,
) -> dict[str, Any]:
    """Solve one IMO AnswerBench task with the IMO25 iterative pipeline."""
    files = task_files(output_dir, task_id)
    call_logger = TaskCallLogger(task_id=task_id, task_log_path=files.llm_outputs, global_log_path=global_llm_log_path)

    progress: dict[str, Any] = {
        "task_id": task_id,
        "started_at": utc_now_iso(),
        "pipeline_config": asdict(config),
        "problem_statement": problem_statement,
        "runs": [],
        "status": "failed",
        "final_run_index": None,
        "final_iteration": None,
        "completed_at": None,
    }

    latest_solution = ""
    latest_solver_completion: ChatCompletion | None = None

    for run_index in range(config.max_runs):
        LOGGER.info("Task %s run %s/%s", task_id, run_index + 1, config.max_runs)
        run_payload: dict[str, Any] = {
            "run_index": run_index,
            "status": "running",
            "initial": {},
            "iterations": [],
        }
        progress["runs"].append(run_payload)

        solver_base_messages = build_solver_messages(problem_statement, config.other_prompts)
        try:
            initial_result = call_model(
                model=config.solver_model,
                messages=[
                    {"role": "system", "content": IMO25_STEP1_SYSTEM_PROMPT},
                    *solver_base_messages,
                ],
                max_tokens=config.solver_max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                phase="initial_solution",
                run_index=run_index,
                iteration=None,
                call_logger=call_logger,
            )
            self_improved_result = call_model(
                model=config.solver_model,
                messages=[
                    {"role": "system", "content": IMO25_STEP1_SYSTEM_PROMPT},
                    *solver_base_messages,
                    {"role": "assistant", "content": initial_result.text},
                    {"role": "user", "content": IMO25_SELF_IMPROVEMENT_PROMPT},
                ],
                max_tokens=config.solver_max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                phase="self_improvement",
                run_index=run_index,
                iteration=None,
                call_logger=call_logger,
            )
            latest_solution = self_improved_result.text
            latest_solver_completion = self_improved_result.completion
            run_payload["initial"] = {
                "initial_solution_call_id": initial_result.call_id,
                "self_improvement_call_id": self_improved_result.call_id,
            }
            verification = run_verification(
                task_id=task_id,
                run_index=run_index,
                iteration=None,
                problem_statement=problem_statement,
                solution_text=latest_solution,
                config=config,
                call_logger=call_logger,
            )
            run_payload["initial"]["verification"] = {
                "is_pass": verification.is_pass,
                "bug_report": verification.bug_report,
                "verifier_output": verification.verifier_output,
                "classifier_output": verification.classifier_output,
                "verifier_call_id": verification.verifier_call_id,
                "classifier_call_id": verification.classifier_call_id,
            }
        except Exception:
            LOGGER.exception("Task %s run %s failed during initialization", task_id, run_index)
            run_payload["status"] = "initialization_error"
            continue

        consecutive_passes = 1
        consecutive_failures = 0

        for iteration in range(config.max_iterations):
            iteration_payload: dict[str, Any] = {
                "iteration": iteration,
                "verification_before_iteration": verification.is_pass,
            }
            try:
                if not verification.is_pass:
                    consecutive_passes = 0
                    consecutive_failures += 1
                    correction_prompt = f"{IMO25_CORRECTION_PROMPT}\n\n{verification.bug_report}"
                    correction_result = call_model(
                        model=config.solver_model,
                        messages=[
                            {"role": "system", "content": IMO25_STEP1_SYSTEM_PROMPT},
                            *solver_base_messages,
                            {"role": "assistant", "content": latest_solution},
                            {"role": "user", "content": correction_prompt},
                        ],
                        max_tokens=config.solver_max_tokens,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        phase="correction",
                        run_index=run_index,
                        iteration=iteration,
                        call_logger=call_logger,
                    )
                    latest_solution = correction_result.text
                    latest_solver_completion = correction_result.completion
                    iteration_payload["correction_call_id"] = correction_result.call_id

                verification = run_verification(
                    task_id=task_id,
                    run_index=run_index,
                    iteration=iteration,
                    problem_statement=problem_statement,
                    solution_text=latest_solution,
                    config=config,
                    call_logger=call_logger,
                )
            except Exception:
                LOGGER.exception("Task %s run %s iteration %s failed", task_id, run_index, iteration)
                iteration_payload["status"] = "iteration_error"
                run_payload["iterations"].append(iteration_payload)
                run_payload["status"] = "iteration_error"
                break

            if verification.is_pass:
                consecutive_passes += 1
                consecutive_failures = 0

            iteration_payload["verification"] = {
                "is_pass": verification.is_pass,
                "bug_report": verification.bug_report,
                "verifier_output": verification.verifier_output,
                "classifier_output": verification.classifier_output,
                "verifier_call_id": verification.verifier_call_id,
                "classifier_call_id": verification.classifier_call_id,
            }
            iteration_payload["consecutive_passes"] = consecutive_passes
            iteration_payload["consecutive_failures"] = consecutive_failures
            run_payload["iterations"].append(iteration_payload)

            if consecutive_passes >= config.required_consecutive_passes:
                run_payload["status"] = "success"
                progress["status"] = "success"
                progress["final_run_index"] = run_index
                progress["final_iteration"] = iteration
                LOGGER.info(
                    "Task %s solved at run %s iteration %s",
                    task_id,
                    run_index,
                    iteration,
                )
                break

            if consecutive_failures >= config.max_consecutive_failures:
                run_payload["status"] = "max_consecutive_failures"
                LOGGER.info(
                    "Task %s run %s reached max consecutive failures (%s)",
                    task_id,
                    run_index,
                    config.max_consecutive_failures,
                )
                break

        if run_payload["status"] == "running":
            run_payload["status"] = "max_iterations_reached"

        if progress["status"] == "success":
            break

    progress["completed_at"] = utc_now_iso()
    persisted_solution = extract_solution(latest_solver_completion) if latest_solver_completion is not None else ""
    if not persisted_solution:
        persisted_solution = latest_solution
    save_task_outputs(
        files=files,
        solution_text=persisted_solution,
        latest_solver_completion=latest_solver_completion,
        progress_payload=progress,
    )

    return {
        "task_id": task_id,
        "status": progress["status"],
        "solution_path": str(files.solution),
        "progress_path": str(files.progress),
        "llm_outputs_path": str(files.llm_outputs),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Reproduce IMO25 verification-and-refinement pipeline")
    parser.add_argument("--start", type=int, required=True, help="Starting task index (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="Ending task index (exclusive)")
    parser.add_argument("--model", type=str, required=True, help="Solver model name")
    parser.add_argument("--output_path", type=str, required=True, help="Base output directory")
    parser.add_argument("--verifier_model", type=str, help="Verifier model name (default: same as --model)")
    parser.add_argument("--classifier_model", type=str, help="Binary checker model name (default: verifier model)")
    parser.add_argument("--solver_max_tokens", type=int, default=32768, help="Maximum solver output tokens")
    parser.add_argument("--verifier_max_tokens", type=int, default=32768, help="Maximum verifier output tokens")
    parser.add_argument("--classifier_max_tokens", type=int, default=2048, help="Maximum checker output tokens")
    parser.add_argument("--max_runs", type=int, default=10, help="Number of independent retries per task")
    parser.add_argument("--max_iterations", type=int, default=30, help="Maximum refinement iterations per run")
    parser.add_argument(
        "--required_consecutive_passes",
        type=int,
        default=5,
        help="Consecutive positive verification checks required for success",
    )
    parser.add_argument(
        "--max_consecutive_failures",
        type=int,
        default=10,
        help="Stop a run after this many consecutive failed checks",
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature (official default: 0.1)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top_p (official default: 1.0)")
    parser.add_argument(
        "--other_prompt",
        action="append",
        default=[],
        help="Additional user prompt appended after the problem statement. Repeat for multiple prompts.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Hwilner/imo-answerbench",
        help="Hugging Face dataset name",
    )
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to evaluate")
    parser.add_argument(
        "--run_name",
        type=str,
        help="Optional run directory name (default: UTC timestamp)",
    )
    return parser.parse_args()


def configure_logging(log_file: Path) -> None:
    """Configure file and stdout logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def main() -> None:
    """Run the IMO25 pipeline over a task range."""
    args = parse_args()
    run_started_at = utc_now_iso()

    if args.start < 0:
        msg = "--start must be non-negative"
        raise ValueError(msg)
    if args.end <= args.start:
        msg = "--end must be greater than --start"
        raise ValueError(msg)

    solver_model = args.model
    verifier_model = args.verifier_model or solver_model
    classifier_model = args.classifier_model or verifier_model

    run_name = args.run_name or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_path) / "imo25" / sanitize_model_name(solver_model) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(run_dir / "run.log")
    LOGGER.info("Loading dataset: %s (%s)", args.dataset_name, args.dataset_split)
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    LOGGER.info("Loaded %s tasks", len(dataset))

    if args.end > len(dataset):
        msg = f"--end ({args.end}) exceeds dataset size ({len(dataset)})"
        raise ValueError(msg)

    config = PipelineConfig(
        solver_model=solver_model,
        verifier_model=verifier_model,
        classifier_model=classifier_model,
        solver_max_tokens=args.solver_max_tokens,
        verifier_max_tokens=args.verifier_max_tokens,
        classifier_max_tokens=args.classifier_max_tokens,
        max_runs=args.max_runs,
        max_iterations=args.max_iterations,
        required_consecutive_passes=args.required_consecutive_passes,
        max_consecutive_failures=args.max_consecutive_failures,
        temperature=args.temperature,
        top_p=args.top_p,
        other_prompts=tuple(args.other_prompt),
    )
    write_json(run_dir / "config.json", asdict(config))

    global_llm_log = run_dir / "all_llm_outputs.jsonl"
    global_llm_log.write_text("", encoding="utf-8")

    results: list[dict[str, Any]] = []
    for task_id in range(args.start, args.end):
        problem_statement = dataset[task_id]["Problem"]
        LOGGER.info("Starting task %s", task_id)
        task_result = solve_task(
            task_id=task_id,
            problem_statement=problem_statement,
            config=config,
            output_dir=run_dir,
            global_llm_log_path=global_llm_log,
        )
        results.append(task_result)
        LOGGER.info("Finished task %s with status=%s", task_id, task_result["status"])

    summary = {
        "started_at": run_started_at,
        "completed_at": utc_now_iso(),
        "run_dir": str(run_dir),
        "task_range": {"start": args.start, "end": args.end},
        "results": results,
        "success_count": sum(result["status"] == "success" for result in results),
        "failure_count": sum(result["status"] != "success" for result in results),
    }
    write_json(run_dir / "run_summary.json", summary)
    LOGGER.info("Run complete. Summary saved to %s", run_dir / "run_summary.json")


if __name__ == "__main__":
    main()
