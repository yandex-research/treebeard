r"""Self-improvement ablation: iteratively improve baseline candidates.

For each task and candidate, loads the baseline candidate produced by
:mod:`generate_baseline` and runs ``n_rounds`` of self-improvement.

Two modes (controlled by ``--use_verification``):

1. **Without verification** (default):
   Each round creates a fresh context with the baseline problem prompt,
   the current solution (as an assistant turn), and
   :data:`IMO25_SELF_IMPROVEMENT_PROMPT`.  The improved solution becomes
   the input for the next round.

2. **With verification** (``--use_verification``):
   Each round first verifies the current solution using
   :data:`IMO25_VERIFICATION_SYSTEM_PROMPT` + the binary classifier,
   then — *regardless of the verification result* — runs correction with
   :data:`IMO25_CORRECTION_PROMPT` and the verification report.

Tasks are processed concurrently via a thread pool; candidates within a
task are processed sequentially (rounds are inherently sequential).
Each task writes a single JSONL file containing LLM call records.

Usage::

    uv run python -m open_deep_think.scripts.ablations.self_improve \\
        --candidates_dir ../data/ablations/baseline_candidates/default \\
        --start 0 --end 10 \\
        --model openai/gpt-oss-120b \\
        --n_rounds 3 \\
        --concurrency 4 \\
        --output_path ../data/ablations/self_improve
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import urllib3
from datasets import load_dataset

from open_deep_think.api import chat_api_call
from open_deep_think.imo_answer_bench.templates import (
    IMO25_BINARY_CORRECTNESS_PROMPT,
    IMO25_CORRECTION_PROMPT,
    IMO25_SELF_IMPROVEMENT_PROMPT,
    IMO25_VERIFICATION_REMINDER,
    IMO25_VERIFICATION_SYSTEM_PROMPT,
    build_problem_prompt,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

LOGGER = logging.getLogger(__name__)
_YES_PATTERN = re.compile(r"\byes\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelfImproveConfig:
    """Configuration for the self-improvement ablation pipeline."""

    solver_model: str
    solver_max_tokens: int
    verifier_model: str
    verifier_max_tokens: int
    classifier_model: str
    classifier_max_tokens: int
    n_rounds: int
    use_verification: bool
    temperature: float | None
    top_p: float | None


@dataclass(frozen=True)
class CallResult:
    """Result of a single model call."""

    completion: ChatCompletion | None
    text: str
    call_id: int


@dataclass(frozen=True)
class VerificationResult:
    """Result of a verification pass (verifier + binary classifier)."""

    is_pass: bool
    bug_report: str
    verifier_output: str
    classifier_output: str
    verifier_call_id: int
    classifier_call_id: int


@dataclass(frozen=True)
class RoundResult:
    """Result of a single self-improvement round for one candidate."""

    candidate_id: int
    round_index: int
    input_solution: str
    output_solution: str
    verification_report: str | None
    verification_is_pass: bool | None
    improvement_call_id: int


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


class TaskCallLogger:
    """Persist every LLM call to a single per-task JSONL file.

    Each record has a ``record_type`` field set to ``"llm_call"``.
    """

    def __init__(self, task_id: int, log_path: Path) -> None:
        self._task_id = task_id
        self._log_path = log_path
        self._next_call_id = 1
        # Truncate / create the file.
        self._log_path.write_text("", encoding="utf-8")

    def record(  # noqa: PLR0913
        self,
        *,
        phase: str,
        candidate_index: int | None,
        round_index: int | None,
        model: str,
        messages: list[dict[str, str]],
        completion: ChatCompletion | None,
        response_text: str,
        error: str | None = None,
    ) -> int:
        """Record an LLM call to the per-task JSONL log.

        Returns:
            The auto-incremented call ID for this record.

        """
        call_id = self._next_call_id
        self._next_call_id += 1
        payload: dict[str, Any] = {
            "record_type": "llm_call",
            "timestamp": utc_now_iso(),
            "task_id": self._task_id,
            "call_id": call_id,
            "phase": phase,
            "candidate_index": candidate_index,
            "round_index": round_index,
            "model": model,
            "messages": messages,
            "response_text": response_text,
            "completion": completion.model_dump() if completion is not None else None,
            "error": error,
        }
        append_jsonl(self._log_path, payload)
        return call_id


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object as a single line to a JSONL file."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
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
    """Extract text before or after *marker*.

    Args:
        text: Source text.
        marker: Marker string to search for.
        after: If ``True``, return everything after the marker;
            otherwise return everything before it.

    Returns:
        Trimmed slice, or an empty string if the marker is absent.

    """
    marker_index = text.find(marker)
    if marker_index == -1:
        return ""
    if after:
        return text[marker_index + len(marker) :].strip()
    return text[:marker_index].strip()


def is_yes_response(text: str) -> bool:
    """Return ``True`` if *text* contains a standalone ``yes`` token."""
    return _YES_PATTERN.search(text) is not None


def load_task_ids_from_file(path: str) -> list[int]:
    """Read task IDs from a plain-text file (one integer per line).

    Blank lines and lines starting with ``#`` are ignored.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a non-blank, non-comment line is not an integer.

    """
    ids: list[int] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                ids.append(int(line))
            except ValueError as exc:
                msg = f"Cannot parse task ID on line {lineno} of {path!r}: {line!r}"
                raise ValueError(msg) from exc
    return ids


def load_candidates(candidates_dir: Path, task_id: int) -> list[dict[str, Any]]:
    """Load candidate solutions for a task from the baseline LLM outputs JSONL.

    Reads ``Task_{task_id}_llm_outputs.jsonl`` and extracts records with
    ``phase == "initial_solution"``, converting each into a dict with
    ``"index"`` and ``"solution_text"`` keys expected by the rest of the
    pipeline.

    Args:
        candidates_dir: Directory containing ``Task_*_llm_outputs.jsonl``
            files produced by :mod:`generate_baseline`.
        task_id: Task identifier.

    Returns:
        List of candidate dicts sorted by ``"index"``, each with
        ``"index"`` and ``"solution_text"`` keys.

    Raises:
        FileNotFoundError: If the LLM outputs file does not exist.
        ValueError: If no initial_solution records are found.

    """
    llm_log_file = candidates_dir / f"Task_{task_id}_llm_outputs.jsonl"
    if not llm_log_file.exists():
        msg = f"LLM outputs file not found: {llm_log_file}"
        raise FileNotFoundError(msg)

    candidates: list[dict[str, Any]] = []
    for line in llm_log_file.read_text(encoding="utf-8").strip().splitlines():
        record = json.loads(line)
        if record.get("phase") == "initial_solution":
            candidates.append(
                {
                    "index": record["candidate_index"],
                    "solution_text": record.get("response_text", ""),
                }
            )

    if not candidates:
        msg = f"No initial_solution records found in {llm_log_file}"
        raise ValueError(msg)

    candidates.sort(key=lambda c: c["index"])
    return candidates


# ---------------------------------------------------------------------------
# Model calling
# ---------------------------------------------------------------------------


def call_model(  # noqa: PLR0913
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    phase: str,
    candidate_index: int | None,
    round_index: int | None,
    call_logger: TaskCallLogger,
) -> CallResult:
    """Call a chat model and persist the full request/response via *call_logger*."""
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
            candidate_index=candidate_index,
            round_index=round_index,
            model=model,
            messages=messages,
            completion=completion,
            response_text=response_text,
        )
    except Exception as error:
        call_id = call_logger.record(
            phase=phase,
            candidate_index=candidate_index,
            round_index=round_index,
            model=model,
            messages=messages,
            completion=completion,
            response_text=response_text,
            error=str(error),
        )
        raise
    return CallResult(completion=completion, text=response_text, call_id=call_id)


# ---------------------------------------------------------------------------
# Verification sub-pipeline
# ---------------------------------------------------------------------------


def build_verification_prompt(problem_statement: str, solution_text: str) -> str:
    """Build the verifier user prompt, mirroring the official pipeline."""
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


def run_verification(  # noqa: PLR0913
    *,
    task_id: int,
    candidate_index: int,
    round_index: int,
    problem_statement: str,
    solution_text: str,
    config: SelfImproveConfig,
    call_logger: TaskCallLogger,
) -> VerificationResult:
    """Run the verifier + binary classifier and derive a bug report.

    Unlike the tournament pipeline, the bug report is *always* extracted
    (even when the classifier says "yes") because the correction step
    runs regardless of the verification result.
    """
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
        candidate_index=candidate_index,
        round_index=round_index,
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
        candidate_index=candidate_index,
        round_index=round_index,
        call_logger=call_logger,
    )

    passed = is_yes_response(classifier_result.text)
    # Always extract the summary section as the bug report, even on pass,
    # because correction runs regardless.
    bug_report = extract_section(verifier_result.text, marker="Detailed Verification", after=False)
    if not bug_report:
        # Marker absent — fall back to the full verifier output.
        bug_report = verifier_result.text

    LOGGER.info(
        "Task %s candidate %s round %s verification=%s",
        task_id,
        candidate_index,
        round_index,
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


# ---------------------------------------------------------------------------
# Self-improvement rounds
# ---------------------------------------------------------------------------


def run_improvement_round_no_verification(  # noqa: PLR0913
    *,
    task_id: int,
    candidate_index: int,
    round_index: int,
    problem_statement: str,
    current_solution: str,
    config: SelfImproveConfig,
    call_logger: TaskCallLogger,
) -> RoundResult:
    """Run one self-improvement round **without** verification.

    Each round creates a fresh context:
    ``[system, baseline_prompt, assistant(solution), self_improvement_prompt]``.
    """
    baseline_prompt = build_problem_prompt(problem_statement)
    messages = [
        {"role": "user", "content": baseline_prompt},
        {"role": "assistant", "content": current_solution},
        {"role": "user", "content": IMO25_SELF_IMPROVEMENT_PROMPT},
    ]

    result = call_model(
        model=config.solver_model,
        messages=messages,
        max_tokens=config.solver_max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        phase="self_improvement",
        candidate_index=candidate_index,
        round_index=round_index,
        call_logger=call_logger,
    )

    output_solution = result.text or current_solution

    LOGGER.info(
        "Task %s candidate %s round %s self-improvement complete",
        task_id,
        candidate_index,
        round_index,
    )

    return RoundResult(
        candidate_id=candidate_index,
        round_index=round_index,
        input_solution=current_solution,
        output_solution=output_solution,
        verification_report=None,
        verification_is_pass=None,
        improvement_call_id=result.call_id,
    )


def run_improvement_round_with_verification(  # noqa: PLR0913
    *,
    task_id: int,
    candidate_index: int,
    round_index: int,
    problem_statement: str,
    current_solution: str,
    config: SelfImproveConfig,
    call_logger: TaskCallLogger,
) -> RoundResult:
    """Run one self-improvement round **with** verification.

    1. Verify the current solution (verifier + classifier).
    2. Regardless of the verification result, run correction with
       :data:`IMO25_CORRECTION_PROMPT` and the verification report.
    """
    verification = run_verification(
        task_id=task_id,
        candidate_index=candidate_index,
        round_index=round_index,
        problem_statement=problem_statement,
        solution_text=current_solution,
        config=config,
        call_logger=call_logger,
    )

    baseline_prompt = build_problem_prompt(problem_statement)
    correction_prompt = f"{IMO25_CORRECTION_PROMPT}\n\n{verification.bug_report}"
    messages = [
        {"role": "user", "content": baseline_prompt},
        {"role": "assistant", "content": current_solution},
        {"role": "user", "content": correction_prompt},
    ]

    result = call_model(
        model=config.solver_model,
        messages=messages,
        max_tokens=config.solver_max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        phase="correction",
        candidate_index=candidate_index,
        round_index=round_index,
        call_logger=call_logger,
    )

    output_solution = result.text or current_solution

    LOGGER.info(
        "Task %s candidate %s round %s correction complete (verification=%s)",
        task_id,
        candidate_index,
        round_index,
        "pass" if verification.is_pass else "fail",
    )

    return RoundResult(
        candidate_id=candidate_index,
        round_index=round_index,
        input_solution=current_solution,
        output_solution=output_solution,
        verification_report=verification.verifier_output,
        verification_is_pass=verification.is_pass,
        improvement_call_id=result.call_id,
    )


def improve_candidate(  # noqa: PLR0913
    *,
    task_id: int,
    candidate_index: int,
    initial_solution: str,
    problem_statement: str,
    config: SelfImproveConfig,
    call_logger: TaskCallLogger,
) -> list[RoundResult]:
    """Run ``n_rounds`` of self-improvement for one candidate.

    Each round feeds the *previous* round's output as input, forming an
    iterative refinement chain.  The mode (with or without verification)
    is determined by ``config.use_verification``.

    Args:
        task_id: Dataset task identifier (for logging).
        candidate_index: Zero-based candidate index within the task.
        initial_solution: The baseline solution to start improving.
        problem_statement: Raw problem text.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls and round results.

    Returns:
        Ordered list of :class:`RoundResult` (one per round).

    """
    current_solution = initial_solution
    round_results: list[RoundResult] = []

    for round_index in range(config.n_rounds):
        if config.use_verification:
            round_result = run_improvement_round_with_verification(
                task_id=task_id,
                candidate_index=candidate_index,
                round_index=round_index,
                problem_statement=problem_statement,
                current_solution=current_solution,
                config=config,
                call_logger=call_logger,
            )
        else:
            round_result = run_improvement_round_no_verification(
                task_id=task_id,
                candidate_index=candidate_index,
                round_index=round_index,
                problem_statement=problem_statement,
                current_solution=current_solution,
                config=config,
                call_logger=call_logger,
            )

        round_results.append(round_result)
        current_solution = round_result.output_solution

    return round_results


# ---------------------------------------------------------------------------
# Task processing
# ---------------------------------------------------------------------------


def is_task_done(output_dir: Path, task_id: int) -> bool:
    """Return ``True`` if the task output JSONL already exists and is non-empty.

    A non-empty ``Task_{task_id}_self_improve.jsonl`` indicates that the
    task was successfully processed in a previous run and can be skipped.
    """
    output_file = output_dir / f"Task_{task_id}_self_improve.jsonl"
    return output_file.exists() and output_file.stat().st_size > 0


def process_task(
    *,
    task_id: int,
    problem_statement: str,
    candidates_dir: Path,
    config: SelfImproveConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Process all candidates for one task sequentially.

    For each candidate, loads the baseline solution and runs
    ``config.n_rounds`` of self-improvement.  All LLM calls are
    written to a single JSONL file per task.

    Args:
        task_id: Dataset task identifier.
        problem_statement: Raw problem text.
        candidates_dir: Directory with baseline LLM output JSONL logs.
        config: Pipeline configuration.
        output_dir: Directory for output files.

    Returns:
        A summary dict with ``task_id``, ``status``, and ``output_path``.

    """
    log_path = output_dir / f"Task_{task_id}_self_improve.jsonl"
    call_logger = TaskCallLogger(task_id=task_id, log_path=log_path)

    candidates = load_candidates(candidates_dir, task_id)

    for candidate in candidates:
        candidate_index = candidate["index"]
        initial_solution = candidate["solution_text"]

        if not initial_solution:
            LOGGER.warning(
                "Task %s candidate %s has empty solution — skipping.",
                task_id,
                candidate_index,
            )
            continue

        LOGGER.info(
            "Task %s candidate %s starting %s rounds of self-improvement",
            task_id,
            candidate_index,
            config.n_rounds,
        )

        try:
            improve_candidate(
                task_id=task_id,
                candidate_index=candidate_index,
                initial_solution=initial_solution,
                problem_statement=problem_statement,
                config=config,
                call_logger=call_logger,
            )
        except Exception:
            LOGGER.exception(
                "Task %s candidate %s failed during self-improvement",
                task_id,
                candidate_index,
            )

    LOGGER.info("Task %s complete. Output: %s", task_id, log_path)
    return {
        "task_id": task_id,
        "status": "success",
        "output_path": str(log_path),
    }


def _process_task_wrapper(
    task_id: int,
    problem_statement: str,
    candidates_dir: Path,
    config: SelfImproveConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Worker function for the thread pool (skip-if-done guard included)."""
    if is_task_done(output_dir, task_id):
        LOGGER.info("Task %s already done — skipping.", task_id)
        return {"task_id": task_id, "status": "skipped"}
    LOGGER.info("Starting task %s", task_id)
    result = process_task(
        task_id=task_id,
        problem_statement=problem_statement,
        candidates_dir=candidates_dir,
        config=config,
        output_dir=output_dir,
    )
    LOGGER.info("Finished task %s with status=%s", task_id, result["status"])
    return result


def run_tasks_concurrent(  # noqa: PLR0913
    *,
    task_ids: list[int],
    problems: list[str],
    candidates_dir: Path,
    config: SelfImproveConfig,
    output_dir: Path,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Run self-improvement for multiple tasks using a thread pool.

    Tasks are submitted to a :class:`~concurrent.futures.ThreadPoolExecutor`
    and processed concurrently (up to *concurrency* threads).  Each task
    writes to its own output files, so there are no shared file handles.

    Args:
        task_ids: Ordered list of task identifiers to process.
        problems: Problem statements corresponding to *task_ids*.
        candidates_dir: Directory with baseline LLM output JSONL logs.
        config: Pipeline configuration.
        output_dir: Directory for per-task output files.
        concurrency: Maximum concurrent worker threads.

    Returns:
        List of per-task result dicts (one per task, in completion order).

    """
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_task = {
            executor.submit(
                _process_task_wrapper,
                tid,
                prob,
                candidates_dir,
                config,
                output_dir,
            ): tid
            for tid, prob in zip(task_ids, problems)
        }
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
    """Parse command-line arguments.

    Task source is specified via one of two mutually exclusive modes:

    * ``--task_ids_file PATH`` — read explicit task IDs from a text file.
    * ``--start N --end M`` — process all tasks in ``[N, M)``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Self-improvement ablation: iteratively improve baseline candidates and log solutions per round."
        ),
    )
    parser.add_argument(
        "--candidates_dir",
        type=str,
        required=True,
        help="Directory with baseline LLM output JSONL logs (from generate_baseline).",
    )
    parser.add_argument(
        "--task_ids_file",
        type=str,
        default=None,
        help="Text file with one task ID per line.  Mutually exclusive with --start/--end.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Starting task index (inclusive).  Requires --end.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending task index (exclusive).  Requires --start.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Solver model name.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Base output directory.",
    )
    parser.add_argument(
        "--verifier_model",
        type=str,
        default=None,
        help="Verifier model (default: same as --model).",
    )
    parser.add_argument(
        "--classifier_model",
        type=str,
        default=None,
        help="Binary classifier model (default: verifier model).",
    )
    parser.add_argument(
        "--solver_max_tokens",
        type=int,
        default=100000,
        help="Maximum solver output tokens.",
    )
    parser.add_argument(
        "--verifier_max_tokens",
        type=int,
        default=100000,
        help="Maximum verifier output tokens.",
    )
    parser.add_argument(
        "--classifier_max_tokens",
        type=int,
        default=2048,
        help="Maximum classifier output tokens.",
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=8,
        help="Number of self-improvement rounds per candidate.",
    )
    parser.add_argument(
        "--use_verification",
        action="store_true",
        help="Verify before each improvement round (uses IMO25_CORRECTION_PROMPT).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of tasks to process in parallel (default: 1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling top_p.",
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
    parser.add_argument(
        "--run_name",
        type=str,
        default="default",
        help="Run directory name (default: 'default').",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate task-source and other CLI argument constraints.

    Raises:
        ValueError: On any invalid combination.

    """
    using_file = args.task_ids_file is not None
    using_range = args.start is not None or args.end is not None

    if using_file and using_range:
        msg = "--task_ids_file is mutually exclusive with --start/--end"
        raise ValueError(msg)

    if not using_file:
        if args.start is None or args.end is None:
            msg = "Either --task_ids_file or both --start and --end must be provided"
            raise ValueError(msg)
        if args.start < 0:
            msg = "--start must be non-negative"
            raise ValueError(msg)
        if args.end <= args.start:
            msg = "--end must be greater than --start"
            raise ValueError(msg)

    if args.n_rounds < 1:
        msg = f"--n_rounds must be >= 1, got {args.n_rounds}"
        raise ValueError(msg)
    if args.concurrency < 1:
        msg = f"--concurrency must be >= 1, got {args.concurrency}"
        raise ValueError(msg)


def configure_logging() -> None:
    """Configure stdout-only logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def main() -> None:
    """Run the self-improvement ablation over baseline candidates."""
    args = parse_args()
    validate_args(args)

    candidates_dir = Path(args.candidates_dir)
    run_dir = Path(args.output_path) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    solver_model = args.model
    verifier_model = args.verifier_model or solver_model
    classifier_model = args.classifier_model or verifier_model

    configure_logging()
    LOGGER.info("Loading dataset: %s (%s)", args.dataset_name, args.dataset_split)
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    LOGGER.info("Loaded %s tasks", len(dataset))

    # Resolve task IDs.
    if args.task_ids_file is not None:
        task_ids = load_task_ids_from_file(args.task_ids_file)
        LOGGER.info("Loaded %s task IDs from %s", len(task_ids), args.task_ids_file)
    else:
        if args.end > len(dataset):
            msg = f"--end ({args.end}) exceeds dataset size ({len(dataset)})"
            raise ValueError(msg)
        task_ids = list(range(args.start, args.end))

    config = SelfImproveConfig(
        solver_model=solver_model,
        solver_max_tokens=args.solver_max_tokens,
        verifier_model=verifier_model,
        verifier_max_tokens=args.verifier_max_tokens,
        classifier_model=classifier_model,
        classifier_max_tokens=args.classifier_max_tokens,
        n_rounds=args.n_rounds,
        use_verification=args.use_verification,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Persist config once for the run.
    config_path = run_dir / "config.json"
    config_dict = json.loads(json.dumps(asdict(config)))
    if not config_path.exists():
        write_json(config_path, config_dict)
        LOGGER.info("Config written to %s", config_path)

    problems = [dataset[tid]["Problem"] for tid in task_ids]

    LOGGER.info(
        "Processing %s tasks with concurrency=%s, n_rounds=%s, use_verification=%s",
        len(task_ids),
        args.concurrency,
        config.n_rounds,
        config.use_verification,
    )
    run_tasks_concurrent(
        task_ids=task_ids,
        problems=problems,
        candidates_dir=candidates_dir,
        config=config,
        output_dir=run_dir,
        concurrency=args.concurrency,
    )
    LOGGER.info("Run complete. Output dir: %s", run_dir)


if __name__ == "__main__":
    main()
