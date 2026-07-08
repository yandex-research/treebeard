"""Generate baseline candidates for ablation studies (solver-only, no verification).

This script generates N independent solution candidates per task using only
the solver model — no verification, no self-improvement.  Each task produces
exactly ``num_solutions`` LLM calls.

Tasks are processed concurrently using a thread pool.  Each task writes its
own per-task JSONL log — there is no shared global output file.

Usage::

    uv run python -m open_deep_think.scripts.ablations.generate_baseline \
        --start 0 --end 10 \
        --model openai/gpt-oss-120b \
        --num_solutions 8 \
        --concurrency 4 \
        --output_path ../data/ablations/baseline_candidates
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import urllib3
from datasets import load_dataset

from open_deep_think.api import chat_api_call
from open_deep_think.imo_answer_bench.templates import build_problem_prompt

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerateBaselineConfig:
    """Configuration for the baseline candidate-generation pipeline."""

    solver_model: str
    solver_max_tokens: int
    num_solutions: int
    temperature: float | None
    top_p: float | None


@dataclass(frozen=True)
class CallResult:
    """Result of a single model call."""

    completion: ChatCompletion | None
    text: str
    call_id: int


@dataclass
class Candidate:
    """A solution candidate produced by the solver."""

    index: int
    solution_text: str
    completion: ChatCompletion | None


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


class TaskCallLogger:
    """Persist every per-task LLM call to a per-task JSONL file."""

    def __init__(self, task_id: int, task_log_path: Path) -> None:
        self._task_id = task_id
        self._task_log_path = task_log_path
        self._next_call_id = 1
        self._task_log_path.write_text("", encoding="utf-8")

    def record(  # noqa: PLR0913
        self,
        *,
        phase: str,
        candidate_index: int | None,
        model: str,
        messages: list[dict[str, str]],
        completion: ChatCompletion | None,
        response_text: str,
        error: str | None = None,
    ) -> int:
        """Record a model interaction to the per-task JSONL log."""
        call_id = self._next_call_id
        self._next_call_id += 1
        payload = {
            "timestamp": utc_now_iso(),
            "task_id": self._task_id,
            "call_id": call_id,
            "phase": phase,
            "candidate_index": candidate_index,
            "model": model,
            "messages": messages,
            "response_text": response_text,
            "completion": completion.model_dump() if completion is not None else None,
            "error": error,
        }
        append_jsonl(self._task_log_path, payload)
        return call_id


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object line to a JSONL file."""
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


def is_power_of_two(n: int) -> bool:
    """Return True if n is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def load_task_ids_from_file(path: str) -> list[int]:
    """Read task IDs from a plain-text file, one ID per line.

    Blank lines and lines starting with ``#`` are ignored.

    Args:
        path: Path to the task-IDs file.

    Returns:
        Ordered list of integer task IDs as they appear in the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any non-blank, non-comment line cannot be parsed as int.

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


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------


def build_solver_messages(problem_statement: str) -> list[dict[str, str]]:
    r"""Build the single-turn user message for baseline solver calls.

    Uses :func:`build_problem_prompt` which prepends the standard
    "reason step by step … \boxed{}" instruction to the problem statement.
    No system message is emitted — the baseline prompt relies on a single
    user turn.
    """
    return [{"role": "user", "content": build_problem_prompt(problem_statement)}]


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
            candidate_index=candidate_index,
            model=model,
            messages=messages,
            completion=completion,
            response_text=response_text,
        )
    except Exception as error:
        call_id = call_logger.record(
            phase=phase,
            candidate_index=candidate_index,
            model=model,
            messages=messages,
            completion=completion,
            response_text=response_text,
            error=str(error),
        )
        raise
    return CallResult(completion=completion, text=response_text, call_id=call_id)


# ---------------------------------------------------------------------------
# Candidate generation (solver-only, no verification)
# ---------------------------------------------------------------------------


def generate_candidate(
    *,
    task_id: int,
    candidate_index: int,
    problem_statement: str,
    config: GenerateBaselineConfig,
    call_logger: TaskCallLogger,
) -> Candidate:
    """Generate one solution candidate using only the solver model.

    A single LLM call is made per candidate — no verification or
    self-improvement is performed.

    Args:
        task_id: Identifier of the current task (for logging).
        candidate_index: Zero-based index of this candidate in the pool.
        problem_statement: The problem to solve.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.

    Returns:
        A :class:`Candidate` with the solver's raw output.

    """
    LOGGER.info("Task %s generating candidate %s", task_id, candidate_index)
    solver_messages = build_solver_messages(problem_statement)
    result = call_model(
        model=config.solver_model,
        messages=solver_messages,
        max_tokens=config.solver_max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        phase="initial_solution",
        candidate_index=candidate_index,
        call_logger=call_logger,
    )
    return Candidate(
        index=candidate_index,
        solution_text=result.text,
        completion=result.completion,
    )


def generate_all_candidates(
    *,
    task_id: int,
    problem_statement: str,
    config: GenerateBaselineConfig,
    call_logger: TaskCallLogger,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Generate all candidates sequentially and collect payloads.

    Failed candidates are represented as dummy :class:`Candidate` objects with an
    empty ``solution_text``, so the output always has exactly
    ``config.num_solutions`` entries.

    Args:
        task_id: Dataset task identifier (for logging).
        problem_statement: Raw problem text.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.

    Returns:
        A tuple of (candidates, candidate_payloads).  ``candidates`` always has
        exactly ``config.num_solutions`` entries in index order.

    """
    candidates: list[Candidate] = []
    candidate_payloads: list[dict[str, Any]] = []

    for idx in range(config.num_solutions):
        try:
            candidate = generate_candidate(
                task_id=task_id,
                candidate_index=idx,
                problem_statement=problem_statement,
                config=config,
                call_logger=call_logger,
            )
        except Exception as exc:  # noqa: PERF203
            LOGGER.exception("Task %s candidate %s generation failed", task_id, idx)
            LOGGER.warning("Task %s candidate %s replaced with empty dummy after generation error", task_id, idx)
            dummy = Candidate(
                index=idx,
                solution_text="",
                completion=None,
            )
            candidates.append(dummy)
            candidate_payloads.append({"candidate_index": idx, "status": "generation_error", "error": str(exc)})
        else:
            candidates.append(candidate)
            candidate_payloads.append({"candidate_index": idx, "status": "ok"})

    return candidates, candidate_payloads


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def is_task_done(output_dir: Path, task_id: int) -> bool:
    """Return True if the task LLM outputs JSONL already exists.

    A non-empty ``Task_{task_id}_llm_outputs.jsonl`` indicates that the task
    was successfully completed in a previous run and can be skipped.

    Args:
        output_dir: Directory containing per-task output files.
        task_id: Task identifier.

    Returns:
        True if the LLM outputs JSONL file exists and is non-empty.

    """
    llm_log_file = output_dir / f"Task_{task_id}_llm_outputs.jsonl"
    return llm_log_file.exists() and llm_log_file.stat().st_size > 0


def generate_and_save(
    *,
    task_id: int,
    problem_statement: str,
    config: GenerateBaselineConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate all candidates for one task and persist LLM calls to JSONL.

    Produces one file per task:

    - ``Task_{task_id}_llm_outputs.jsonl`` — per-call LLM log containing
      the full completion, response text, messages, and metadata for every
      candidate generation call.

    Each task uses its own JSONL log file, so this function is safe to call
    from multiple threads concurrently (no shared file handles).

    Args:
        task_id: Dataset task identifier.
        problem_statement: Raw problem text.
        config: Pipeline configuration.
        output_dir: Directory for per-task output files.

    Returns:
        A summary dict with task_id, status, and output file path.

    """
    llm_log_path = output_dir / f"Task_{task_id}_llm_outputs.jsonl"
    call_logger = TaskCallLogger(task_id=task_id, task_log_path=llm_log_path)

    candidates, _candidate_payloads = generate_all_candidates(
        task_id=task_id,
        problem_statement=problem_statement,
        config=config,
        call_logger=call_logger,
    )

    LOGGER.info("Task %s: saved %s candidates to %s", task_id, len(candidates), llm_log_path)
    return {
        "task_id": task_id,
        "status": "success",
        "llm_outputs_path": str(llm_log_path),
    }


# ---------------------------------------------------------------------------
# Concurrent task processing
# ---------------------------------------------------------------------------


def _process_task(
    task_id: int,
    problem_statement: str,
    config: GenerateBaselineConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Process a single task — worker function for the thread pool.

    Skips the task if it was already completed in a previous run.

    Args:
        task_id: Dataset task identifier.
        problem_statement: Raw problem text.
        config: Pipeline configuration.
        output_dir: Directory for per-task output files.

    Returns:
        A summary dict with task_id and status.

    """
    if is_task_done(output_dir, task_id):
        LOGGER.info("Task %s already done — skipping.", task_id)
        return {"task_id": task_id, "status": "skipped"}
    LOGGER.info("Starting task %s", task_id)
    result = generate_and_save(
        task_id=task_id,
        problem_statement=problem_statement,
        config=config,
        output_dir=output_dir,
    )
    LOGGER.info("Finished task %s with status=%s", task_id, result["status"])
    return result


def run_tasks_concurrent(
    *,
    task_ids: list[int],
    problems: list[str],
    config: GenerateBaselineConfig,
    output_dir: Path,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Run generate_and_save for multiple tasks using a thread pool.

    Tasks are submitted to a :class:`ThreadPoolExecutor` and processed
    concurrently up to ``concurrency`` threads.  Each task writes to its own
    JSONL log file, so there are no shared file handles across threads.

    Args:
        task_ids: Ordered list of task identifiers to process.
        problems: Problem statements corresponding to ``task_ids``.
        config: Pipeline configuration.
        output_dir: Directory for per-task output files.
        concurrency: Maximum number of concurrent worker threads.

    Returns:
        A list of per-task result dicts (one per task_id, in completion order).

    """
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_task = {
            executor.submit(_process_task, tid, prob, config, output_dir): tid for tid, prob in zip(task_ids, problems)
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

    * ``--task_ids_file PATH`` — read explicit task IDs from a text file
      (one integer per line).  ``--start`` and ``--end`` must not be given.
    * ``--start N --end M`` — run all tasks in the range ``[N, M)``.
      ``--task_ids_file`` must not be given.

    Mutual exclusivity is enforced in :func:`validate_args`.
    """
    parser = argparse.ArgumentParser(
        description="Generate baseline candidates (solver-only, no verification) and save to JSON.",
    )
    parser.add_argument(
        "--task_ids_file",
        type=str,
        default=None,
        help="Path to a text file with one task ID per line. Mutually exclusive with --start/--end.",
    )
    parser.add_argument("--start", type=int, default=None, help="Starting task index (inclusive). Requires --end.")
    parser.add_argument("--end", type=int, default=None, help="Ending task index (exclusive). Requires --start.")
    parser.add_argument("--model", type=str, required=True, help="Solver model name")
    parser.add_argument("--output_path", type=str, required=True, help="Base output directory")
    parser.add_argument("--solver_max_tokens", type=int, default=64000, help="Maximum solver output tokens")
    parser.add_argument(
        "--num_solutions",
        type=int,
        default=8,
        help="Number of independent solutions to generate (must be a power of 2)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of tasks to process in parallel (default: 1, sequential)",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top_p")
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
        default="default",
        help="Run directory name (default: 'default')",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate task-source and other launcher argument constraints.

    Exactly one of ``--task_ids_file`` or ``--start``/``--end`` must be
    provided.  When using the range form both ``--start`` and ``--end`` are
    required and ``--end`` must be strictly greater than ``--start``.

    Raises:
        ValueError: On any invalid combination or out-of-range value.

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

    if not is_power_of_two(args.num_solutions):
        msg = f"--num_solutions must be a power of 2, got {args.num_solutions}"
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
    """Generate baseline candidates for a range of tasks and save to JSON."""
    args = parse_args()
    validate_args(args)

    run_dir = Path(args.output_path) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    configure_logging()
    LOGGER.info("Loading dataset: %s (%s)", args.dataset_name, args.dataset_split)
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    LOGGER.info("Loaded %s tasks", len(dataset))

    # Resolve task IDs from either --task_ids_file or --start/--end.
    if args.task_ids_file is not None:
        task_ids = load_task_ids_from_file(args.task_ids_file)
        LOGGER.info("Loaded %s task IDs from %s", len(task_ids), args.task_ids_file)
    else:
        if args.end > len(dataset):
            msg = f"--end ({args.end}) exceeds dataset size ({len(dataset)})"
            raise ValueError(msg)
        task_ids = list(range(args.start, args.end))

    config = GenerateBaselineConfig(
        solver_model=args.model,
        solver_max_tokens=args.solver_max_tokens,
        num_solutions=args.num_solutions,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Save config once for the run.
    config_path = run_dir / "config.json"
    config_dict = json.loads(json.dumps(asdict(config)))
    if not config_path.exists():
        write_json(config_path, config_dict)
        LOGGER.info("Config written to %s", config_path)

    problems = [dataset[tid]["Problem"] for tid in task_ids]

    LOGGER.info(
        "Processing %s tasks with concurrency=%s",
        len(task_ids),
        args.concurrency,
    )
    run_tasks_concurrent(
        task_ids=task_ids,
        problems=problems,
        config=config,
        output_dir=run_dir,
        concurrency=args.concurrency,
    )
    LOGGER.info("Run complete. Output dir: %s", run_dir)


if __name__ == "__main__":
    main()
