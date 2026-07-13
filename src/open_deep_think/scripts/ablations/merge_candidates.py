r"""Merge-candidates ablation: iteratively merge baseline candidates.

For each task, loads the baseline candidates produced by
:mod:`generate_baseline` and runs ``n_rounds`` of pairwise merges.

At each round, every candidate *i* is merged with a randomly chosen
partner *j* (``j != i``) from the same round, using
:func:`~open_deep_think.imo_answer_bench.templates.build_tournament_merge_prompt`.
Candidate *i*'s solution is always placed as Solution 1 in the merge
prompt (preserving authorship order).  The merged solution replaces
candidate *i* for the next round.

No verification or self-improvement is performed.

Tasks are processed concurrently via a thread pool; rounds within a
task are inherently sequential (each round depends on all candidates
from the previous round).  Each task writes a single JSONL file
containing LLM call records.

Usage::

    uv run python -m open_deep_think.scripts.ablations.merge_candidates \
        --candidates_dir ../data/ablations/baseline_candidates/default \
        --start 0 --end 10 \
        --model openai/gpt-oss-120b \
        --n_rounds 3 \
        --concurrency 4 \
        --output_path ../data/ablations/merge_candidates
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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
    TOURNAMENT_MERGE_SYSTEM_PROMPT,
    build_tournament_merge_prompt,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

LOGGER = logging.getLogger(__name__)

_MIN_CANDIDATES_FOR_MERGE = 2


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeCandidatesConfig:
    """Configuration for the merge-candidates ablation pipeline."""

    merger_model: str
    merger_max_tokens: int
    n_rounds: int
    temperature: float | None
    top_p: float | None
    seed: int | None


@dataclass(frozen=True)
class CallResult:
    """Result of a single model call."""

    completion: ChatCompletion | None
    text: str
    call_id: int


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


class TaskCallLogger:
    """Persist every LLM call to a single per-task JSONL file.

    Each record has a ``record_type`` field set to ``"llm_call"`` and
    includes a ``merge_partner_index`` field recording which candidate
    was used as the merge partner.
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
        merge_partner_index: int | None,
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
            "merge_partner_index": merge_partner_index,
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
    merge_partner_index: int | None,
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
            merge_partner_index=merge_partner_index,
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
            merge_partner_index=merge_partner_index,
            model=model,
            messages=messages,
            completion=completion,
            response_text=response_text,
            error=str(error),
        )
        raise
    return CallResult(completion=completion, text=response_text, call_id=call_id)


# ---------------------------------------------------------------------------
# Merge rounds
# ---------------------------------------------------------------------------


def run_merge_round(  # noqa: PLR0913
    *,
    task_id: int,
    round_index: int,
    problem_statement: str,
    candidates: list[dict[str, Any]],
    config: MergeCandidatesConfig,
    call_logger: TaskCallLogger,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Run one round of pairwise merges for all candidates.

    For each candidate *i*, a random partner *j* (``j != i``) is chosen
    from the current population.  The merge prompt places candidate *i*'s
    solution as Solution 1 and candidate *j*'s solution as Solution 2.

    If the API call for a single candidate fails, the candidate retains
    its current solution for the next round and processing continues.

    Args:
        task_id: Dataset task identifier (for logging).
        round_index: Zero-based round index.
        problem_statement: Raw problem text.
        candidates: Current population of candidate dicts with
            ``"index"`` and ``"solution_text"`` keys.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.
        rng: Random number generator for partner selection.

    Returns:
        New population of candidate dicts (same length as *candidates*).

    """
    n = len(candidates)
    new_candidates: list[dict[str, Any]] = []

    for i in range(n):
        # Pick a random partner j != i.
        possible_partners = [j for j in range(n) if j != i]
        j = rng.choice(possible_partners)

        merge_prompt = build_tournament_merge_prompt(
            problem=problem_statement,
            solution_1=candidates[i]["solution_text"],
            solution_2=candidates[j]["solution_text"],
        )
        messages = [
            {"role": "system", "content": TOURNAMENT_MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": merge_prompt},
        ]

        try:
            result = call_model(
                model=config.merger_model,
                messages=messages,
                max_tokens=config.merger_max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                phase="merge",
                candidate_index=i,
                round_index=round_index,
                merge_partner_index=j,
                call_logger=call_logger,
            )
            output_solution = result.text or candidates[i]["solution_text"]
        except Exception:
            LOGGER.exception(
                "Task %s candidate %s round %s merge with %s failed",
                task_id,
                i,
                round_index,
                j,
            )
            output_solution = candidates[i]["solution_text"]

        LOGGER.info(
            "Task %s candidate %s round %s merged with candidate %s",
            task_id,
            i,
            round_index,
            j,
        )

        new_candidates.append(
            {
                "index": candidates[i]["index"],
                "solution_text": output_solution,
            }
        )

    return new_candidates


def run_all_merge_rounds(  # noqa: PLR0913
    *,
    task_id: int,
    problem_statement: str,
    candidates: list[dict[str, Any]],
    config: MergeCandidatesConfig,
    call_logger: TaskCallLogger,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Run all merge rounds for a single task.

    Each round uses the output population from the previous round as
    input, forming an iterative refinement process through merging.

    Args:
        task_id: Dataset task identifier (for logging).
        problem_statement: Raw problem text.
        candidates: Initial population of candidate dicts with
            ``"index"`` and ``"solution_text"`` keys.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.
        rng: Random number generator for partner selection.

    Returns:
        Final population of candidate dicts after all rounds.

    """
    current_candidates = candidates
    for round_index in range(config.n_rounds):
        LOGGER.info(
            "Task %s starting merge round %s/%s",
            task_id,
            round_index + 1,
            config.n_rounds,
        )
        current_candidates = run_merge_round(
            task_id=task_id,
            round_index=round_index,
            problem_statement=problem_statement,
            candidates=current_candidates,
            config=config,
            call_logger=call_logger,
            rng=rng,
        )
    return current_candidates


# ---------------------------------------------------------------------------
# Task processing
# ---------------------------------------------------------------------------


def is_task_done(output_dir: Path, task_id: int) -> bool:
    """Return ``True`` if the task output JSONL already exists and is non-empty.

    A non-empty ``Task_{task_id}_merge_candidates.jsonl`` indicates that
    the task was successfully processed in a previous run and can be
    skipped.
    """
    output_file = output_dir / f"Task_{task_id}_merge_candidates.jsonl"
    return output_file.exists() and output_file.stat().st_size > 0


def process_task(
    *,
    task_id: int,
    problem_statement: str,
    candidates_dir: Path,
    config: MergeCandidatesConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Process all candidates for one task through merge rounds.

    Loads baseline candidates, filters out those with empty solutions
    (requiring at least two non-empty candidates for merging), and runs
    ``config.n_rounds`` of pairwise merges.  All LLM calls are written
    to a single JSONL file per task.

    Args:
        task_id: Dataset task identifier.
        problem_statement: Raw problem text.
        candidates_dir: Directory with baseline LLM output JSONL logs.
        config: Pipeline configuration.
        output_dir: Directory for output files.

    Returns:
        A summary dict with ``task_id``, ``status``, and ``output_path``.

    """
    log_path = output_dir / f"Task_{task_id}_merge_candidates.jsonl"
    call_logger = TaskCallLogger(task_id=task_id, log_path=log_path)

    candidates = load_candidates(candidates_dir, task_id)

    # Filter out empty candidates — merging with an empty solution is pointless.
    valid_candidates = [c for c in candidates if c["solution_text"]]
    if len(valid_candidates) < _MIN_CANDIDATES_FOR_MERGE:
        LOGGER.warning(
            "Task %s has fewer than %s non-empty candidates (%s) — skipping merge.",
            task_id,
            _MIN_CANDIDATES_FOR_MERGE,
            len(valid_candidates),
        )
        return {
            "task_id": task_id,
            "status": "skipped_too_few_candidates",
            "output_path": str(log_path),
        }

    rng = random.Random(config.seed + task_id if config.seed is not None else None)  # noqa: S311

    LOGGER.info(
        "Task %s starting %s rounds of merging with %s candidates",
        task_id,
        config.n_rounds,
        len(valid_candidates),
    )

    run_all_merge_rounds(
        task_id=task_id,
        problem_statement=problem_statement,
        candidates=valid_candidates,
        config=config,
        call_logger=call_logger,
        rng=rng,
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
    config: MergeCandidatesConfig,
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
    config: MergeCandidatesConfig,
    output_dir: Path,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Run merge-candidates for multiple tasks using a thread pool.

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
        description=("Merge-candidates ablation: iteratively merge baseline candidates and log solutions per round."),
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
        help="Merger model name.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Base output directory.",
    )
    parser.add_argument(
        "--merger_max_tokens",
        type=int,
        default=100000,
        help="Maximum merger output tokens.",
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=8,
        help="Number of merge rounds per task.",
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
        "--seed",
        type=int,
        default=None,
        help="Random seed for partner selection reproducibility (default: non-deterministic).",
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
    """Run the merge-candidates ablation over baseline candidates."""
    args = parse_args()
    validate_args(args)

    candidates_dir = Path(args.candidates_dir)
    run_dir = Path(args.output_path) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

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

    config = MergeCandidatesConfig(
        merger_model=args.model,
        merger_max_tokens=args.merger_max_tokens,
        n_rounds=args.n_rounds,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    # Persist config once for the run.
    config_path = run_dir / "config.json"
    config_dict = json.loads(json.dumps(asdict(config)))
    if not config_path.exists():
        write_json(config_path, config_dict)
        LOGGER.info("Config written to %s", config_path)

    problems = [dataset[tid]["Problem"] for tid in task_ids]

    LOGGER.info(
        "Processing %s tasks with concurrency=%s, n_rounds=%s",
        len(task_ids),
        args.concurrency,
        config.n_rounds,
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
