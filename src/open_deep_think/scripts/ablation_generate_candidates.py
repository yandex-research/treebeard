r"""Ablation: generate independent candidates with verification reports.

This script implements the candidate-generation phase of the
``tournament_merge_improve`` pipeline as a standalone ablation study.  For each
task it generates *n_solutions* independent solutions, verifies each one (and
optionally runs self-improvement rounds) — exactly as
``tournament_merge_improve.py`` does before entering its tournament bracket.

All candidates across all tasks are stored in a **single JSONL file**
(``candidates.jsonl``), where every record contains ``task_id``,
``candidate_index``, the full ``solution_text``, and a nested ``verification``
dict.  Per-candidate LLM call logs are also written for debuggability.

Concurrency is managed via a ``ThreadPoolExecutor`` operating on individual
``(task_id, candidate_index)`` pairs — not on whole tasks.  This ensures high
utilisation even when some tasks are much harder than others: easy tasks finish
quickly and free up worker slots for harder ones.

Resumability: individual ``(task_id, candidate_index)`` pairs already present
in the output JSONL are skipped automatically.

Usage::

    python -m open_deep_think.scripts.ablation_generate_candidates \
        --task_ids_file src/exps/subset_1.txt \
        --model openai/gpt-oss-120b \
        --n_solutions 8 \
        --concurrency 15 \
        --output_path data/ \
        --run_name ablation_subset_1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import urllib3
from datasets import load_dataset

from open_deep_think.scripts.parallel_solve import load_task_ids_from_file
from open_deep_think.scripts.tournament_merge_improve import (
    _FAILED_VERIFICATION,
    TaskCallLogger,
    TournamentMergeImproveConfig,
    generate_candidate,
    utc_now_iso,
    write_json,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkItem:
    """A single (task_id, candidate_index) unit of work."""

    task_id: int
    candidate_index: int
    problem_statement: str


# ---------------------------------------------------------------------------
# Thread-safe JSONL writer
# ---------------------------------------------------------------------------


class ThreadSafeJSONLWriter:
    """Append-only JSONL writer safe for concurrent use from multiple threads.

    Each :meth:`append` call serialises the payload to JSON, acquires an
    internal lock, opens the file in append mode, writes one line, and closes
    the file — ensuring atomic per-record writes even under high concurrency.

    Args:
        path: Filesystem path to the JSONL file.

    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Return the underlying file path."""
        return self._path

    def append(self, payload: dict[str, Any]) -> None:
        """Serialise *payload* and append it as a single JSON line."""
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)


# ---------------------------------------------------------------------------
# Argument parsing & validation
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ablation candidate generation."""
    parser = argparse.ArgumentParser(
        description="Ablation: generate independent candidates with verification for IMO AnswerBench",
    )

    # Task source: either file or range (mutually exclusive).
    parser.add_argument(
        "--task_ids_file",
        type=str,
        default=None,
        help="Path to a text file with one task ID per line. Mutually exclusive with --start/--end.",
    )
    parser.add_argument("--start", type=int, default=None, help="Starting task index (inclusive). Requires --end.")
    parser.add_argument("--end", type=int, default=None, help="Ending task index (exclusive). Requires --start.")

    # Core settings.
    parser.add_argument("--model", type=str, required=True, help="Solver model name")
    parser.add_argument("--output_path", type=str, required=True, help="Base output directory")
    parser.add_argument("--run_name", type=str, default="default", help="Run directory name (default: 'default')")
    parser.add_argument("--n_solutions", type=int, default=8, help="Number of independent solutions per task")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent (task, candidate) workers (default: 1, sequential)",
    )

    # Model overrides.
    parser.add_argument("--verifier_model", type=str, help="Verifier model name (default: same as --model)")
    parser.add_argument("--classifier_model", type=str, help="Binary checker model name (default: verifier model)")

    # Token limits.
    parser.add_argument("--solver_max_tokens", type=int, default=64000, help="Maximum solver output tokens")
    parser.add_argument("--verifier_max_tokens", type=int, default=64000, help="Maximum verifier output tokens")
    parser.add_argument("--classifier_max_tokens", type=int, default=64000, help="Maximum checker output tokens")

    # Sampling.
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top_p")

    # Self-improvement (0 = skip entirely).
    parser.add_argument(
        "--si_rounds",
        type=int,
        default=1,
        help="Maximum self-improvement rounds per verification failure (0 to disable, default: 1)",
    )

    # Extra prompts.
    parser.add_argument(
        "--other_prompt",
        action="append",
        default=[],
        help="Additional user prompt appended after the problem statement. Repeat for multiple.",
    )

    # Dataset.
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Hwilner/imo-answerbench",
        help="Hugging Face dataset name",
    )
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to evaluate")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate mutual exclusivity and value constraints.

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

    if args.n_solutions < 1:
        msg = f"--n_solutions must be >= 1, got {args.n_solutions}"
        raise ValueError(msg)

    if args.si_rounds < 0:
        msg = f"--si_rounds must be >= 0, got {args.si_rounds}"
        raise ValueError(msg)

    if args.concurrency < 1:
        msg = f"--concurrency must be >= 1, got {args.concurrency}"
        raise ValueError(msg)


def resolve_task_ids(args: argparse.Namespace) -> list[int]:
    """Resolve task IDs from ``--task_ids_file`` or ``--start``/``--end``.

    Args:
        args: Parsed namespace (already validated).

    Returns:
        Ordered list of integer task IDs.

    """
    if args.task_ids_file is not None:
        return load_task_ids_from_file(args.task_ids_file)
    return list(range(args.start, args.end))


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def load_completed_task_ids(candidates_jsonl_path: Path, n_solutions: int) -> set[int]:
    """Return task IDs that already have *n_solutions* candidates in the JSONL.

    This enables simple resumability: if a previous run was interrupted, tasks
    whose candidates are fully recorded are skipped on the next invocation.

    Args:
        candidates_jsonl_path: Path to the ``candidates.jsonl`` file.
        n_solutions: Expected number of candidates per task.

    Returns:
        Set of task IDs that are already complete.

    """
    if not candidates_jsonl_path.exists():
        return set()

    counts: dict[int, int] = {}
    with candidates_jsonl_path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            tid = record["task_id"]
            counts[tid] = counts.get(tid, 0) + 1

    return {tid for tid, count in counts.items() if count >= n_solutions}


def load_completed_pairs(candidates_jsonl_path: Path) -> set[tuple[int, int]]:
    """Return ``(task_id, candidate_index)`` pairs already in the JSONL.

    This provides fine-grained resumability at the individual candidate level,
    which is important when running concurrently over ``(task, candidate)``
    pairs: a partially completed task can be resumed without re-generating
    candidates that already succeeded.

    Args:
        candidates_jsonl_path: Path to the ``candidates.jsonl`` file.

    Returns:
        Set of ``(task_id, candidate_index)`` tuples already recorded.

    """
    if not candidates_jsonl_path.exists():
        return set()

    pairs: set[tuple[int, int]] = set()
    with candidates_jsonl_path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            pairs.add((record["task_id"], record["candidate_index"]))

    return pairs


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_config(args: argparse.Namespace) -> TournamentMergeImproveConfig:
    """Build a :class:`TournamentMergeImproveConfig` from CLI arguments.

    Merger-related fields are set to the solver model for compatibility with
    :func:`generate_candidate`; they are never actually used during ablation.

    Args:
        args: Parsed and validated namespace.

    Returns:
        A frozen pipeline configuration.

    """
    solver_model = args.model
    verifier_model = args.verifier_model or solver_model
    classifier_model = args.classifier_model or verifier_model

    return TournamentMergeImproveConfig(
        solver_model=solver_model,
        verifier_model=verifier_model,
        classifier_model=classifier_model,
        merger_model=solver_model,  # unused — set for dataclass compatibility
        solver_max_tokens=args.solver_max_tokens,
        verifier_max_tokens=args.verifier_max_tokens,
        classifier_max_tokens=args.classifier_max_tokens,
        merger_max_tokens=args.solver_max_tokens,  # unused
        num_solutions=args.n_solutions,
        temperature=args.temperature,
        top_p=args.top_p,
        other_prompts=tuple(args.other_prompt),
        si_rounds=args.si_rounds,
    )


# ---------------------------------------------------------------------------
# Work item builders
# ---------------------------------------------------------------------------


def build_work_items(
    task_ids: list[int],
    n_solutions: int,
    problem_statements: dict[int, str],
    completed_pairs: set[tuple[int, int]],
) -> list[WorkItem]:
    """Build the list of ``(task_id, candidate_index)`` work items to process.

    Work items whose ``(task_id, candidate_index)`` pair is already in
    *completed_pairs* are excluded, enabling fine-grained resume.

    Args:
        task_ids: Ordered list of task IDs to process.
        n_solutions: Number of candidates per task.
        problem_statements: Mapping from task ID to problem text.
        completed_pairs: Set of already-completed ``(task_id, candidate_index)`` pairs.

    Returns:
        Ordered list of :class:`WorkItem` objects.

    """
    items: list[WorkItem] = []
    for tid in task_ids:
        items.extend(
            WorkItem(task_id=tid, candidate_index=idx, problem_statement=problem_statements[tid])
            for idx in range(n_solutions)
            if (tid, idx) not in completed_pairs
        )
    return items


# ---------------------------------------------------------------------------
# Single-candidate worker
# ---------------------------------------------------------------------------


def generate_single_candidate(
    *,
    work_item: WorkItem,
    config: TournamentMergeImproveConfig,
    run_dir: Path,
    jsonl_writer: ThreadSafeJSONLWriter,
) -> dict[str, Any]:
    """Generate one candidate for a single ``(task_id, candidate_index)`` pair.

    This is the unit of work for the concurrent pool.  Each invocation:

    1. Creates its own :class:`TaskCallLogger` with a per-candidate log file
       (``Task_{task_id}_cand_{idx}_llm_outputs.jsonl``) to avoid file-level
       conflicts between concurrent workers operating on the same task.
    2. Calls :func:`generate_candidate` from ``tournament_merge_improve``.
    3. Appends the result record to the shared ``candidates.jsonl`` via the
       thread-safe *jsonl_writer*.

    Args:
        work_item: The ``(task_id, candidate_index, problem_statement)`` to process.
        config: Pipeline configuration.
        run_dir: Output directory for per-candidate LLM logs.
        jsonl_writer: Thread-safe writer for the shared ``candidates.jsonl``.

    Returns:
        The record dict that was written to the JSONL.

    """
    task_id = work_item.task_id
    idx = work_item.candidate_index

    # Per-candidate log file avoids conflicts when multiple candidates for
    # the same task run concurrently.
    llm_log_path = run_dir / f"Task_{task_id}_cand_{idx}_llm_outputs.jsonl"
    call_logger = TaskCallLogger(task_id=task_id, task_log_path=llm_log_path)

    try:
        candidate = generate_candidate(
            task_id=task_id,
            candidate_index=idx,
            problem_statement=work_item.problem_statement,
            config=config,
            call_logger=call_logger,
        )
        record: dict[str, Any] = {
            "task_id": task_id,
            "candidate_index": idx,
            "solution_text": candidate.solution_text,
            "verification": {
                "is_pass": candidate.verification.is_pass,
                "bug_report": candidate.verification.bug_report,
                "verifier_output": candidate.verification.verifier_output,
                "classifier_output": candidate.verification.classifier_output,
                "verifier_call_id": candidate.verification.verifier_call_id,
                "classifier_call_id": candidate.verification.classifier_call_id,
            },
            "status": "ok",
            "timestamp": utc_now_iso(),
        }
    except Exception as exc:
        LOGGER.exception("Task %s candidate %s generation failed", task_id, idx)
        record = {
            "task_id": task_id,
            "candidate_index": idx,
            "solution_text": "",
            "verification": asdict(_FAILED_VERIFICATION),
            "status": "error",
            "error": str(exc),
            "timestamp": utc_now_iso(),
        }

    jsonl_writer.append(record)
    LOGGER.info(
        "Task %s candidate %s: status=%s verification=%s",
        task_id,
        idx,
        record["status"],
        record.get("verification", {}).get("is_pass", "N/A"),
    )
    return record


# ---------------------------------------------------------------------------
# Sequential fallback (kept for backwards compatibility with tests)
# ---------------------------------------------------------------------------


def generate_candidates_for_task(
    *,
    task_id: int,
    problem_statement: str,
    config: TournamentMergeImproveConfig,
    run_dir: Path,
    candidates_jsonl_path: Path,
) -> list[dict[str, Any]]:
    """Generate all candidates for one task sequentially, appending to JSONL.

    This is the non-concurrent code path, preserved for simple use-cases and
    testing.  The concurrent :func:`main` uses :func:`generate_single_candidate`
    instead.

    Each candidate record is appended to *candidates_jsonl_path* immediately
    after generation so that partial progress is preserved if the process is
    interrupted.

    Args:
        task_id: Dataset task identifier.
        problem_statement: Raw problem text from the dataset.
        config: Pipeline configuration.
        run_dir: Output directory for per-task LLM logs.
        candidates_jsonl_path: Path to the shared ``candidates.jsonl``.

    Returns:
        List of dicts written to the JSONL (one per candidate).

    """
    writer = ThreadSafeJSONLWriter(candidates_jsonl_path)
    records: list[dict[str, Any]] = []

    for idx in range(config.num_solutions):
        item = WorkItem(task_id=task_id, candidate_index=idx, problem_statement=problem_statement)
        record = generate_single_candidate(
            work_item=item,
            config=config,
            run_dir=run_dir,
            jsonl_writer=writer,
        )
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Concurrent execution
# ---------------------------------------------------------------------------


def run_generation(
    *,
    work_items: list[WorkItem],
    config: TournamentMergeImproveConfig,
    run_dir: Path,
    jsonl_writer: ThreadSafeJSONLWriter,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Execute all work items with up to *concurrency* parallel workers.

    Uses :class:`ThreadPoolExecutor` because the work is I/O-bound (LLM API
    calls).  Each worker handles exactly one ``(task_id, candidate_index)``
    pair, ensuring fine-grained scheduling: easy tasks free up slots quickly,
    keeping utilisation high even when difficulty varies across tasks.

    Args:
        work_items: Ordered list of work items to process.
        config: Pipeline configuration.
        run_dir: Output directory for per-candidate LLM logs.
        jsonl_writer: Thread-safe writer for the shared ``candidates.jsonl``.
        concurrency: Maximum number of parallel workers.

    Returns:
        List of result dicts (one per work item), in completion order.

    """
    results: list[dict[str, Any]] = []

    if concurrency <= 1:
        # Sequential fallback — avoid thread pool overhead.
        for item in work_items:
            record = generate_single_candidate(
                work_item=item,
                config=config,
                run_dir=run_dir,
                jsonl_writer=jsonl_writer,
            )
            results.append(record)
        return results

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_item = {
            pool.submit(
                generate_single_candidate,
                work_item=item,
                config=config,
                run_dir=run_dir,
                jsonl_writer=jsonl_writer,
            ): item
            for item in work_items
        }
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                record = future.result()
            except Exception:
                LOGGER.exception(
                    "Unexpected worker error for task %s candidate %s",
                    item.task_id,
                    item.candidate_index,
                )
                record = {
                    "task_id": item.task_id,
                    "candidate_index": item.candidate_index,
                    "status": "worker_error",
                }
            results.append(record)

    return results


# ---------------------------------------------------------------------------
# Logging & main
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    """Configure stdout-only logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def main() -> None:
    """Run ablation candidate generation over specified tasks."""
    args = parse_args()
    validate_args(args)
    configure_logging()

    task_ids = resolve_task_ids(args)
    config = build_config(args)

    run_dir = Path(args.output_path) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    candidates_jsonl_path = run_dir / "candidates.jsonl"
    jsonl_writer = ThreadSafeJSONLWriter(candidates_jsonl_path)

    # Write config (idempotent — skips if already present).
    config_path = run_dir / "config.json"
    config_dict: dict[str, Any] = asdict(config)
    config_dict["task_ids"] = task_ids
    config_dict["script"] = "ablation_generate_candidates"
    config_dict["concurrency"] = args.concurrency
    if not config_path.exists():
        write_json(config_path, config_dict)
        LOGGER.info("Config written to %s", config_path)
    else:
        LOGGER.info("Config already exists at %s — skipping write.", config_path)

    LOGGER.info("Loading dataset: %s (%s)", args.dataset_name, args.dataset_split)
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    LOGGER.info("Loaded %s tasks", len(dataset))

    # Validate task IDs against dataset size.
    for tid in task_ids:
        if tid < 0 or tid >= len(dataset):
            msg = f"Task ID {tid} out of range [0, {len(dataset)})"
            raise ValueError(msg)

    # Build problem statement lookup.
    problem_statements = {tid: dataset[tid]["Problem"] for tid in task_ids}

    # Determine which (task_id, candidate_index) pairs are already done.
    completed_pairs = load_completed_pairs(candidates_jsonl_path)
    work_items = build_work_items(task_ids, config.num_solutions, problem_statements, completed_pairs)

    total_pairs = len(task_ids) * config.num_solutions
    LOGGER.info(
        "%s total (task, candidate) pairs — %s already complete, %s remaining. Concurrency: %s. Output: %s",
        total_pairs,
        len(completed_pairs),
        len(work_items),
        args.concurrency,
        candidates_jsonl_path,
    )

    if not work_items:
        LOGGER.info("Nothing to do — all candidates already generated.")
        return

    results = run_generation(
        work_items=work_items,
        config=config,
        run_dir=run_dir,
        jsonl_writer=jsonl_writer,
        concurrency=args.concurrency,
    )

    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_pass = sum(1 for r in results if r.get("verification", {}).get("is_pass", False))
    LOGGER.info(
        "Generation complete. %s/%s succeeded, %s/%s passed verification. Results in %s",
        n_ok,
        len(results),
        n_pass,
        len(results),
        candidates_jsonl_path,
    )


if __name__ == "__main__":
    main()
