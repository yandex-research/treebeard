"""Launch parallel task runs of solver scripts using a global work queue.

This wrapper keeps orchestration intentionally simple:
- Build a queue of individual task indices from ``[start, end)`` or from a
  file of explicit task IDs (``--task_ids_file``).
- Maintain a pool of up to ``--concurrency`` active child processes.
- As each process finishes, immediately launch the next task from the queue.

This ensures full CPU/API utilisation even when individual tasks vary widely
in duration — no worker ever sits idle while tasks remain in the queue.

All child processes write into the same output directory (``--run_name`` is
shared).  The first task in the queue is assigned ``--shard_index 0`` so that
it writes ``config.json``; all other tasks receive a non-zero shard index and
skip that write.

Logs are written to a timestamped subdirectory under the launcher log dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def sanitize_model_name(model: str) -> str:
    """Convert a model identifier to a filesystem-safe slug."""
    return model.replace("/", "__").replace(":", "_")


@dataclass(frozen=True)
class TaskSlot:
    """A single task to be solved in one child process."""

    task_index: int
    shard_index: int


@dataclass(frozen=True)
class TaskResult:
    """Execution result for one task process."""

    slot: TaskSlot
    returncode: int
    command: list[str]
    log_path: str
    elapsed_seconds: float


def add_optional_arg(command: list[str], flag: str, value: str | float | None) -> None:
    """Append a CLI flag and value if the value is not None."""
    if value is None:
        return
    command.extend([flag, str(value)])


def _append_common_args(command: list[str], args: argparse.Namespace, base_run_name: str, slot: TaskSlot) -> None:
    """Append args shared by all child scripts (run name, shard index, sampling, dataset)."""
    command.extend(["--run_name", base_run_name, "--shard_index", str(slot.shard_index)])
    add_optional_arg(command, "--temperature", args.temperature)
    add_optional_arg(command, "--top_p", args.top_p)
    add_optional_arg(command, "--dataset_name", args.dataset_name)
    add_optional_arg(command, "--dataset_split", args.dataset_split)


def _append_common_tournament_args(command: list[str], args: argparse.Namespace) -> None:
    """Append shared tournament args (verifier, classifier, num_solutions, prompts)."""
    add_optional_arg(command, "--verifier_model", args.verifier_model)
    add_optional_arg(command, "--classifier_model", args.classifier_model)
    add_optional_arg(command, "--solver_max_tokens", args.solver_max_tokens)
    add_optional_arg(command, "--verifier_max_tokens", args.verifier_max_tokens)
    add_optional_arg(command, "--classifier_max_tokens", args.classifier_max_tokens)
    add_optional_arg(command, "--num_solutions", args.num_solutions)
    for other_prompt in args.other_prompt:
        command.extend(["--other_prompt", other_prompt])


def build_child_command(args: argparse.Namespace, slot: TaskSlot, base_run_name: str) -> list[str]:
    """Build the child solver command for one task slot.

    Each child processes exactly one task (``--start task_index --end task_index+1``).
    The ``--shard_index`` is set to ``slot.shard_index``; only shard 0 writes
    ``config.json``, so the first task in the range gets shard index 0.

    Args:
        args: Parsed launcher arguments.
        slot: The task slot describing which task to run and its shard index.
        base_run_name: Shared run name forwarded to all child scripts.

    Returns:
        A list of strings forming the child process command.

    """
    module_map = {
        "imo25": "open_deep_think.scripts.imo25_solve",
        "simple_tournament": "open_deep_think.scripts.simple_tournament",
        "tournament_merge": "open_deep_think.scripts.tournament_merge",
        "tournament_merge_improve": "open_deep_think.scripts.tournament_merge_improve",
        "baseline": "open_deep_think.scripts.baseline_solve",
    }
    module_name = module_map[args.script]
    command = [
        sys.executable,
        "-m",
        module_name,
        "--start",
        str(slot.task_index),
        "--end",
        str(slot.task_index + 1),
        "--model",
        args.model,
        "--output_path",
        args.output_path,
    ]

    _append_common_args(command, args, base_run_name, slot)

    if args.script == "imo25":
        add_optional_arg(command, "--verifier_model", args.verifier_model)
        add_optional_arg(command, "--classifier_model", args.classifier_model)
        add_optional_arg(command, "--solver_max_tokens", args.solver_max_tokens)
        add_optional_arg(command, "--verifier_max_tokens", args.verifier_max_tokens)
        add_optional_arg(command, "--classifier_max_tokens", args.classifier_max_tokens)
        add_optional_arg(command, "--max_runs", args.max_runs)
        add_optional_arg(command, "--max_iterations", args.max_iterations)
        add_optional_arg(command, "--required_consecutive_passes", args.required_consecutive_passes)
        add_optional_arg(command, "--max_consecutive_failures", args.max_consecutive_failures)
        for other_prompt in args.other_prompt:
            command.extend(["--other_prompt", other_prompt])
    elif args.script == "simple_tournament":
        _append_common_tournament_args(command, args)
        add_optional_arg(command, "--judge_model", args.judge_model)
        add_optional_arg(command, "--judge_max_tokens", args.judge_max_tokens)
    elif args.script in {"tournament_merge", "tournament_merge_improve"}:
        _append_common_tournament_args(command, args)
        add_optional_arg(command, "--merger_model", args.merger_model)
        add_optional_arg(command, "--merger_max_tokens", args.merger_max_tokens)
        if args.script == "tournament_merge_improve":
            add_optional_arg(command, "--si_rounds", args.si_rounds)
    else:
        add_optional_arg(command, "--max_tokens", args.baseline_max_tokens)
    return command


def run_worker_pool(
    args: argparse.Namespace,
    slots: list[TaskSlot],
    base_run_name: str,
    launcher_log_dir: Path,
) -> list[TaskResult]:
    """Run all task slots through a bounded worker pool.

    Maintains up to ``args.concurrency`` active child processes at all times.
    As each process finishes, the next slot from the queue is launched
    immediately, ensuring full parallelism regardless of per-task duration.

    Args:
        args: Parsed launcher arguments (used for ``concurrency`` and child command building).
        slots: Ordered list of task slots to execute.
        base_run_name: Shared run name forwarded to child scripts.
        launcher_log_dir: Directory where per-task log files are written.

    Returns:
        List of :class:`TaskResult` objects in completion order.

    """
    pending: deque[TaskSlot] = deque(slots)
    # active: list of (slot, command, process, log_path, start_monotonic)
    active: list[tuple[TaskSlot, list[str], subprocess.Popen[str], str, float]] = []
    results: list[TaskResult] = []

    while pending or active:
        # Fill up to concurrency slots.
        while pending and len(active) < args.concurrency:
            slot = pending.popleft()
            command = build_child_command(args=args, slot=slot, base_run_name=base_run_name)
            log_path = launcher_log_dir / f"task_{slot.task_index:06d}.log"
            log_file = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(  # noqa: S603
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_file.close()
            LOGGER.info(
                "Launched task %s (shard_index=%s), log=%s",
                slot.task_index,
                slot.shard_index,
                log_path,
            )
            active.append((slot, command, process, str(log_path), time.monotonic()))

        # Poll for any finished process.
        still_active = []
        for slot, command, process, log_path, started_monotonic in active:
            returncode = process.poll()
            if returncode is None:
                still_active.append((slot, command, process, log_path, started_monotonic))
            else:
                elapsed = time.monotonic() - started_monotonic
                result = TaskResult(
                    slot=slot,
                    returncode=returncode,
                    command=command,
                    log_path=log_path,
                    elapsed_seconds=elapsed,
                )
                results.append(result)
                status = "ok" if returncode == 0 else "failed"
                LOGGER.info(
                    "Task %s finished (%s) in %.1fs, log=%s",
                    slot.task_index,
                    status,
                    elapsed,
                    log_path,
                )
        active = still_active

        # Avoid busy-waiting when all slots are occupied.
        if active and (not pending or len(active) >= args.concurrency):
            time.sleep(0.5)

    return results


def parse_args() -> argparse.Namespace:
    """Parse launcher and child-forwarded options.

    Task source is specified via one of two mutually exclusive modes:

    * ``--task_ids_file PATH`` — read explicit task IDs from a text file
      (one integer per line).  ``--start`` and ``--end`` must not be given.
    * ``--start N --end M`` — run all tasks in the range ``[N, M)``.
      ``--task_ids_file`` must not be given.

    Mutual exclusivity is enforced in :func:`validate_args`.
    """
    parser = argparse.ArgumentParser(description="Run parallel task solver using a global work queue")
    parser.add_argument(
        "--task_ids_file",
        type=str,
        default=None,
        help=("Path to a text file with one task ID per line. Mutually exclusive with --start/--end."),
    )
    parser.add_argument("--start", type=int, default=None, help="Starting task index (inclusive). Requires --end.")
    parser.add_argument("--end", type=int, default=None, help="Ending task index (exclusive). Requires --start.")
    parser.add_argument("--model", type=str, required=True, help="Solver model name")
    parser.add_argument("--output_path", type=str, required=True, help="Base output directory for child runs")
    parser.add_argument(
        "--script",
        choices=["baseline", "imo25", "simple_tournament", "tournament_merge", "tournament_merge_improve"],
        default="baseline",
        help="Child script to run in parallel",
    )

    parser.add_argument("--concurrency", type=int, default=1, help="Maximum number of parallel worker processes")
    parser.add_argument(
        "--run_name",
        type=str,
        default="default",
        help="Logical run name forwarded to child scripts and used in log path (default: 'default')",
    )
    parser.add_argument(
        "--launcher_log_dir",
        type=str,
        help=(
            "Directory for launcher task logs and summary "
            "(default: <output_path>/<script>_parallel_launcher/<model>/<run_name>/<timestamp>/)"
        ),
    )
    parser.add_argument("--dry_run", action="store_true", help="Print task commands without executing them")

    # Forwarded imo25 arguments.
    parser.add_argument("--verifier_model", type=str)
    parser.add_argument("--classifier_model", type=str)
    parser.add_argument("--solver_max_tokens", type=int)
    parser.add_argument("--verifier_max_tokens", type=int)
    parser.add_argument("--classifier_max_tokens", type=int)
    parser.add_argument("--max_runs", type=int)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--required_consecutive_passes", type=int)
    parser.add_argument("--max_consecutive_failures", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_p", type=float)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Hwilner/imo-answerbench",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--other_prompt",
        action="append",
        default=[],
        help="Additional user prompt forwarded to child solver. Repeat for multiple prompts.",
    )
    # Forwarded simple_tournament arguments.
    parser.add_argument("--judge_model", type=str, help="Forwarded as --judge_model for simple_tournament")
    parser.add_argument("--judge_max_tokens", type=int, help="Forwarded as --judge_max_tokens for simple_tournament")
    parser.add_argument("--num_solutions", type=int, help="Forwarded as --num_solutions for tournament scripts")
    # Forwarded tournament_merge arguments.
    parser.add_argument("--merger_model", type=str, help="Forwarded as --merger_model for tournament_merge")
    parser.add_argument("--merger_max_tokens", type=int, help="Forwarded as --merger_max_tokens for tournament_merge")
    # Forwarded tournament_merge_improve arguments.
    parser.add_argument("--si_rounds", type=int, help="Forwarded as --si_rounds for tournament_merge_improve")
    # Forwarded baseline arguments.
    parser.add_argument("--baseline_max_tokens", type=int, help="Forwarded as --max_tokens for baseline_solve")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate essential launcher argument constraints.

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
        # Range mode: both --start and --end are required.
        if args.start is None or args.end is None:
            msg = "Either --task_ids_file or both --start and --end must be provided"
            raise ValueError(msg)
        if args.start < 0:
            msg = "--start must be non-negative"
            raise ValueError(msg)
        if args.end <= args.start:
            msg = "--end must be greater than --start"
            raise ValueError(msg)

    if args.concurrency <= 0:
        msg = "--concurrency must be positive"
        raise ValueError(msg)


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


def build_task_slots(start: int, end: int) -> list[TaskSlot]:
    """Build an ordered list of task slots for the range ``[start, end)``.

    The first task (``start``) receives ``shard_index=0`` so that it writes
    ``config.json``; all subsequent tasks receive a non-zero shard index and
    skip that write.

    Args:
        start: Start index (inclusive).
        end: End index (exclusive).

    Returns:
        Ordered list of :class:`TaskSlot` objects, one per task.

    """
    return [TaskSlot(task_index=idx, shard_index=0 if idx == start else idx - start) for idx in range(start, end)]


def build_task_slots_from_ids(task_ids: list[int]) -> list[TaskSlot]:
    """Build an ordered list of task slots from an explicit list of task IDs.

    The first ID in the list receives ``shard_index=0`` so that it writes
    ``config.json``; all subsequent IDs receive a non-zero shard index.

    Args:
        task_ids: Ordered list of task IDs to run.

    Returns:
        Ordered list of :class:`TaskSlot` objects, one per task ID.

    """
    return [TaskSlot(task_index=tid, shard_index=i) for i, tid in enumerate(task_ids)]


def main() -> int:
    """Execute queue-based task orchestration and return process exit code."""
    args = parse_args()
    validate_args(args)

    base_run_name = args.run_name or "default"
    launch_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model_slug = sanitize_model_name(args.model)
    launcher_log_dir = (
        Path(args.launcher_log_dir)
        if args.launcher_log_dir
        else Path(args.output_path)
        / f"{args.script}_parallel_launcher"
        / model_slug
        / base_run_name
        / launch_timestamp
    )
    launcher_log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s", force=True)

    if args.task_ids_file is not None:
        task_ids = load_task_ids_from_file(args.task_ids_file)
        slots = build_task_slots_from_ids(task_ids)
        task_source_info: dict[str, Any] = {"task_ids_file": args.task_ids_file, "task_ids": task_ids}
        LOGGER.info(
            "Script: %s | tasks: %s | concurrency: %s | source: %s",
            args.script,
            len(slots),
            args.concurrency,
            args.task_ids_file,
        )
    else:
        slots = build_task_slots(start=args.start, end=args.end)
        task_source_info = {"start": args.start, "end": args.end}
        LOGGER.info(
            "Script: %s | tasks: %s | concurrency: %s | range: [%s, %s)",
            args.script,
            len(slots),
            args.concurrency,
            args.start,
            args.end,
        )

    if args.dry_run:
        for slot in slots:
            command = build_child_command(args=args, slot=slot, base_run_name=base_run_name)
            LOGGER.info("DRY RUN task %s (shard_index=%s): %s", slot.task_index, slot.shard_index, " ".join(command))
        return 0

    started_at = datetime.now(UTC)
    results = run_worker_pool(
        args=args,
        slots=slots,
        base_run_name=base_run_name,
        launcher_log_dir=launcher_log_dir,
    )
    completed_at = datetime.now(UTC)

    results.sort(key=lambda r: r.slot.task_index)
    failed_count = sum(r.returncode != 0 for r in results)
    summary: dict[str, Any] = {
        "base_run_name": base_run_name,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": (completed_at - started_at).total_seconds(),
        "concurrency": args.concurrency,
        "task_source": task_source_info,
        "failed_count": failed_count,
        "succeeded_count": len(results) - failed_count,
        "results": [asdict(r) for r in results],
    }
    summary_path = launcher_log_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)
    LOGGER.info("Launcher summary saved to %s", summary_path)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
