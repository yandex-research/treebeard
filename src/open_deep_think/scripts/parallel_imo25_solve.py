"""Launch parallel task-interval runs of solver scripts.

This wrapper keeps orchestration intentionally simple:
- Split `[start, end)` into `--concurrency` contiguous shards.
- Run one solver process per shard.
- Start all shard commands together.

Each shard gets a unique `--run_name` suffix to avoid output collisions.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Shard:
    """A task interval to solve in one child process."""

    index: int
    start: int
    end: int


@dataclass(frozen=True)
class ShardResult:
    """Execution result for one shard process."""

    shard: Shard
    returncode: int
    command: list[str]
    log_path: str
    elapsed_seconds: float


def split_by_concurrency(start: int, end: int, concurrency: int) -> list[Shard]:
    """Split `[start, end)` into contiguous shards based on concurrency.

    Args:
        start: Start index (inclusive).
        end: End index (exclusive).
        concurrency: Requested number of parallel shards.

    Returns:
        Ordered shard list with contiguous non-overlapping intervals.

    """
    if concurrency <= 0:
        msg = "concurrency must be positive"
        raise ValueError(msg)
    if end <= start:
        return []

    task_count = end - start
    shard_count = min(concurrency, task_count)
    base_size = task_count // shard_count
    remainder = task_count % shard_count

    shards: list[Shard] = []
    shard_start = start
    for shard_index in range(shard_count):
        shard_size = base_size + (1 if shard_index < remainder else 0)
        shard_end = shard_start + shard_size
        shards.append(Shard(index=shard_index, start=shard_start, end=shard_end))
        shard_start = shard_end
    return shards


def add_optional_arg(command: list[str], flag: str, value: str | float | None) -> None:
    """Append a CLI flag and value if the value is not None."""
    if value is None:
        return
    command.extend([flag, str(value)])


def _append_common_tournament_args(command: list[str], args: argparse.Namespace) -> None:
    """Append shared tournament args (verifier, classifier, num_solutions, sampling, dataset)."""
    add_optional_arg(command, "--verifier_model", args.verifier_model)
    add_optional_arg(command, "--classifier_model", args.classifier_model)
    add_optional_arg(command, "--solver_max_tokens", args.solver_max_tokens)
    add_optional_arg(command, "--verifier_max_tokens", args.verifier_max_tokens)
    add_optional_arg(command, "--classifier_max_tokens", args.classifier_max_tokens)
    add_optional_arg(command, "--num_solutions", args.num_solutions)
    add_optional_arg(command, "--temperature", args.temperature)
    add_optional_arg(command, "--top_p", args.top_p)
    add_optional_arg(command, "--dataset_name", args.dataset_name)
    add_optional_arg(command, "--dataset_split", args.dataset_split)
    for other_prompt in args.other_prompt:
        command.extend(["--other_prompt", other_prompt])


def build_child_command(args: argparse.Namespace, shard: Shard, base_run_name: str) -> list[str]:
    """Build the child solver command for one shard."""
    run_name = f"{base_run_name}_shard_{shard.index:03d}"
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
        str(shard.start),
        "--end",
        str(shard.end),
        "--model",
        args.model,
        "--output_path",
        args.output_path,
    ]

    if args.script == "imo25":
        command.extend(["--run_name", run_name])
        add_optional_arg(command, "--verifier_model", args.verifier_model)
        add_optional_arg(command, "--classifier_model", args.classifier_model)
        add_optional_arg(command, "--solver_max_tokens", args.solver_max_tokens)
        add_optional_arg(command, "--verifier_max_tokens", args.verifier_max_tokens)
        add_optional_arg(command, "--classifier_max_tokens", args.classifier_max_tokens)
        add_optional_arg(command, "--max_runs", args.max_runs)
        add_optional_arg(command, "--max_iterations", args.max_iterations)
        add_optional_arg(command, "--required_consecutive_passes", args.required_consecutive_passes)
        add_optional_arg(command, "--max_consecutive_failures", args.max_consecutive_failures)
        add_optional_arg(command, "--temperature", args.temperature)
        add_optional_arg(command, "--top_p", args.top_p)
        add_optional_arg(command, "--dataset_name", args.dataset_name)
        add_optional_arg(command, "--dataset_split", args.dataset_split)
        for other_prompt in args.other_prompt:
            command.extend(["--other_prompt", other_prompt])
    elif args.script == "simple_tournament":
        command.extend(["--run_name", run_name])
        _append_common_tournament_args(command, args)
        add_optional_arg(command, "--judge_model", args.judge_model)
        add_optional_arg(command, "--judge_max_tokens", args.judge_max_tokens)
    elif args.script in {"tournament_merge", "tournament_merge_improve"}:
        command.extend(["--run_name", run_name])
        _append_common_tournament_args(command, args)
        add_optional_arg(command, "--merger_model", args.merger_model)
        add_optional_arg(command, "--merger_max_tokens", args.merger_max_tokens)
    else:
        add_optional_arg(command, "--max_tokens", args.baseline_max_tokens)
        add_optional_arg(command, "--temperature", args.temperature)
        add_optional_arg(command, "--top_p", args.top_p)
    return command


def run_shard(args: argparse.Namespace, shard: Shard, base_run_name: str, launcher_log_dir: Path) -> ShardResult:
    """Execute one shard process and return completion metadata."""
    command = build_child_command(args=args, shard=shard, base_run_name=base_run_name)
    log_path = launcher_log_dir / f"shard_{shard.index:03d}.log"
    started = time.monotonic()

    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(  # noqa: S603
            command,
            check=False,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    elapsed_seconds = time.monotonic() - started
    return ShardResult(
        shard=shard,
        returncode=completed.returncode,
        command=command,
        log_path=str(log_path),
        elapsed_seconds=elapsed_seconds,
    )


def parse_args() -> argparse.Namespace:
    """Parse launcher and child-forwarded options."""
    parser = argparse.ArgumentParser(description="Run parallel task-interval solver shards")
    parser.add_argument("--start", type=int, required=True, help="Starting task index (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="Ending task index (exclusive)")
    parser.add_argument("--model", type=str, required=True, help="Solver model name")
    parser.add_argument("--output_path", type=str, required=True, help="Base output directory for child runs")
    parser.add_argument(
        "--script",
        choices=["imo25", "simple_tournament", "tournament_merge", "tournament_merge_improve", "baseline"],
        default="imo25",
        help="Child script to run in parallel shards",
    )

    parser.add_argument("--concurrency", type=int, default=1, help="Number of parallel shards/processes")
    parser.add_argument("--run_name", type=str, help="Base run name for all shards")
    parser.add_argument(
        "--launcher_log_dir",
        type=str,
        help=(
            "Directory for launcher shard logs and summary (default: <output_path>/imo25_parallel_launcher/<run_name>)"
        ),
    )
    parser.add_argument("--dry_run", action="store_true", help="Print shard commands without executing them")

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
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_split", type=str)
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
    # Forwarded baseline arguments.
    parser.add_argument("--baseline_max_tokens", type=int, help="Forwarded as --max_tokens for baseline_solve")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate essential launcher argument constraints."""
    if args.start < 0:
        msg = "--start must be non-negative"
        raise ValueError(msg)
    if args.end <= args.start:
        msg = "--end must be greater than --start"
        raise ValueError(msg)
    if args.concurrency <= 0:
        msg = "--concurrency must be positive"
        raise ValueError(msg)


def main() -> int:
    """Execute shard orchestration and return process exit code."""
    args = parse_args()
    validate_args(args)

    base_run_name = args.run_name or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    launcher_log_dir = (
        Path(args.launcher_log_dir)
        if args.launcher_log_dir
        else Path(args.output_path) / f"{args.script}_parallel_launcher" / base_run_name
    )
    launcher_log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s", force=True)

    shards = split_by_concurrency(start=args.start, end=args.end, concurrency=args.concurrency)
    if not shards:
        LOGGER.info("No shards to run.")
        return 0

    LOGGER.info("Script: %s | shards: %s (requested concurrency: %s)", args.script, len(shards), args.concurrency)
    for shard in shards:
        LOGGER.info("Shard %03d interval [%s, %s)", shard.index, shard.start, shard.end)

    if args.dry_run:
        for shard in shards:
            command = build_child_command(args=args, shard=shard, base_run_name=base_run_name)
            LOGGER.info("DRY RUN shard %03d: %s", shard.index, " ".join(command))
        return 0

    started_at = datetime.now(UTC)
    processes: list[tuple[Shard, list[str], subprocess.Popen[str], str, float]] = []
    for shard in shards:
        command = build_child_command(args=args, shard=shard, base_run_name=base_run_name)
        log_path = launcher_log_dir / f"shard_{shard.index:03d}.log"
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(  # noqa: S603
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_file.close()
        processes.append((shard, command, process, str(log_path), time.monotonic()))

    results: list[ShardResult] = []
    for shard, command, process, log_path, started_monotonic in processes:
        returncode = process.wait()
        result = ShardResult(
            shard=shard,
            returncode=returncode,
            command=command,
            log_path=log_path,
            elapsed_seconds=time.monotonic() - started_monotonic,
        )
        results.append(result)
        status = "ok" if result.returncode == 0 else "failed"
        LOGGER.info(
            "Shard %03d finished (%s) in %.1fs, log=%s",
            result.shard.index,
            status,
            result.elapsed_seconds,
            result.log_path,
        )

    completed_at = datetime.now(UTC)
    results.sort(key=lambda item: item.shard.index)
    failed_count = sum(result.returncode != 0 for result in results)
    summary: dict[str, Any] = {
        "base_run_name": base_run_name,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": (completed_at - started_at).total_seconds(),
        "concurrency": args.concurrency,
        "task_range": {"start": args.start, "end": args.end},
        "failed_count": failed_count,
        "succeeded_count": len(results) - failed_count,
        "results": [asdict(result) for result in results],
    }
    summary_path = launcher_log_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)
    LOGGER.info("Launcher summary saved to %s", summary_path)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
