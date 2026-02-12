"""Tests for the parallel IMO25 launcher helpers."""

from __future__ import annotations

import argparse

from open_deep_think.scripts.parallel_imo25_solve import (
    Shard,
    build_child_command,
    split_by_concurrency,
)


def _base_args() -> argparse.Namespace:
    """Build a minimal namespace for command construction tests."""
    return argparse.Namespace(
        model="provider/model",
        output_path="./logs",
        verifier_model=None,
        classifier_model=None,
        solver_max_tokens=None,
        verifier_max_tokens=None,
        classifier_max_tokens=None,
        max_runs=None,
        max_iterations=None,
        required_consecutive_passes=None,
        max_consecutive_failures=None,
        temperature=None,
        top_p=None,
        dataset_name=None,
        dataset_split=None,
        other_prompt=[],
    )


def test_split_by_concurrency_splits_full_range() -> None:
    shards = split_by_concurrency(start=0, end=10, concurrency=3)
    assert [(shard.index, shard.start, shard.end) for shard in shards] == [
        (0, 0, 4),
        (1, 4, 7),
        (2, 7, 10),
    ]


def test_split_by_concurrency_returns_empty_for_invalid_range() -> None:
    assert split_by_concurrency(start=5, end=5, concurrency=2) == []


def test_split_by_concurrency_caps_shards_to_task_count() -> None:
    shards = split_by_concurrency(start=2, end=5, concurrency=10)
    assert [(shard.start, shard.end) for shard in shards] == [(2, 3), (3, 4), (4, 5)]


def test_build_child_command_includes_run_name_suffix() -> None:
    args = _base_args()
    command = build_child_command(args=args, shard=Shard(index=2, start=7, end=9), base_run_name="batch")
    assert "--run_name" in command
    assert "batch_shard_002" in command
    assert "--start" in command
    assert "--end" in command
    assert "7" in command
    assert "9" in command


def test_build_child_command_includes_optional_forwarded_args() -> None:
    expected_other_prompt_count = 2

    args = _base_args()
    args.verifier_model = "judge/model"
    args.max_runs = 4
    args.temperature = 0.2
    args.other_prompt = ["Hint A", "Hint B"]

    command = build_child_command(args=args, shard=Shard(index=0, start=0, end=1), base_run_name="r")

    assert "--verifier_model" in command
    assert "judge/model" in command
    assert "--max_runs" in command
    assert "4" in command
    assert "--temperature" in command
    assert "0.2" in command
    assert command.count("--other_prompt") == expected_other_prompt_count
