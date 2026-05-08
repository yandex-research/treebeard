"""Tests for the parallel solver launcher helpers."""

from __future__ import annotations

import argparse

import pytest

from open_deep_think.scripts.parallel_solve import (
    TaskSlot,
    build_child_command,
    build_task_slots,
    sanitize_model_name,
    validate_args,
)


def _base_args() -> argparse.Namespace:
    """Build a minimal namespace for command construction tests."""
    return argparse.Namespace(
        script="imo25",
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
        baseline_max_tokens=None,
        # simple_tournament-specific
        judge_model=None,
        judge_max_tokens=None,
        num_solutions=None,
        # tournament_merge-specific
        merger_model=None,
        merger_max_tokens=None,
    )


# ── build_task_slots ──────────────────────────────────────────────────────────


def test_build_task_slots_produces_one_slot_per_task() -> None:
    """Each task index in [start, end) gets exactly one slot."""
    slots = build_task_slots(start=3, end=7)
    assert [s.task_index for s in slots] == [3, 4, 5, 6]


def test_build_task_slots_first_task_gets_shard_index_zero() -> None:
    """The first task always receives shard_index=0 so it writes config.json."""
    start = 5
    slots = build_task_slots(start=start, end=10)
    assert slots[0].shard_index == 0
    assert slots[0].task_index == start


def test_build_task_slots_subsequent_tasks_get_nonzero_shard_index() -> None:
    """Tasks after the first must have shard_index > 0 to skip config.json write."""
    slots = build_task_slots(start=0, end=5)
    for slot in slots[1:]:
        assert slot.shard_index > 0


def test_build_task_slots_shard_indices_are_unique() -> None:
    """Every slot in a range must have a distinct shard_index."""
    slots = build_task_slots(start=10, end=15)
    shard_indices = [s.shard_index for s in slots]
    assert len(shard_indices) == len(set(shard_indices))


def test_build_task_slots_empty_range() -> None:
    """An empty range produces no slots."""
    assert build_task_slots(start=5, end=5) == []


def test_build_task_slots_single_task() -> None:
    """A single-task range produces one slot with shard_index=0."""
    start = 7
    slots = build_task_slots(start=start, end=start + 1)
    assert len(slots) == 1
    assert slots[0].task_index == start
    assert slots[0].shard_index == 0


# ── build_child_command ───────────────────────────────────────────────────────


def test_build_child_command_imo25_uses_single_task_range() -> None:
    """Each child command covers exactly one task (end = start + 1)."""
    args = _base_args()
    slot = TaskSlot(task_index=7, shard_index=2)
    command = build_child_command(args=args, slot=slot, base_run_name="batch")

    start_idx = command.index("--start") + 1
    end_idx = command.index("--end") + 1
    assert command[start_idx] == "7"
    assert command[end_idx] == "8"


def test_build_child_command_imo25_uses_shared_run_name_and_shard_index() -> None:
    """imo25 child commands use shared run_name and forward shard_index."""
    args = _base_args()
    slot = TaskSlot(task_index=7, shard_index=2)
    command = build_child_command(args=args, slot=slot, base_run_name="batch")

    assert "--run_name" in command
    assert "batch" in command
    # shard_index must be forwarded
    assert "--shard_index" in command
    assert "2" in command


def test_build_child_command_includes_optional_forwarded_args() -> None:
    """Optional args are forwarded when set."""
    expected_other_prompt_count = 2

    args = _base_args()
    args.verifier_model = "judge/model"
    args.max_runs = 4
    args.temperature = 0.2
    args.other_prompt = ["Hint A", "Hint B"]

    slot = TaskSlot(task_index=0, shard_index=0)
    command = build_child_command(args=args, slot=slot, base_run_name="r")

    assert "--verifier_model" in command
    assert "judge/model" in command
    assert "--max_runs" in command
    assert "4" in command
    assert "--temperature" in command
    assert "0.2" in command
    assert command.count("--other_prompt") == expected_other_prompt_count


def test_build_child_command_baseline_uses_baseline_module_and_max_tokens() -> None:
    """Baseline child commands use the baseline module and forward max_tokens."""
    args = _base_args()
    args.script = "baseline"
    args.baseline_max_tokens = 4096
    args.temperature = 1.0
    args.top_p = 0.95

    slot = TaskSlot(task_index=2, shard_index=2)
    command = build_child_command(args=args, slot=slot, base_run_name="r")

    assert "open_deep_think.scripts.baseline_solve" in command
    assert "--max_tokens" in command
    assert "4096" in command
    assert "--temperature" in command
    assert "1.0" in command
    assert "--top_p" in command
    assert "0.95" in command
    # baseline does not use run_name / shard_index
    assert "--run_name" not in command
    assert "--shard_index" not in command


def test_build_child_command_tournament_uses_tournament_module_and_args() -> None:
    """simple_tournament child commands use shared run_name and forward shard_index."""
    args = _base_args()
    args.script = "simple_tournament"
    args.judge_model = "judge/model"
    args.judge_max_tokens = 16
    args.num_solutions = 4
    args.verifier_model = "verifier/model"
    args.temperature = 0.7

    slot = TaskSlot(task_index=0, shard_index=0)
    command = build_child_command(args=args, slot=slot, base_run_name="run")

    assert "open_deep_think.scripts.simple_tournament" in command
    assert "--run_name" in command
    assert "run" in command
    assert "--shard_index" in command
    assert "0" in command
    assert "--judge_model" in command
    assert "judge/model" in command
    assert "--judge_max_tokens" in command
    assert "16" in command
    assert "--num_solutions" in command
    assert "4" in command
    assert "--verifier_model" in command
    assert "verifier/model" in command
    assert "--temperature" in command
    assert "0.7" in command
    # imo25-only args must not appear
    assert "--max_runs" not in command
    assert "--max_iterations" not in command


def test_build_child_command_tournament_merge_uses_tournament_merge_module_and_args() -> None:
    """tournament_merge child commands use shared run_name and forward shard_index."""
    args = _base_args()
    args.script = "tournament_merge"
    args.merger_model = "merger/model"
    args.merger_max_tokens = 32000
    args.num_solutions = 4
    args.verifier_model = "verifier/model"
    args.temperature = 0.7

    slot = TaskSlot(task_index=5, shard_index=5)
    command = build_child_command(args=args, slot=slot, base_run_name="run")

    assert "open_deep_think.scripts.tournament_merge" in command
    assert "--run_name" in command
    assert "run" in command
    assert "--shard_index" in command
    assert "5" in command
    assert "--merger_model" in command
    assert "merger/model" in command
    assert "--merger_max_tokens" in command
    assert "32000" in command
    assert "--num_solutions" in command
    assert "4" in command
    assert "--verifier_model" in command
    assert "verifier/model" in command
    assert "--temperature" in command
    assert "0.7" in command
    # simple_tournament-only and imo25-only args must not appear
    assert "--judge_model" not in command
    assert "--judge_max_tokens" not in command
    assert "--max_runs" not in command
    assert "--max_iterations" not in command


def test_build_child_command_tournament_merge_improve_uses_correct_module_and_args() -> None:
    """tournament_merge_improve child commands use shared run_name and forward shard_index."""
    args = _base_args()
    args.script = "tournament_merge_improve"
    args.merger_model = "merger/model"
    args.merger_max_tokens = 32000
    args.num_solutions = 4
    args.verifier_model = "verifier/model"
    args.temperature = 0.7

    slot = TaskSlot(task_index=3, shard_index=3)
    command = build_child_command(args=args, slot=slot, base_run_name="run")

    assert "open_deep_think.scripts.tournament_merge_improve" in command
    assert "--run_name" in command
    assert "run" in command
    assert "--shard_index" in command
    assert "3" in command
    assert "--merger_model" in command
    assert "merger/model" in command
    assert "--merger_max_tokens" in command
    assert "32000" in command
    assert "--num_solutions" in command
    assert "4" in command
    assert "--verifier_model" in command
    assert "verifier/model" in command
    assert "--temperature" in command
    assert "0.7" in command
    # simple_tournament-only and imo25-only args must not appear
    assert "--judge_model" not in command
    assert "--judge_max_tokens" not in command
    assert "--max_runs" not in command
    assert "--max_iterations" not in command


# ── sanitize_model_name ───────────────────────────────────────────────────────


def test_sanitize_model_name_replaces_slash_and_colon() -> None:
    """sanitize_model_name must produce a filesystem-safe slug."""
    assert sanitize_model_name("provider/model:latest") == "provider__model_latest"


def test_sanitize_model_name_no_special_chars() -> None:
    assert sanitize_model_name("mymodel") == "mymodel"


# ── validate_args ─────────────────────────────────────────────────────────────


def test_validate_args_rejects_negative_start() -> None:
    """validate_args must raise ValueError for negative --start."""
    args = argparse.Namespace(start=-1, end=5, concurrency=2)
    with pytest.raises(ValueError, match="--start must be non-negative"):
        validate_args(args)


def test_validate_args_rejects_end_not_greater_than_start() -> None:
    """validate_args must raise ValueError when --end <= --start."""
    args = argparse.Namespace(start=5, end=5, concurrency=2)
    with pytest.raises(ValueError, match="--end must be greater than --start"):
        validate_args(args)


def test_validate_args_rejects_zero_concurrency() -> None:
    """validate_args must raise ValueError for --concurrency <= 0."""
    args = argparse.Namespace(start=0, end=5, concurrency=0)
    with pytest.raises(ValueError, match="--concurrency must be positive"):
        validate_args(args)
