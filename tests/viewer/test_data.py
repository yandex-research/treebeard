"""Tests for the viewer data-access layer."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from open_deep_think.viewer.data import (
    calls_for_node,
    discover_runs,
    enrich_node_from_calls,
    load_calls,
    load_task,
    parse_progress,
)

if TYPE_CHECKING:
    from pathlib import Path

_TASK_ID = 42
_FINAL_INDEX = -1001
_ROUND_ZERO_MERGED_A = -1
_ROUND_ZERO_MERGED_B = -2
_EXPECTED_INITIAL_PASS_COUNT = 3


def test_parse_progress_extracts_candidates_and_matches(run_dir: Path) -> None:
    task = parse_progress(run_dir / "Task_42_progress.json", run_name="sample_run")

    assert task.task_id == _TASK_ID
    initial_indices = sorted(c.index for c in task.candidates if c.kind == "initial")
    assert initial_indices == [0, 1, 2, 3]
    merged_indices = sorted(c.index for c in task.candidates if c.kind == "merged")
    assert merged_indices == [_FINAL_INDEX, _ROUND_ZERO_MERGED_B, _ROUND_ZERO_MERGED_A]
    assert task.summary.final_candidate_index == _FINAL_INDEX
    assert task.summary.final_verification_pass is True
    assert task.summary.num_initial_passed == _EXPECTED_INITIAL_PASS_COUNT
    assert task.summary.solver_model == "openai/gpt-oss-120b"
    assert task.solution_text == "Final answer is 49."

    # Round structure preserved.
    assert [r[0].round_index for r in task.rounds if r] == [0, 1]
    round_zero_matches = task.rounds[0]
    assert sorted(m.merged_index for m in round_zero_matches) == [_ROUND_ZERO_MERGED_B, _ROUND_ZERO_MERGED_A]
    final_match = task.rounds[1][0]
    assert (final_match.candidate_a, final_match.candidate_b) == (_ROUND_ZERO_MERGED_A, _ROUND_ZERO_MERGED_B)


def test_parse_progress_records_parents_for_merged_candidates(run_dir: Path) -> None:
    task = parse_progress(run_dir / "Task_42_progress.json", run_name="sample_run")
    by_index = {c.index: c for c in task.candidates}
    assert by_index[-1].parents == (0, 1)
    assert by_index[-2].parents == (2, 3)
    assert by_index[-1001].parents == (-1, -2)
    assert by_index[0].parents == ()


def test_discover_runs_recognises_single_run(run_dir: Path) -> None:
    runs = discover_runs(run_dir)
    assert len(runs) == 1
    run = runs[0]
    assert run.name == run_dir.name
    assert run.task_ids == (_TASK_ID,)
    assert _TASK_ID in run.task_summaries


def test_discover_runs_recognises_multi_run(tmp_path: Path, run_dir: Path) -> None:
    parent = tmp_path / "logs"
    parent.mkdir()
    moved = parent / "run_a"
    run_dir.rename(moved)
    (parent / "empty").mkdir()  # should be ignored

    runs = discover_runs(parent)
    names = [run.name for run in runs]
    assert names == ["run_a"]


def test_discover_runs_raises_when_root_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_runs(tmp_path / "does_not_exist")


def test_load_task_round_trip(run_dir: Path) -> None:
    runs = discover_runs(run_dir)
    task = load_task(runs[0], _TASK_ID)
    assert task.task_id == _TASK_ID


def test_load_calls_returns_chronological_order(run_dir: Path) -> None:
    load_calls.cache_clear()
    calls = load_calls(str(run_dir), _TASK_ID)
    assert [c.call_id for c in calls] == sorted(c.call_id for c in calls)
    assert calls[0].phase == "initial_solution"
    assert calls[0].reasoning == "reasoning-1"
    assert calls[0].usage == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }


def test_load_calls_handles_missing_file(tmp_path: Path) -> None:
    load_calls.cache_clear()
    assert load_calls(str(tmp_path), 999) == ()


def test_load_calls_skips_malformed_lines(tmp_path: Path) -> None:
    run = tmp_path / "run_malformed"
    run.mkdir()
    progress = {
        "task_id": 1,
        "candidates": [],
        "tournament_rounds": [],
        "pipeline_config": {},
        "problem_statement": "",
        "started_at": "",
    }
    (run / "Task_1_progress.json").write_text(json.dumps(progress), encoding="utf-8")
    jsonl_path = run / "Task_1_llm_outputs.jsonl"
    jsonl_path.write_text(
        '\n{"call_id":1, "phase":"x", "messages":[], "completion":null}\nNOT JSON\n',
        encoding="utf-8",
    )
    load_calls.cache_clear()
    calls = load_calls(str(run), 1)
    assert len(calls) == 1
    assert calls[0].call_id == 1


def test_calls_for_node_initial_filters_to_candidate_phases(run_dir: Path) -> None:
    task = parse_progress(run_dir / "Task_42_progress.json", run_name="sample_run")
    load_calls.cache_clear()
    calls = load_calls(str(run_dir), _TASK_ID)
    initial_candidate = next(c for c in task.candidates if c.kind == "initial" and c.index == 0)
    related = calls_for_node(calls, initial_candidate)
    assert [c.phase for c in related] == ["initial_solution", "verification", "verification_check"]
    assert all(c.candidate_index == 0 for c in related)


def test_calls_for_node_merged_associates_correct_merge_call(run_dir: Path) -> None:
    task = parse_progress(run_dir / "Task_42_progress.json", run_name="sample_run")
    load_calls.cache_clear()
    calls = load_calls(str(run_dir), _TASK_ID)

    merged_minus_one = next(c for c in task.candidates if c.index == _ROUND_ZERO_MERGED_A)
    related = calls_for_node(calls, merged_minus_one)
    phases = [c.phase for c in related]
    assert phases == ["tournament_merge", "verification", "verification_check"]
    merge_call = related[0]
    assert merge_call.round_index == 0
    # The merge call for -1 is the *first* tournament_merge in round 0.
    all_round_zero_merges = [c for c in calls if c.phase == "tournament_merge" and c.round_index == 0]
    assert merge_call.call_id == all_round_zero_merges[0].call_id

    merged_minus_two = next(c for c in task.candidates if c.index == _ROUND_ZERO_MERGED_B)
    related_two = calls_for_node(calls, merged_minus_two)
    assert related_two[0].call_id == all_round_zero_merges[1].call_id


def test_enrich_node_from_calls_populates_merged_verifier_text(run_dir: Path) -> None:
    task = parse_progress(run_dir / "Task_42_progress.json", run_name="sample_run")
    load_calls.cache_clear()
    calls = load_calls(str(run_dir), _TASK_ID)
    merged = next(c for c in task.candidates if c.index == _ROUND_ZERO_MERGED_A)
    assert merged.verifier_output == ""

    enriched = enrich_node_from_calls(merged, calls)
    assert enriched.verifier_output.startswith("response-")
    assert enriched.classifier_output.startswith("response-")
    assert enriched.verifier_call_id is not None
    assert enriched.classifier_call_id is not None


def test_enrich_node_from_calls_passes_through_initials(run_dir: Path) -> None:
    task = parse_progress(run_dir / "Task_42_progress.json", run_name="sample_run")
    load_calls.cache_clear()
    calls = load_calls(str(run_dir), _TASK_ID)
    initial = next(c for c in task.candidates if c.kind == "initial" and c.index == 0)
    assert enrich_node_from_calls(initial, calls) is initial


def test_calls_for_node_final_uses_round_one_merge(run_dir: Path) -> None:
    task = parse_progress(run_dir / "Task_42_progress.json", run_name="sample_run")
    load_calls.cache_clear()
    calls = load_calls(str(run_dir), _TASK_ID)

    final_node = next(c for c in task.candidates if c.index == _FINAL_INDEX)
    related = calls_for_node(calls, final_node)
    assert related[0].phase == "tournament_merge"
    assert related[0].round_index == 1
