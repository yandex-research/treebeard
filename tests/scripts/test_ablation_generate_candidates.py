"""Tests for the ablation_generate_candidates script."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from open_deep_think.scripts.ablation_generate_candidates import (
    ThreadSafeJSONLWriter,
    WorkItem,
    build_config,
    build_work_items,
    generate_candidates_for_task,
    generate_single_candidate,
    load_completed_pairs,
    load_completed_task_ids,
    resolve_task_ids,
    run_generation,
    validate_args,
)
from open_deep_think.scripts.tournament_merge_improve import (
    Candidate,
    TournamentMergeImproveConfig,
    VerificationResult,
)

if TYPE_CHECKING:
    from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_args(**overrides: object) -> MagicMock:
    """Build a mock argparse.Namespace with sensible defaults."""
    defaults = {
        "task_ids_file": None,
        "start": 0,
        "end": 10,
        "model": "test-solver",
        "output_path": "test_output",
        "run_name": "default",
        "n_solutions": 4,
        "concurrency": 1,
        "verifier_model": None,
        "classifier_model": None,
        "solver_max_tokens": 100,
        "verifier_max_tokens": 100,
        "classifier_max_tokens": 100,
        "temperature": 0.7,
        "top_p": 1.0,
        "si_rounds": 1,
        "other_prompt": [],
        "dataset_name": "test-dataset",
        "dataset_split": "train",
    }
    defaults.update(overrides)
    ns = MagicMock()
    for k, v in defaults.items():
        setattr(ns, k, v)
    return ns


def _make_verification(*, is_pass: bool) -> VerificationResult:
    """Return a VerificationResult stub."""
    return VerificationResult(
        is_pass=is_pass,
        bug_report="" if is_pass else "bug found",
        verifier_output="ok" if is_pass else "fail",
        classifier_output="yes" if is_pass else "no",
        verifier_call_id=10,
        classifier_call_id=11,
    )


def _make_config(n_solutions: int = 2, si_rounds: int = 1) -> TournamentMergeImproveConfig:
    """Return a minimal config for testing."""
    return TournamentMergeImproveConfig(
        solver_model="solver",
        verifier_model="verifier",
        classifier_model="classifier",
        merger_model="solver",
        solver_max_tokens=100,
        verifier_max_tokens=100,
        classifier_max_tokens=100,
        merger_max_tokens=100,
        num_solutions=n_solutions,
        temperature=None,
        top_p=None,
        other_prompts=(),
        si_rounds=si_rounds,
    )


# ── validate_args ─────────────────────────────────────────────────────────────


def test_validate_args_accepts_range_mode() -> None:
    """validate_args must accept valid --start/--end without error."""
    args = _make_args(start=0, end=5, task_ids_file=None)
    validate_args(args)  # should not raise


def test_validate_args_accepts_file_mode() -> None:
    """validate_args must accept --task_ids_file without --start/--end."""
    args = _make_args(task_ids_file="tasks.txt", start=None, end=None)
    validate_args(args)  # should not raise


def test_validate_args_rejects_both_file_and_range() -> None:
    """validate_args must reject --task_ids_file combined with --start/--end."""
    args = _make_args(task_ids_file="tasks.txt", start=0, end=5)
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_args(args)


def test_validate_args_rejects_missing_range() -> None:
    """validate_args must reject missing --start/--end when no file is given."""
    args = _make_args(task_ids_file=None, start=None, end=None)
    with pytest.raises(ValueError, match="must be provided"):
        validate_args(args)


def test_validate_args_rejects_negative_start() -> None:
    """validate_args must reject negative --start."""
    args = _make_args(start=-1, end=5)
    with pytest.raises(ValueError, match="non-negative"):
        validate_args(args)


def test_validate_args_rejects_end_not_greater_than_start() -> None:
    """validate_args must reject --end <= --start."""
    args = _make_args(start=5, end=5)
    with pytest.raises(ValueError, match="greater than"):
        validate_args(args)


def test_validate_args_rejects_zero_n_solutions() -> None:
    """validate_args must reject --n_solutions < 1."""
    args = _make_args(n_solutions=0)
    with pytest.raises(ValueError, match="n_solutions"):
        validate_args(args)


def test_validate_args_rejects_negative_si_rounds() -> None:
    """validate_args must reject --si_rounds < 0."""
    args = _make_args(si_rounds=-1)
    with pytest.raises(ValueError, match="si_rounds"):
        validate_args(args)


def test_validate_args_accepts_zero_si_rounds() -> None:
    """validate_args must accept --si_rounds=0 (disables self-improvement)."""
    args = _make_args(si_rounds=0)
    validate_args(args)  # should not raise


def test_validate_args_rejects_zero_concurrency() -> None:
    """validate_args must reject --concurrency < 1."""
    args = _make_args(concurrency=0)
    with pytest.raises(ValueError, match="concurrency"):
        validate_args(args)


def test_validate_args_accepts_concurrency_greater_than_one() -> None:
    """validate_args must accept --concurrency > 1."""
    args = _make_args(concurrency=10)
    validate_args(args)  # should not raise


# ── resolve_task_ids ──────────────────────────────────────────────────────────


def test_resolve_task_ids_from_range() -> None:
    """resolve_task_ids must return range(start, end) when no file is given."""
    args = _make_args(start=3, end=7, task_ids_file=None)
    assert resolve_task_ids(args) == [3, 4, 5, 6]


def test_resolve_task_ids_from_file(tmp_path: Path) -> None:
    """resolve_task_ids must read IDs from a file when --task_ids_file is given."""
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("10\n20\n30\n", encoding="utf-8")
    args = _make_args(task_ids_file=str(ids_file), start=None, end=None)
    assert resolve_task_ids(args) == [10, 20, 30]


# ── build_config ──────────────────────────────────────────────────────────────


def test_build_config_uses_solver_model_as_default_verifier() -> None:
    """build_config must fall back to the solver model for verifier and classifier."""
    args = _make_args(model="my-solver", verifier_model=None, classifier_model=None)
    cfg = build_config(args)
    assert cfg.solver_model == "my-solver"
    assert cfg.verifier_model == "my-solver"
    assert cfg.classifier_model == "my-solver"


def test_build_config_respects_explicit_verifier() -> None:
    """build_config must use the explicit verifier and classifier when provided."""
    args = _make_args(model="solver", verifier_model="verifier", classifier_model="classifier")
    cfg = build_config(args)
    assert cfg.verifier_model == "verifier"
    assert cfg.classifier_model == "classifier"


def test_build_config_num_solutions_matches_n_solutions() -> None:
    """build_config must map --n_solutions to config.num_solutions."""
    args = _make_args(n_solutions=16)
    cfg = build_config(args)
    assert cfg.num_solutions == 16


def test_build_config_si_rounds_zero() -> None:
    """build_config must pass si_rounds=0 through to the config."""
    args = _make_args(si_rounds=0)
    cfg = build_config(args)
    assert cfg.si_rounds == 0


# ── load_completed_task_ids ───────────────────────────────────────────────────


def test_load_completed_task_ids_empty_file(tmp_path: Path) -> None:
    """load_completed_task_ids must return an empty set for an empty JSONL."""
    jsonl = tmp_path / "candidates.jsonl"
    jsonl.write_text("", encoding="utf-8")
    assert load_completed_task_ids(jsonl, n_solutions=4) == set()


def test_load_completed_task_ids_missing_file(tmp_path: Path) -> None:
    """load_completed_task_ids must return an empty set when the file does not exist."""
    jsonl = tmp_path / "nonexistent.jsonl"
    assert load_completed_task_ids(jsonl, n_solutions=4) == set()


def test_load_completed_task_ids_partial_and_full(tmp_path: Path) -> None:
    """load_completed_task_ids must only return task IDs that are fully complete."""
    jsonl = tmp_path / "candidates.jsonl"
    # Task 42 has 4 records → complete (n_solutions=4).
    lines = [json.dumps({"task_id": 42, "candidate_index": i}) for i in range(4)]
    # Task 99 has only 2 records → incomplete.
    lines.extend(json.dumps({"task_id": 99, "candidate_index": i}) for i in range(2))
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = load_completed_task_ids(jsonl, n_solutions=4)
    assert result == {42}


def test_load_completed_task_ids_multiple_complete(tmp_path: Path) -> None:
    """load_completed_task_ids must return all fully-completed task IDs."""
    jsonl = tmp_path / "candidates.jsonl"
    lines = [json.dumps({"task_id": tid, "candidate_index": i}) for tid in [10, 20] for i in range(2)]
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = load_completed_task_ids(jsonl, n_solutions=2)
    assert result == {10, 20}


# ── load_completed_pairs ─────────────────────────────────────────────────────


def test_load_completed_pairs_missing_file(tmp_path: Path) -> None:
    """load_completed_pairs must return empty set when file does not exist."""
    jsonl = tmp_path / "nonexistent.jsonl"
    assert load_completed_pairs(jsonl) == set()


def test_load_completed_pairs_empty_file(tmp_path: Path) -> None:
    """load_completed_pairs must return empty set for an empty JSONL."""
    jsonl = tmp_path / "candidates.jsonl"
    jsonl.write_text("", encoding="utf-8")
    assert load_completed_pairs(jsonl) == set()


def test_load_completed_pairs_returns_all_pairs(tmp_path: Path) -> None:
    """load_completed_pairs must return every (task_id, candidate_index) present."""
    jsonl = tmp_path / "candidates.jsonl"
    lines = [
        json.dumps({"task_id": 1, "candidate_index": 0}),
        json.dumps({"task_id": 1, "candidate_index": 1}),
        json.dumps({"task_id": 2, "candidate_index": 0}),
    ]
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = load_completed_pairs(jsonl)
    assert result == {(1, 0), (1, 1), (2, 0)}


# ── ThreadSafeJSONLWriter ────────────────────────────────────────────────────


def test_thread_safe_writer_creates_file_on_first_append(tmp_path: Path) -> None:
    """ThreadSafeJSONLWriter must create the file if it doesn't exist."""
    jsonl = tmp_path / "out.jsonl"
    writer = ThreadSafeJSONLWriter(jsonl)
    writer.append({"key": "value"})
    assert jsonl.exists()
    lines = jsonl.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"key": "value"}


def test_thread_safe_writer_appends_multiple_records(tmp_path: Path) -> None:
    """ThreadSafeJSONLWriter must append without overwriting existing data."""
    jsonl = tmp_path / "out.jsonl"
    writer = ThreadSafeJSONLWriter(jsonl)
    writer.append({"idx": 0})
    writer.append({"idx": 1})
    writer.append({"idx": 2})
    lines = jsonl.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    assert [json.loads(line)["idx"] for line in lines] == [0, 1, 2]


def test_thread_safe_writer_path_property(tmp_path: Path) -> None:
    """ThreadSafeJSONLWriter.path must return the underlying file path."""
    jsonl = tmp_path / "out.jsonl"
    writer = ThreadSafeJSONLWriter(jsonl)
    assert writer.path == jsonl


# ── build_work_items ─────────────────────────────────────────────────────────


def test_build_work_items_generates_all_pairs() -> None:
    """build_work_items must produce (task, candidate) pairs for all tasks and solutions."""
    problems = {1: "P1", 2: "P2"}
    items = build_work_items([1, 2], n_solutions=3, problem_statements=problems, completed_pairs=set())
    assert len(items) == 6
    pairs = [(item.task_id, item.candidate_index) for item in items]
    expected = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    assert pairs == expected


def test_build_work_items_skips_completed_pairs() -> None:
    """build_work_items must exclude pairs present in completed_pairs."""
    problems = {1: "P1", 2: "P2"}
    completed = {(1, 0), (1, 2), (2, 1)}
    items = build_work_items([1, 2], n_solutions=3, problem_statements=problems, completed_pairs=completed)
    pairs = [(item.task_id, item.candidate_index) for item in items]
    assert pairs == [(1, 1), (2, 0), (2, 2)]


def test_build_work_items_returns_empty_when_all_done() -> None:
    """build_work_items must return empty list when everything is completed."""
    problems = {5: "P5"}
    completed = {(5, 0), (5, 1)}
    items = build_work_items([5], n_solutions=2, problem_statements=problems, completed_pairs=completed)
    assert items == []


def test_build_work_items_carries_problem_statement() -> None:
    """Each WorkItem must carry the correct problem statement for its task."""
    problems = {10: "Problem ten", 20: "Problem twenty"}
    items = build_work_items([10, 20], n_solutions=1, problem_statements=problems, completed_pairs=set())
    assert items[0].problem_statement == "Problem ten"
    assert items[1].problem_statement == "Problem twenty"


# ── generate_single_candidate ────────────────────────────────────────────────


def test_generate_single_candidate_writes_to_jsonl(tmp_path: Path) -> None:
    """generate_single_candidate must append one record to the JSONL writer."""
    config = _make_config(n_solutions=1)
    jsonl_path = tmp_path / "candidates.jsonl"
    writer = ThreadSafeJSONLWriter(jsonl_path)
    item = WorkItem(task_id=42, candidate_index=0, problem_statement="Prove x=0.")

    fake_candidate = Candidate(
        index=0,
        solution_text="My solution",
        completion=None,
        verification=_make_verification(is_pass=True),
    )
    with patch(
        "open_deep_think.scripts.ablation_generate_candidates.generate_candidate",
        return_value=fake_candidate,
    ):
        record = generate_single_candidate(
            work_item=item,
            config=config,
            run_dir=tmp_path,
            jsonl_writer=writer,
        )

    assert record["task_id"] == 42
    assert record["candidate_index"] == 0
    assert record["status"] == "ok"
    assert record["verification"]["is_pass"] is True

    # Verify the JSONL was written.
    lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["task_id"] == 42


def test_generate_single_candidate_handles_errors(tmp_path: Path) -> None:
    """generate_single_candidate must record an error entry without crashing."""
    config = _make_config(n_solutions=1)
    jsonl_path = tmp_path / "candidates.jsonl"
    writer = ThreadSafeJSONLWriter(jsonl_path)
    item = WorkItem(task_id=7, candidate_index=0, problem_statement="Find n.")

    def _mock_fail(**_kwargs: object) -> Candidate:
        msg = "API timeout"
        raise RuntimeError(msg)

    with patch(
        "open_deep_think.scripts.ablation_generate_candidates.generate_candidate",
        side_effect=_mock_fail,
    ):
        record = generate_single_candidate(
            work_item=item,
            config=config,
            run_dir=tmp_path,
            jsonl_writer=writer,
        )

    assert record["task_id"] == 7
    assert record["status"] == "error"
    assert "API timeout" in record["error"]
    assert record["solution_text"] == ""


def test_generate_single_candidate_creates_per_candidate_log(tmp_path: Path) -> None:
    """generate_single_candidate must create a per-candidate LLM log file."""
    config = _make_config(n_solutions=1)
    jsonl_path = tmp_path / "candidates.jsonl"
    writer = ThreadSafeJSONLWriter(jsonl_path)
    item = WorkItem(task_id=3, candidate_index=2, problem_statement="P")

    fake_candidate = Candidate(
        index=2,
        solution_text="Sol",
        completion=None,
        verification=_make_verification(is_pass=False),
    )
    with patch(
        "open_deep_think.scripts.ablation_generate_candidates.generate_candidate",
        return_value=fake_candidate,
    ):
        generate_single_candidate(
            work_item=item,
            config=config,
            run_dir=tmp_path,
            jsonl_writer=writer,
        )

    log_path = tmp_path / "Task_3_cand_2_llm_outputs.jsonl"
    assert log_path.exists()


# ── generate_candidates_for_task (sequential compatibility) ───────────────────


def test_generate_candidates_for_task_writes_all_records(tmp_path: Path) -> None:
    """generate_candidates_for_task must write n_solutions records to the JSONL."""
    n_solutions = 3
    config = _make_config(n_solutions=n_solutions)
    jsonl_path = tmp_path / "candidates.jsonl"

    def _mock_generate(*, candidate_index: int, **_kwargs: object) -> Candidate:
        return Candidate(
            index=candidate_index,
            solution_text=f"Solution {candidate_index}",
            completion=None,
            verification=_make_verification(is_pass=(candidate_index % 2 == 0)),
        )

    with patch(
        "open_deep_think.scripts.ablation_generate_candidates.generate_candidate",
        side_effect=_mock_generate,
    ):
        records = generate_candidates_for_task(
            task_id=42,
            problem_statement="Prove x=0.",
            config=config,
            run_dir=tmp_path,
            candidates_jsonl_path=jsonl_path,
        )

    assert len(records) == n_solutions

    # Read back the JSONL and verify structure.
    with jsonl_path.open(encoding="utf-8") as fh:
        written = [json.loads(line) for line in fh]
    assert len(written) == n_solutions

    for i, rec in enumerate(written):
        assert rec["task_id"] == 42
        assert rec["candidate_index"] == i
        assert rec["status"] == "ok"
        assert "verification" in rec
        assert "solution_text" in rec


def test_generate_candidates_for_task_handles_errors(tmp_path: Path) -> None:
    """generate_candidates_for_task must record error entries without crashing."""
    config = _make_config(n_solutions=2)
    jsonl_path = tmp_path / "candidates.jsonl"

    def _mock_generate_fail(**_kwargs: object) -> Candidate:
        msg = "API timeout"
        raise RuntimeError(msg)

    with patch(
        "open_deep_think.scripts.ablation_generate_candidates.generate_candidate",
        side_effect=_mock_generate_fail,
    ):
        records = generate_candidates_for_task(
            task_id=7,
            problem_statement="Find n.",
            config=config,
            run_dir=tmp_path,
            candidates_jsonl_path=jsonl_path,
        )

    assert len(records) == 2
    for rec in records:
        assert rec["task_id"] == 7
        assert rec["status"] == "error"
        assert "API timeout" in rec["error"]
        assert rec["solution_text"] == ""


def test_generate_candidates_for_task_records_contain_verification_fields(tmp_path: Path) -> None:
    """Each candidate record must contain all expected verification sub-fields."""
    config = _make_config(n_solutions=1, si_rounds=0)
    jsonl_path = tmp_path / "candidates.jsonl"

    fake_candidate = Candidate(
        index=0,
        solution_text="Sol",
        completion=None,
        verification=_make_verification(is_pass=False),
    )
    with patch(
        "open_deep_think.scripts.ablation_generate_candidates.generate_candidate",
        return_value=fake_candidate,
    ):
        records = generate_candidates_for_task(
            task_id=1,
            problem_statement="P",
            config=config,
            run_dir=tmp_path,
            candidates_jsonl_path=jsonl_path,
        )

    assert len(records) == 1
    v = records[0]["verification"]
    assert "is_pass" in v
    assert "bug_report" in v
    assert "verifier_output" in v
    assert "classifier_output" in v
    assert "verifier_call_id" in v
    assert "classifier_call_id" in v


def test_generate_candidates_appends_to_existing_jsonl(tmp_path: Path) -> None:
    """generate_candidates_for_task must append (not overwrite) existing content."""
    config = _make_config(n_solutions=1, si_rounds=0)
    jsonl_path = tmp_path / "candidates.jsonl"

    # Pre-populate with a record from a different task.
    existing = {"task_id": 0, "candidate_index": 0, "solution_text": "old"}
    jsonl_path.write_text(json.dumps(existing) + "\n", encoding="utf-8")

    fake_candidate = Candidate(
        index=0,
        solution_text="New",
        completion=None,
        verification=_make_verification(is_pass=True),
    )
    with patch(
        "open_deep_think.scripts.ablation_generate_candidates.generate_candidate",
        return_value=fake_candidate,
    ):
        generate_candidates_for_task(
            task_id=5,
            problem_statement="P",
            config=config,
            run_dir=tmp_path,
            candidates_jsonl_path=jsonl_path,
        )

    lines = [json.loads(entry) for entry in jsonl_path.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == 2
    assert lines[0]["task_id"] == 0
    assert lines[1]["task_id"] == 5


# ── run_generation ────────────────────────────────────────────────────────────


def test_run_generation_sequential_produces_all_records(tmp_path: Path) -> None:
    """run_generation with concurrency=1 must process all work items sequentially."""
    config = _make_config(n_solutions=2)
    jsonl_path = tmp_path / "candidates.jsonl"
    writer = ThreadSafeJSONLWriter(jsonl_path)

    items = [
        WorkItem(task_id=1, candidate_index=0, problem_statement="P1"),
        WorkItem(task_id=1, candidate_index=1, problem_statement="P1"),
        WorkItem(task_id=2, candidate_index=0, problem_statement="P2"),
    ]

    def _mock_generate(*, candidate_index: int, task_id: int, **_kwargs: object) -> Candidate:
        return Candidate(
            index=candidate_index,
            solution_text=f"Sol t={task_id} c={candidate_index}",
            completion=None,
            verification=_make_verification(is_pass=True),
        )

    with patch(
        "open_deep_think.scripts.ablation_generate_candidates.generate_candidate",
        side_effect=_mock_generate,
    ):
        results = run_generation(
            work_items=items,
            config=config,
            run_dir=tmp_path,
            jsonl_writer=writer,
            concurrency=1,
        )

    assert len(results) == 3
    assert all(r["status"] == "ok" for r in results)

    # JSONL should have 3 lines.
    lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3


def test_run_generation_concurrent_produces_all_records(tmp_path: Path) -> None:
    """run_generation with concurrency>1 must process all work items."""
    config = _make_config(n_solutions=2)
    jsonl_path = tmp_path / "candidates.jsonl"
    writer = ThreadSafeJSONLWriter(jsonl_path)

    items = [
        WorkItem(task_id=1, candidate_index=0, problem_statement="P1"),
        WorkItem(task_id=1, candidate_index=1, problem_statement="P1"),
        WorkItem(task_id=2, candidate_index=0, problem_statement="P2"),
        WorkItem(task_id=2, candidate_index=1, problem_statement="P2"),
    ]

    def _mock_generate(*, candidate_index: int, task_id: int, **_kwargs: object) -> Candidate:
        return Candidate(
            index=candidate_index,
            solution_text=f"Sol t={task_id} c={candidate_index}",
            completion=None,
            verification=_make_verification(is_pass=True),
        )

    with patch(
        "open_deep_think.scripts.ablation_generate_candidates.generate_candidate",
        side_effect=_mock_generate,
    ):
        results = run_generation(
            work_items=items,
            config=config,
            run_dir=tmp_path,
            jsonl_writer=writer,
            concurrency=4,
        )

    assert len(results) == 4
    assert all(r["status"] == "ok" for r in results)

    # All task/candidate pairs should be present in JSONL.
    lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 4
    written_pairs = {(json.loads(line)["task_id"], json.loads(line)["candidate_index"]) for line in lines}
    assert written_pairs == {(1, 0), (1, 1), (2, 0), (2, 1)}


def test_run_generation_handles_worker_errors(tmp_path: Path) -> None:
    """run_generation must handle exceptions from workers without crashing."""
    config = _make_config(n_solutions=1)
    jsonl_path = tmp_path / "candidates.jsonl"
    writer = ThreadSafeJSONLWriter(jsonl_path)

    items = [
        WorkItem(task_id=1, candidate_index=0, problem_statement="P1"),
    ]

    def _mock_fail(**_kwargs: object) -> Candidate:
        msg = "Boom"
        raise RuntimeError(msg)

    with patch(
        "open_deep_think.scripts.ablation_generate_candidates.generate_candidate",
        side_effect=_mock_fail,
    ):
        results = run_generation(
            work_items=items,
            config=config,
            run_dir=tmp_path,
            jsonl_writer=writer,
            concurrency=1,
        )

    # Error should be recorded in the result, not propagated.
    assert len(results) == 1
    assert results[0]["status"] == "error"
