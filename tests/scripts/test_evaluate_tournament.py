"""Tests for the evaluate_tournament script."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from open_deep_think.imo_answer_bench.templates import JudgeType
from open_deep_think.scripts.evaluate_tournament import (
    EXCLUDED_PHASES,
    _filter_solution_records,
    _load_task_llm_outputs,
    evaluate_tournament_solutions,
)

if TYPE_CHECKING:
    from pathlib import Path

# Named constants to satisfy PLR2004
_EXPECT_2 = 2
_EXPECT_3 = 3
_EXPECT_4 = 4

# ── _filter_solution_records ──────────────────────────────────────────────────


def test_filter_solution_records_keeps_solution_phases() -> None:
    """Solution-producing phases (initial_solution, tournament_merge, etc.) are kept."""
    records = [
        {"phase": "initial_solution", "call_id": 1},
        {"phase": "self_improvement", "call_id": 2},
        {"phase": "tournament_merge", "call_id": 3},
        {"phase": "tournament_match", "call_id": 4},
    ]
    filtered = _filter_solution_records(records)
    assert len(filtered) == _EXPECT_4
    assert all(r["phase"] not in EXCLUDED_PHASES for r in filtered)


def test_filter_solution_records_removes_verification_phases() -> None:
    """Verification and verification_check phases are excluded."""
    records = [
        {"phase": "verification", "call_id": 10},
        {"phase": "verification_check", "call_id": 11},
        {"phase": "initial_solution", "call_id": 1},
    ]
    filtered = _filter_solution_records(records)
    assert len(filtered) == 1
    assert filtered[0]["call_id"] == 1


def test_filter_solution_records_empty_input() -> None:
    """An empty list returns an empty list."""
    assert _filter_solution_records([]) == []


def test_filter_solution_records_all_excluded() -> None:
    """When all records are verification phases, result is empty."""
    records = [
        {"phase": "verification", "call_id": 1},
        {"phase": "verification_check", "call_id": 2},
    ]
    assert _filter_solution_records(records) == []


def test_filter_solution_records_missing_phase_key_kept() -> None:
    """Records without a 'phase' key are kept (phase=None is not in EXCLUDED_PHASES)."""
    records = [{"call_id": 99}]
    assert len(_filter_solution_records(records)) == 1


# ── _load_task_llm_outputs ────────────────────────────────────────────────────


def test_load_task_llm_outputs_reads_single_file(tmp_path: Path) -> None:
    """Reads a single Task_*_llm_outputs.jsonl file correctly."""
    record_1 = {"call_id": 1, "phase": "initial_solution", "response_text": "ans"}
    record_2 = {"call_id": 2, "phase": "verification", "response_text": "ok"}
    jsonl_file = tmp_path / "Task_0_llm_outputs.jsonl"
    jsonl_file.write_text(
        json.dumps(record_1) + "\n" + json.dumps(record_2) + "\n",
        encoding="utf-8",
    )

    result = _load_task_llm_outputs(tmp_path)
    assert "0" in result
    assert len(result["0"]) == _EXPECT_2
    assert result["0"][0]["call_id"] == 1
    assert result["0"][1]["call_id"] == _EXPECT_2


def test_load_task_llm_outputs_merges_shards(tmp_path: Path) -> None:
    """Records from multiple shard directories for the same task_id are merged."""
    shard_a = tmp_path / "shard_000"
    shard_b = tmp_path / "shard_001"
    shard_a.mkdir()
    shard_b.mkdir()

    (shard_a / "Task_5_llm_outputs.jsonl").write_text(
        json.dumps({"call_id": 1, "phase": "initial_solution"}) + "\n",
        encoding="utf-8",
    )
    (shard_b / "Task_5_llm_outputs.jsonl").write_text(
        json.dumps({"call_id": 2, "phase": "tournament_merge"}) + "\n",
        encoding="utf-8",
    )

    result = _load_task_llm_outputs(tmp_path)
    assert "5" in result
    assert len(result["5"]) == _EXPECT_2


def test_load_task_llm_outputs_skips_malformed_json(tmp_path: Path) -> None:
    """Malformed JSON lines are skipped with a warning, valid lines are kept."""
    jsonl_file = tmp_path / "Task_3_llm_outputs.jsonl"
    jsonl_file.write_text(
        '{"call_id": 1, "phase": "initial_solution"}\nNOT VALID JSON\n{"call_id": 2, "phase": "self_improvement"}\n',
        encoding="utf-8",
    )

    result = _load_task_llm_outputs(tmp_path)
    assert len(result["3"]) == _EXPECT_2


def test_load_task_llm_outputs_empty_directory(tmp_path: Path) -> None:
    """An empty directory returns an empty dict."""
    assert _load_task_llm_outputs(tmp_path) == {}


def test_load_task_llm_outputs_skips_blank_lines(tmp_path: Path) -> None:
    """Blank lines in the JSONL file are skipped."""
    jsonl_file = tmp_path / "Task_7_llm_outputs.jsonl"
    jsonl_file.write_text(
        '{"call_id": 1}\n\n   \n{"call_id": 2}\n',
        encoding="utf-8",
    )

    result = _load_task_llm_outputs(tmp_path)
    assert len(result["7"]) == _EXPECT_2


# ── evaluate_tournament_solutions ─────────────────────────────────────────────


def _build_dataset_row(
    *,
    problem: str = "Prove that 1+1=2.",
    short_answer: str = "2",
    solution: str = "Proof: trivially 1+1=2.",
    grading_guidelines: str = "Award 7 if correct.",
) -> dict:
    """Build a fake HuggingFace dataset row."""
    return {
        "Problem": problem,
        "Short Answer": short_answer,
        "Solution": solution,
        "Grading guidelines": grading_guidelines,
    }


def _setup_solutions_dir(tmp_path: Path, records: list[dict]) -> Path:
    """Write *records* into ``Task_0_llm_outputs.jsonl`` inside *tmp_path*."""
    jsonl_file = tmp_path / "Task_0_llm_outputs.jsonl"
    lines = [json.dumps(r) for r in records]
    jsonl_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


@patch("open_deep_think.scripts.evaluate_tournament.load_dataset")
@patch("open_deep_think.scripts.evaluate_tournament.judge_answer")
def test_evaluate_tournament_solutions_answer_mode(
    mock_judge: MagicMock,
    mock_load_dataset: MagicMock,
    tmp_path: Path,
) -> None:
    """Answer-mode evaluation passes JudgeType.ANSWER and Short Answer ground truth."""
    mock_judge.return_value = (True, "\\boxed{Correct}")
    mock_load_dataset.return_value = [_build_dataset_row()]

    records = [
        {"call_id": 1, "phase": "initial_solution", "response_text": "My answer is 2."},
    ]
    solutions_dir = _setup_solutions_dir(tmp_path, records)

    results = evaluate_tournament_solutions(
        solutions_dir=str(solutions_dir),
        judge_type=JudgeType.ANSWER,
        judge_model_name="test-model",
    )

    assert "0" in results
    assert results["0"]["1"]["is_correct"] is True
    assert results["0"]["1"]["phase"] == "initial_solution"

    # Verify judge_answer was called with correct args
    mock_judge.assert_called_once()
    call_kwargs = mock_judge.call_args
    assert call_kwargs.kwargs["judge_type"] == JudgeType.ANSWER
    assert call_kwargs.kwargs["ground_truth"] == "2"
    assert call_kwargs.kwargs["guidelines"] is None


@patch("open_deep_think.scripts.evaluate_tournament.load_dataset")
@patch("open_deep_think.scripts.evaluate_tournament.judge_answer")
def test_evaluate_tournament_solutions_proof_mode(
    mock_judge: MagicMock,
    mock_load_dataset: MagicMock,
    tmp_path: Path,
) -> None:
    """Proof-mode evaluation passes JudgeType.PROOF, Solution ground truth, and guidelines."""
    mock_judge.return_value = (True, "<points>7 out of 7</points>")
    row = _build_dataset_row(
        solution="Full proof here.",
        grading_guidelines="Award 7 for complete proof.",
    )
    mock_load_dataset.return_value = [row]

    records = [
        {"call_id": 1, "phase": "initial_solution", "response_text": "My proof text."},
    ]
    solutions_dir = _setup_solutions_dir(tmp_path, records)

    results = evaluate_tournament_solutions(
        solutions_dir=str(solutions_dir),
        judge_type=JudgeType.PROOF,
        judge_model_name="test-model",
    )

    assert "0" in results
    assert results["0"]["1"]["is_correct"] is True

    # Verify judge_answer was called with proof-specific args
    mock_judge.assert_called_once()
    call_kwargs = mock_judge.call_args
    assert call_kwargs.kwargs["judge_type"] == JudgeType.PROOF
    assert call_kwargs.kwargs["ground_truth"] == "Full proof here."
    assert call_kwargs.kwargs["guidelines"] == "Award 7 for complete proof."


@patch("open_deep_think.scripts.evaluate_tournament.load_dataset")
@patch("open_deep_think.scripts.evaluate_tournament.judge_answer")
def test_evaluate_tournament_skips_verification_phases(
    mock_judge: MagicMock,
    mock_load_dataset: MagicMock,
    tmp_path: Path,
) -> None:
    """Verification and verification_check phases are not sent to the judge."""
    mock_judge.return_value = (True, "correct")
    mock_load_dataset.return_value = [_build_dataset_row()]

    records = [
        {"call_id": 1, "phase": "initial_solution", "response_text": "ans"},
        {"call_id": 2, "phase": "verification", "response_text": "report"},
        {"call_id": 3, "phase": "verification_check", "response_text": "yes"},
    ]
    solutions_dir = _setup_solutions_dir(tmp_path, records)

    results = evaluate_tournament_solutions(
        solutions_dir=str(solutions_dir),
        judge_type=JudgeType.ANSWER,
    )

    # Only call_id 1 should have been judged
    assert mock_judge.call_count == 1
    assert "1" in results["0"]
    assert "2" not in results["0"]
    assert "3" not in results["0"]


@patch("open_deep_think.scripts.evaluate_tournament.load_dataset")
@patch("open_deep_think.scripts.evaluate_tournament.judge_answer")
def test_evaluate_tournament_skips_empty_response_text(
    mock_judge: MagicMock,
    mock_load_dataset: MagicMock,
    tmp_path: Path,
) -> None:
    """Records with empty or whitespace-only response_text are skipped."""
    mock_judge.return_value = (True, "correct")
    mock_load_dataset.return_value = [_build_dataset_row()]

    records = [
        {"call_id": 1, "phase": "initial_solution", "response_text": ""},
        {"call_id": 2, "phase": "initial_solution", "response_text": "   "},
        {"call_id": 3, "phase": "initial_solution", "response_text": None},
        {"call_id": 4, "phase": "initial_solution", "response_text": "actual answer"},
    ]
    solutions_dir = _setup_solutions_dir(tmp_path, records)

    results = evaluate_tournament_solutions(
        solutions_dir=str(solutions_dir),
        judge_type=JudgeType.ANSWER,
    )

    assert mock_judge.call_count == 1
    assert "4" in results["0"]


@patch("open_deep_think.scripts.evaluate_tournament.load_dataset")
@patch("open_deep_think.scripts.evaluate_tournament.judge_answer")
def test_evaluate_tournament_skips_unknown_task_ids(
    mock_judge: MagicMock,
    mock_load_dataset: MagicMock,
    tmp_path: Path,
) -> None:
    """Tasks whose id is not in the dataset are skipped."""
    mock_load_dataset.return_value = []  # empty dataset — no task_ids exist

    records = [
        {"call_id": 1, "phase": "initial_solution", "response_text": "ans"},
    ]
    solutions_dir = _setup_solutions_dir(tmp_path, records)

    results = evaluate_tournament_solutions(
        solutions_dir=str(solutions_dir),
        judge_type=JudgeType.ANSWER,
    )

    mock_judge.assert_not_called()
    assert results == {}


@patch("open_deep_think.scripts.evaluate_tournament.load_dataset")
@patch("open_deep_think.scripts.evaluate_tournament.judge_answer")
def test_evaluate_tournament_multiple_solution_phases(
    mock_judge: MagicMock,
    mock_load_dataset: MagicMock,
    tmp_path: Path,
) -> None:
    """Multiple solution-producing phases for the same task are all evaluated."""
    mock_judge.side_effect = [
        (True, "correct"),
        (False, "incorrect"),
        (True, "correct again"),
    ]
    mock_load_dataset.return_value = [_build_dataset_row()]

    records = [
        {"call_id": 1, "phase": "initial_solution", "response_text": "sol1"},
        {"call_id": 2, "phase": "self_improvement", "response_text": "sol2"},
        {"call_id": 3, "phase": "tournament_merge", "response_text": "sol3"},
    ]
    solutions_dir = _setup_solutions_dir(tmp_path, records)

    results = evaluate_tournament_solutions(
        solutions_dir=str(solutions_dir),
        judge_type=JudgeType.ANSWER,
    )

    assert mock_judge.call_count == _EXPECT_3
    assert results["0"]["1"]["is_correct"] is True
    assert results["0"]["2"]["is_correct"] is False
    assert results["0"]["3"]["is_correct"] is True


def test_evaluate_tournament_raises_on_missing_directory() -> None:
    """ValueError is raised when the solutions directory does not exist."""
    with pytest.raises(ValueError, match="does not exist"):
        evaluate_tournament_solutions(
            solutions_dir="/nonexistent/path",
            judge_type=JudgeType.ANSWER,
        )


@patch("open_deep_think.scripts.evaluate_tournament.load_dataset")
@patch("open_deep_think.scripts.evaluate_tournament.judge_answer")
def test_evaluate_tournament_no_solution_records(
    mock_judge: MagicMock,
    mock_load_dataset: MagicMock,
    tmp_path: Path,
) -> None:
    """Tasks with only verification records produce an empty result for that task."""
    mock_load_dataset.return_value = [_build_dataset_row()]

    records = [
        {"call_id": 1, "phase": "verification", "response_text": "report"},
        {"call_id": 2, "phase": "verification_check", "response_text": "yes"},
    ]
    solutions_dir = _setup_solutions_dir(tmp_path, records)

    results = evaluate_tournament_solutions(
        solutions_dir=str(solutions_dir),
        judge_type=JudgeType.ANSWER,
    )

    mock_judge.assert_not_called()
    # Task 0 should not appear in results since there are no solution records
    assert "0" not in results
