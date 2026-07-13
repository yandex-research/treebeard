"""Tests for the ablations.generate_baseline script."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path


from open_deep_think.scripts.ablations.generate_baseline import (
    Candidate,
    GenerateBaselineConfig,
    generate_all_candidates,
    generate_and_save,
    is_task_done,
    load_task_ids_from_file,
    run_tasks_concurrent,
    validate_args,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

_NUM_SOLUTIONS_4 = 4
_NUM_SOLUTIONS_2 = 2
_TASK_ID_7 = 7
_TASK_ID_FAIL = 31


def _make_config(num_solutions: int = 2) -> GenerateBaselineConfig:
    """Return a minimal GenerateBaselineConfig for testing."""
    return GenerateBaselineConfig(
        solver_model="solver",
        solver_max_tokens=100,
        num_solutions=num_solutions,
        temperature=None,
        top_p=None,
    )


def _make_candidate(index: int) -> Candidate:
    """Return a Candidate stub."""
    return Candidate(
        index=index,
        solution_text=f"Solution {index}",
        completion=None,
    )


# ── is_task_done ──────────────────────────────────────────────────────────────


def test_is_task_done_returns_false_when_file_missing(tmp_path: Path) -> None:
    """is_task_done returns False when no LLM outputs JSONL exists."""
    assert is_task_done(tmp_path, 42) is False


def test_is_task_done_returns_false_when_file_empty(tmp_path: Path) -> None:
    """is_task_done returns False when LLM outputs JSONL is empty."""
    (tmp_path / "Task_42_llm_outputs.jsonl").write_text("", encoding="utf-8")
    assert is_task_done(tmp_path, 42) is False


def test_is_task_done_returns_true_when_file_has_content(tmp_path: Path) -> None:
    """is_task_done returns True when LLM outputs JSONL is non-empty."""
    (tmp_path / "Task_42_llm_outputs.jsonl").write_text('{"ok": true}\n', encoding="utf-8")
    assert is_task_done(tmp_path, 42) is True


# ── generate_all_candidates ───────────────────────────────────────────────────


def test_generate_all_candidates_returns_correct_count() -> None:
    """generate_all_candidates must return exactly num_solutions candidates."""
    config = _make_config(num_solutions=_NUM_SOLUTIONS_4)
    call_logger = MagicMock()

    with patch(
        "open_deep_think.scripts.ablations.generate_baseline.generate_candidate",
        side_effect=lambda **kwargs: Candidate(
            index=kwargs["candidate_index"],
            solution_text=f"Solution {kwargs['candidate_index']}",
            completion=None,
        ),
    ):
        candidates, payloads = generate_all_candidates(
            task_id=0,
            problem_statement="Solve x.",
            config=config,
            call_logger=call_logger,
        )

    assert len(candidates) == _NUM_SOLUTIONS_4
    assert len(payloads) == _NUM_SOLUTIONS_4
    assert all(p["status"] == "ok" for p in payloads)


def test_generate_all_candidates_replaces_failed_with_dummy() -> None:
    """When generate_candidate raises, a dummy candidate must be inserted."""
    config = _make_config(num_solutions=_NUM_SOLUTIONS_2)
    call_logger = MagicMock()

    def _side_effect(**kwargs: object) -> Candidate:
        idx = kwargs["candidate_index"]
        if idx == 1:
            msg = "API error"
            raise RuntimeError(msg)
        return _make_candidate(idx)

    with patch(
        "open_deep_think.scripts.ablations.generate_baseline.generate_candidate",
        side_effect=_side_effect,
    ):
        candidates, payloads = generate_all_candidates(
            task_id=5,
            problem_statement="Prove P.",
            config=config,
            call_logger=call_logger,
        )

    assert len(candidates) == _NUM_SOLUTIONS_2
    # First candidate is ok.
    assert payloads[0]["status"] == "ok"
    # Second candidate failed.
    assert payloads[1]["status"] == "generation_error"
    assert "API error" in payloads[1]["error"]
    # The dummy candidate has empty solution_text.
    assert candidates[1].solution_text == ""
    assert candidates[1].completion is None


def test_generate_all_candidates_payloads_have_no_verification() -> None:
    """Baseline candidate payloads must not contain verification details."""
    config = _make_config(num_solutions=1)
    call_logger = MagicMock()

    with patch(
        "open_deep_think.scripts.ablations.generate_baseline.generate_candidate",
        return_value=_make_candidate(0),
    ):
        _, payloads = generate_all_candidates(
            task_id=0,
            problem_statement="Problem.",
            config=config,
            call_logger=call_logger,
        )

    assert payloads[0]["status"] == "ok"
    assert "verification" not in payloads[0]


def test_generate_candidate_uses_baseline_prompt_without_system_message() -> None:
    """generate_candidate must use build_problem_prompt and omit any system message."""
    from open_deep_think.scripts.ablations.generate_baseline import generate_candidate  # noqa: PLC0415

    config = _make_config(num_solutions=1)
    call_logger = MagicMock()
    call_logger.record.return_value = 1

    captured_messages: list[list[dict[str, str]]] = []

    def _fake_chat_api_call(*, messages: list, **_kwargs: object) -> MagicMock:
        captured_messages.append(messages)
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Answer: 42"))]
        return mock_completion

    with patch(
        "open_deep_think.scripts.ablations.generate_baseline.chat_api_call",
        side_effect=_fake_chat_api_call,
    ):
        generate_candidate(
            task_id=0,
            candidate_index=0,
            problem_statement="Solve x + 1 = 2.",
            config=config,
            call_logger=call_logger,
        )

    assert len(captured_messages) == 1
    msgs = captured_messages[0]
    # No system message at all.
    assert all(m["role"] != "system" for m in msgs), "Expected no system message in baseline prompt"
    # The user message must start with the baseline prefix.
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"].startswith("Please reason step by step")
    assert "Solve x + 1 = 2." in msgs[0]["content"]


# ── generate_and_save ─────────────────────────────────────────────────────────


def test_generate_and_save_creates_llm_log_file(tmp_path: Path) -> None:
    """generate_and_save must create a Task_*_llm_outputs.jsonl file."""
    config = _make_config(num_solutions=1)

    with patch(
        "open_deep_think.scripts.ablations.generate_baseline.generate_all_candidates",
        return_value=(
            [_make_candidate(0)],
            [{"candidate_index": 0, "status": "ok"}],
        ),
    ):
        result = generate_and_save(
            task_id=3,
            problem_statement="P",
            config=config,
            output_dir=tmp_path,
        )

    assert result["status"] == "success"
    llm_log = tmp_path / "Task_3_llm_outputs.jsonl"
    assert llm_log.exists()
    assert "llm_outputs_path" in result


def test_generate_and_save_does_not_create_candidates_json(tmp_path: Path) -> None:
    """generate_and_save must NOT write a Task_*_candidates.json file."""
    config = _make_config(num_solutions=_NUM_SOLUTIONS_2)

    with patch(
        "open_deep_think.scripts.ablations.generate_baseline.generate_all_candidates",
        return_value=(
            [_make_candidate(0), _make_candidate(1)],
            [
                {"candidate_index": 0, "status": "ok"},
                {"candidate_index": 1, "status": "ok"},
            ],
        ),
    ):
        generate_and_save(
            task_id=_TASK_ID_7,
            problem_statement="Prove that 2+2=4.",
            config=config,
            output_dir=tmp_path,
        )

    candidates_file = tmp_path / f"Task_{_TASK_ID_7}_candidates.json"
    assert not candidates_file.exists(), "candidates JSON should no longer be written"


def test_generate_and_save_result_has_no_candidates_path(tmp_path: Path) -> None:
    """The result dict from generate_and_save must not include candidates_path."""
    config = _make_config(num_solutions=1)

    with patch(
        "open_deep_think.scripts.ablations.generate_baseline.generate_all_candidates",
        return_value=(
            [_make_candidate(0)],
            [{"candidate_index": 0, "status": "ok"}],
        ),
    ):
        result = generate_and_save(
            task_id=0,
            problem_statement="P",
            config=config,
            output_dir=tmp_path,
        )

    assert "candidates_path" not in result
    assert "llm_outputs_path" in result


# ── run_tasks_concurrent ──────────────────────────────────────────────────────

_NUM_CONCURRENT_TASKS = 3


def test_run_tasks_concurrent_processes_all_tasks(tmp_path: Path) -> None:
    """run_tasks_concurrent must produce LLM log JSONL for each task."""
    config = _make_config(num_solutions=1)

    with patch(
        "open_deep_think.scripts.ablations.generate_baseline.generate_all_candidates",
        return_value=(
            [_make_candidate(0)],
            [{"candidate_index": 0, "status": "ok"}],
        ),
    ):
        results = run_tasks_concurrent(
            task_ids=[10, 11, 12],
            problems=["P1", "P2", "P3"],
            config=config,
            output_dir=tmp_path,
            concurrency=2,
        )

    assert len(results) == _NUM_CONCURRENT_TASKS
    for tid in [10, 11, 12]:
        assert (tmp_path / f"Task_{tid}_llm_outputs.jsonl").exists()


def test_run_tasks_concurrent_skips_done_tasks(tmp_path: Path) -> None:
    """run_tasks_concurrent must skip tasks that already have output."""
    config = _make_config(num_solutions=1)

    # Pre-create LLM log for task 20 to mark it as done.
    (tmp_path / "Task_20_llm_outputs.jsonl").write_text('{"done": true}\n', encoding="utf-8")

    with patch(
        "open_deep_think.scripts.ablations.generate_baseline.generate_all_candidates",
        return_value=(
            [_make_candidate(0)],
            [{"candidate_index": 0, "status": "ok"}],
        ),
    ) as mock_gen:
        results = run_tasks_concurrent(
            task_ids=[20, 21],
            problems=["P1", "P2"],
            config=config,
            output_dir=tmp_path,
            concurrency=1,
        )

    # Task 20 was skipped; only task 21 called generate_all_candidates.
    assert mock_gen.call_count == 1
    statuses = {r["task_id"]: r["status"] for r in results}
    assert statuses[20] == "skipped"
    assert statuses[21] == "success"


def test_run_tasks_concurrent_handles_errors_gracefully(tmp_path: Path) -> None:
    """Tasks that raise unrecoverable errors must produce an error result."""
    config = _make_config(num_solutions=1)

    def _side_effect(*, task_id: int, **_kwargs: object) -> tuple:
        if task_id == _TASK_ID_FAIL:
            msg = "boom"
            raise RuntimeError(msg)
        return ([_make_candidate(0)], [{"candidate_index": 0, "status": "ok"}])

    with patch(
        "open_deep_think.scripts.ablations.generate_baseline.generate_all_candidates",
        side_effect=_side_effect,
    ):
        results = run_tasks_concurrent(
            task_ids=[30, _TASK_ID_FAIL],
            problems=["P1", "P2"],
            config=config,
            output_dir=tmp_path,
            concurrency=1,
        )

    statuses = {r["task_id"]: r["status"] for r in results}
    assert statuses[30] == "success"
    assert statuses[_TASK_ID_FAIL] == "error"


def test_run_tasks_concurrent_each_task_has_own_log(tmp_path: Path) -> None:
    """Each task must write its own JSONL log file — no shared global log."""
    config = _make_config(num_solutions=1)

    with patch(
        "open_deep_think.scripts.ablations.generate_baseline.generate_all_candidates",
        return_value=(
            [_make_candidate(0)],
            [{"candidate_index": 0, "status": "ok"}],
        ),
    ):
        run_tasks_concurrent(
            task_ids=[40, 41],
            problems=["P1", "P2"],
            config=config,
            output_dir=tmp_path,
            concurrency=2,
        )

    assert (tmp_path / "Task_40_llm_outputs.jsonl").exists()
    assert (tmp_path / "Task_41_llm_outputs.jsonl").exists()
    # No global output file.
    assert not (tmp_path / "all_llm_outputs.jsonl").exists()
    assert not (tmp_path / "run_summary.json").exists()


# ── load_task_ids_from_file ───────────────────────────────────────────────────


def test_load_task_ids_from_file_reads_ids(tmp_path: Path) -> None:
    """load_task_ids_from_file must parse one integer per line."""
    ids_file = tmp_path / "tasks.txt"
    ids_file.write_text("10\n20\n30\n", encoding="utf-8")
    result = load_task_ids_from_file(str(ids_file))
    assert result == [10, 20, 30]


def test_load_task_ids_from_file_skips_blanks_and_comments(tmp_path: Path) -> None:
    """Blank lines and comment lines (starting with #) must be ignored."""
    ids_file = tmp_path / "tasks.txt"
    ids_file.write_text("# header comment\n5\n\n# another comment\n15\n", encoding="utf-8")
    result = load_task_ids_from_file(str(ids_file))
    assert result == [5, 15]


def test_load_task_ids_from_file_raises_on_invalid_line(tmp_path: Path) -> None:
    """Non-integer lines must raise ValueError."""
    ids_file = tmp_path / "tasks.txt"
    ids_file.write_text("1\nnot_a_number\n3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot parse task ID"):
        load_task_ids_from_file(str(ids_file))


def test_load_task_ids_from_file_raises_on_missing_file(tmp_path: Path) -> None:
    """A missing file must raise FileNotFoundError."""
    missing = tmp_path / "nonexistent_task_ids_file.txt"
    with pytest.raises(FileNotFoundError):
        load_task_ids_from_file(str(missing))


# ── validate_args ─────────────────────────────────────────────────────────────

_VALID_DEFAULTS = {
    "model": "m",
    "output_path": "output_dir",
    "solver_max_tokens": 100,
    "num_solutions": 8,
    "concurrency": 1,
    "temperature": 0.7,
    "top_p": 1.0,
    "dataset_name": "ds",
    "dataset_split": "train",
    "run_name": "default",
}


def _ns(**overrides: object) -> argparse.Namespace:
    """Build an argparse.Namespace with sensible defaults and overrides."""
    values = {
        "task_ids_file": None,
        "start": None,
        "end": None,
        **_VALID_DEFAULTS,
        **overrides,
    }
    return argparse.Namespace(**values)


def test_validate_args_range_mode_ok() -> None:
    """Range mode (--start/--end) must pass when both are valid."""
    validate_args(_ns(start=0, end=10))


def test_validate_args_file_mode_ok() -> None:
    """File mode (--task_ids_file) must pass when start/end are absent."""
    validate_args(_ns(task_ids_file="tasks.txt"))


def test_validate_args_rejects_both_modes() -> None:
    """Providing both --task_ids_file and --start/--end must raise ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_args(_ns(task_ids_file="tasks.txt", start=0, end=10))


def test_validate_args_rejects_no_source() -> None:
    """Providing neither --task_ids_file nor --start/--end must raise ValueError."""
    with pytest.raises(ValueError, match="Either --task_ids_file or both"):
        validate_args(_ns())


def test_validate_args_rejects_start_only() -> None:
    """Providing only --start without --end must raise ValueError."""
    with pytest.raises(ValueError, match="Either --task_ids_file or both"):
        validate_args(_ns(start=0))


def test_validate_args_rejects_negative_start() -> None:
    """Negative --start must raise ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        validate_args(_ns(start=-1, end=5))


def test_validate_args_rejects_end_le_start() -> None:
    """--end <= --start must raise ValueError."""
    with pytest.raises(ValueError, match="greater than"):
        validate_args(_ns(start=5, end=5))
