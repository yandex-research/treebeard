"""Tests for the ablations.self_improve script."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from open_deep_think.scripts.ablations.self_improve import (
    CallResult,
    RoundResult,
    SelfImproveConfig,
    VerificationResult,
    build_verification_prompt,
    improve_candidate,
    is_task_done,
    is_yes_response,
    load_candidates,
    load_task_ids_from_file,
    process_task,
    run_improvement_round_no_verification,
    run_improvement_round_with_verification,
    run_tasks_concurrent,
    validate_args,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

_N_ROUNDS_2 = 2
_N_ROUNDS_3 = 3
_N_CANDIDATES_3 = 3
_TASK_ID_FAIL = 31


def _make_config(
    *,
    n_rounds: int = 2,
    use_verification: bool = False,
) -> SelfImproveConfig:
    """Return a minimal SelfImproveConfig for testing."""
    return SelfImproveConfig(
        solver_model="solver",
        solver_max_tokens=100,
        verifier_model="verifier",
        verifier_max_tokens=100,
        classifier_model="classifier",
        classifier_max_tokens=100,
        n_rounds=n_rounds,
        use_verification=use_verification,
        temperature=None,
        top_p=None,
    )


def _make_llm_outputs_file(tmp_path: Path, task_id: int, n_candidates: int = 2) -> None:
    """Write a minimal Task_{id}_llm_outputs.jsonl with initial_solution records."""
    path = tmp_path / f"Task_{task_id}_llm_outputs.jsonl"
    lines: list[str] = []
    for i in range(n_candidates):
        record = {
            "record_type": "llm_call",
            "task_id": task_id,
            "call_id": i + 1,
            "phase": "initial_solution",
            "candidate_index": i,
            "model": "solver",
            "messages": [{"role": "user", "content": "Solve x."}],
            "response_text": f"Solution {i}",
            "completion": None,
            "error": None,
        }
        lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fake_call_result(text: str = "Improved solution.", call_id: int = 1) -> CallResult:
    """Return a stub CallResult."""
    return CallResult(completion=None, text=text, call_id=call_id)


def _fake_verification(*, is_pass: bool = False) -> VerificationResult:
    """Return a stub VerificationResult."""
    return VerificationResult(
        is_pass=is_pass,
        bug_report="bug report" if not is_pass else "",
        verifier_output="verifier output",
        classifier_output="yes" if is_pass else "no",
        verifier_call_id=10,
        classifier_call_id=11,
    )


# ── is_yes_response ──────────────────────────────────────────────────────────


def test_is_yes_response_standalone() -> None:
    """Standalone 'yes' must be detected."""
    assert is_yes_response("yes") is True


def test_is_yes_response_case_insensitive() -> None:
    """Case-insensitive matching must work."""
    assert is_yes_response("YES, the solution is correct") is True


def test_is_yes_response_rejects_substring() -> None:
    """Substrings like 'yesterday' must not match."""
    assert is_yes_response("yesterday") is False


# ── build_verification_prompt ─────────────────────────────────────────────────


def test_build_verification_prompt_includes_problem() -> None:
    """The verification prompt must include the problem statement."""
    prompt = build_verification_prompt(
        problem_statement="Find all primes p.",
        solution_text="Summary\nDetailed Solution\nHence p=2.",
    )
    assert "Find all primes p." in prompt


def test_build_verification_prompt_extracts_detailed_solution() -> None:
    """The verification prompt must extract the body after 'Detailed Solution'."""
    prompt = build_verification_prompt(
        problem_statement="P",
        solution_text="Summary\nDetailed Solution\nProof body here.",
    )
    assert "Proof body here." in prompt


# ── load_candidates ───────────────────────────────────────────────────────────


def test_load_candidates_reads_candidates(tmp_path: Path) -> None:
    """load_candidates must return candidate dicts from the JSONL file."""
    _make_llm_outputs_file(tmp_path, task_id=5, n_candidates=_N_CANDIDATES_3)
    candidates = load_candidates(tmp_path, task_id=5)
    assert len(candidates) == _N_CANDIDATES_3
    assert candidates[0]["index"] == 0
    assert candidates[2]["solution_text"] == "Solution 2"


def test_load_candidates_raises_on_missing_file(tmp_path: Path) -> None:
    """load_candidates must raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_candidates(tmp_path, task_id=99)


def test_load_candidates_raises_on_no_initial_solution_records(tmp_path: Path) -> None:
    """load_candidates must raise ValueError when no initial_solution records exist."""
    path = tmp_path / "Task_1_llm_outputs.jsonl"
    record = {
        "record_type": "llm_call",
        "task_id": 1,
        "call_id": 1,
        "phase": "verification",
        "candidate_index": 0,
        "model": "verifier",
        "messages": [],
        "response_text": "Verified.",
        "completion": None,
        "error": None,
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No initial_solution"):
        load_candidates(tmp_path, task_id=1)


def test_load_candidates_sorts_by_index(tmp_path: Path) -> None:
    """load_candidates must return candidates sorted by index."""
    path = tmp_path / "Task_2_llm_outputs.jsonl"
    lines: list[str] = []
    # Write in reverse order.
    for i in [2, 0, 1]:
        record = {
            "phase": "initial_solution",
            "candidate_index": i,
            "response_text": f"Solution {i}",
        }
        lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    candidates = load_candidates(tmp_path, task_id=2)
    assert [c["index"] for c in candidates] == [0, 1, 2]


# ── is_task_done ──────────────────────────────────────────────────────────────


def test_is_task_done_false_when_missing(tmp_path: Path) -> None:
    """is_task_done returns False when output JSONL does not exist."""
    assert is_task_done(tmp_path, 42) is False


def test_is_task_done_false_when_empty(tmp_path: Path) -> None:
    """is_task_done returns False when output JSONL is empty."""
    (tmp_path / "Task_42_self_improve.jsonl").write_text("", encoding="utf-8")
    assert is_task_done(tmp_path, 42) is False


def test_is_task_done_true_when_non_empty(tmp_path: Path) -> None:
    """is_task_done returns True when output JSONL has content."""
    (tmp_path / "Task_42_self_improve.jsonl").write_text('{"ok":true}\n', encoding="utf-8")
    assert is_task_done(tmp_path, 42) is True


# ── run_improvement_round_no_verification ─────────────────────────────────────


def test_no_verification_round_returns_improved_solution() -> None:
    """Without verification, round must return the model's improved text."""
    config = _make_config(use_verification=False)
    call_logger = MagicMock()

    with patch(
        "open_deep_think.scripts.ablations.self_improve.call_model",
        return_value=_fake_call_result("Better solution."),
    ):
        result = run_improvement_round_no_verification(
            task_id=0,
            candidate_index=0,
            round_index=0,
            problem_statement="Find x.",
            current_solution="Initial solution.",
            config=config,
            call_logger=call_logger,
        )

    assert result.output_solution == "Better solution."
    assert result.verification_report is None
    assert result.verification_is_pass is None


def test_no_verification_round_uses_self_improvement_prompt() -> None:
    """The self-improvement round must include IMO25_SELF_IMPROVEMENT_PROMPT."""
    config = _make_config(use_verification=False)
    call_logger = MagicMock()
    captured_messages: list[list[dict[str, str]]] = []

    def _capture_call(*, messages: list, **_kwargs: object) -> CallResult:
        captured_messages.append(messages)
        return _fake_call_result()

    with patch(
        "open_deep_think.scripts.ablations.self_improve.call_model",
        side_effect=_capture_call,
    ):
        run_improvement_round_no_verification(
            task_id=0,
            candidate_index=0,
            round_index=0,
            problem_statement="Find x.",
            current_solution="Solution text.",
            config=config,
            call_logger=call_logger,
        )

    assert len(captured_messages) == 1
    msgs = captured_messages[0]
    # Must have user (problem), assistant (solution), user (improve prompt).
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Solution text."
    assert msgs[2]["role"] == "user"
    assert "improve" in msgs[2]["content"].lower()


def test_no_verification_round_falls_back_to_input_on_empty_response() -> None:
    """If the model returns empty text, the round should keep the input solution."""
    config = _make_config(use_verification=False)
    call_logger = MagicMock()

    with patch(
        "open_deep_think.scripts.ablations.self_improve.call_model",
        return_value=_fake_call_result(""),
    ):
        result = run_improvement_round_no_verification(
            task_id=0,
            candidate_index=0,
            round_index=0,
            problem_statement="P",
            current_solution="Original solution.",
            config=config,
            call_logger=call_logger,
        )

    assert result.output_solution == "Original solution."


# ── run_improvement_round_with_verification ───────────────────────────────────


def test_with_verification_round_calls_verify_then_correct() -> None:
    """With verification, round must first verify then call the correction model."""
    config = _make_config(use_verification=True)
    call_logger = MagicMock()

    with (
        patch(
            "open_deep_think.scripts.ablations.self_improve.run_verification",
            return_value=_fake_verification(is_pass=False),
        ) as mock_verify,
        patch(
            "open_deep_think.scripts.ablations.self_improve.call_model",
            return_value=_fake_call_result("Corrected solution."),
        ) as mock_call,
    ):
        result = run_improvement_round_with_verification(
            task_id=0,
            candidate_index=0,
            round_index=0,
            problem_statement="Find x.",
            current_solution="Initial.",
            config=config,
            call_logger=call_logger,
        )

    mock_verify.assert_called_once()
    mock_call.assert_called_once()
    assert result.output_solution == "Corrected solution."
    assert result.verification_is_pass is False
    assert result.verification_report == "verifier output"


def test_with_verification_round_runs_even_on_pass() -> None:
    """Correction runs regardless of verification result."""
    config = _make_config(use_verification=True)
    call_logger = MagicMock()

    with (
        patch(
            "open_deep_think.scripts.ablations.self_improve.run_verification",
            return_value=_fake_verification(is_pass=True),
        ),
        patch(
            "open_deep_think.scripts.ablations.self_improve.call_model",
            return_value=_fake_call_result("Still corrected."),
        ) as mock_call,
    ):
        result = run_improvement_round_with_verification(
            task_id=0,
            candidate_index=0,
            round_index=0,
            problem_statement="P",
            current_solution="S",
            config=config,
            call_logger=call_logger,
        )

    mock_call.assert_called_once()
    assert result.verification_is_pass is True
    assert result.output_solution == "Still corrected."


def test_with_verification_round_uses_correction_prompt() -> None:
    """The correction call must include IMO25_CORRECTION_PROMPT + bug report."""
    config = _make_config(use_verification=True)
    call_logger = MagicMock()
    captured_messages: list[list[dict[str, str]]] = []

    def _capture_call(*, messages: list, **_kwargs: object) -> CallResult:
        captured_messages.append(messages)
        return _fake_call_result()

    with (
        patch(
            "open_deep_think.scripts.ablations.self_improve.run_verification",
            return_value=_fake_verification(is_pass=False),
        ),
        patch(
            "open_deep_think.scripts.ablations.self_improve.call_model",
            side_effect=_capture_call,
        ),
    ):
        run_improvement_round_with_verification(
            task_id=0,
            candidate_index=0,
            round_index=0,
            problem_statement="Find x.",
            current_solution="Old solution.",
            config=config,
            call_logger=call_logger,
        )

    assert len(captured_messages) == 1
    msgs = captured_messages[0]
    # user (problem), assistant (solution), user (correction+report).
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Old solution."
    assert msgs[2]["role"] == "user"
    assert "bug report" in msgs[2]["content"].lower()


# ── improve_candidate ─────────────────────────────────────────────────────────


def test_improve_candidate_chains_rounds() -> None:
    """Each round's output must become the next round's input."""
    config = _make_config(n_rounds=_N_ROUNDS_3, use_verification=False)
    call_logger = MagicMock()
    call_counter = {"n": 0}

    def _sequential_call(*, messages: list, **_kwargs: object) -> CallResult:
        call_counter["n"] += 1
        # The assistant content is the current solution fed to the model.
        solution_in = messages[2]["content"]
        return _fake_call_result(f"{solution_in} -> improved_{call_counter['n']}")

    with patch(
        "open_deep_think.scripts.ablations.self_improve.call_model",
        side_effect=_sequential_call,
    ):
        results = improve_candidate(
            task_id=0,
            candidate_index=0,
            initial_solution="base",
            problem_statement="P",
            config=config,
            call_logger=call_logger,
        )

    assert len(results) == _N_ROUNDS_3
    # First round input is the initial solution.
    assert results[0].input_solution == "base"
    # Second round input is first round output.
    assert results[1].input_solution == results[0].output_solution
    # Third round input is second round output.
    assert results[2].input_solution == results[1].output_solution


def test_improve_candidate_uses_verification_when_configured() -> None:
    """improve_candidate must dispatch to the verification variant when configured."""
    config = _make_config(n_rounds=1, use_verification=True)
    call_logger = MagicMock()

    with patch(
        "open_deep_think.scripts.ablations.self_improve.run_improvement_round_with_verification",
        return_value=RoundResult(
            candidate_id=0,
            round_index=0,
            input_solution="S",
            output_solution="S improved",
            verification_report="report",
            verification_is_pass=True,
            improvement_call_id=1,
        ),
    ) as mock_with_verify:
        results = improve_candidate(
            task_id=0,
            candidate_index=0,
            initial_solution="S",
            problem_statement="P",
            config=config,
            call_logger=call_logger,
        )

    mock_with_verify.assert_called_once()
    assert len(results) == 1
    assert results[0].verification_report == "report"


def test_improve_candidate_uses_no_verification_when_configured() -> None:
    """improve_candidate must dispatch to the no-verification variant by default."""
    config = _make_config(n_rounds=1, use_verification=False)
    call_logger = MagicMock()

    with patch(
        "open_deep_think.scripts.ablations.self_improve.run_improvement_round_no_verification",
        return_value=RoundResult(
            candidate_id=0,
            round_index=0,
            input_solution="S",
            output_solution="S improved",
            verification_report=None,
            verification_is_pass=None,
            improvement_call_id=1,
        ),
    ) as mock_no_verify:
        results = improve_candidate(
            task_id=0,
            candidate_index=0,
            initial_solution="S",
            problem_statement="P",
            config=config,
            call_logger=call_logger,
        )

    mock_no_verify.assert_called_once()
    assert len(results) == 1
    assert results[0].verification_report is None


# ── process_task ──────────────────────────────────────────────────────────────


def test_process_task_writes_output_jsonl(tmp_path: Path) -> None:
    """process_task must create a Task_*_self_improve.jsonl file."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _make_llm_outputs_file(candidates_dir, task_id=5, n_candidates=1)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _make_config(n_rounds=1, use_verification=False)

    with patch(
        "open_deep_think.scripts.ablations.self_improve.call_model",
        return_value=_fake_call_result("Improved."),
    ):
        result = process_task(
            task_id=5,
            problem_statement="Find x.",
            candidates_dir=candidates_dir,
            config=config,
            output_dir=output_dir,
        )

    assert result["status"] == "success"
    output_file = output_dir / "Task_5_self_improve.jsonl"
    assert output_file.exists()
    lines = output_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 1


def test_process_task_processes_all_candidates(tmp_path: Path) -> None:
    """process_task must process each non-empty candidate from the baseline."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _make_llm_outputs_file(candidates_dir, task_id=1, n_candidates=3)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _make_config(n_rounds=1, use_verification=False)

    improve_calls: list[int] = []

    def _track_improve(**kwargs: object) -> list[RoundResult]:
        improve_calls.append(kwargs["candidate_index"])
        return [
            RoundResult(
                candidate_id=kwargs["candidate_index"],
                round_index=0,
                input_solution="in",
                output_solution="out",
                verification_report=None,
                verification_is_pass=None,
                improvement_call_id=1,
            )
        ]

    with patch(
        "open_deep_think.scripts.ablations.self_improve.improve_candidate",
        side_effect=_track_improve,
    ):
        process_task(
            task_id=1,
            problem_statement="P",
            candidates_dir=candidates_dir,
            config=config,
            output_dir=output_dir,
        )

    assert improve_calls == [0, 1, 2]


def test_process_task_skips_empty_candidates(tmp_path: Path) -> None:
    """Candidates with empty solution_text must be skipped."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    # Write JSONL with one empty and one valid candidate.
    path = candidates_dir / "Task_2_llm_outputs.jsonl"
    records = [
        {
            "phase": "initial_solution",
            "candidate_index": 0,
            "response_text": "",
        },
        {
            "phase": "initial_solution",
            "candidate_index": 1,
            "response_text": "Valid solution.",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _make_config(n_rounds=1, use_verification=False)

    improve_calls: list[int] = []

    def _track_improve(**kwargs: object) -> list[RoundResult]:
        improve_calls.append(kwargs["candidate_index"])
        return [
            RoundResult(
                candidate_id=kwargs["candidate_index"],
                round_index=0,
                input_solution="in",
                output_solution="out",
                verification_report=None,
                verification_is_pass=None,
                improvement_call_id=1,
            )
        ]

    with patch(
        "open_deep_think.scripts.ablations.self_improve.improve_candidate",
        side_effect=_track_improve,
    ):
        process_task(
            task_id=2,
            problem_statement="P",
            candidates_dir=candidates_dir,
            config=config,
            output_dir=output_dir,
        )

    # Only candidate 1 should have been processed.
    assert improve_calls == [1]


def test_process_task_handles_candidate_failure_gracefully(tmp_path: Path) -> None:
    """If improve_candidate raises, the task should still finish successfully."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _make_llm_outputs_file(candidates_dir, task_id=3, n_candidates=2)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _make_config(n_rounds=1, use_verification=False)

    def _failing_improve(**kwargs: object) -> list[RoundResult]:
        if kwargs["candidate_index"] == 0:
            msg = "API failure"
            raise RuntimeError(msg)
        return [
            RoundResult(
                candidate_id=kwargs["candidate_index"],
                round_index=0,
                input_solution="in",
                output_solution="out",
                verification_report=None,
                verification_is_pass=None,
                improvement_call_id=1,
            )
        ]

    with patch(
        "open_deep_think.scripts.ablations.self_improve.improve_candidate",
        side_effect=_failing_improve,
    ):
        result = process_task(
            task_id=3,
            problem_statement="P",
            candidates_dir=candidates_dir,
            config=config,
            output_dir=output_dir,
        )

    # Task still reports success (individual candidate failures are logged).
    assert result["status"] == "success"


# ── run_tasks_concurrent ──────────────────────────────────────────────────────

_NUM_TASKS_CONCURRENT = 2


def test_run_tasks_concurrent_processes_all_tasks(tmp_path: Path) -> None:
    """run_tasks_concurrent must produce output for each task."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _make_llm_outputs_file(candidates_dir, task_id=10, n_candidates=1)
    _make_llm_outputs_file(candidates_dir, task_id=11, n_candidates=1)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _make_config(n_rounds=1)

    with patch(
        "open_deep_think.scripts.ablations.self_improve.call_model",
        return_value=_fake_call_result("Improved."),
    ):
        results = run_tasks_concurrent(
            task_ids=[10, 11],
            problems=["P1", "P2"],
            candidates_dir=candidates_dir,
            config=config,
            output_dir=output_dir,
            concurrency=2,
        )

    assert len(results) == _NUM_TASKS_CONCURRENT
    for tid in [10, 11]:
        assert (output_dir / f"Task_{tid}_self_improve.jsonl").exists()


def test_run_tasks_concurrent_skips_done_tasks(tmp_path: Path) -> None:
    """run_tasks_concurrent must skip tasks that already have output."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _make_llm_outputs_file(candidates_dir, task_id=20, n_candidates=1)
    _make_llm_outputs_file(candidates_dir, task_id=21, n_candidates=1)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    # Pre-create output for task 20.
    (output_dir / "Task_20_self_improve.jsonl").write_text('{"done": true}\n', encoding="utf-8")

    config = _make_config(n_rounds=1)

    with patch(
        "open_deep_think.scripts.ablations.self_improve.improve_candidate",
        return_value=[
            RoundResult(
                candidate_id=0,
                round_index=0,
                input_solution="in",
                output_solution="out",
                verification_report=None,
                verification_is_pass=None,
                improvement_call_id=1,
            )
        ],
    ) as mock_improve:
        results = run_tasks_concurrent(
            task_ids=[20, 21],
            problems=["P1", "P2"],
            candidates_dir=candidates_dir,
            config=config,
            output_dir=output_dir,
            concurrency=1,
        )

    statuses = {r["task_id"]: r["status"] for r in results}
    assert statuses[20] == "skipped"
    assert statuses[21] == "success"
    # improve_candidate should only have been called for task 21.
    assert mock_improve.call_count == 1


def test_run_tasks_concurrent_handles_errors_gracefully(tmp_path: Path) -> None:
    """Tasks that raise unrecoverable errors must produce an error result."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _make_llm_outputs_file(candidates_dir, task_id=30, n_candidates=1)
    _make_llm_outputs_file(candidates_dir, task_id=31, n_candidates=1)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _make_config(n_rounds=1)

    def _failing_process(task_id: int, *_args: object, **_kwargs: object) -> dict:
        if task_id == _TASK_ID_FAIL:
            msg = "boom"
            raise RuntimeError(msg)
        return {"task_id": task_id, "status": "success", "output_path": "x"}

    with patch(
        "open_deep_think.scripts.ablations.self_improve._process_task_wrapper",
        side_effect=_failing_process,
    ):
        results = run_tasks_concurrent(
            task_ids=[30, 31],
            problems=["P1", "P2"],
            candidates_dir=candidates_dir,
            config=config,
            output_dir=output_dir,
            concurrency=1,
        )

    statuses = {r["task_id"]: r["status"] for r in results}
    assert statuses[30] == "success"
    assert statuses[31] == "error"


# ── load_task_ids_from_file ───────────────────────────────────────────────────


def test_load_task_ids_from_file_reads_ids(tmp_path: Path) -> None:
    """load_task_ids_from_file must parse one integer per line."""
    ids_file = tmp_path / "tasks.txt"
    ids_file.write_text("10\n20\n30\n", encoding="utf-8")
    result = load_task_ids_from_file(str(ids_file))
    assert result == [10, 20, 30]


def test_load_task_ids_from_file_skips_blanks_and_comments(tmp_path: Path) -> None:
    """Blank lines and comment lines must be ignored."""
    ids_file = tmp_path / "tasks.txt"
    ids_file.write_text("# comment\n5\n\n15\n", encoding="utf-8")
    result = load_task_ids_from_file(str(ids_file))
    assert result == [5, 15]


def test_load_task_ids_from_file_raises_on_invalid_line(tmp_path: Path) -> None:
    """Non-integer lines must raise ValueError."""
    ids_file = tmp_path / "tasks.txt"
    ids_file.write_text("1\nnot_a_number\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot parse task ID"):
        load_task_ids_from_file(str(ids_file))


# ── validate_args ─────────────────────────────────────────────────────────────

_VALID_DEFAULTS = {
    "candidates_dir": "/data/candidates",
    "model": "m",
    "output_path": "/data/output",
    "verifier_model": None,
    "classifier_model": None,
    "solver_max_tokens": 100,
    "verifier_max_tokens": 100,
    "classifier_max_tokens": 100,
    "n_rounds": 3,
    "use_verification": False,
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
    """Range mode (--start/--end) must pass when valid."""
    validate_args(_ns(start=0, end=10))


def test_validate_args_file_mode_ok() -> None:
    """File mode (--task_ids_file) must pass when start/end are absent."""
    validate_args(_ns(task_ids_file="tasks.txt"))


def test_validate_args_rejects_both_modes() -> None:
    """Providing both task source modes must raise ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_args(_ns(task_ids_file="tasks.txt", start=0, end=10))


def test_validate_args_rejects_no_source() -> None:
    """Providing neither task source must raise ValueError."""
    with pytest.raises(ValueError, match="Either --task_ids_file or both"):
        validate_args(_ns())


def test_validate_args_rejects_negative_start() -> None:
    """Negative --start must raise ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        validate_args(_ns(start=-1, end=5))


def test_validate_args_rejects_end_le_start() -> None:
    """--end <= --start must raise ValueError."""
    with pytest.raises(ValueError, match="greater than"):
        validate_args(_ns(start=5, end=5))


def test_validate_args_rejects_zero_n_rounds() -> None:
    """--n_rounds < 1 must raise ValueError."""
    with pytest.raises(ValueError, match="n_rounds"):
        validate_args(_ns(start=0, end=1, n_rounds=0))


def test_validate_args_rejects_zero_concurrency() -> None:
    """--concurrency < 1 must raise ValueError."""
    with pytest.raises(ValueError, match="concurrency"):
        validate_args(_ns(start=0, end=1, concurrency=0))


# ── End-to-end round chaining with verification ──────────────────────────────


def test_improve_candidate_chains_rounds_with_verification() -> None:
    """With verification, each round's output becomes the next round's input."""
    config = _make_config(n_rounds=_N_ROUNDS_2, use_verification=True)
    call_logger = MagicMock()

    call_counter = {"n": 0}

    def _fake_verify(**_kwargs: object) -> VerificationResult:
        return _fake_verification(is_pass=False)

    def _fake_correct(*, messages: list, **_kwargs: object) -> CallResult:
        call_counter["n"] += 1
        solution_in = messages[1]["content"]
        return _fake_call_result(f"{solution_in}_corrected_{call_counter['n']}")

    with (
        patch(
            "open_deep_think.scripts.ablations.self_improve.run_verification",
            side_effect=_fake_verify,
        ),
        patch(
            "open_deep_think.scripts.ablations.self_improve.call_model",
            side_effect=_fake_correct,
        ),
    ):
        results = improve_candidate(
            task_id=0,
            candidate_index=0,
            initial_solution="base",
            problem_statement="P",
            config=config,
            call_logger=call_logger,
        )

    assert len(results) == _N_ROUNDS_2
    assert results[0].input_solution == "base"
    assert results[0].output_solution == "base_corrected_1"
    assert results[1].input_solution == "base_corrected_1"
    assert results[1].output_solution == "base_corrected_1_corrected_2"


# ── Output JSONL format ──────────────────────────────────────────────────────


def test_output_jsonl_contains_only_llm_calls(tmp_path: Path) -> None:
    """The output JSONL must contain only 'llm_call' records (no round_result)."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _make_llm_outputs_file(candidates_dir, task_id=7, n_candidates=1)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _make_config(n_rounds=1, use_verification=False)

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Improved."))]
    mock_completion.model_dump.return_value = {"id": "c1"}

    with patch(
        "open_deep_think.scripts.ablations.self_improve.chat_api_call",
        return_value=mock_completion,
    ):
        process_task(
            task_id=7,
            problem_statement="Find x.",
            candidates_dir=candidates_dir,
            config=config,
            output_dir=output_dir,
        )

    output_file = output_dir / "Task_7_self_improve.jsonl"
    lines = output_file.read_text(encoding="utf-8").strip().split("\n")
    records = [json.loads(line) for line in lines]

    record_types = {r["record_type"] for r in records}
    assert record_types == {"llm_call"}
    assert len(records) >= 1
