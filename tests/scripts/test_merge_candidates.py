"""Tests for the ablations.merge_candidates script."""

from __future__ import annotations

import argparse
import json
import random
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from open_deep_think.scripts.ablations.merge_candidates import (
    CallResult,
    MergeCandidatesConfig,
    is_task_done,
    load_candidates,
    load_task_ids_from_file,
    process_task,
    run_all_merge_rounds,
    run_merge_round,
    run_tasks_concurrent,
    validate_args,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

_N_ROUNDS_2 = 2
_N_ROUNDS_3 = 3
_N_CANDIDATES_3 = 3
_N_CANDIDATES_4 = 4
_TASK_ID_FAIL = 31


def _make_config(
    *,
    n_rounds: int = 2,
    seed: int | None = 42,
) -> MergeCandidatesConfig:
    """Return a minimal MergeCandidatesConfig for testing."""
    return MergeCandidatesConfig(
        merger_model="merger",
        merger_max_tokens=100,
        n_rounds=n_rounds,
        temperature=None,
        top_p=None,
        seed=seed,
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


def _fake_call_result(text: str = "Merged solution.", call_id: int = 1) -> CallResult:
    """Return a stub CallResult."""
    return CallResult(completion=None, text=text, call_id=call_id)


def _make_candidates(n: int) -> list[dict]:
    """Build a list of candidate dicts for testing."""
    return [{"index": i, "solution_text": f"Solution {i}"} for i in range(n)]


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
    (tmp_path / "Task_42_merge_candidates.jsonl").write_text("", encoding="utf-8")
    assert is_task_done(tmp_path, 42) is False


def test_is_task_done_true_when_non_empty(tmp_path: Path) -> None:
    """is_task_done returns True when output JSONL has content."""
    (tmp_path / "Task_42_merge_candidates.jsonl").write_text(
        '{"ok":true}\n', encoding="utf-8"
    )
    assert is_task_done(tmp_path, 42) is True


# ── run_merge_round ───────────────────────────────────────────────────────────


def test_merge_round_returns_new_candidates() -> None:
    """A merge round must return a new list of candidates with merged solutions."""
    config = _make_config()
    call_logger = MagicMock()
    candidates = _make_candidates(_N_CANDIDATES_3)
    rng = random.Random(42)

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates.call_model",
        return_value=_fake_call_result("Merged solution."),
    ):
        new_candidates = run_merge_round(
            task_id=0,
            round_index=0,
            problem_statement="Find x.",
            candidates=candidates,
            config=config,
            call_logger=call_logger,
            rng=rng,
        )

    assert len(new_candidates) == _N_CANDIDATES_3
    # All candidates should now have the merged solution.
    for c in new_candidates:
        assert c["solution_text"] == "Merged solution."


def test_merge_round_preserves_candidate_indices() -> None:
    """The merge round must preserve original candidate indices."""
    config = _make_config()
    call_logger = MagicMock()
    candidates = _make_candidates(_N_CANDIDATES_3)
    rng = random.Random(42)

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates.call_model",
        return_value=_fake_call_result("Merged."),
    ):
        new_candidates = run_merge_round(
            task_id=0,
            round_index=0,
            problem_statement="P",
            candidates=candidates,
            config=config,
            call_logger=call_logger,
            rng=rng,
        )

    assert [c["index"] for c in new_candidates] == [0, 1, 2]


def test_merge_round_partner_is_never_self() -> None:
    """The merge partner j must never equal the candidate index i."""
    config = _make_config()
    call_logger = MagicMock()
    candidates = _make_candidates(_N_CANDIDATES_4)
    rng = random.Random(123)
    captured_partners: list[tuple[int, int]] = []

    def _capture_call(
        *, candidate_index: int, merge_partner_index: int, **_kwargs: object
    ) -> CallResult:
        captured_partners.append((candidate_index, merge_partner_index))
        return _fake_call_result("Merged.")

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates.call_model",
        side_effect=_capture_call,
    ):
        run_merge_round(
            task_id=0,
            round_index=0,
            problem_statement="P",
            candidates=candidates,
            config=config,
            call_logger=call_logger,
            rng=rng,
        )

    assert len(captured_partners) == _N_CANDIDATES_4
    for i, j in captured_partners:
        assert i != j, f"Candidate {i} was merged with itself"


def test_merge_round_uses_merge_prompt() -> None:
    """The merge call must use build_tournament_merge_prompt and system prompt."""
    config = _make_config()
    call_logger = MagicMock()
    candidates = _make_candidates(2)
    rng = random.Random(42)
    captured_messages: list[list[dict[str, str]]] = []

    def _capture_call(*, messages: list, **_kwargs: object) -> CallResult:
        captured_messages.append(messages)
        return _fake_call_result("Merged.")

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates.call_model",
        side_effect=_capture_call,
    ):
        run_merge_round(
            task_id=0,
            round_index=0,
            problem_statement="Find x.",
            candidates=candidates,
            config=config,
            call_logger=call_logger,
            rng=rng,
        )

    # Each call should have a system message and a user message with merge content.
    for msgs in captured_messages:
        assert msgs[0]["role"] == "system"
        assert "aggregate" in msgs[0]["content"].lower() or "merge" in msgs[0]["content"].lower() or "candidate" in msgs[0]["content"].lower()
        assert msgs[1]["role"] == "user"
        assert "Solution 1" in msgs[1]["content"]
        assert "Solution 2" in msgs[1]["content"]


def test_merge_round_candidate_i_is_solution_1() -> None:
    """Candidate i's solution must be placed as Solution 1 in the merge prompt."""
    config = _make_config()
    call_logger = MagicMock()
    candidates = [
        {"index": 0, "solution_text": "ALPHA_SOLUTION"},
        {"index": 1, "solution_text": "BETA_SOLUTION"},
    ]
    rng = random.Random(42)
    captured_messages: list[list[dict[str, str]]] = []

    def _capture_call(
        *, messages: list, candidate_index: int, **_kwargs: object
    ) -> CallResult:
        captured_messages.append(messages)
        return _fake_call_result("Merged.")

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates.call_model",
        side_effect=_capture_call,
    ):
        run_merge_round(
            task_id=0,
            round_index=0,
            problem_statement="P",
            candidates=candidates,
            config=config,
            call_logger=call_logger,
            rng=rng,
        )

    # For candidate 0, Solution 1 should be "ALPHA_SOLUTION".
    user_msg_0 = captured_messages[0][1]["content"]
    sol1_pos_0 = user_msg_0.index("Solution 1")
    sol2_pos_0 = user_msg_0.index("Solution 2")
    alpha_pos = user_msg_0.index("ALPHA_SOLUTION")
    # ALPHA should appear between Solution 1 and Solution 2 markers.
    assert sol1_pos_0 < alpha_pos < sol2_pos_0


def test_merge_round_falls_back_on_api_error() -> None:
    """If the API call fails, the candidate should keep its original solution."""
    config = _make_config()
    call_logger = MagicMock()
    candidates = _make_candidates(_N_CANDIDATES_3)
    rng = random.Random(42)
    call_count = {"n": 0}

    def _failing_call(**_kwargs: object) -> CallResult:
        call_count["n"] += 1
        if call_count["n"] == 2:
            msg = "API error"
            raise RuntimeError(msg)
        return _fake_call_result("Merged.")

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates.call_model",
        side_effect=_failing_call,
    ):
        new_candidates = run_merge_round(
            task_id=0,
            round_index=0,
            problem_statement="P",
            candidates=candidates,
            config=config,
            call_logger=call_logger,
            rng=rng,
        )

    assert len(new_candidates) == _N_CANDIDATES_3
    # The second candidate (index 1) should retain its original solution.
    assert new_candidates[1]["solution_text"] == "Solution 1"
    # The others should have merged solutions.
    assert new_candidates[0]["solution_text"] == "Merged."
    assert new_candidates[2]["solution_text"] == "Merged."


def test_merge_round_falls_back_on_empty_response() -> None:
    """If the model returns empty text, the candidate keeps its solution."""
    config = _make_config()
    call_logger = MagicMock()
    candidates = _make_candidates(2)
    rng = random.Random(42)

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates.call_model",
        return_value=_fake_call_result(""),
    ):
        new_candidates = run_merge_round(
            task_id=0,
            round_index=0,
            problem_statement="P",
            candidates=candidates,
            config=config,
            call_logger=call_logger,
            rng=rng,
        )

    # Both candidates should retain their original solutions.
    for i, c in enumerate(new_candidates):
        assert c["solution_text"] == f"Solution {i}"


# ── run_all_merge_rounds ─────────────────────────────────────────────────────


def test_all_merge_rounds_chains_rounds() -> None:
    """Each round's output candidates must become the next round's input."""
    config = _make_config(n_rounds=_N_ROUNDS_3)
    call_logger = MagicMock()
    candidates = _make_candidates(2)
    rng = random.Random(42)
    round_inputs: list[list[str]] = []

    original_run_merge_round = run_merge_round

    def _tracking_round(*, candidates: list, **kwargs: object) -> list[dict]:
        round_inputs.append([c["solution_text"] for c in candidates])
        return original_run_merge_round(candidates=candidates, **kwargs)

    with (
        patch(
            "open_deep_think.scripts.ablations.merge_candidates.run_merge_round",
            side_effect=_tracking_round,
        ),
        patch(
            "open_deep_think.scripts.ablations.merge_candidates.call_model",
            return_value=_fake_call_result("Merged."),
        ),
    ):
        result = run_all_merge_rounds(
            task_id=0,
            problem_statement="P",
            candidates=candidates,
            config=config,
            call_logger=call_logger,
            rng=rng,
        )

    assert len(round_inputs) == _N_ROUNDS_3
    # First round gets the original solutions.
    assert round_inputs[0] == ["Solution 0", "Solution 1"]
    # Subsequent rounds get the merged outputs from the previous round.
    assert round_inputs[1] == ["Merged.", "Merged."]
    assert round_inputs[2] == ["Merged.", "Merged."]
    # Final output should have merged solutions.
    assert all(c["solution_text"] == "Merged." for c in result)


def test_all_merge_rounds_returns_correct_count() -> None:
    """run_all_merge_rounds must return the same number of candidates."""
    config = _make_config(n_rounds=_N_ROUNDS_2)
    call_logger = MagicMock()
    candidates = _make_candidates(_N_CANDIDATES_4)
    rng = random.Random(42)

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates.call_model",
        return_value=_fake_call_result("Merged."),
    ):
        result = run_all_merge_rounds(
            task_id=0,
            problem_statement="P",
            candidates=candidates,
            config=config,
            call_logger=call_logger,
            rng=rng,
        )

    assert len(result) == _N_CANDIDATES_4


# ── process_task ──────────────────────────────────────────────────────────────


def test_process_task_writes_output_jsonl(tmp_path: Path) -> None:
    """process_task must create a Task_*_merge_candidates.jsonl file."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _make_llm_outputs_file(candidates_dir, task_id=5, n_candidates=2)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _make_config(n_rounds=1)

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates.call_model",
        return_value=_fake_call_result("Merged."),
    ):
        result = process_task(
            task_id=5,
            problem_statement="Find x.",
            candidates_dir=candidates_dir,
            config=config,
            output_dir=output_dir,
        )

    assert result["status"] == "success"
    output_file = output_dir / "Task_5_merge_candidates.jsonl"
    assert output_file.exists()
    lines = output_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 1


def test_process_task_skips_empty_candidates(tmp_path: Path) -> None:
    """Candidates with empty solution_text must be filtered out."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
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
            "response_text": "Valid solution A.",
        },
        {
            "phase": "initial_solution",
            "candidate_index": 2,
            "response_text": "Valid solution B.",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _make_config(n_rounds=1)

    merge_call_count = {"n": 0}

    def _counting_call(**_kwargs: object) -> CallResult:
        merge_call_count["n"] += 1
        return _fake_call_result("Merged.")

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates.call_model",
        side_effect=_counting_call,
    ):
        result = process_task(
            task_id=2,
            problem_statement="P",
            candidates_dir=candidates_dir,
            config=config,
            output_dir=output_dir,
        )

    assert result["status"] == "success"
    # Only 2 valid candidates, so 2 merge calls per round × 1 round.
    assert merge_call_count["n"] == 2


def test_process_task_skips_when_fewer_than_two_valid(tmp_path: Path) -> None:
    """If only one non-empty candidate exists, skip merge processing."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    path = candidates_dir / "Task_3_llm_outputs.jsonl"
    records = [
        {
            "phase": "initial_solution",
            "candidate_index": 0,
            "response_text": "",
        },
        {
            "phase": "initial_solution",
            "candidate_index": 1,
            "response_text": "Only valid solution.",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _make_config(n_rounds=1)

    result = process_task(
        task_id=3,
        problem_statement="P",
        candidates_dir=candidates_dir,
        config=config,
        output_dir=output_dir,
    )

    assert result["status"] == "skipped_too_few_candidates"


def test_process_task_output_records_have_merge_phase(tmp_path: Path) -> None:
    """Output JSONL records must have phase='merge' and merge_partner_index set."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _make_llm_outputs_file(candidates_dir, task_id=7, n_candidates=2)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _make_config(n_rounds=1)

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Merged."))]
    mock_completion.model_dump.return_value = {"id": "c1"}

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates.chat_api_call",
        return_value=mock_completion,
    ):
        process_task(
            task_id=7,
            problem_statement="Find x.",
            candidates_dir=candidates_dir,
            config=config,
            output_dir=output_dir,
        )

    output_file = output_dir / "Task_7_merge_candidates.jsonl"
    lines = output_file.read_text(encoding="utf-8").strip().split("\n")
    records = [json.loads(line) for line in lines]

    # All records should have record_type "llm_call" and phase "merge".
    for rec in records:
        assert rec["record_type"] == "llm_call"
        assert rec["phase"] == "merge"
        assert rec["merge_partner_index"] is not None
        assert rec["candidate_index"] != rec["merge_partner_index"]


# ── run_tasks_concurrent ──────────────────────────────────────────────────────

_NUM_TASKS_CONCURRENT = 2


def test_run_tasks_concurrent_processes_all_tasks(tmp_path: Path) -> None:
    """run_tasks_concurrent must produce output for each task."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _make_llm_outputs_file(candidates_dir, task_id=10, n_candidates=2)
    _make_llm_outputs_file(candidates_dir, task_id=11, n_candidates=2)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _make_config(n_rounds=1)

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates.call_model",
        return_value=_fake_call_result("Merged."),
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
        assert (output_dir / f"Task_{tid}_merge_candidates.jsonl").exists()


def test_run_tasks_concurrent_skips_done_tasks(tmp_path: Path) -> None:
    """run_tasks_concurrent must skip tasks that already have output."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _make_llm_outputs_file(candidates_dir, task_id=20, n_candidates=2)
    _make_llm_outputs_file(candidates_dir, task_id=21, n_candidates=2)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    # Pre-create output for task 20.
    (output_dir / "Task_20_merge_candidates.jsonl").write_text(
        '{"done": true}\n', encoding="utf-8"
    )

    config = _make_config(n_rounds=1)

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates.call_model",
        return_value=_fake_call_result("Merged."),
    ):
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


def test_run_tasks_concurrent_handles_errors_gracefully(tmp_path: Path) -> None:
    """Tasks that raise unrecoverable errors must produce an error result."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _make_llm_outputs_file(candidates_dir, task_id=30, n_candidates=2)
    _make_llm_outputs_file(candidates_dir, task_id=31, n_candidates=2)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _make_config(n_rounds=1)

    def _failing_process(task_id: int, *_args: object, **_kwargs: object) -> dict:
        if task_id == _TASK_ID_FAIL:
            msg = "boom"
            raise RuntimeError(msg)
        return {"task_id": task_id, "status": "success", "output_path": "x"}

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates._process_task_wrapper",
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


# ── Determinism with seed ─────────────────────────────────────────────────────


def test_merge_round_is_deterministic_with_seed() -> None:
    """Two runs with the same seed must pick the same merge partners."""
    config = _make_config(seed=42)
    call_logger = MagicMock()
    candidates = _make_candidates(_N_CANDIDATES_4)

    partners_run_1: list[int] = []
    partners_run_2: list[int] = []

    def _capture_partners_1(*, merge_partner_index: int, **_kwargs: object) -> CallResult:
        partners_run_1.append(merge_partner_index)
        return _fake_call_result("Merged.")

    def _capture_partners_2(*, merge_partner_index: int, **_kwargs: object) -> CallResult:
        partners_run_2.append(merge_partner_index)
        return _fake_call_result("Merged.")

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates.call_model",
        side_effect=_capture_partners_1,
    ):
        run_merge_round(
            task_id=0,
            round_index=0,
            problem_statement="P",
            candidates=candidates,
            config=config,
            call_logger=call_logger,
            rng=random.Random(42),
        )

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates.call_model",
        side_effect=_capture_partners_2,
    ):
        run_merge_round(
            task_id=0,
            round_index=0,
            problem_statement="P",
            candidates=candidates,
            config=config,
            call_logger=call_logger,
            rng=random.Random(42),
        )

    assert partners_run_1 == partners_run_2


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
    "merger_max_tokens": 100,
    "n_rounds": 3,
    "concurrency": 1,
    "temperature": 0.7,
    "top_p": 1.0,
    "seed": None,
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


# ── Output JSONL format ──────────────────────────────────────────────────────


def test_output_jsonl_contains_only_llm_calls(tmp_path: Path) -> None:
    """The output JSONL must contain only 'llm_call' records."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _make_llm_outputs_file(candidates_dir, task_id=8, n_candidates=2)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _make_config(n_rounds=1)

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Merged."))]
    mock_completion.model_dump.return_value = {"id": "c1"}

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates.chat_api_call",
        return_value=mock_completion,
    ):
        process_task(
            task_id=8,
            problem_statement="Find x.",
            candidates_dir=candidates_dir,
            config=config,
            output_dir=output_dir,
        )

    output_file = output_dir / "Task_8_merge_candidates.jsonl"
    lines = output_file.read_text(encoding="utf-8").strip().split("\n")
    records = [json.loads(line) for line in lines]

    record_types = {r["record_type"] for r in records}
    assert record_types == {"llm_call"}
    assert len(records) >= 1


def test_output_jsonl_logs_merge_partner_index(tmp_path: Path) -> None:
    """Each JSONL record must include the merge_partner_index field."""
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _make_llm_outputs_file(candidates_dir, task_id=9, n_candidates=_N_CANDIDATES_3)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _make_config(n_rounds=1)

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Merged."))]
    mock_completion.model_dump.return_value = {"id": "c1"}

    with patch(
        "open_deep_think.scripts.ablations.merge_candidates.chat_api_call",
        return_value=mock_completion,
    ):
        process_task(
            task_id=9,
            problem_statement="Find x.",
            candidates_dir=candidates_dir,
            config=config,
            output_dir=output_dir,
        )

    output_file = output_dir / "Task_9_merge_candidates.jsonl"
    lines = output_file.read_text(encoding="utf-8").strip().split("\n")
    records = [json.loads(line) for line in lines]

    for rec in records:
        assert "merge_partner_index" in rec
        assert isinstance(rec["merge_partner_index"], int)
        # Partner must be different from the candidate.
        assert rec["merge_partner_index"] != rec["candidate_index"]
