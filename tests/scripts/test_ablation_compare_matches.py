"""Tests for the ablation_compare_matches script."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from open_deep_think.scripts.ablation_compare_matches import (
    _METHOD_RUNNERS,
    ALL_METHODS,
    AblationConfig,
    CandidateData,
    PairWorkItem,
    ThreadSafeJSONLWriter,
    build_ordered_pairs,
    build_work_items,
    load_candidates,
    load_completed_pair_keys,
    load_completed_pairs,
    parse_pick,
    process_pair,
    run_clean_merge,
    run_clean_select,
    run_clean_select_improve,
    run_comparison,
    run_method,
    run_verification_based_merge,
    run_verification_based_select,
    run_verification_based_select_improve,
    validate_args,
)
from open_deep_think.scripts.tournament_merge_improve import TaskCallLogger

if TYPE_CHECKING:
    from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_candidate(task_id: int, idx: int, *, is_pass: bool = True) -> CandidateData:
    """Build a CandidateData stub for testing."""
    return CandidateData(
        task_id=task_id,
        candidate_index=idx,
        solution_text=f"Solution {idx} for task {task_id}",
        verification={
            "is_pass": is_pass,
            "verifier_output": f"Verification for solution {idx}",
            "classifier_output": "yes" if is_pass else "no",
            "bug_report": "" if is_pass else "bug found",
            "verifier_call_id": 10,
            "classifier_call_id": 11,
        },
    )


def _make_config() -> AblationConfig:
    """Build a minimal config for testing."""
    return AblationConfig(
        model="test-model",
        max_tokens=100,
        temperature=None,
        top_p=None,
    )


def _make_work_item(
    task_id: int = 1,
    a_idx: int = 0,
    b_idx: int = 1,
) -> PairWorkItem:
    """Build a PairWorkItem stub for testing."""
    return PairWorkItem(
        task_id=task_id,
        solution_a=_make_candidate(task_id, a_idx),
        solution_b=_make_candidate(task_id, b_idx, is_pass=False),
        problem_statement=f"Problem for task {task_id}",
    )


def _make_args(**overrides: object) -> MagicMock:
    """Build a mock argparse.Namespace with sensible defaults."""
    defaults = {
        "inputs_path": "test_inputs_nonexistent",
        "model": "test-model",
        "output_path": "test_output_nonexistent",
        "run_name": "default",
        "concurrency": 1,
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 1.0,
        "dataset_name": "test",
        "dataset_split": "train",
    }
    defaults.update(overrides)
    ns = MagicMock()
    for k, v in defaults.items():
        setattr(ns, k, v)
    return ns


def _mock_call_model_select_1(**_kwargs: object) -> MagicMock:
    """Mock call_model that returns '1' (select solution 1)."""
    result = MagicMock()
    result.text = "1"
    result.call_id = 1
    return result


def _mock_call_model_select_2(**_kwargs: object) -> MagicMock:
    """Mock call_model that returns '2' (select solution 2)."""
    result = MagicMock()
    result.text = "2"
    result.call_id = 1
    return result


def _mock_call_model_merge(**_kwargs: object) -> MagicMock:
    """Mock call_model that returns a merged solution."""
    result = MagicMock()
    result.text = "Merged solution text"
    result.call_id = 1
    return result


# ── parse_pick ────────────────────────────────────────────────────────────────


def test_parse_pick_returns_1() -> None:
    """parse_pick must return 1 when the response contains '1'."""
    assert parse_pick("1") == 1


def test_parse_pick_returns_2() -> None:
    """parse_pick must return 2 when the response contains '2'."""
    assert parse_pick("2") == 2


def test_parse_pick_returns_first_match() -> None:
    """parse_pick must return the first match when both 1 and 2 appear."""
    assert parse_pick("1 then 2") == 1


def test_parse_pick_defaults_to_1_on_no_match() -> None:
    """parse_pick must default to 1 when no digit is found."""
    assert parse_pick("neither") == 1


def test_parse_pick_handles_whitespace() -> None:
    """parse_pick must handle leading/trailing whitespace."""
    assert parse_pick("  2  ") == 2


# ── build_ordered_pairs ──────────────────────────────────────────────────────


def test_build_ordered_pairs_produces_correct_pairs() -> None:
    """build_ordered_pairs must produce (0,1), (1,0), (2,3), (3,2) from 4 candidates."""
    cands = [_make_candidate(1, i) for i in range(4)]
    pairs = build_ordered_pairs(cands)
    pair_indices = [(a.candidate_index, b.candidate_index) for a, b in pairs]
    assert pair_indices == [(0, 1), (1, 0), (2, 3), (3, 2)]


def test_build_ordered_pairs_two_candidates() -> None:
    """build_ordered_pairs with 2 candidates must produce (0,1) and (1,0)."""
    cands = [_make_candidate(1, 0), _make_candidate(1, 1)]
    pairs = build_ordered_pairs(cands)
    pair_indices = [(a.candidate_index, b.candidate_index) for a, b in pairs]
    assert pair_indices == [(0, 1), (1, 0)]


def test_build_ordered_pairs_rejects_odd_count() -> None:
    """build_ordered_pairs must raise ValueError for odd number of candidates."""
    cands = [_make_candidate(1, i) for i in range(3)]
    with pytest.raises(ValueError, match="even number"):
        build_ordered_pairs(cands)


def test_build_ordered_pairs_six_candidates() -> None:
    """build_ordered_pairs with 6 candidates must produce 6 pairs."""
    cands = [_make_candidate(1, i) for i in range(6)]
    pairs = build_ordered_pairs(cands)
    assert len(pairs) == 6
    pair_indices = [(a.candidate_index, b.candidate_index) for a, b in pairs]
    assert pair_indices == [(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)]


# ── load_candidates ─────────────────────────────────────────────────────────


def test_load_candidates_reads_jsonl(tmp_path: Path) -> None:
    """load_candidates must parse JSONL records into CandidateData grouped by task."""
    jsonl = tmp_path / "candidates.jsonl"
    records = [
        {
            "task_id": 1,
            "candidate_index": 0,
            "solution_text": "Sol 0",
            "verification": {"is_pass": True, "verifier_output": "ok"},
            "status": "ok",
        },
        {
            "task_id": 1,
            "candidate_index": 1,
            "solution_text": "Sol 1",
            "verification": {"is_pass": False, "verifier_output": "fail"},
            "status": "ok",
        },
        {
            "task_id": 2,
            "candidate_index": 0,
            "solution_text": "Sol 2-0",
            "verification": {"is_pass": True, "verifier_output": "ok"},
            "status": "ok",
        },
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    result = load_candidates(jsonl)
    assert set(result.keys()) == {1, 2}
    assert len(result[1]) == 2
    assert len(result[2]) == 1
    assert result[1][0].candidate_index == 0
    assert result[1][1].candidate_index == 1


def test_load_candidates_sorts_by_index(tmp_path: Path) -> None:
    """load_candidates must sort candidates by candidate_index within each task."""
    jsonl = tmp_path / "candidates.jsonl"
    # Write in reverse order.
    records = [
        {"task_id": 1, "candidate_index": 2, "solution_text": "S2", "verification": {}},
        {"task_id": 1, "candidate_index": 0, "solution_text": "S0", "verification": {}},
        {"task_id": 1, "candidate_index": 1, "solution_text": "S1", "verification": {}},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    result = load_candidates(jsonl)
    indices = [c.candidate_index for c in result[1]]
    assert indices == [0, 1, 2]


def test_load_candidates_skips_blank_lines(tmp_path: Path) -> None:
    """load_candidates must skip blank lines in the JSONL."""
    jsonl = tmp_path / "candidates.jsonl"
    content = (
        json.dumps({"task_id": 1, "candidate_index": 0, "solution_text": "S", "verification": {}})
        + "\n\n"
        + json.dumps({"task_id": 1, "candidate_index": 1, "solution_text": "S", "verification": {}})
        + "\n"
    )
    jsonl.write_text(content, encoding="utf-8")

    result = load_candidates(jsonl)
    assert len(result[1]) == 2


# ── load_completed_pairs ─────────────────────────────────────────────────────


def test_load_completed_pairs_missing_file(tmp_path: Path) -> None:
    """load_completed_pairs must return empty set when file does not exist."""
    jsonl = tmp_path / "nonexistent.jsonl"
    assert load_completed_pairs(jsonl) == set()


def test_load_completed_pairs_returns_all_tuples(tmp_path: Path) -> None:
    """load_completed_pairs must return all (task_id, a, b, method) tuples."""
    jsonl = tmp_path / "results.jsonl"
    records = [
        {"task_id": 1, "solution_a_index": 0, "solution_b_index": 1, "method": "clean_select"},
        {"task_id": 1, "solution_a_index": 0, "solution_b_index": 1, "method": "clean_merge"},
        {"task_id": 2, "solution_a_index": 2, "solution_b_index": 3, "method": "clean_select"},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    result = load_completed_pairs(jsonl)
    assert result == {
        (1, 0, 1, "clean_select"),
        (1, 0, 1, "clean_merge"),
        (2, 2, 3, "clean_select"),
    }


# ── load_completed_pair_keys ────────────────────────────────────────────────


def test_load_completed_pair_keys_only_returns_fully_done(tmp_path: Path) -> None:
    """load_completed_pair_keys must only return pairs with all 6 methods done."""
    jsonl = tmp_path / "results.jsonl"
    # Pair (1, 0, 1) has all 6 methods.
    records = [
        {"task_id": 1, "solution_a_index": 0, "solution_b_index": 1, "method": method} for method in ALL_METHODS
    ]
    # Pair (1, 1, 0) has only 2 methods -- incomplete.
    records.extend(
        [
            {"task_id": 1, "solution_a_index": 1, "solution_b_index": 0, "method": "clean_select"},
            {"task_id": 1, "solution_a_index": 1, "solution_b_index": 0, "method": "clean_merge"},
        ]
    )
    jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    result = load_completed_pair_keys(jsonl)
    assert result == {(1, 0, 1)}


def test_load_completed_pair_keys_missing_file(tmp_path: Path) -> None:
    """load_completed_pair_keys must return empty set when file does not exist."""
    jsonl = tmp_path / "nonexistent.jsonl"
    assert load_completed_pair_keys(jsonl) == set()


# ── build_work_items ─────────────────────────────────────────────────────────


def test_build_work_items_generates_all_pairs() -> None:
    """build_work_items must produce work items for all pairs across all tasks."""
    candidates = {
        1: [_make_candidate(1, 0), _make_candidate(1, 1)],
        2: [_make_candidate(2, 0), _make_candidate(2, 1)],
    }
    problems = {1: "P1", 2: "P2"}
    items = build_work_items(candidates, problems, set())
    # 2 candidates per task -> 2 pairs per task -> 4 total.
    assert len(items) == 4


def test_build_work_items_skips_completed() -> None:
    """build_work_items must skip pairs in the completed set."""
    candidates = {1: [_make_candidate(1, 0), _make_candidate(1, 1)]}
    problems = {1: "P1"}
    completed = {(1, 0, 1)}  # skip the (0, 1) pair
    items = build_work_items(candidates, problems, completed)
    # Only (1, 0) pair remains.
    assert len(items) == 1
    assert items[0].solution_a.candidate_index == 1
    assert items[0].solution_b.candidate_index == 0


def test_build_work_items_returns_empty_when_all_done() -> None:
    """build_work_items must return empty list when all pairs are completed."""
    candidates = {1: [_make_candidate(1, 0), _make_candidate(1, 1)]}
    problems = {1: "P1"}
    completed = {(1, 0, 1), (1, 1, 0)}
    items = build_work_items(candidates, problems, completed)
    assert items == []


def test_build_work_items_carries_problem_statement() -> None:
    """Each PairWorkItem must carry the correct problem statement."""
    candidates = {5: [_make_candidate(5, 0), _make_candidate(5, 1)]}
    problems = {5: "Problem five"}
    items = build_work_items(candidates, problems, set())
    assert all(item.problem_statement == "Problem five" for item in items)


# ── validate_args ────────────────────────────────────────────────────────────


def test_validate_args_rejects_zero_concurrency() -> None:
    """validate_args must reject --concurrency < 1."""
    args = _make_args(concurrency=0)
    with pytest.raises(ValueError, match="concurrency"):
        validate_args(args)


def test_validate_args_rejects_missing_inputs_path() -> None:
    """validate_args must reject non-existent --inputs_path."""
    args = _make_args(inputs_path="/nonexistent/path/ablation_test")
    with pytest.raises(ValueError, match="does not exist"):
        validate_args(args)


def test_validate_args_rejects_missing_candidates_jsonl(tmp_path: Path) -> None:
    """validate_args must reject --inputs_path without candidates.jsonl."""
    args = _make_args(inputs_path=str(tmp_path))
    with pytest.raises(ValueError, match=re.escape("candidates.jsonl")):
        validate_args(args)


def test_validate_args_accepts_valid_inputs(tmp_path: Path) -> None:
    """validate_args must accept valid arguments without raising."""
    (tmp_path / "candidates.jsonl").write_text("", encoding="utf-8")
    args = _make_args(inputs_path=str(tmp_path), concurrency=4)
    validate_args(args)  # should not raise


# ── ThreadSafeJSONLWriter ────────────────────────────────────────────────────


def test_writer_creates_file_on_first_append(tmp_path: Path) -> None:
    """ThreadSafeJSONLWriter must create the file if it doesn't exist."""
    jsonl = tmp_path / "out.jsonl"
    writer = ThreadSafeJSONLWriter(jsonl)
    writer.append({"key": "value"})
    assert jsonl.exists()
    lines = jsonl.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"key": "value"}


def test_writer_appends_multiple_records(tmp_path: Path) -> None:
    """ThreadSafeJSONLWriter must append without overwriting existing data."""
    jsonl = tmp_path / "out.jsonl"
    writer = ThreadSafeJSONLWriter(jsonl)
    writer.append({"idx": 0})
    writer.append({"idx": 1})
    lines = jsonl.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


# ── run_clean_select ─────────────────────────────────────────────────────────


def test_run_clean_select_picks_solution_1(tmp_path: Path) -> None:
    """run_clean_select must return solution_a when judge picks 1."""
    config = _make_config()
    work_item = _make_work_item()
    log_path = tmp_path / "log.jsonl"
    call_logger = TaskCallLogger(task_id=1, task_log_path=log_path)

    with patch(
        "open_deep_think.scripts.ablation_compare_matches.call_model",
        side_effect=_mock_call_model_select_1,
    ):
        result = run_clean_select(
            work_item=work_item,
            config=config,
            call_logger=call_logger,
        )

    assert result["selected_index"] == 1
    assert result["selected_candidate_index"] == work_item.solution_a.candidate_index
    assert result["result_solution_text"] == work_item.solution_a.solution_text


def test_run_clean_select_picks_solution_2(tmp_path: Path) -> None:
    """run_clean_select must return solution_b when judge picks 2."""
    config = _make_config()
    work_item = _make_work_item()
    log_path = tmp_path / "log.jsonl"
    call_logger = TaskCallLogger(task_id=1, task_log_path=log_path)

    with patch(
        "open_deep_think.scripts.ablation_compare_matches.call_model",
        side_effect=_mock_call_model_select_2,
    ):
        result = run_clean_select(
            work_item=work_item,
            config=config,
            call_logger=call_logger,
        )

    assert result["selected_index"] == 2
    assert result["selected_candidate_index"] == work_item.solution_b.candidate_index


# ── run_verification_based_select ────────────────────────────────────────────


def test_run_verification_based_select_picks_solution_1(tmp_path: Path) -> None:
    """run_verification_based_select must return solution_a when judge picks 1."""
    config = _make_config()
    work_item = _make_work_item()
    log_path = tmp_path / "log.jsonl"
    call_logger = TaskCallLogger(task_id=1, task_log_path=log_path)

    with patch(
        "open_deep_think.scripts.ablation_compare_matches.call_model",
        side_effect=_mock_call_model_select_1,
    ):
        result = run_verification_based_select(
            work_item=work_item,
            config=config,
            call_logger=call_logger,
        )

    assert result["selected_index"] == 1
    assert result["result_solution_text"] == work_item.solution_a.solution_text


# ── run_clean_merge ──────────────────────────────────────────────────────────


def test_run_clean_merge_returns_merged_text(tmp_path: Path) -> None:
    """run_clean_merge must return the merged solution text."""
    config = _make_config()
    work_item = _make_work_item()
    log_path = tmp_path / "log.jsonl"
    call_logger = TaskCallLogger(task_id=1, task_log_path=log_path)

    with patch(
        "open_deep_think.scripts.ablation_compare_matches.call_model",
        side_effect=_mock_call_model_merge,
    ):
        result = run_clean_merge(
            work_item=work_item,
            config=config,
            call_logger=call_logger,
        )

    assert result["result_solution_text"] == "Merged solution text"


# ── run_verification_based_merge ─────────────────────────────────────────────


def test_run_verification_based_merge_returns_merged_text(tmp_path: Path) -> None:
    """run_verification_based_merge must return the merged solution text."""
    config = _make_config()
    work_item = _make_work_item()
    log_path = tmp_path / "log.jsonl"
    call_logger = TaskCallLogger(task_id=1, task_log_path=log_path)

    with patch(
        "open_deep_think.scripts.ablation_compare_matches.call_model",
        side_effect=_mock_call_model_merge,
    ):
        result = run_verification_based_merge(
            work_item=work_item,
            config=config,
            call_logger=call_logger,
        )

    assert result["result_solution_text"] == "Merged solution text"


# ── select_improve variants ──────────────────────────────────────────────────


def test_run_clean_select_improve_calls_select_then_improve(tmp_path: Path) -> None:
    """run_clean_select_improve must first select, then self-improve."""
    config = _make_config()
    work_item = _make_work_item()
    log_path = tmp_path / "log.jsonl"
    call_logger = TaskCallLogger(task_id=1, task_log_path=log_path)

    call_count = 0

    def _mock_call(**_kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        if call_count == 1:
            # Selection call -> pick 1.
            result.text = "1"
        else:
            # Self-improvement call.
            result.text = "Improved solution"
        result.call_id = call_count
        return result

    with patch(
        "open_deep_think.scripts.ablation_compare_matches.call_model",
        side_effect=_mock_call,
    ):
        result = run_clean_select_improve(
            work_item=work_item,
            config=config,
            call_logger=call_logger,
        )

    assert result["selected_index"] == 1
    assert result["result_solution_text"] == "Improved solution"
    assert call_count == 2  # one select + one SI


def test_run_verification_based_select_improve_calls_select_then_improve(tmp_path: Path) -> None:
    """run_verification_based_select_improve must first select, then self-improve."""
    config = _make_config()
    work_item = _make_work_item()
    log_path = tmp_path / "log.jsonl"
    call_logger = TaskCallLogger(task_id=1, task_log_path=log_path)

    call_count = 0

    def _mock_call(**_kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        if call_count == 1:
            result.text = "2"
        else:
            result.text = "Improved B"
        result.call_id = call_count
        return result

    with patch(
        "open_deep_think.scripts.ablation_compare_matches.call_model",
        side_effect=_mock_call,
    ):
        result = run_verification_based_select_improve(
            work_item=work_item,
            config=config,
            call_logger=call_logger,
        )

    assert result["selected_index"] == 2
    assert result["result_solution_text"] == "Improved B"
    assert call_count == 2


def test_run_clean_select_improve_includes_verification_report(tmp_path: Path) -> None:
    """run_clean_select_improve must pass the selected solution's verification report to the SI call."""
    config = _make_config()
    work_item = _make_work_item()  # solution_a has verifier_output="Verification for solution 0"
    log_path = tmp_path / "log.jsonl"
    call_logger = TaskCallLogger(task_id=1, task_log_path=log_path)

    captured_messages: list[list[dict[str, str]]] = []

    def _mock_call(**kwargs: object) -> MagicMock:
        captured_messages.append(kwargs["messages"])
        result = MagicMock()
        phase = kwargs.get("phase", "")
        if phase == "clean_select":
            result.text = "1"  # select solution_a
        else:
            result.text = "Improved"
        result.call_id = 1
        return result

    with patch(
        "open_deep_think.scripts.ablation_compare_matches.call_model",
        side_effect=_mock_call,
    ):
        run_clean_select_improve(
            work_item=work_item,
            config=config,
            call_logger=call_logger,
        )

    # Second call is self-improvement; its last user message must contain the
    # verification report for the selected candidate (solution_a, index 0).
    si_messages = captured_messages[1]
    last_user_content = si_messages[-1]["content"]
    assert "Verification for solution 0" in last_user_content


def test_run_verification_based_select_improve_includes_verification_report(tmp_path: Path) -> None:
    """run_verification_based_select_improve must pass the selected solution's verification report."""
    config = _make_config()
    work_item = _make_work_item()  # solution_b has verifier_output="Verification for solution 1"
    log_path = tmp_path / "log.jsonl"
    call_logger = TaskCallLogger(task_id=1, task_log_path=log_path)

    captured_messages: list[list[dict[str, str]]] = []

    def _mock_call(**kwargs: object) -> MagicMock:
        captured_messages.append(kwargs["messages"])
        result = MagicMock()
        phase = kwargs.get("phase", "")
        if phase == "verification_based_select":
            result.text = "2"  # select solution_b
        else:
            result.text = "Improved B"
        result.call_id = 1
        return result

    with patch(
        "open_deep_think.scripts.ablation_compare_matches.call_model",
        side_effect=_mock_call,
    ):
        run_verification_based_select_improve(
            work_item=work_item,
            config=config,
            call_logger=call_logger,
        )

    # Second call is self-improvement; its last user message must contain the
    # verification report for the selected candidate (solution_b, index 1).
    si_messages = captured_messages[1]
    last_user_content = si_messages[-1]["content"]
    assert "Verification for solution 1" in last_user_content


# ── run_method dispatch ──────────────────────────────────────────────────────


def test_run_method_dispatches_to_correct_runner(tmp_path: Path) -> None:
    """run_method must dispatch to the correct method runner."""
    config = _make_config()
    work_item = _make_work_item()
    log_path = tmp_path / "log.jsonl"
    call_logger = TaskCallLogger(task_id=1, task_log_path=log_path)

    with patch(
        "open_deep_think.scripts.ablation_compare_matches.call_model",
        side_effect=_mock_call_model_merge,
    ):
        result = run_method(
            method="clean_merge",
            work_item=work_item,
            config=config,
            call_logger=call_logger,
        )

    assert "result_solution_text" in result


def test_run_method_raises_on_unknown_method(tmp_path: Path) -> None:
    """run_method must raise ValueError for unknown method names."""
    config = _make_config()
    work_item = _make_work_item()
    log_path = tmp_path / "log.jsonl"
    call_logger = TaskCallLogger(task_id=1, task_log_path=log_path)

    with pytest.raises(ValueError, match="Unknown method"):
        run_method(
            method="nonexistent_method",
            work_item=work_item,
            config=config,
            call_logger=call_logger,
        )


# ── process_pair ─────────────────────────────────────────────────────────────


def test_process_pair_runs_all_six_methods(tmp_path: Path) -> None:
    """process_pair must produce exactly 6 result records (one per method)."""
    config = _make_config()
    work_item = _make_work_item()
    results_path = tmp_path / "results.jsonl"
    llm_path = tmp_path / "llm_outputs.jsonl"
    results_writer = ThreadSafeJSONLWriter(results_path)
    llm_writer = ThreadSafeJSONLWriter(llm_path)

    call_count = 0

    def _mock_call(**_kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        phase = _kwargs.get("phase", "")
        if "select" in str(phase) and "si" not in str(phase):
            result.text = "1"
        else:
            result.text = "Solution output"
        result.call_id = call_count
        return result

    with patch(
        "open_deep_think.scripts.ablation_compare_matches.call_model",
        side_effect=_mock_call,
    ):
        records = process_pair(
            work_item=work_item,
            config=config,
            run_dir=tmp_path,
            results_writer=results_writer,
            llm_writer=llm_writer,
        )

    assert len(records) == 6
    methods_run = [r["method"] for r in records]
    assert methods_run == list(ALL_METHODS)
    assert all(r["status"] == "ok" for r in records)

    # Results should also be written to the JSONL file.
    written = results_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(written) == 6


def test_process_pair_handles_method_errors(tmp_path: Path) -> None:
    """process_pair must record errors without stopping other methods."""
    config = _make_config()
    work_item = _make_work_item()
    results_path = tmp_path / "results.jsonl"
    llm_path = tmp_path / "llm_outputs.jsonl"
    results_writer = ThreadSafeJSONLWriter(results_path)
    llm_writer = ThreadSafeJSONLWriter(llm_path)

    call_count = 0

    def _mock_call(**_kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        phase = _kwargs.get("phase", "")
        if phase == "clean_select":
            msg = "API error"
            raise RuntimeError(msg)
        result = MagicMock()
        result.text = "1" if "select" in str(phase) and "si" not in str(phase) else "Solution"
        result.call_id = call_count
        return result

    with patch(
        "open_deep_think.scripts.ablation_compare_matches.call_model",
        side_effect=_mock_call,
    ):
        records = process_pair(
            work_item=work_item,
            config=config,
            run_dir=tmp_path,
            results_writer=results_writer,
            llm_writer=llm_writer,
        )

    # All 6 methods should still be attempted.
    assert len(records) == 6
    # clean_select should have error status.
    clean_select_rec = next(r for r in records if r["method"] == "clean_select")
    assert clean_select_rec["status"] == "error"
    assert "API error" in clean_select_rec["error"]


def test_process_pair_logs_correct_metadata(tmp_path: Path) -> None:
    """process_pair must log task_id and candidate indices in each record."""
    config = _make_config()
    work_item = _make_work_item(task_id=42, a_idx=3, b_idx=5)
    results_path = tmp_path / "results.jsonl"
    llm_path = tmp_path / "llm_outputs.jsonl"
    results_writer = ThreadSafeJSONLWriter(results_path)
    llm_writer = ThreadSafeJSONLWriter(llm_path)

    def _mock_call(**_kwargs: object) -> MagicMock:
        result = MagicMock()
        phase = _kwargs.get("phase", "")
        result.text = "1" if "select" in str(phase) and "si" not in str(phase) else "Solution"
        result.call_id = 1
        return result

    with patch(
        "open_deep_think.scripts.ablation_compare_matches.call_model",
        side_effect=_mock_call,
    ):
        records = process_pair(
            work_item=work_item,
            config=config,
            run_dir=tmp_path,
            results_writer=results_writer,
            llm_writer=llm_writer,
        )

    for rec in records:
        assert rec["task_id"] == 42
        assert rec["solution_a_index"] == 3
        assert rec["solution_b_index"] == 5


# ── run_comparison ───────────────────────────────────────────────────────────


def test_run_comparison_sequential(tmp_path: Path) -> None:
    """run_comparison with concurrency=1 must process all items sequentially."""
    config = _make_config()
    items = [_make_work_item(task_id=1), _make_work_item(task_id=2)]
    results_path = tmp_path / "results.jsonl"
    llm_path = tmp_path / "llm_outputs.jsonl"
    results_writer = ThreadSafeJSONLWriter(results_path)
    llm_writer = ThreadSafeJSONLWriter(llm_path)

    def _mock_call(**_kwargs: object) -> MagicMock:
        result = MagicMock()
        phase = _kwargs.get("phase", "")
        result.text = "1" if "select" in str(phase) and "si" not in str(phase) else "Sol"
        result.call_id = 1
        return result

    with patch(
        "open_deep_think.scripts.ablation_compare_matches.call_model",
        side_effect=_mock_call,
    ):
        results = run_comparison(
            work_items=items,
            config=config,
            run_dir=tmp_path,
            results_writer=results_writer,
            llm_writer=llm_writer,
            concurrency=1,
        )

    # 2 work items x 6 methods = 12 results.
    assert len(results) == 12


def test_run_comparison_concurrent(tmp_path: Path) -> None:
    """run_comparison with concurrency>1 must process all items."""
    config = _make_config()
    items = [_make_work_item(task_id=1), _make_work_item(task_id=2)]
    results_path = tmp_path / "results.jsonl"
    llm_path = tmp_path / "llm_outputs.jsonl"
    results_writer = ThreadSafeJSONLWriter(results_path)
    llm_writer = ThreadSafeJSONLWriter(llm_path)

    def _mock_call(**_kwargs: object) -> MagicMock:
        result = MagicMock()
        phase = _kwargs.get("phase", "")
        result.text = "1" if "select" in str(phase) and "si" not in str(phase) else "Sol"
        result.call_id = 1
        return result

    with patch(
        "open_deep_think.scripts.ablation_compare_matches.call_model",
        side_effect=_mock_call,
    ):
        results = run_comparison(
            work_items=items,
            config=config,
            run_dir=tmp_path,
            results_writer=results_writer,
            llm_writer=llm_writer,
            concurrency=4,
        )

    assert len(results) == 12
    assert all(r["status"] == "ok" for r in results)


# ── ALL_METHODS constant ────────────────────────────────────────────────────


def test_all_methods_has_six_entries() -> None:
    """ALL_METHODS must contain exactly 6 method names."""
    assert len(ALL_METHODS) == 6


def test_all_methods_are_unique() -> None:
    """ALL_METHODS must not contain duplicates."""
    assert len(set(ALL_METHODS)) == len(ALL_METHODS)


def test_all_methods_match_dispatch_table() -> None:
    """Every method in ALL_METHODS must be in the dispatch table."""
    for method in ALL_METHODS:
        assert method in _METHOD_RUNNERS, f"Method {method!r} missing from _METHOD_RUNNERS"
