"""Tests for the tournament_merge_improve script helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from open_deep_think.imo_answer_bench.templates import (
    TOURNAMENT_MERGE_SYSTEM_PROMPT,
    build_tournament_merge_prompt,
)
from open_deep_think.scripts.tournament_merge_improve import (
    Candidate,
    TaskCallLogger,
    TournamentMergeImproveConfig,
    VerificationResult,
    _finish_reason,
    build_solver_messages,
    build_verification_prompt,
    call_model,
    extract_section,
    generate_candidate,
    is_power_of_two,
    is_yes_response,
    run_match,
    run_self_improvement,
    sanitize_model_name,
    write_config_if_shard_zero,
)

# ── extract_section ───────────────────────────────────────────────────────────


def test_extract_section_after_returns_text_after_marker() -> None:
    text = "Intro\nDetailed Solution\nProof body."
    assert extract_section(text, "Detailed Solution", after=True) == "Proof body."


def test_extract_section_before_returns_text_before_marker() -> None:
    text = "Summary\nDetailed Verification\nLog"
    assert extract_section(text, "Detailed Verification", after=False) == "Summary"


def test_extract_section_returns_empty_when_marker_missing() -> None:
    assert extract_section("No marker here", "Detailed Solution", after=True) == ""


# ── is_yes_response ───────────────────────────────────────────────────────────


def test_is_yes_response_detects_standalone_yes() -> None:
    assert is_yes_response("yes") is True


def test_is_yes_response_case_insensitive() -> None:
    assert is_yes_response("YES, the solution is correct") is True


def test_is_yes_response_rejects_substring() -> None:
    assert is_yes_response("yesterday") is False


# ── is_power_of_two ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64])
def test_is_power_of_two_valid(n: int) -> None:
    assert is_power_of_two(n) is True


@pytest.mark.parametrize("n", [0, 3, 5, 6, 7, 9, 10, -1, -4])
def test_is_power_of_two_invalid(n: int) -> None:
    assert is_power_of_two(n) is False


# ── sanitize_model_name ───────────────────────────────────────────────────────


def test_sanitize_model_name_replaces_slash_and_colon() -> None:
    assert sanitize_model_name("provider/model:latest") == "provider__model_latest"


def test_sanitize_model_name_no_special_chars() -> None:
    assert sanitize_model_name("mymodel") == "mymodel"


# ── build_solver_messages ─────────────────────────────────────────────────────


def test_build_solver_messages_single_problem() -> None:
    messages = build_solver_messages("Find x.", ())
    assert messages == [{"role": "user", "content": "Find x."}]


def test_build_solver_messages_with_extra_prompts() -> None:
    messages = build_solver_messages("Problem", ("Hint A", "Hint B"))
    assert messages == [
        {"role": "user", "content": "Problem"},
        {"role": "user", "content": "Hint A"},
        {"role": "user", "content": "Hint B"},
    ]


# ── build_verification_prompt ─────────────────────────────────────────────────


def test_build_verification_prompt_includes_problem() -> None:
    prompt = build_verification_prompt(
        problem_statement="Find all integers n.",
        solution_text="Method Sketch\nOverview\nDetailed Solution\nHence n=0.",
    )
    assert "Find all integers n." in prompt


def test_build_verification_prompt_passes_full_solution_text() -> None:
    """The verifier prompt must contain the entire solver response, not just the Detailed Solution."""
    full_text = "Method Sketch\nOverview of proof\nDetailed Solution\nHence n=0."
    prompt = build_verification_prompt(
        problem_statement="Find all integers n.",
        solution_text=full_text,
    )
    # Both the Method Sketch and the Detailed Solution body must appear.
    assert "Method Sketch" in prompt
    assert "Overview of proof" in prompt
    assert "Hence n=0." in prompt


def test_build_verification_prompt_works_without_detailed_solution_marker() -> None:
    """When the model omits the marker, the full text is still passed through."""
    prompt = build_verification_prompt(
        problem_statement="Find x.",
        solution_text="Just a plain solution without markers.",
    )
    assert "Find x." in prompt
    assert "Just a plain solution without markers." in prompt


# ── build_tournament_merge_prompt ─────────────────────────────────────────────


def test_build_tournament_merge_prompt_contains_all_sections() -> None:
    prompt = build_tournament_merge_prompt(
        problem="Prove that 1+1=2.",
        solution_1="Solution A text.",
        verification_1="Verification A text.",
        solution_2="Solution B text.",
        verification_2="Verification B text.",
    )
    assert "Prove that 1+1=2." in prompt
    assert "Solution A text." in prompt
    assert "Verification A text." in prompt
    assert "Solution B text." in prompt
    assert "Verification B text." in prompt


def test_build_tournament_merge_prompt_labels_solutions() -> None:
    """The prompt must clearly label Solution 1 and Solution 2."""
    prompt = build_tournament_merge_prompt(
        problem="P",
        solution_1="S1",
        verification_1="V1",
        solution_2="S2",
        verification_2="V2",
    )
    assert "Solution 1" in prompt
    assert "Solution 2" in prompt


def test_build_tournament_merge_prompt_labels_verification_reports() -> None:
    """The prompt must clearly label both verification reports."""
    prompt = build_tournament_merge_prompt(
        problem="P",
        solution_1="S1",
        verification_1="V1",
        solution_2="S2",
        verification_2="V2",
    )
    assert "Verification Report for Solution 1" in prompt
    assert "Verification Report for Solution 2" in prompt


# ── TOURNAMENT_MERGE_SYSTEM_PROMPT ────────────────────────────────────────────


def test_tournament_merge_system_prompt_is_non_empty_string() -> None:
    assert isinstance(TOURNAMENT_MERGE_SYSTEM_PROMPT, str)
    assert len(TOURNAMENT_MERGE_SYSTEM_PROMPT) > 0


def test_tournament_merge_system_prompt_mentions_merge_goal() -> None:
    """The system prompt should instruct the model to synthesise / merge solutions."""
    lower = TOURNAMENT_MERGE_SYSTEM_PROMPT.lower()
    assert "merge" in lower or "synthesis" in lower or "combin" in lower


# ── TournamentMergeImproveConfig ──────────────────────────────────────────────

_NUM_SOLUTIONS = 4


def test_tournament_merge_improve_config_fields() -> None:
    """TournamentMergeImproveConfig must expose all expected fields."""
    cfg = TournamentMergeImproveConfig(
        solver_model="solver",
        verifier_model="verifier",
        classifier_model="classifier",
        merger_model="merger",
        solver_max_tokens=1000,
        verifier_max_tokens=2000,
        classifier_max_tokens=500,
        merger_max_tokens=3000,
        num_solutions=_NUM_SOLUTIONS,
        temperature=0.7,
        top_p=1.0,
        other_prompts=(),
    )
    assert cfg.solver_model == "solver"
    assert cfg.merger_model == "merger"
    assert cfg.num_solutions == _NUM_SOLUTIONS
    assert cfg.si_rounds == 1  # default


def test_tournament_merge_improve_config_si_rounds_custom() -> None:
    """TournamentMergeImproveConfig must accept a custom si_rounds value."""
    cfg = TournamentMergeImproveConfig(
        solver_model="s",
        verifier_model="v",
        classifier_model="c",
        merger_model="m",
        solver_max_tokens=100,
        verifier_max_tokens=100,
        classifier_max_tokens=100,
        merger_max_tokens=100,
        num_solutions=2,
        temperature=None,
        top_p=None,
        other_prompts=(),
        si_rounds=3,
    )
    assert cfg.si_rounds == 3


# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_config() -> TournamentMergeImproveConfig:
    """Return a minimal TournamentMergeImproveConfig for testing."""
    return TournamentMergeImproveConfig(
        solver_model="solver",
        verifier_model="verifier",
        classifier_model="classifier",
        merger_model="merger",
        solver_max_tokens=100,
        verifier_max_tokens=100,
        classifier_max_tokens=100,
        merger_max_tokens=100,
        num_solutions=2,
        temperature=None,
        top_p=None,
        other_prompts=(),
    )


def _make_verification(*, is_pass: bool) -> VerificationResult:
    """Return a VerificationResult stub."""
    return VerificationResult(
        is_pass=is_pass,
        bug_report="" if is_pass else "bug",
        verifier_output="ok" if is_pass else "fail",
        classifier_output="yes" if is_pass else "no",
        verifier_call_id=10,
        classifier_call_id=11,
    )


# ── run_self_improvement (unit-level, mocked) ─────────────────────────────────


def test_run_self_improvement_calls_solver_and_verifier() -> None:
    """run_self_improvement must call the solver (self-improve) then run_verification."""
    config = _make_config()

    fake_call_result = MagicMock()
    fake_call_result.text = "Improved solution text."
    fake_call_result.completion = None
    fake_call_result.call_id = 42

    fake_verification = _make_verification(is_pass=True)
    call_logger = MagicMock()

    with (
        patch(
            "open_deep_think.scripts.tournament_merge_improve.call_model",
            return_value=fake_call_result,
        ) as mock_call,
        patch(
            "open_deep_think.scripts.tournament_merge_improve.run_verification",
            return_value=fake_verification,
        ) as mock_verify,
    ):
        improved, verification = run_self_improvement(
            task_id=0,
            candidate_index=0,
            problem_statement="Solve x.",
            solution_text="Initial solution.",
            config=config,
            call_logger=call_logger,
            round_index=None,
        )

    mock_call.assert_called_once()
    mock_verify.assert_called_once()
    assert improved.text == "Improved solution text."
    assert verification.is_pass is True


def test_generate_candidate_calls_self_improvement_when_first_verify_fails() -> None:
    """generate_candidate must call run_self_improvement when the first verify fails."""
    config = _make_config()

    fake_initial = MagicMock()
    fake_initial.text = "Initial solution."
    fake_initial.completion = None
    fake_initial.call_id = 1

    fake_improved = MagicMock()
    fake_improved.text = "Improved solution."
    fake_improved.completion = None
    fake_improved.call_id = 5

    fake_verification = _make_verification(is_pass=False)
    fake_post_verification = _make_verification(is_pass=True)

    call_logger = MagicMock()

    with (
        patch(
            "open_deep_think.scripts.tournament_merge_improve.call_model",
            return_value=fake_initial,
        ),
        patch(
            "open_deep_think.scripts.tournament_merge_improve.run_verification",
            return_value=fake_verification,
        ),
        patch(
            "open_deep_think.scripts.tournament_merge_improve.run_self_improvement",
            return_value=(fake_improved, fake_post_verification),
        ) as mock_improve,
    ):
        candidate = generate_candidate(
            task_id=0,
            candidate_index=1,
            problem_statement="Solve x.",
            config=config,
            call_logger=call_logger,
        )

    mock_improve.assert_called_once()
    assert candidate.solution_text == "Improved solution."
    assert candidate.verification.is_pass is True


def test_generate_candidate_skips_self_improvement_when_first_verify_passes() -> None:
    """generate_candidate must skip self-improvement when the first verify passes."""
    config = _make_config()

    fake_initial = MagicMock()
    fake_initial.text = "Initial solution."
    fake_initial.completion = None
    fake_initial.call_id = 1

    fake_verification = _make_verification(is_pass=True)

    call_logger = MagicMock()

    with (
        patch(
            "open_deep_think.scripts.tournament_merge_improve.call_model",
            return_value=fake_initial,
        ),
        patch(
            "open_deep_think.scripts.tournament_merge_improve.run_verification",
            return_value=fake_verification,
        ),
        patch(
            "open_deep_think.scripts.tournament_merge_improve.run_self_improvement",
        ) as mock_improve,
    ):
        candidate = generate_candidate(
            task_id=0,
            candidate_index=1,
            problem_statement="Solve x.",
            config=config,
            call_logger=call_logger,
        )

    mock_improve.assert_not_called()
    assert candidate.solution_text == "Initial solution."
    assert candidate.verification.is_pass is True


def test_run_match_calls_self_improvement_when_first_verify_fails() -> None:
    """run_match must call run_self_improvement when the initial merge verify fails."""
    config = _make_config()

    fake_verification = _make_verification(is_pass=False)
    fake_post_verification = _make_verification(is_pass=True)

    fake_merger_result = MagicMock()
    fake_merger_result.text = "Merged solution."
    fake_merger_result.completion = None
    fake_merger_result.call_id = 9

    fake_improved = MagicMock()
    fake_improved.text = "Improved merged solution."
    fake_improved.completion = None
    fake_improved.call_id = 13

    cand_a = Candidate(
        index=0,
        solution_text="Solution A",
        completion=None,
        verification=fake_verification,
    )
    cand_b = Candidate(
        index=1,
        solution_text="Solution B",
        completion=None,
        verification=fake_verification,
    )

    call_logger = MagicMock()

    with (
        patch(
            "open_deep_think.scripts.tournament_merge_improve.call_model",
            return_value=fake_merger_result,
        ),
        patch(
            "open_deep_think.scripts.tournament_merge_improve.run_verification",
            return_value=fake_verification,
        ),
        patch(
            "open_deep_think.scripts.tournament_merge_improve.run_self_improvement",
            return_value=(fake_improved, fake_post_verification),
        ) as mock_improve,
    ):
        result = run_match(
            task_id=0,
            round_index=0,
            match_index=0,
            candidate_a=cand_a,
            candidate_b=cand_b,
            problem_statement="Solve x.",
            config=config,
            call_logger=call_logger,
        )

    mock_improve.assert_called_once()
    assert result.solution_text == "Improved merged solution."
    assert result.verification.is_pass is True


def test_run_match_skips_self_improvement_when_first_verify_passes() -> None:
    """run_match must skip self-improvement when the initial merge verify passes."""
    config = _make_config()

    fake_verification = _make_verification(is_pass=True)

    fake_merger_result = MagicMock()
    fake_merger_result.text = "Merged solution."
    fake_merger_result.completion = None
    fake_merger_result.call_id = 9

    cand_a = Candidate(
        index=0,
        solution_text="Solution A",
        completion=None,
        verification=fake_verification,
    )
    cand_b = Candidate(
        index=1,
        solution_text="Solution B",
        completion=None,
        verification=fake_verification,
    )

    call_logger = MagicMock()

    with (
        patch(
            "open_deep_think.scripts.tournament_merge_improve.call_model",
            return_value=fake_merger_result,
        ),
        patch(
            "open_deep_think.scripts.tournament_merge_improve.run_verification",
            return_value=fake_verification,
        ),
        patch(
            "open_deep_think.scripts.tournament_merge_improve.run_self_improvement",
        ) as mock_improve,
    ):
        result = run_match(
            task_id=0,
            round_index=0,
            match_index=0,
            candidate_a=cand_a,
            candidate_b=cand_b,
            problem_statement="Solve x.",
            config=config,
            call_logger=call_logger,
        )

    mock_improve.assert_not_called()
    assert result.solution_text == "Merged solution."
    assert result.verification.is_pass is True


# ── write_config_if_shard_zero ────────────────────────────────────────────────


def test_write_config_if_shard_zero_writes_file_on_first_run(tmp_path: Path) -> None:
    """Shard 0 writes config.json when the file does not yet exist."""
    config = {"solver_model": "m", "num_solutions": 4}
    write_config_if_shard_zero(tmp_path, config, shard_index=0)
    written = json.loads((tmp_path / "config.json").read_text())
    assert written == config


def test_write_config_if_shard_zero_skips_for_non_zero_shard(tmp_path: Path) -> None:
    """Non-zero shards must not write config.json."""
    config = {"solver_model": "m", "num_solutions": 4}
    write_config_if_shard_zero(tmp_path, config, shard_index=1)
    assert not (tmp_path / "config.json").exists()


def test_write_config_if_shard_zero_passes_when_config_matches(tmp_path: Path) -> None:
    """Shard 0 must not raise when the existing config matches the current one."""
    config = {"solver_model": "m", "num_solutions": 4}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    # Should not raise.
    write_config_if_shard_zero(tmp_path, config, shard_index=0)


def test_write_config_if_shard_zero_raises_on_config_mismatch(tmp_path: Path) -> None:
    """Shard 0 must raise ValueError when the existing config differs from the current one."""
    old_config = {"solver_model": "m", "num_solutions": 4}
    new_config = {"solver_model": "m", "num_solutions": 8}
    (tmp_path / "config.json").write_text(json.dumps(old_config), encoding="utf-8")
    with pytest.raises(ValueError, match="Config mismatch"):
        write_config_if_shard_zero(tmp_path, new_config, shard_index=0)


# ── _finish_reason ────────────────────────────────────────────────────────────


def test_finish_reason_returns_none_for_none_completion() -> None:
    assert _finish_reason(None) is None


def test_finish_reason_returns_none_for_empty_choices() -> None:
    completion = MagicMock()
    completion.choices = []
    assert _finish_reason(completion) is None


def test_finish_reason_returns_stop() -> None:
    choice = MagicMock()
    choice.finish_reason = "stop"
    completion = MagicMock()
    completion.choices = [choice]
    assert _finish_reason(completion) == "stop"


def test_finish_reason_returns_length() -> None:
    choice = MagicMock()
    choice.finish_reason = "length"
    completion = MagicMock()
    completion.choices = [choice]
    assert _finish_reason(completion) == "length"


# ── call_model retry on finish_reason='length' ───────────────────────────────


def _make_completion(*, finish_reason: str, content: str) -> MagicMock:
    """Build a minimal ChatCompletion-like mock with JSON-serializable model_dump."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    completion.model_dump.return_value = {
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
    }
    return completion


def test_call_model_retries_once_on_length(tmp_path: Path) -> None:
    """call_model must retry exactly once when finish_reason is 'length'."""
    first = _make_completion(finish_reason="length", content="truncated")
    second = _make_completion(finish_reason="stop", content="complete")
    log_path = tmp_path / "log.jsonl"
    logger = TaskCallLogger(task_id=0, task_log_path=log_path)

    with patch("open_deep_think.scripts.tournament_merge_improve.chat_api_call", side_effect=[first, second]) as mock:
        result = call_model(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=None,
            top_p=None,
            phase="test",
            candidate_index=0,
            round_index=None,
            call_logger=logger,
        )

    assert mock.call_count == 2
    assert result.text == "complete"


def test_call_model_no_retry_on_stop(tmp_path: Path) -> None:
    """call_model must NOT retry when finish_reason is 'stop'."""
    comp = _make_completion(finish_reason="stop", content="done")
    log_path = tmp_path / "log.jsonl"
    logger = TaskCallLogger(task_id=0, task_log_path=log_path)

    with patch("open_deep_think.scripts.tournament_merge_improve.chat_api_call", return_value=comp) as mock:
        result = call_model(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=None,
            top_p=None,
            phase="test",
            candidate_index=0,
            round_index=None,
            call_logger=logger,
        )

    assert mock.call_count == 1
    assert result.text == "done"


def test_call_model_accepts_double_length(tmp_path: Path) -> None:
    """If both attempts return finish_reason='length', the result is accepted."""
    first = _make_completion(finish_reason="length", content="trunc1")
    second = _make_completion(finish_reason="length", content="trunc2")
    log_path = tmp_path / "log.jsonl"
    logger = TaskCallLogger(task_id=0, task_log_path=log_path)

    with patch("open_deep_think.scripts.tournament_merge_improve.chat_api_call", side_effect=[first, second]) as mock:
        result = call_model(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=None,
            top_p=None,
            phase="test",
            candidate_index=0,
            round_index=None,
            call_logger=logger,
        )

    assert mock.call_count == 2
    assert result.text == "trunc2"


# ── Multi-round self-improvement tests ────────────────────────────────────────


def _make_config_with_si_rounds(si_rounds: int) -> TournamentMergeImproveConfig:
    """Return a minimal TournamentMergeImproveConfig with custom si_rounds."""
    return TournamentMergeImproveConfig(
        solver_model="solver",
        verifier_model="verifier",
        classifier_model="classifier",
        merger_model="merger",
        solver_max_tokens=100,
        verifier_max_tokens=100,
        classifier_max_tokens=100,
        merger_max_tokens=100,
        num_solutions=2,
        temperature=None,
        top_p=None,
        other_prompts=(),
        si_rounds=si_rounds,
    )


def test_generate_candidate_multiple_si_rounds_stops_on_pass() -> None:
    """generate_candidate with si_rounds=3 must stop after the first passing SI round."""
    config = _make_config_with_si_rounds(si_rounds=3)

    fake_initial = MagicMock()
    fake_initial.text = "Initial solution."
    fake_initial.completion = None
    fake_initial.call_id = 1

    fake_improved_1 = MagicMock()
    fake_improved_1.text = "Improved round 1."
    fake_improved_1.completion = None
    fake_improved_1.call_id = 5

    fail_verification = _make_verification(is_pass=False)
    pass_verification = _make_verification(is_pass=True)
    call_logger = MagicMock()

    with (
        patch(
            "open_deep_think.scripts.tournament_merge_improve.call_model",
            return_value=fake_initial,
        ),
        patch(
            "open_deep_think.scripts.tournament_merge_improve.run_verification",
            return_value=fail_verification,
        ),
        patch(
            "open_deep_think.scripts.tournament_merge_improve.run_self_improvement",
            return_value=(fake_improved_1, pass_verification),
        ) as mock_improve,
    ):
        candidate = generate_candidate(
            task_id=0,
            candidate_index=0,
            problem_statement="Solve x.",
            config=config,
            call_logger=call_logger,
        )

    # Only 1 SI round needed — the second round sees pass and breaks.
    mock_improve.assert_called_once()
    assert candidate.solution_text == "Improved round 1."
    assert candidate.verification.is_pass is True


def test_generate_candidate_exhausts_all_si_rounds_when_always_failing() -> None:
    """generate_candidate with si_rounds=3 must run all 3 rounds when all verifications fail."""
    config = _make_config_with_si_rounds(si_rounds=3)

    fake_initial = MagicMock()
    fake_initial.text = "Initial solution."
    fake_initial.completion = None
    fake_initial.call_id = 1

    fake_improved = MagicMock()
    fake_improved.text = "Still wrong."
    fake_improved.completion = None
    fake_improved.call_id = 10

    fail_verification = _make_verification(is_pass=False)
    call_logger = MagicMock()

    with (
        patch(
            "open_deep_think.scripts.tournament_merge_improve.call_model",
            return_value=fake_initial,
        ),
        patch(
            "open_deep_think.scripts.tournament_merge_improve.run_verification",
            return_value=fail_verification,
        ),
        patch(
            "open_deep_think.scripts.tournament_merge_improve.run_self_improvement",
            return_value=(fake_improved, fail_verification),
        ) as mock_improve,
    ):
        candidate = generate_candidate(
            task_id=0,
            candidate_index=0,
            problem_statement="Solve x.",
            config=config,
            call_logger=call_logger,
        )

    assert mock_improve.call_count == 3
    assert candidate.solution_text == "Still wrong."
    assert candidate.verification.is_pass is False


def test_run_match_multiple_si_rounds_stops_on_pass() -> None:
    """run_match with si_rounds=2 must stop after the first passing SI round."""
    config = _make_config_with_si_rounds(si_rounds=2)

    fail_verification = _make_verification(is_pass=False)
    pass_verification = _make_verification(is_pass=True)

    fake_merger_result = MagicMock()
    fake_merger_result.text = "Merged solution."
    fake_merger_result.completion = None
    fake_merger_result.call_id = 9

    fake_improved = MagicMock()
    fake_improved.text = "Improved merged."
    fake_improved.completion = None
    fake_improved.call_id = 13

    cand_a = Candidate(index=0, solution_text="A", completion=None, verification=fail_verification)
    cand_b = Candidate(index=1, solution_text="B", completion=None, verification=fail_verification)
    call_logger = MagicMock()

    with (
        patch(
            "open_deep_think.scripts.tournament_merge_improve.call_model",
            return_value=fake_merger_result,
        ),
        patch(
            "open_deep_think.scripts.tournament_merge_improve.run_verification",
            return_value=fail_verification,
        ),
        patch(
            "open_deep_think.scripts.tournament_merge_improve.run_self_improvement",
            return_value=(fake_improved, pass_verification),
        ) as mock_improve,
    ):
        result = run_match(
            task_id=0,
            round_index=0,
            match_index=0,
            candidate_a=cand_a,
            candidate_b=cand_b,
            problem_statement="Solve x.",
            config=config,
            call_logger=call_logger,
        )

    # First SI round passes → second round is skipped.
    mock_improve.assert_called_once()
    assert result.solution_text == "Improved merged."
    assert result.verification.is_pass is True


def test_run_match_exhausts_all_si_rounds_when_always_failing() -> None:
    """run_match with si_rounds=3 must run all 3 rounds when verification never passes."""
    config = _make_config_with_si_rounds(si_rounds=3)

    fail_verification = _make_verification(is_pass=False)

    fake_merger_result = MagicMock()
    fake_merger_result.text = "Merged solution."
    fake_merger_result.completion = None
    fake_merger_result.call_id = 9

    fake_improved = MagicMock()
    fake_improved.text = "Still broken."
    fake_improved.completion = None
    fake_improved.call_id = 20

    cand_a = Candidate(index=0, solution_text="A", completion=None, verification=fail_verification)
    cand_b = Candidate(index=1, solution_text="B", completion=None, verification=fail_verification)
    call_logger = MagicMock()

    with (
        patch(
            "open_deep_think.scripts.tournament_merge_improve.call_model",
            return_value=fake_merger_result,
        ),
        patch(
            "open_deep_think.scripts.tournament_merge_improve.run_verification",
            return_value=fail_verification,
        ),
        patch(
            "open_deep_think.scripts.tournament_merge_improve.run_self_improvement",
            return_value=(fake_improved, fail_verification),
        ) as mock_improve,
    ):
        result = run_match(
            task_id=0,
            round_index=0,
            match_index=0,
            candidate_a=cand_a,
            candidate_b=cand_b,
            problem_statement="Solve x.",
            config=config,
            call_logger=call_logger,
        )

    assert mock_improve.call_count == 3
    assert result.solution_text == "Still broken."
    assert result.verification.is_pass is False
