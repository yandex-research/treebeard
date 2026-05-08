"""Tests for the IMO25 solver script helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from open_deep_think.scripts.imo25_solve import (
    build_solver_messages,
    build_verification_prompt,
    extract_section,
    is_yes_response,
    sanitize_model_name,
    write_config_if_shard_zero,
)


def test_extract_section_after_returns_text_after_marker() -> None:
    text = "Intro\nDetailed Solution\nProof body."
    assert extract_section(text, "Detailed Solution", after=True) == "Proof body."


def test_extract_section_before_returns_text_before_marker() -> None:
    text = "Summary\nDetailed Verification\nLog"
    assert extract_section(text, "Detailed Verification", after=False) == "Summary"


def test_extract_section_returns_empty_when_marker_missing() -> None:
    assert extract_section("No marker", "Detailed Solution", after=True) == ""


def test_build_solver_messages_preserves_prompt_order() -> None:
    messages = build_solver_messages("Problem", ("Prompt A", "Prompt B"))
    assert messages == [
        {"role": "user", "content": "Problem"},
        {"role": "user", "content": "Prompt A"},
        {"role": "user", "content": "Prompt B"},
    ]


def test_build_verification_prompt_includes_problem_and_solution() -> None:
    prompt = build_verification_prompt(
        problem_statement="Find x.",
        solution_text="Summary\nDetailed Solution\nHence x=3.",
    )
    assert "Find x." in prompt
    assert "Hence x=3." in prompt


def test_is_yes_response_requires_word_boundary() -> None:
    assert is_yes_response("yes") is True
    assert is_yes_response("YES, correct") is True
    assert is_yes_response("yesterday") is False


def test_sanitize_model_name_replaces_path_and_colon() -> None:
    assert sanitize_model_name("provider/model:latest") == "provider__model_latest"


# ── write_config_if_shard_zero ────────────────────────────────────────────────


def test_write_config_if_shard_zero_writes_file_on_first_run(tmp_path: Path) -> None:
    """Shard 0 writes config.json when the file does not yet exist."""
    config = {"solver_model": "m", "max_runs": 10}
    write_config_if_shard_zero(tmp_path, config, shard_index=0)
    written = json.loads((tmp_path / "config.json").read_text())
    assert written == config


def test_write_config_if_shard_zero_skips_for_non_zero_shard(tmp_path: Path) -> None:
    """Non-zero shards must not write config.json."""
    config = {"solver_model": "m", "max_runs": 10}
    write_config_if_shard_zero(tmp_path, config, shard_index=2)
    assert not (tmp_path / "config.json").exists()


def test_write_config_if_shard_zero_passes_when_config_matches(tmp_path: Path) -> None:
    """Shard 0 must not raise when the existing config matches the current one."""
    config = {"solver_model": "m", "max_runs": 10}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    # Should not raise.
    write_config_if_shard_zero(tmp_path, config, shard_index=0)


def test_write_config_if_shard_zero_raises_on_config_mismatch(tmp_path: Path) -> None:
    """Shard 0 must raise ValueError when the existing config differs from the current one."""
    old_config = {"solver_model": "m", "max_runs": 10}
    new_config = {"solver_model": "m", "max_runs": 5}
    (tmp_path / "config.json").write_text(json.dumps(old_config), encoding="utf-8")
    with pytest.raises(ValueError, match="Config mismatch"):
        write_config_if_shard_zero(tmp_path, new_config, shard_index=0)
