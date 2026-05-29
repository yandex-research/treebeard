"""Shared fixtures for the tournament viewer tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _make_initial_candidate(index: int, *, verifier_pass: bool) -> dict[str, Any]:
    """Build a minimal initial-candidate payload for ``progress.json``."""
    return {
        "candidate_index": index,
        "status": "ok",
        "verification": {
            "is_pass": verifier_pass,
            "bug_report": "" if verifier_pass else f"bug-{index}",
            "verifier_output": f"verifier-{index}",
            "classifier_output": "yes" if verifier_pass else "no",
            "verifier_call_id": index * 3 + 2,
            "classifier_call_id": index * 3 + 3,
        },
    }


@pytest.fixture
def sample_progress() -> dict[str, Any]:
    """Return a canonical 4-candidate tournament progress payload (two rounds)."""
    return {
        "task_id": 42,
        "started_at": "2026-05-22T19:58:00+00:00",
        "completed_at": "2026-05-22T20:27:59+00:00",
        "pipeline_config": {
            "solver_model": "openai/gpt-oss-120b",
            "verifier_model": "openai/gpt-oss-120b",
            "classifier_model": "openai/gpt-oss-120b",
            "merger_model": "openai/gpt-oss-120b",
            "solver_max_tokens": 100000,
            "verifier_max_tokens": 100000,
            "classifier_max_tokens": 100000,
            "merger_max_tokens": 100000,
            "num_solutions": 4,
            "temperature": 1.0,
            "top_p": 1.0,
            "other_prompts": [],
        },
        "problem_statement": "Determine the maximum value of $m$.",
        "candidates": [
            _make_initial_candidate(0, verifier_pass=True),
            _make_initial_candidate(1, verifier_pass=False),
            _make_initial_candidate(2, verifier_pass=True),
            _make_initial_candidate(3, verifier_pass=True),
        ],
        "tournament_rounds": [
            {
                "round_index": 0,
                "matches": [
                    {
                        "match_index": 0,
                        "candidate_a": 0,
                        "candidate_b": 1,
                        "merged_index": -1,
                        "merged_verification_pass": True,
                        "status": "ok",
                    },
                    {
                        "match_index": 1,
                        "candidate_a": 2,
                        "candidate_b": 3,
                        "merged_index": -2,
                        "merged_verification_pass": True,
                        "status": "ok",
                    },
                ],
            },
            {
                "round_index": 1,
                "matches": [
                    {
                        "match_index": 0,
                        "candidate_a": -1,
                        "candidate_b": -2,
                        "merged_index": -1001,
                        "merged_verification_pass": True,
                        "status": "ok",
                    },
                ],
            },
        ],
        "status": "success",
        "final_candidate_index": -1001,
    }


def _make_call(call_id: int, **overrides: Any) -> dict[str, Any]:  # noqa: ANN401
    """Build a minimal JSONL call payload, with overridable fields."""
    base = {
        "timestamp": "2026-05-22T20:00:00+00:00",
        "task_id": 42,
        "call_id": call_id,
        "phase": "initial_solution",
        "candidate_index": 0,
        "round_index": None,
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": "You are a math solver."},
            {"role": "user", "content": "Solve me."},
        ],
        "response_text": f"response-{call_id}",
        "completion": {
            "choices": [
                {
                    "message": {
                        "content": f"response-{call_id}",
                        "reasoning": f"reasoning-{call_id}",
                    },
                },
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        },
        "error": None,
    }
    base.update(overrides)
    return base


@pytest.fixture
def sample_calls() -> list[dict[str, Any]]:
    """Return a small JSONL payload exercising every phase used by the viewer."""
    calls: list[dict[str, Any]] = []
    call_id = 1
    for candidate_index in range(4):
        calls.append(_make_call(call_id, phase="initial_solution", candidate_index=candidate_index))
        calls.append(_make_call(call_id + 1, phase="verification", candidate_index=candidate_index))
        calls.append(_make_call(call_id + 2, phase="verification_check", candidate_index=candidate_index))
        call_id += 3
    # Round 0 merges
    calls.append(_make_call(call_id, phase="tournament_merge", candidate_index=None, round_index=0))
    calls.append(_make_call(call_id + 1, phase="verification", candidate_index=-1))
    calls.append(_make_call(call_id + 2, phase="verification_check", candidate_index=-1))
    call_id += 3
    calls.append(_make_call(call_id, phase="tournament_merge", candidate_index=None, round_index=0))
    calls.append(_make_call(call_id + 1, phase="verification", candidate_index=-2))
    calls.append(_make_call(call_id + 2, phase="verification_check", candidate_index=-2))
    call_id += 3
    # Round 1 merge
    calls.append(_make_call(call_id, phase="tournament_merge", candidate_index=None, round_index=1))
    calls.append(_make_call(call_id + 1, phase="verification", candidate_index=-1001))
    calls.append(_make_call(call_id + 2, phase="verification_check", candidate_index=-1001))
    return calls


@pytest.fixture
def run_dir(tmp_path: Path, sample_progress: dict[str, Any], sample_calls: list[dict[str, Any]]) -> Path:
    """Materialise a single-run directory on disk for one task (id 42)."""
    run_path = tmp_path / "sample_run"
    run_path.mkdir()
    (run_path / "Task_42_progress.json").write_text(json.dumps(sample_progress), encoding="utf-8")
    with (run_path / "Task_42_llm_outputs.jsonl").open("w", encoding="utf-8") as fh:
        for call in sample_calls:
            fh.write(json.dumps(call) + "\n")
    (run_path / "Task_42_solution.txt").write_text("Final answer is 49.", encoding="utf-8")
    return run_path
