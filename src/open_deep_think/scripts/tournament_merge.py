"""Tournament-merge solution pipeline on IMO AnswerBench.

This script generates N independent solutions (N must be a power of 2), verifies each
one, then runs a single-elimination tournament where, instead of selecting the better
solution, a merger model synthesises a *new* solution from the two candidates and their
verification reports.  The merged solution is then verified before advancing to the next
round.

Pipeline overview:
1. Generate N independent solutions (same first-iteration prompt as IMO25).
2. Verify every solution with the IMO25 verifier + binary correctness classifier.
3. Run ceil(log2(N)) tournament rounds:
   - Split the current pool into consecutive pairs.
   - For each pair, call the merger model to produce a combined solution.
   - Verify the merged solution.
   - The merged candidate advances to the next round.
4. The last remaining (merged) solution is the final answer.

Outputs are saved per task with files compatible with `scripts/evaluate.py`:
- `Task_{id}_solution.txt`
- `Task_{id}_reasoning.txt`
- `Task_{id}_response.json`
- `Task_{id}_progress.json`
- `Task_{id}_llm_outputs.jsonl`
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import urllib3
from datasets import load_dataset

from open_deep_think.api import chat_api_call
from open_deep_think.imo_answer_bench.extract import extract_reasoning, extract_solution
from open_deep_think.imo_answer_bench.templates import (
    IMO25_BINARY_CORRECTNESS_PROMPT,
    IMO25_STEP1_SYSTEM_PROMPT,
    IMO25_VERIFICATION_REMINDER,
    IMO25_VERIFICATION_SYSTEM_PROMPT,
    TOURNAMENT_MERGE_SYSTEM_PROMPT,
    build_tournament_merge_prompt,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

LOGGER = logging.getLogger(__name__)
_YES_PATTERN = re.compile(r"\byes\b", re.IGNORECASE)
_MIN_CANDIDATES_FOR_TOURNAMENT = 2


@dataclass(frozen=True)
class TournamentMergeConfig:
    """Configuration for the tournament-merge pipeline."""

    solver_model: str
    verifier_model: str
    classifier_model: str
    merger_model: str
    solver_max_tokens: int
    verifier_max_tokens: int
    classifier_max_tokens: int
    merger_max_tokens: int
    num_solutions: int
    temperature: float | None
    top_p: float | None
    other_prompts: tuple[str, ...]


@dataclass(frozen=True)
class TaskFiles:
    """Filesystem paths for per-task outputs."""

    solution: Path
    reasoning: Path
    response: Path
    progress: Path
    llm_outputs: Path


@dataclass(frozen=True)
class VerificationResult:
    """Result of a verification pass."""

    is_pass: bool
    bug_report: str
    verifier_output: str
    classifier_output: str
    verifier_call_id: int
    classifier_call_id: int


@dataclass(frozen=True)
class CallResult:
    """Result of a single model call."""

    completion: ChatCompletion | None
    text: str
    call_id: int


@dataclass
class Candidate:
    """A solution candidate with its verification result."""

    index: int
    solution_text: str
    completion: ChatCompletion | None
    verification: VerificationResult


class TaskCallLogger:
    """Persist every per-task LLM call to JSONL files."""

    def __init__(self, task_id: int, task_log_path: Path, global_log_path: Path) -> None:
        self._task_id = task_id
        self._task_log_path = task_log_path
        self._global_log_path = global_log_path
        self._next_call_id = 1
        self._task_log_path.write_text("", encoding="utf-8")

    def record(  # noqa: PLR0913
        self,
        *,
        phase: str,
        candidate_index: int | None,
        round_index: int | None,
        model: str,
        messages: list[dict[str, str]],
        completion: ChatCompletion | None,
        response_text: str,
        error: str | None = None,
    ) -> int:
        """Record a model interaction to task and global JSONL logs."""
        call_id = self._next_call_id
        self._next_call_id += 1
        payload = {
            "timestamp": utc_now_iso(),
            "task_id": self._task_id,
            "call_id": call_id,
            "phase": phase,
            "candidate_index": candidate_index,
            "round_index": round_index,
            "model": model,
            "messages": messages,
            "response_text": response_text,
            "completion": completion.model_dump() if completion is not None else None,
            "error": error,
        }
        append_jsonl(self._task_log_path, payload)
        append_jsonl(self._global_log_path, payload)
        return call_id


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def sanitize_model_name(model: str) -> str:
    """Convert a model identifier to a filesystem-safe slug."""
    return model.replace("/", "__").replace(":", "_")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object line to a JSONL file."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON data with UTF-8 encoding and indentation."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def completion_text(completion: ChatCompletion) -> str:
    """Extract the first assistant message content from a completion."""
    if not completion.choices:
        return ""
    message = completion.choices[0].message
    return message.content if message.content is not None else ""


def extract_section(text: str, marker: str, *, after: bool) -> str:
    """Extract a section before or after a marker.

    Args:
        text: Source text.
        marker: Marker to look up.
        after: If True, return text after the marker; else return text before it.

    Returns:
        Trimmed slice or empty string if marker is absent.

    """
    marker_index = text.find(marker)
    if marker_index == -1:
        return ""
    if after:
        return text[marker_index + len(marker) :].strip()
    return text[:marker_index].strip()


def is_yes_response(text: str) -> bool:
    """Return True if the text contains a standalone 'yes' token."""
    return _YES_PATTERN.search(text) is not None


def is_power_of_two(n: int) -> bool:
    """Return True if n is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def build_solver_messages(problem_statement: str, other_prompts: tuple[str, ...]) -> list[dict[str, str]]:
    """Build the base user message sequence for solver calls."""
    messages: list[dict[str, str]] = [{"role": "user", "content": problem_statement}]
    messages.extend({"role": "user", "content": prompt} for prompt in other_prompts)
    return messages


def build_verification_prompt(problem_statement: str, solution_text: str) -> str:
    """Build the verifier user prompt exactly as in the official pipeline."""
    detailed_solution = extract_section(solution_text, marker="Detailed Solution", after=True)
    return f"""
======================================================================
### Problem ###

{problem_statement}

======================================================================
### Solution ###

{detailed_solution}

{IMO25_VERIFICATION_REMINDER}
"""


def call_model(  # noqa: PLR0913
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    phase: str,
    candidate_index: int | None,
    round_index: int | None,
    call_logger: TaskCallLogger,
) -> CallResult:
    """Call a chat model and persist the full request/response to JSONL logs."""
    completion: ChatCompletion | None = None
    response_text = ""
    try:
        completion = chat_api_call(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        response_text = completion_text(completion)
        call_id = call_logger.record(
            phase=phase,
            candidate_index=candidate_index,
            round_index=round_index,
            model=model,
            messages=messages,
            completion=completion,
            response_text=response_text,
        )
    except Exception as error:
        call_id = call_logger.record(
            phase=phase,
            candidate_index=candidate_index,
            round_index=round_index,
            model=model,
            messages=messages,
            completion=completion,
            response_text=response_text,
            error=str(error),
        )
        raise
    return CallResult(completion=completion, text=response_text, call_id=call_id)


def run_verification(  # noqa: PLR0913
    *,
    task_id: int,
    candidate_index: int,
    problem_statement: str,
    solution_text: str,
    config: TournamentMergeConfig,
    call_logger: TaskCallLogger,
) -> VerificationResult:
    """Run the verifier + binary correctness checks and derive a bug report.

    Args:
        task_id: Identifier of the current task (for logging).
        candidate_index: Zero-based index of the candidate being verified.
        problem_statement: The problem to verify against.
        solution_text: Full solution text produced by the solver or merger.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.

    Returns:
        A :class:`VerificationResult` with pass/fail status and bug report.

    """
    verification_prompt = build_verification_prompt(problem_statement, solution_text)
    verifier_messages = [
        {"role": "system", "content": IMO25_VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": verification_prompt},
    ]
    verifier_result = call_model(
        model=config.verifier_model,
        messages=verifier_messages,
        max_tokens=config.verifier_max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        phase="verification",
        candidate_index=candidate_index,
        round_index=None,
        call_logger=call_logger,
    )

    classifier_prompt = f"{IMO25_BINARY_CORRECTNESS_PROMPT}\n\n{verifier_result.text}"
    classifier_messages = [{"role": "user", "content": classifier_prompt}]
    classifier_result = call_model(
        model=config.classifier_model,
        messages=classifier_messages,
        max_tokens=config.classifier_max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        phase="verification_check",
        candidate_index=candidate_index,
        round_index=None,
        call_logger=call_logger,
    )

    passed = is_yes_response(classifier_result.text)
    bug_report = "" if passed else extract_section(verifier_result.text, marker="Detailed Verification", after=False)
    LOGGER.info(
        "Task %s candidate %s verification=%s",
        task_id,
        candidate_index,
        "pass" if passed else "fail",
    )
    return VerificationResult(
        is_pass=passed,
        bug_report=bug_report,
        verifier_output=verifier_result.text,
        classifier_output=classifier_result.text,
        verifier_call_id=verifier_result.call_id,
        classifier_call_id=classifier_result.call_id,
    )


def generate_candidate(
    *,
    task_id: int,
    candidate_index: int,
    problem_statement: str,
    config: TournamentMergeConfig,
    call_logger: TaskCallLogger,
) -> Candidate:
    """Generate one solution candidate and verify it.

    Args:
        task_id: Identifier of the current task (for logging).
        candidate_index: Zero-based index of this candidate in the pool.
        problem_statement: The problem to solve.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.

    Returns:
        A fully populated :class:`Candidate` with solution text and verification.

    """
    LOGGER.info("Task %s generating candidate %s", task_id, candidate_index)
    solver_base_messages = build_solver_messages(problem_statement, config.other_prompts)
    result = call_model(
        model=config.solver_model,
        messages=[
            {"role": "system", "content": IMO25_STEP1_SYSTEM_PROMPT},
            *solver_base_messages,
        ],
        max_tokens=config.solver_max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        phase="initial_solution",
        candidate_index=candidate_index,
        round_index=None,
        call_logger=call_logger,
    )
    verification = run_verification(
        task_id=task_id,
        candidate_index=candidate_index,
        problem_statement=problem_statement,
        solution_text=result.text,
        config=config,
        call_logger=call_logger,
    )
    return Candidate(
        index=candidate_index,
        solution_text=result.text,
        completion=result.completion,
        verification=verification,
    )


def run_match(  # noqa: PLR0913
    *,
    task_id: int,
    round_index: int,
    match_index: int,
    candidate_a: Candidate,
    candidate_b: Candidate,
    problem_statement: str,
    config: TournamentMergeConfig,
    call_logger: TaskCallLogger,
) -> Candidate:
    """Run a single tournament-merge match between two candidates.

    The merger model is asked to synthesise a new solution from both candidates
    and their verification reports.  The merged solution is then verified before
    being returned as the match winner.

    Args:
        task_id: Identifier of the current task (for logging).
        round_index: Zero-based tournament round number.
        match_index: Zero-based match index within the round.
        candidate_a: First candidate (presented as Solution 1).
        candidate_b: Second candidate (presented as Solution 2).
        problem_statement: The original problem statement.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.

    Returns:
        A new :class:`Candidate` containing the merged solution and its verification.

    """
    LOGGER.info(
        "Task %s round %s match %s: merging candidate %s and candidate %s",
        task_id,
        round_index,
        match_index,
        candidate_a.index,
        candidate_b.index,
    )
    merge_prompt = build_tournament_merge_prompt(
        problem=problem_statement,
        solution_1=candidate_a.solution_text,
        verification_1=candidate_a.verification.verifier_output,
        solution_2=candidate_b.solution_text,
        verification_2=candidate_b.verification.verifier_output,
    )
    merger_messages = [
        {"role": "system", "content": TOURNAMENT_MERGE_SYSTEM_PROMPT},
        {"role": "user", "content": merge_prompt},
    ]
    merger_result = call_model(
        model=config.merger_model,
        messages=merger_messages,
        max_tokens=config.merger_max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        phase="tournament_merge",
        candidate_index=None,
        round_index=round_index,
        call_logger=call_logger,
    )

    # Assign a synthetic index that encodes the round and match for traceability.
    merged_index = -(round_index * 1000 + match_index + 1)
    LOGGER.info(
        "Task %s round %s match %s: verifying merged solution (index %s)",
        task_id,
        round_index,
        match_index,
        merged_index,
    )
    merged_verification = run_verification(
        task_id=task_id,
        candidate_index=merged_index,
        problem_statement=problem_statement,
        solution_text=merger_result.text,
        config=config,
        call_logger=call_logger,
    )
    LOGGER.info(
        "Task %s round %s match %s: merged solution verification=%s",
        task_id,
        round_index,
        match_index,
        "pass" if merged_verification.is_pass else "fail",
    )
    return Candidate(
        index=merged_index,
        solution_text=merger_result.text,
        completion=merger_result.completion,
        verification=merged_verification,
    )


def task_files(output_dir: Path, task_id: int) -> TaskFiles:
    """Build per-task output file paths."""
    return TaskFiles(
        solution=output_dir / f"Task_{task_id}_solution.txt",
        reasoning=output_dir / f"Task_{task_id}_reasoning.txt",
        response=output_dir / f"Task_{task_id}_response.json",
        progress=output_dir / f"Task_{task_id}_progress.json",
        llm_outputs=output_dir / f"Task_{task_id}_llm_outputs.jsonl",
    )


def save_task_outputs(
    *,
    files: TaskFiles,
    solution_text: str,
    latest_solver_completion: ChatCompletion | None,
    progress_payload: dict[str, Any],
) -> None:
    """Persist final task outputs to disk."""
    files.solution.write_text(solution_text, encoding="utf-8")

    reasoning_text = ""
    response_payload: dict[str, Any] = {}
    if latest_solver_completion is not None:
        reasoning_text = extract_reasoning(latest_solver_completion)
        response_payload = latest_solver_completion.model_dump()
    files.reasoning.write_text(reasoning_text, encoding="utf-8")
    write_json(files.response, response_payload)
    write_json(files.progress, progress_payload)


_FAILED_VERIFICATION = VerificationResult(
    is_pass=False,
    bug_report="Candidate generation failed — no solution produced.",
    verifier_output="Candidate generation failed — no solution produced.",
    classifier_output="no",
    verifier_call_id=-1,
    classifier_call_id=-1,
)
"""Sentinel :class:`VerificationResult` used for candidates that failed to generate.

A failed candidate always produces a poor merged solution because its verifier output
clearly states that no solution was produced.
"""


def _generate_all_candidates(
    *,
    task_id: int,
    problem_statement: str,
    config: TournamentMergeConfig,
    call_logger: TaskCallLogger,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Generate all candidates sequentially and collect payloads.

    Failed candidates are represented as dummy :class:`Candidate` objects with an
    empty ``solution_text`` and a failing :data:`_FAILED_VERIFICATION`, so the
    tournament can always proceed with the full bracket.

    Args:
        task_id: Dataset task identifier (for logging).
        problem_statement: Raw problem text.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.

    Returns:
        A tuple of (candidates, candidate_payloads).  ``candidates`` always has
        exactly ``config.num_solutions`` entries in index order.

    """
    candidates: list[Candidate] = []
    candidate_payloads: list[dict[str, Any]] = []

    for idx in range(config.num_solutions):
        try:
            candidate = generate_candidate(
                task_id=task_id,
                candidate_index=idx,
                problem_statement=problem_statement,
                config=config,
                call_logger=call_logger,
            )
        except Exception as exc:  # noqa: PERF203
            LOGGER.exception("Task %s candidate %s generation failed", task_id, idx)
            LOGGER.warning("Task %s candidate %s replaced with empty dummy after generation error", task_id, idx)
            dummy = Candidate(
                index=idx,
                solution_text="",
                completion=None,
                verification=_FAILED_VERIFICATION,
            )
            candidates.append(dummy)
            candidate_payloads.append(
                {"candidate_index": idx, "status": "generation_error", "error": str(exc)}
            )
        else:
            candidates.append(candidate)
            candidate_payloads.append(
                {
                    "candidate_index": idx,
                    "status": "ok",
                    "verification": {
                        "is_pass": candidate.verification.is_pass,
                        "bug_report": candidate.verification.bug_report,
                        "verifier_output": candidate.verification.verifier_output,
                        "classifier_output": candidate.verification.classifier_output,
                        "verifier_call_id": candidate.verification.verifier_call_id,
                        "classifier_call_id": candidate.verification.classifier_call_id,
                    },
                }
            )

    return candidates, candidate_payloads


def _run_tournament_rounds(
    *,
    task_id: int,
    candidates: list[Candidate],
    problem_statement: str,
    config: TournamentMergeConfig,
    call_logger: TaskCallLogger,
) -> tuple[Candidate, list[dict[str, Any]]]:
    """Run all tournament-merge rounds and return the final candidate and round payloads.

    Args:
        task_id: Dataset task identifier (for logging).
        candidates: Sorted list of candidates entering the bracket.
        problem_statement: The original problem statement.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.

    Returns:
        A tuple of (final_candidate, tournament_round_payloads).

    """
    num_rounds = int(math.log2(len(candidates)))
    pool = list(candidates)
    round_payloads: list[dict[str, Any]] = []

    for round_index in range(num_rounds):
        LOGGER.info(
            "Task %s tournament round %s/%s (%s candidates)",
            task_id,
            round_index + 1,
            num_rounds,
            len(pool),
        )
        round_payload: dict[str, Any] = {"round_index": round_index, "matches": []}
        next_pool: list[Candidate] = []

        for match_index, (cand_a, cand_b) in enumerate(zip(pool[::2], pool[1::2])):
            merged, match_status = _play_match(
                task_id=task_id,
                round_index=round_index,
                match_index=match_index,
                candidate_a=cand_a,
                candidate_b=cand_b,
                problem_statement=problem_statement,
                config=config,
                call_logger=call_logger,
            )
            round_payload["matches"].append(
                {
                    "match_index": match_index,
                    "candidate_a": cand_a.index,
                    "candidate_b": cand_b.index,
                    "merged_index": merged.index,
                    "merged_verification_pass": merged.verification.is_pass,
                    "status": match_status,
                }
            )
            next_pool.append(merged)

        round_payloads.append(round_payload)
        pool = next_pool

    assert len(pool) == 1, f"Expected exactly 1 candidate after tournament, got {len(pool)}"  # noqa: S101
    return pool[0], round_payloads


def _play_match(  # noqa: PLR0913
    *,
    task_id: int,
    round_index: int,
    match_index: int,
    candidate_a: Candidate,
    candidate_b: Candidate,
    problem_statement: str,
    config: TournamentMergeConfig,
    call_logger: TaskCallLogger,
) -> tuple[Candidate, str]:
    """Play one merge match, returning (merged_candidate, status_string).

    On error, falls back to ``candidate_a`` and returns ``"match_error"``.

    Args:
        task_id: Dataset task identifier (for logging).
        round_index: Zero-based tournament round number.
        match_index: Zero-based match index within the round.
        candidate_a: First candidate (presented as Solution 1).
        candidate_b: Second candidate (presented as Solution 2).
        problem_statement: The original problem statement.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.

    Returns:
        Tuple of (merged_or_fallback_candidate, status_string).

    """
    try:
        merged = run_match(
            task_id=task_id,
            round_index=round_index,
            match_index=match_index,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            problem_statement=problem_statement,
            config=config,
            call_logger=call_logger,
        )
    except Exception:
        LOGGER.exception(
            "Task %s round %s match %s failed — defaulting to candidate %s",
            task_id,
            round_index,
            match_index,
            candidate_a.index,
        )
        return candidate_a, "match_error"
    else:
        return merged, "ok"


def _persist_best_candidate(
    *,
    candidates: list[Candidate],
    files: TaskFiles,
    progress: dict[str, Any],
) -> None:
    """Save the best available candidate when the tournament cannot run."""
    progress["completed_at"] = utc_now_iso()
    best = candidates[0] if candidates else None
    persisted_solution = ""
    latest_completion: ChatCompletion | None = None
    if best is not None:
        persisted_solution = extract_solution(best.completion) if best.completion is not None else ""
        if not persisted_solution:
            persisted_solution = best.solution_text
        latest_completion = best.completion
    save_task_outputs(
        files=files,
        solution_text=persisted_solution,
        latest_solver_completion=latest_completion,
        progress_payload=progress,
    )


def _task_result(*, task_id: int, status: str, files: TaskFiles) -> dict[str, Any]:
    """Build the standard per-task result dict."""
    return {
        "task_id": task_id,
        "status": status,
        "solution_path": str(files.solution),
        "progress_path": str(files.progress),
        "llm_outputs_path": str(files.llm_outputs),
    }


def solve_task(
    *,
    task_id: int,
    problem_statement: str,
    config: TournamentMergeConfig,
    output_dir: Path,
    global_llm_log_path: Path,
) -> dict[str, Any]:
    """Solve one IMO AnswerBench task with the tournament-merge pipeline.

    Steps:
    1. Generate ``config.num_solutions`` independent candidates in parallel.
    2. Verify each candidate.
    3. Run a single-elimination tournament where each match merges two candidates
       into a new, verified solution.

    Args:
        task_id: Dataset task identifier.
        problem_statement: Raw problem text.
        config: Pipeline configuration.
        output_dir: Directory for per-task output files.
        global_llm_log_path: Path to the global JSONL log shared across tasks.

    Returns:
        A summary dict with task_id, status, and output file paths.

    """
    files = task_files(output_dir, task_id)
    call_logger = TaskCallLogger(task_id=task_id, task_log_path=files.llm_outputs, global_log_path=global_llm_log_path)

    progress: dict[str, Any] = {
        "task_id": task_id,
        "started_at": utc_now_iso(),
        "pipeline_config": asdict(config),
        "problem_statement": problem_statement,
        "candidates": [],
        "tournament_rounds": [],
        "status": "failed",
        "final_candidate_index": None,
        "completed_at": None,
    }

    candidates, candidate_payloads = _generate_all_candidates(
        task_id=task_id,
        problem_statement=problem_statement,
        config=config,
        call_logger=call_logger,
    )
    progress["candidates"] = candidate_payloads

    # candidates is always exactly num_solutions long (failed ones are dummies).
    candidates.sort(key=lambda c: c.index)

    final_candidate, round_payloads = _run_tournament_rounds(
        task_id=task_id,
        candidates=candidates,
        problem_statement=problem_statement,
        config=config,
        call_logger=call_logger,
    )
    progress["tournament_rounds"] = round_payloads

    LOGGER.info("Task %s tournament-merge final candidate index: %s", task_id, final_candidate.index)
    progress["status"] = "success"
    progress["final_candidate_index"] = final_candidate.index
    progress["completed_at"] = utc_now_iso()

    persisted_solution = extract_solution(final_candidate.completion) if final_candidate.completion is not None else ""
    if not persisted_solution:
        persisted_solution = final_candidate.solution_text

    save_task_outputs(
        files=files,
        solution_text=persisted_solution,
        latest_solver_completion=final_candidate.completion,
        progress_payload=progress,
    )
    return _task_result(task_id=task_id, status=progress["status"], files=files)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Tournament-merge solution pipeline for IMO AnswerBench")
    parser.add_argument("--start", type=int, required=True, help="Starting task index (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="Ending task index (exclusive)")
    parser.add_argument("--model", type=str, required=True, help="Solver model name")
    parser.add_argument("--output_path", type=str, required=True, help="Base output directory")
    parser.add_argument("--verifier_model", type=str, help="Verifier model name (default: same as --model)")
    parser.add_argument("--classifier_model", type=str, help="Binary checker model name (default: verifier model)")
    parser.add_argument("--merger_model", type=str, help="Merger model name (default: verifier model)")
    parser.add_argument("--solver_max_tokens", type=int, default=64000, help="Maximum solver output tokens")
    parser.add_argument("--verifier_max_tokens", type=int, default=64000, help="Maximum verifier output tokens")
    parser.add_argument("--classifier_max_tokens", type=int, default=64000, help="Maximum checker output tokens")
    parser.add_argument(
        "--merger_max_tokens",
        type=int,
        default=64000,
        help="Maximum merger output tokens",
    )
    parser.add_argument(
        "--num_solutions",
        type=int,
        default=8,
        help="Number of independent solutions to generate (must be a power of 2)",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top_p")
    parser.add_argument(
        "--other_prompt",
        action="append",
        default=[],
        help="Additional user prompt appended after the problem statement. Repeat for multiple prompts.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Hwilner/imo-answerbench",
        help="Hugging Face dataset name",
    )
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to evaluate")
    parser.add_argument(
        "--run_name",
        type=str,
        help="Optional run directory name (default: UTC timestamp)",
    )
    return parser.parse_args()


def configure_logging(log_file: Path) -> None:
    """Configure file and stdout logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def main() -> None:
    """Run the tournament-merge pipeline over a task range."""
    args = parse_args()
    run_started_at = utc_now_iso()

    if args.start < 0:
        msg = "--start must be non-negative"
        raise ValueError(msg)
    if args.end <= args.start:
        msg = "--end must be greater than --start"
        raise ValueError(msg)
    if not is_power_of_two(args.num_solutions):
        msg = f"--num_solutions must be a power of 2, got {args.num_solutions}"
        raise ValueError(msg)

    solver_model = args.model
    verifier_model = args.verifier_model or solver_model
    classifier_model = args.classifier_model or verifier_model
    merger_model = args.merger_model or verifier_model

    run_name = args.run_name or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_path) / "tournament_merge" / sanitize_model_name(solver_model) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(run_dir / "run.log")
    LOGGER.info("Loading dataset: %s (%s)", args.dataset_name, args.dataset_split)
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    LOGGER.info("Loaded %s tasks", len(dataset))

    if args.end > len(dataset):
        msg = f"--end ({args.end}) exceeds dataset size ({len(dataset)})"
        raise ValueError(msg)

    config = TournamentMergeConfig(
        solver_model=solver_model,
        verifier_model=verifier_model,
        classifier_model=classifier_model,
        merger_model=merger_model,
        solver_max_tokens=args.solver_max_tokens,
        verifier_max_tokens=args.verifier_max_tokens,
        classifier_max_tokens=args.classifier_max_tokens,
        merger_max_tokens=args.merger_max_tokens,
        num_solutions=args.num_solutions,
        temperature=args.temperature,
        top_p=args.top_p,
        other_prompts=tuple(args.other_prompt),
    )
    write_json(run_dir / "config.json", asdict(config))

    global_llm_log = run_dir / "all_llm_outputs.jsonl"
    global_llm_log.write_text("", encoding="utf-8")

    results: list[dict[str, Any]] = []
    for task_id in range(args.start, args.end):
        problem_statement = dataset[task_id]["Problem"]
        LOGGER.info("Starting task %s", task_id)
        task_result = solve_task(
            task_id=task_id,
            problem_statement=problem_statement,
            config=config,
            output_dir=run_dir,
            global_llm_log_path=global_llm_log,
        )
        results.append(task_result)
        LOGGER.info("Finished task %s with status=%s", task_id, task_result["status"])

    summary = {
        "started_at": run_started_at,
        "completed_at": utc_now_iso(),
        "run_dir": str(run_dir),
        "task_range": {"start": args.start, "end": args.end},
        "results": results,
        "success_count": sum(result["status"] == "success" for result in results),
        "failure_count": sum(result["status"] != "success" for result in results),
    }
    write_json(run_dir / "run_summary.json", summary)
    LOGGER.info("Run complete. Summary saved to %s", run_dir / "run_summary.json")


if __name__ == "__main__":
    main()
