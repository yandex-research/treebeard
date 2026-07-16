r"""Merge-compare ablation: alternate merge and comparison selection rounds.

For each task, loads the baseline candidates produced by
:mod:`generate_baseline` and runs ``n_rounds`` alternating between
merge rounds and comparison (selection) rounds.

Round types alternate starting with merge:

- **Merge round** (indices 0, 2, 4, …):

  - Round 0 (simple merge): each candidate *i* is merged with a
    randomly chosen partner *j* (``j != i``).  Pool size stays at *n*.
  - Rounds 2, 4, … (expanding merge): each candidate *i* produces
    **two** new candidates by merging with two different random
    partners.  Candidate *i* produces output candidates ``2*i`` and
    ``2*i + 1``.  Pool doubles from ``n/2`` back to ``n``.

- **Comparison round** (indices 1, 3, 5, …): candidates are randomly
  paired and a judge model selects the better solution.  Only winners
  advance; pool halves from ``n`` to ``n/2``.  Winners are
  re-indexed ``0 .. n/2 - 1``.

The population oscillates between *n* and *n/2*::

    merge(n) → compare(n/2) → merge(n) → compare(n/2) → …

No verification or self-improvement is performed.

Tasks are processed concurrently via a thread pool; rounds within a
task are inherently sequential.  Each task writes a single JSONL file
containing LLM call records.

Usage::

    uv run python -m open_deep_think.scripts.ablations.merge_compare \
        --candidates_dir ../data/ablations/baseline_candidates/default \
        --start 0 --end 10 \
        --model openai/gpt-oss-120b \
        --n_rounds 4 \
        --concurrency 4 \
        --output_path ../data/ablations/merge_compare
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import urllib3
from datasets import load_dataset

from open_deep_think.api import chat_api_call
from open_deep_think.imo_answer_bench.templates import (
    TOURNAMENT_COMPARISON_SYSTEM_PROMPT,
    TOURNAMENT_MERGE_SYSTEM_PROMPT,
    build_tournament_comparison_prompt,
    build_tournament_merge_prompt,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

LOGGER = logging.getLogger(__name__)

_MIN_CANDIDATES = 2
_SOLUTION_1 = 1
_SOLUTION_2 = 2


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeCompareConfig:
    """Configuration for the merge-compare ablation pipeline."""

    model: str
    max_tokens: int
    n_rounds: int
    temperature: float | None
    top_p: float | None
    seed: int | None


@dataclass(frozen=True)
class CallResult:
    """Result of a single model call."""

    completion: ChatCompletion | None
    text: str
    call_id: int


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


class TaskCallLogger:
    """Persist every LLM call to a single per-task JSONL file.

    Supports both merge and comparison phases.  Merge records include
    ``candidate_index``, ``merge_partner_index``, and optionally
    ``source_candidate_index`` (for expanding merges).  Comparison
    records include ``candidate_a_index`` and ``candidate_b_index``.
    """

    def __init__(self, task_id: int, log_path: Path) -> None:
        self._task_id = task_id
        self._log_path = log_path
        self._next_call_id = 1
        # Truncate / create the file.
        self._log_path.write_text("", encoding="utf-8")

    def record(  # noqa: PLR0913
        self,
        *,
        phase: str,
        round_index: int | None,
        model: str,
        messages: list[dict[str, str]],
        completion: ChatCompletion | None,
        response_text: str,
        # Merge-specific fields
        candidate_index: int | None = None,
        merge_partner_index: int | None = None,
        source_candidate_index: int | None = None,
        # Comparison-specific fields
        candidate_a_index: int | None = None,
        candidate_b_index: int | None = None,
        error: str | None = None,
    ) -> int:
        """Record an LLM call to the per-task JSONL log.

        Returns:
            The auto-incremented call ID for this record.

        """
        call_id = self._next_call_id
        self._next_call_id += 1
        payload: dict[str, Any] = {
            "record_type": "llm_call",
            "timestamp": utc_now_iso(),
            "task_id": self._task_id,
            "call_id": call_id,
            "phase": phase,
            "round_index": round_index,
            "candidate_index": candidate_index,
            "merge_partner_index": merge_partner_index,
            "source_candidate_index": source_candidate_index,
            "candidate_a_index": candidate_a_index,
            "candidate_b_index": candidate_b_index,
            "model": model,
            "messages": messages,
            "response_text": response_text,
            "completion": completion.model_dump() if completion is not None else None,
            "error": error,
        }
        append_jsonl(self._log_path, payload)
        return call_id


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object as a single line to a JSONL file."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    """Write JSON data with UTF-8 encoding and indentation."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def completion_text(completion: ChatCompletion) -> str:
    """Extract the first assistant message content from a completion."""
    if not completion.choices:
        return ""
    message = completion.choices[0].message
    return message.content if message.content is not None else ""


def load_task_ids_from_file(path: str) -> list[int]:
    """Read task IDs from a plain-text file (one integer per line).

    Blank lines and lines starting with ``#`` are ignored.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a non-blank, non-comment line is not an integer.

    """
    ids: list[int] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                ids.append(int(line))
            except ValueError as exc:
                msg = f"Cannot parse task ID on line {lineno} of {path!r}: {line!r}"
                raise ValueError(msg) from exc
    return ids


def load_candidates(candidates_dir: Path, task_id: int) -> list[dict[str, Any]]:
    """Load candidate solutions for a task from the baseline LLM outputs JSONL.

    Reads ``Task_{task_id}_llm_outputs.jsonl`` and extracts records with
    ``phase == "initial_solution"``, converting each into a dict with
    ``"index"`` and ``"solution_text"`` keys expected by the rest of the
    pipeline.

    Args:
        candidates_dir: Directory containing ``Task_*_llm_outputs.jsonl``
            files produced by :mod:`generate_baseline`.
        task_id: Task identifier.

    Returns:
        List of candidate dicts sorted by ``"index"``, each with
        ``"index"`` and ``"solution_text"`` keys.

    Raises:
        FileNotFoundError: If the LLM outputs file does not exist.
        ValueError: If no initial_solution records are found.

    """
    llm_log_file = candidates_dir / f"Task_{task_id}_llm_outputs.jsonl"
    if not llm_log_file.exists():
        msg = f"LLM outputs file not found: {llm_log_file}"
        raise FileNotFoundError(msg)

    candidates: list[dict[str, Any]] = []
    for line in llm_log_file.read_text(encoding="utf-8").strip().splitlines():
        record = json.loads(line)
        if record.get("phase") == "initial_solution":
            candidates.append(
                {
                    "index": record["candidate_index"],
                    "solution_text": record.get("response_text", ""),
                }
            )

    if not candidates:
        msg = f"No initial_solution records found in {llm_log_file}"
        raise ValueError(msg)

    candidates.sort(key=lambda c: c["index"])
    return candidates


def check_population_requirements(n: int, n_rounds: int) -> str | None:
    """Validate that population size *n* supports the requested rounds.

    Returns:
        An error message string if requirements are not met, or
        ``None`` if the population is valid.

    """
    if n < _MIN_CANDIDATES:
        return f"Need at least {_MIN_CANDIDATES} candidates, got {n}"
    if n_rounds >= _MIN_CANDIDATES and n % 2 != 0:
        return f"Need even number of candidates for comparison rounds, got {n}"
    _min_for_expanding = 3
    _first_expanding_round = 3
    if n_rounds >= _first_expanding_round and n < 2 * _min_for_expanding:
        return (
            f"Need at least {2 * _min_for_expanding} candidates for expanding merge "
            f"rounds (so that pool of n/2 has ≥ {_min_for_expanding} "
            f"for distinct partners), got {n}"
        )
    return None


# ---------------------------------------------------------------------------
# Comparison result parsing
# ---------------------------------------------------------------------------


def parse_comparison_result(response_text: str) -> int | None:
    """Parse the judge response to determine which solution won.

    The judge is expected to respond with only ``1`` or ``2``.

    Args:
        response_text: Raw judge response text.

    Returns:
        ``1`` if Solution 1 won, ``2`` if Solution 2 won,
        or ``None`` if the response could not be parsed.

    """
    text = response_text.strip()
    if text in ("1", "2"):
        return int(text)
    match = re.search(r"\b([12])\b", text)
    if match:
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Model calling
# ---------------------------------------------------------------------------


def call_model(  # noqa: PLR0913
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    phase: str,
    round_index: int | None,
    call_logger: TaskCallLogger,
    candidate_index: int | None = None,
    merge_partner_index: int | None = None,
    source_candidate_index: int | None = None,
    candidate_a_index: int | None = None,
    candidate_b_index: int | None = None,
) -> CallResult:
    """Call a chat model and persist the full request/response via *call_logger*."""
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
            round_index=round_index,
            candidate_index=candidate_index,
            merge_partner_index=merge_partner_index,
            source_candidate_index=source_candidate_index,
            candidate_a_index=candidate_a_index,
            candidate_b_index=candidate_b_index,
            model=model,
            messages=messages,
            completion=completion,
            response_text=response_text,
        )
    except Exception as error:
        call_id = call_logger.record(
            phase=phase,
            round_index=round_index,
            candidate_index=candidate_index,
            merge_partner_index=merge_partner_index,
            source_candidate_index=source_candidate_index,
            candidate_a_index=candidate_a_index,
            candidate_b_index=candidate_b_index,
            model=model,
            messages=messages,
            completion=completion,
            response_text=response_text,
            error=str(error),
        )
        raise
    return CallResult(completion=completion, text=response_text, call_id=call_id)


# ---------------------------------------------------------------------------
# Merge rounds
# ---------------------------------------------------------------------------


def run_simple_merge_round(  # noqa: PLR0913
    *,
    task_id: int,
    round_index: int,
    problem_statement: str,
    candidates: list[dict[str, Any]],
    config: MergeCompareConfig,
    call_logger: TaskCallLogger,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Run the initial merge round (round 0).  Pool size stays at *n*.

    For each candidate *i*, a random partner *j* (``j != i``) is chosen.
    Candidate *i*'s solution is placed as Solution 1 and partner *j*'s
    solution as Solution 2.  The merged output replaces candidate *i*.

    Args:
        task_id: Dataset task identifier (for logging).
        round_index: Zero-based round index.
        problem_statement: Raw problem text.
        candidates: Current population with ``"index"`` and
            ``"solution_text"`` keys.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.
        rng: Random number generator for partner selection.

    Returns:
        New population of candidate dicts (same length as *candidates*).

    """
    n = len(candidates)
    new_candidates: list[dict[str, Any]] = []

    for i in range(n):
        possible_partners = [j for j in range(n) if j != i]
        j = rng.choice(possible_partners)

        merge_prompt = build_tournament_merge_prompt(
            problem=problem_statement,
            solution_1=candidates[i]["solution_text"],
            solution_2=candidates[j]["solution_text"],
        )
        messages = [
            {"role": "system", "content": TOURNAMENT_MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": merge_prompt},
        ]

        try:
            result = call_model(
                model=config.model,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                phase="merge",
                round_index=round_index,
                candidate_index=i,
                merge_partner_index=j,
                call_logger=call_logger,
            )
            output_solution = result.text or candidates[i]["solution_text"]
        except Exception:
            LOGGER.exception(
                "Task %s candidate %s round %s merge with %s failed",
                task_id,
                i,
                round_index,
                j,
            )
            output_solution = candidates[i]["solution_text"]

        LOGGER.info(
            "Task %s candidate %s round %s merged with candidate %s",
            task_id,
            i,
            round_index,
            j,
        )
        new_candidates.append({"index": i, "solution_text": output_solution})

    return new_candidates


def run_expanding_merge_round(  # noqa: PLR0913
    *,
    task_id: int,
    round_index: int,
    problem_statement: str,
    candidates: list[dict[str, Any]],
    config: MergeCompareConfig,
    call_logger: TaskCallLogger,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Run an expanding merge round (rounds 2, 4, …).  Pool doubles.

    Each candidate *i* in the current pool (size ``n/2``) produces two
    new candidates by merging with two random partners from the pool:

    - Output candidate ``2*i``: merge of *i* with partner *j1*.
    - Output candidate ``2*i + 1``: merge of *i* with partner *j2*.

    Partners *j1* and *j2* are guaranteed different from *i* and from
    each other when the pool has ≥ 3 candidates.  If the pool has only
    2 candidates both partners will be the same (the only other
    candidate).

    Args:
        task_id: Dataset task identifier (for logging).
        round_index: Zero-based round index.
        problem_statement: Raw problem text.
        candidates: Current population (size ``n/2``) with ``"index"``
            and ``"solution_text"`` keys.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.
        rng: Random number generator for partner selection.

    Returns:
        New population of candidate dicts (twice the input length).

    """
    n = len(candidates)
    new_candidates: list[dict[str, Any]] = []

    for i in range(n):
        possible_partners = [j for j in range(n) if j != i]
        if len(possible_partners) >= _MIN_CANDIDATES:
            j1, j2 = rng.sample(possible_partners, _MIN_CANDIDATES)
        else:
            # Pool has only 2 candidates — both merges use the same partner.
            j1 = j2 = possible_partners[0]

        for sub_idx, partner_j in enumerate([j1, j2]):
            output_index = 2 * i + sub_idx

            merge_prompt = build_tournament_merge_prompt(
                problem=problem_statement,
                solution_1=candidates[i]["solution_text"],
                solution_2=candidates[partner_j]["solution_text"],
            )
            messages = [
                {"role": "system", "content": TOURNAMENT_MERGE_SYSTEM_PROMPT},
                {"role": "user", "content": merge_prompt},
            ]

            try:
                result = call_model(
                    model=config.model,
                    messages=messages,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    phase="merge",
                    round_index=round_index,
                    candidate_index=output_index,
                    merge_partner_index=partner_j,
                    source_candidate_index=i,
                    call_logger=call_logger,
                )
                output_solution = result.text or candidates[i]["solution_text"]
            except Exception:
                LOGGER.exception(
                    "Task %s source %s → candidate %s round %s merge with %s failed",
                    task_id,
                    i,
                    output_index,
                    round_index,
                    partner_j,
                )
                output_solution = candidates[i]["solution_text"]

            LOGGER.info(
                "Task %s source %s → candidate %s round %s merged with candidate %s",
                task_id,
                i,
                output_index,
                round_index,
                partner_j,
            )
            new_candidates.append({"index": output_index, "solution_text": output_solution})

    return new_candidates


# ---------------------------------------------------------------------------
# Comparison round
# ---------------------------------------------------------------------------


def run_comparison_round(  # noqa: PLR0913
    *,
    task_id: int,
    round_index: int,
    problem_statement: str,
    candidates: list[dict[str, Any]],
    config: MergeCompareConfig,
    call_logger: TaskCallLogger,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Run a comparison/selection round.  Pool halves: *n* → *n/2*.

    Candidates are randomly paired.  For each pair a judge model
    selects the better solution.  Only winners advance and are
    re-indexed ``0 .. n/2 - 1``.

    If the judge response cannot be parsed or the API call fails,
    the first candidate in the pair wins by default.

    Args:
        task_id: Dataset task identifier (for logging).
        round_index: Zero-based round index.
        problem_statement: Raw problem text.
        candidates: Current population (even length) with ``"index"``
            and ``"solution_text"`` keys.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.
        rng: Random number generator for pairing.

    Returns:
        New population of winner dicts (half the input length),
        re-indexed ``0 .. n/2 - 1``.

    """
    n = len(candidates)
    indices = list(range(n))
    rng.shuffle(indices)

    winners: list[dict[str, Any]] = []

    for k in range(0, n, 2):
        a_idx = indices[k]
        b_idx = indices[k + 1]

        comparison_prompt = build_tournament_comparison_prompt(
            problem=problem_statement,
            solution_1=candidates[a_idx]["solution_text"],
            solution_2=candidates[b_idx]["solution_text"],
        )
        messages = [
            {"role": "system", "content": TOURNAMENT_COMPARISON_SYSTEM_PROMPT},
            {"role": "user", "content": comparison_prompt},
        ]

        try:
            result = call_model(
                model=config.model,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                phase="comparison",
                round_index=round_index,
                candidate_a_index=a_idx,
                candidate_b_index=b_idx,
                call_logger=call_logger,
            )
            choice = parse_comparison_result(result.text)
            if choice == _SOLUTION_1:
                winner = candidates[a_idx]
                winner_idx = a_idx
            elif choice == _SOLUTION_2:
                winner = candidates[b_idx]
                winner_idx = b_idx
            else:
                LOGGER.warning(
                    "Task %s round %s comparison %s vs %s: unparseable response %r — defaulting to candidate %s",
                    task_id,
                    round_index,
                    a_idx,
                    b_idx,
                    result.text.strip()[:50],
                    a_idx,
                )
                winner = candidates[a_idx]
                winner_idx = a_idx
        except Exception:
            LOGGER.exception(
                "Task %s round %s comparison %s vs %s failed — defaulting to candidate %s",
                task_id,
                round_index,
                a_idx,
                b_idx,
                a_idx,
            )
            winner = candidates[a_idx]
            winner_idx = a_idx

        LOGGER.info(
            "Task %s round %s: candidate %s vs %s → winner=%s",
            task_id,
            round_index,
            a_idx,
            b_idx,
            winner_idx,
        )
        winners.append(winner)

    # Re-index winners as 0 .. n/2 - 1.
    return [{"index": i, "solution_text": w["solution_text"]} for i, w in enumerate(winners)]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all_rounds(  # noqa: PLR0913
    *,
    task_id: int,
    problem_statement: str,
    candidates: list[dict[str, Any]],
    config: MergeCompareConfig,
    call_logger: TaskCallLogger,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Run all rounds for a single task, alternating merge and comparison.

    Even-indexed rounds are merge rounds; odd-indexed rounds are
    comparison rounds.  Round 0 is a simple merge (pool stays at *n*);
    subsequent merge rounds (2, 4, …) are expanding merges that double
    the pool from ``n/2`` back to ``n``.

    Args:
        task_id: Dataset task identifier (for logging).
        problem_statement: Raw problem text.
        candidates: Initial population with ``"index"`` and
            ``"solution_text"`` keys.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.
        rng: Random number generator.

    Returns:
        Final population of candidate dicts after all rounds.

    """
    current = candidates
    for round_index in range(config.n_rounds):
        is_merge_round = round_index % 2 == 0

        if is_merge_round:
            if round_index == 0:
                LOGGER.info(
                    "Task %s starting simple merge round %s/%s (pool size %s)",
                    task_id,
                    round_index + 1,
                    config.n_rounds,
                    len(current),
                )
                current = run_simple_merge_round(
                    task_id=task_id,
                    round_index=round_index,
                    problem_statement=problem_statement,
                    candidates=current,
                    config=config,
                    call_logger=call_logger,
                    rng=rng,
                )
            else:
                LOGGER.info(
                    "Task %s starting expanding merge round %s/%s (pool size %s → %s)",
                    task_id,
                    round_index + 1,
                    config.n_rounds,
                    len(current),
                    len(current) * 2,
                )
                current = run_expanding_merge_round(
                    task_id=task_id,
                    round_index=round_index,
                    problem_statement=problem_statement,
                    candidates=current,
                    config=config,
                    call_logger=call_logger,
                    rng=rng,
                )
        else:
            LOGGER.info(
                "Task %s starting comparison round %s/%s (pool size %s → %s)",
                task_id,
                round_index + 1,
                config.n_rounds,
                len(current),
                len(current) // 2,
            )
            current = run_comparison_round(
                task_id=task_id,
                round_index=round_index,
                problem_statement=problem_statement,
                candidates=current,
                config=config,
                call_logger=call_logger,
                rng=rng,
            )

    return current


# ---------------------------------------------------------------------------
# Task processing
# ---------------------------------------------------------------------------


def is_task_done(output_dir: Path, task_id: int) -> bool:
    """Return ``True`` if the task output JSONL already exists and is non-empty.

    A non-empty ``Task_{task_id}_merge_compare.jsonl`` indicates that
    the task was successfully processed in a previous run and can be
    skipped.
    """
    output_file = output_dir / f"Task_{task_id}_merge_compare.jsonl"
    return output_file.exists() and output_file.stat().st_size > 0


def process_task(
    *,
    task_id: int,
    problem_statement: str,
    candidates_dir: Path,
    config: MergeCompareConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Process all candidates for one task through merge-compare rounds.

    Loads baseline candidates, filters out those with empty solutions,
    validates that the population size supports the requested number
    of rounds, and runs the alternating merge/comparison pipeline.
    All LLM calls are written to a single JSONL file per task.

    Args:
        task_id: Dataset task identifier.
        problem_statement: Raw problem text.
        candidates_dir: Directory with baseline LLM output JSONL logs.
        config: Pipeline configuration.
        output_dir: Directory for output files.

    Returns:
        A summary dict with ``task_id``, ``status``, and ``output_path``.

    """
    log_path = output_dir / f"Task_{task_id}_merge_compare.jsonl"
    call_logger = TaskCallLogger(task_id=task_id, log_path=log_path)

    candidates = load_candidates(candidates_dir, task_id)

    # Filter out empty candidates.
    valid_candidates = [c for c in candidates if c["solution_text"]]

    # Validate population size against requested rounds.
    error_msg = check_population_requirements(len(valid_candidates), config.n_rounds)
    if error_msg is not None:
        LOGGER.warning("Task %s: %s — skipping.", task_id, error_msg)
        return {
            "task_id": task_id,
            "status": "skipped_insufficient_candidates",
            "output_path": str(log_path),
        }

    # Re-index valid candidates to 0..n-1.
    for i, c in enumerate(valid_candidates):
        c["index"] = i

    rng = random.Random(config.seed + task_id if config.seed is not None else None)  # noqa: S311

    LOGGER.info(
        "Task %s starting %s rounds of merge-compare with %s candidates",
        task_id,
        config.n_rounds,
        len(valid_candidates),
    )

    run_all_rounds(
        task_id=task_id,
        problem_statement=problem_statement,
        candidates=valid_candidates,
        config=config,
        call_logger=call_logger,
        rng=rng,
    )

    LOGGER.info("Task %s complete. Output: %s", task_id, log_path)
    return {
        "task_id": task_id,
        "status": "success",
        "output_path": str(log_path),
    }


def _process_task_wrapper(
    task_id: int,
    problem_statement: str,
    candidates_dir: Path,
    config: MergeCompareConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Worker function for the thread pool (skip-if-done guard included)."""
    if is_task_done(output_dir, task_id):
        LOGGER.info("Task %s already done — skipping.", task_id)
        return {"task_id": task_id, "status": "skipped"}
    LOGGER.info("Starting task %s", task_id)
    result = process_task(
        task_id=task_id,
        problem_statement=problem_statement,
        candidates_dir=candidates_dir,
        config=config,
        output_dir=output_dir,
    )
    LOGGER.info("Finished task %s with status=%s", task_id, result["status"])
    return result


def run_tasks_concurrent(  # noqa: PLR0913
    *,
    task_ids: list[int],
    problems: list[str],
    candidates_dir: Path,
    config: MergeCompareConfig,
    output_dir: Path,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Run merge-compare for multiple tasks using a thread pool.

    Tasks are submitted to a :class:`~concurrent.futures.ThreadPoolExecutor`
    and processed concurrently (up to *concurrency* threads).  Each task
    writes to its own output files, so there are no shared file handles.

    Args:
        task_ids: Ordered list of task identifiers to process.
        problems: Problem statements corresponding to *task_ids*.
        candidates_dir: Directory with baseline LLM output JSONL logs.
        config: Pipeline configuration.
        output_dir: Directory for per-task output files.
        concurrency: Maximum concurrent worker threads.

    Returns:
        List of per-task result dicts (one per task, in completion order).

    """
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_task = {
            executor.submit(
                _process_task_wrapper,
                tid,
                prob,
                candidates_dir,
                config,
                output_dir,
            ): tid
            for tid, prob in zip(task_ids, problems)
        }
        for future in as_completed(future_to_task):
            tid = future_to_task[future]
            try:
                result = future.result()
            except Exception:
                LOGGER.exception("Task %s failed with unrecoverable error", tid)
                result = {"task_id": tid, "status": "error"}
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Task source is specified via one of two mutually exclusive modes:

    * ``--task_ids_file PATH`` — read explicit task IDs from a text file.
    * ``--start N --end M`` — process all tasks in ``[N, M)``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Merge-compare ablation: alternate merge and comparison selection rounds over baseline candidates."
        ),
    )
    parser.add_argument(
        "--candidates_dir",
        type=str,
        required=True,
        help="Directory with baseline LLM output JSONL logs (from generate_baseline).",
    )
    parser.add_argument(
        "--task_ids_file",
        type=str,
        default=None,
        help="Text file with one task ID per line.  Mutually exclusive with --start/--end.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Starting task index (inclusive).  Requires --end.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending task index (exclusive).  Requires --start.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (used for both merging and comparison).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Base output directory.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100000,
        help="Maximum output tokens for both merge and comparison calls.",
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=4,
        help="Total number of rounds (alternating merge/comparison).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of tasks to process in parallel (default: 1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling top_p.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: non-deterministic).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Hwilner/imo-answerbench",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="default",
        help="Run directory name (default: 'default').",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate task-source and other CLI argument constraints.

    Raises:
        ValueError: On any invalid combination.

    """
    using_file = args.task_ids_file is not None
    using_range = args.start is not None or args.end is not None

    if using_file and using_range:
        msg = "--task_ids_file is mutually exclusive with --start/--end"
        raise ValueError(msg)

    if not using_file:
        if args.start is None or args.end is None:
            msg = "Either --task_ids_file or both --start and --end must be provided"
            raise ValueError(msg)
        if args.start < 0:
            msg = "--start must be non-negative"
            raise ValueError(msg)
        if args.end <= args.start:
            msg = "--end must be greater than --start"
            raise ValueError(msg)

    if args.n_rounds < 1:
        msg = f"--n_rounds must be >= 1, got {args.n_rounds}"
        raise ValueError(msg)
    if args.concurrency < 1:
        msg = f"--concurrency must be >= 1, got {args.concurrency}"
        raise ValueError(msg)


def configure_logging() -> None:
    """Configure stdout-only logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def main() -> None:
    """Run the merge-compare ablation over baseline candidates."""
    args = parse_args()
    validate_args(args)

    candidates_dir = Path(args.candidates_dir)
    run_dir = Path(args.output_path) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    configure_logging()
    LOGGER.info("Loading dataset: %s (%s)", args.dataset_name, args.dataset_split)
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    LOGGER.info("Loaded %s tasks", len(dataset))

    # Resolve task IDs.
    if args.task_ids_file is not None:
        task_ids = load_task_ids_from_file(args.task_ids_file)
        LOGGER.info("Loaded %s task IDs from %s", len(task_ids), args.task_ids_file)
    else:
        if args.end > len(dataset):
            msg = f"--end ({args.end}) exceeds dataset size ({len(dataset)})"
            raise ValueError(msg)
        task_ids = list(range(args.start, args.end))

    config = MergeCompareConfig(
        model=args.model,
        max_tokens=args.max_tokens,
        n_rounds=args.n_rounds,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    # Persist config once for the run.
    config_path = run_dir / "config.json"
    config_dict = json.loads(json.dumps(asdict(config)))
    if not config_path.exists():
        write_json(config_path, config_dict)
        LOGGER.info("Config written to %s", config_path)

    problems = [dataset[tid]["Problem"] for tid in task_ids]

    LOGGER.info(
        "Processing %s tasks with concurrency=%s, n_rounds=%s",
        len(task_ids),
        args.concurrency,
        config.n_rounds,
    )
    run_tasks_concurrent(
        task_ids=task_ids,
        problems=problems,
        candidates_dir=candidates_dir,
        config=config,
        output_dir=run_dir,
        concurrency=args.concurrency,
    )
    LOGGER.info("Run complete. Output dir: %s", run_dir)


if __name__ == "__main__":
    main()
