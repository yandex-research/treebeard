r"""Swiss-tournament ablation: rank baseline candidates via pairwise comparisons.

For each task, loads the baseline candidates produced by
:mod:`generate_baseline` and runs ``n_rounds`` of Swiss-system tournament
rounds using pairwise comparisons.

In each round, candidates are paired according to the Swiss system:
players are grouped by score, randomly shuffled within each group,
then paired with adjacent players — avoiding rematches by scanning
forward within the group or floating down to the next score group.

Each match uses
:func:`~open_deep_think.imo_answer_bench.templates.build_tournament_comparison_prompt`
to ask a judge model which of two solutions is better.  The winner
receives 1 point; the loser receives 0.

No verification or self-improvement is performed.

Tasks are processed concurrently via a thread pool; rounds within a
task are inherently sequential (each round depends on match history
from previous rounds).  Each task writes:

- A JSONL file containing all LLM call records.
- A JSON file with the sorted standings table after each round.

Usage::

    uv run python -m open_deep_think.scripts.ablations.swiss_tournament \
        --candidates_dir ../data/ablations/baseline_candidates/default \
        --start 0 --end 10 \
        --model openai/gpt-oss-120b \
        --n_rounds 8 \
        --concurrency 4 \
        --output_path ../data/ablations/swiss_tournament
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
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
    build_tournament_comparison_prompt,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

LOGGER = logging.getLogger(__name__)

_MIN_CANDIDATES_FOR_TOURNAMENT = 2
_SOLUTION_1 = 1
_SOLUTION_2 = 2


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwissTournamentConfig:
    """Configuration for the Swiss-tournament ablation pipeline."""

    judge_model: str
    judge_max_tokens: int
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


@dataclass
class MatchResult:
    """Result of a single pairwise comparison match."""

    candidate_a_index: int
    candidate_b_index: int
    winner_index: int | None
    raw_response: str


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


class TaskCallLogger:
    """Persist every LLM call to a single per-task JSONL file.

    Each record has a ``record_type`` field set to ``"llm_call"`` and
    includes ``candidate_a_index`` and ``candidate_b_index`` fields
    recording which candidates were compared.
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
        candidate_a_index: int | None,
        candidate_b_index: int | None,
        round_index: int | None,
        model: str,
        messages: list[dict[str, str]],
        completion: ChatCompletion | None,
        response_text: str,
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
            "candidate_a_index": candidate_a_index,
            "candidate_b_index": candidate_b_index,
            "round_index": round_index,
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
    # Try exact match first.
    if text in ("1", "2"):
        return int(text)
    # Try to find a standalone digit 1 or 2.
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
    candidate_a_index: int | None,
    candidate_b_index: int | None,
    round_index: int | None,
    call_logger: TaskCallLogger,
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
            candidate_a_index=candidate_a_index,
            candidate_b_index=candidate_b_index,
            round_index=round_index,
            model=model,
            messages=messages,
            completion=completion,
            response_text=response_text,
        )
    except Exception as error:
        call_id = call_logger.record(
            phase=phase,
            candidate_a_index=candidate_a_index,
            candidate_b_index=candidate_b_index,
            round_index=round_index,
            model=model,
            messages=messages,
            completion=completion,
            response_text=response_text,
            error=str(error),
        )
        raise
    return CallResult(completion=completion, text=response_text, call_id=call_id)


# ---------------------------------------------------------------------------
# Swiss pairing
# ---------------------------------------------------------------------------


def _find_opponent_in_candidates(
    player: int,
    candidates: list[int],
    paired: set[int],
    played: dict[int, set[int]],
) -> int | None:
    """Find the first valid (unpaired, not-yet-played) opponent for *player*.

    Args:
        player: The player seeking an opponent.
        candidates: Ordered list of potential opponents to scan.
        paired: Set of already-paired player indices.
        played: Mapping of player index to set of opponents already faced.

    Returns:
        The opponent index if found, or ``None``.

    """
    player_history = played.get(player, set())
    for opponent in candidates:
        if opponent not in paired and opponent not in player_history:
            return opponent
    return None


def _float_down(
    player: int,
    groups: list[list[int]],
    start_group: int,
    paired: set[int],
    played: dict[int, set[int]],
) -> int | None:
    """Search lower score groups for a valid opponent for *player*.

    Args:
        player: The player seeking an opponent via float-down.
        groups: All score groups (ordered descending by score).
        start_group: Index of the first lower group to search.
        paired: Set of already-paired player indices.
        played: Mapping of player index to set of opponents already faced.

    Returns:
        The opponent index if found, or ``None``.

    """
    for lower_idx in range(start_group, len(groups)):
        lower_available = [p for p in groups[lower_idx] if p not in paired]
        opponent = _find_opponent_in_candidates(
            player,
            lower_available,
            paired,
            played,
        )
        if opponent is not None:
            return opponent
    return None


def swiss_pair(
    *,
    scores: dict[int, int],
    played: dict[int, set[int]],
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Pair candidates according to the Swiss system.

    Groups players by score (descending), randomly shuffles each group,
    then pairs adjacent players within each group.  Before finalizing
    a pair, checks if they've already played each other; if so, skips
    that opponent and tries the next available player in the group.
    If no valid opponent exists in the current group, the player floats
    down to the next lower score group.

    Args:
        scores: Mapping of candidate index to current score.
        played: Mapping of candidate index to set of opponent indices
            already faced.
        rng: Random number generator for shuffling.

    Returns:
        List of ``(candidate_a, candidate_b)`` tuples for this round.
        If the total number of candidates is odd, one candidate will
        receive a bye (not included in any pair).

    """
    # Group candidates by score, sorted descending.
    score_groups: dict[int, list[int]] = defaultdict(list)
    for idx, score in scores.items():
        score_groups[score].append(idx)

    sorted_scores = sorted(score_groups.keys(), reverse=True)

    # Build ordered list of groups, each shuffled.
    groups: list[list[int]] = []
    for s in sorted_scores:
        group = score_groups[s][:]
        rng.shuffle(group)
        groups.append(group)

    pairs: list[tuple[int, int]] = []
    paired: set[int] = set()

    for group_idx in range(len(groups)):
        # Collect unpaired players in this group.
        available = [p for p in groups[group_idx] if p not in paired]

        i = 0
        while i < len(available):
            player = available[i]
            if player in paired:
                i += 1
                continue

            # Try to find a valid opponent in the rest of this group.
            rest = available[i + 1 :]
            opponent = _find_opponent_in_candidates(player, rest, paired, played)

            if opponent is not None:
                pairs.append((player, opponent))
                paired.add(player)
                paired.add(opponent)
            else:
                # Float the player down to the next lower group.
                floated_opponent = _float_down(
                    player,
                    groups,
                    group_idx + 1,
                    paired,
                    played,
                )
                if floated_opponent is not None:
                    pairs.append((player, floated_opponent))
                    paired.add(player)
                    paired.add(floated_opponent)
                else:
                    LOGGER.warning(
                        "Candidate %s has no valid opponent — receives a bye.",
                        player,
                    )

            i += 1

    return pairs


# ---------------------------------------------------------------------------
# Tournament rounds
# ---------------------------------------------------------------------------


def run_comparison_match(  # noqa: PLR0913
    *,
    task_id: int,
    round_index: int,
    problem_statement: str,
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
    config: SwissTournamentConfig,
    call_logger: TaskCallLogger,
) -> MatchResult:
    """Run a single comparison match between two candidates.

    Builds a comparison prompt placing *candidate_a* as Solution 1 and
    *candidate_b* as Solution 2, then parses the judge response to
    determine the winner.

    Args:
        task_id: Dataset task identifier (for logging).
        round_index: Zero-based round index.
        problem_statement: Raw problem text.
        candidate_a: First candidate dict with ``"index"`` and ``"solution_text"``.
        candidate_b: Second candidate dict with ``"index"`` and ``"solution_text"``.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.

    Returns:
        A :class:`MatchResult` with the winner index (or ``None`` on
        parse failure / API error).

    """
    comparison_prompt = build_tournament_comparison_prompt(
        problem=problem_statement,
        solution_1=candidate_a["solution_text"],
        solution_2=candidate_b["solution_text"],
    )
    messages = [
        {"role": "system", "content": TOURNAMENT_COMPARISON_SYSTEM_PROMPT},
        {"role": "user", "content": comparison_prompt},
    ]

    idx_a = candidate_a["index"]
    idx_b = candidate_b["index"]

    try:
        result = call_model(
            model=config.judge_model,
            messages=messages,
            max_tokens=config.judge_max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            phase="comparison",
            candidate_a_index=idx_a,
            candidate_b_index=idx_b,
            round_index=round_index,
            call_logger=call_logger,
        )
        raw_response = result.text
    except Exception:
        LOGGER.exception(
            "Task %s round %s comparison %s vs %s failed",
            task_id,
            round_index,
            idx_a,
            idx_b,
        )
        return MatchResult(
            candidate_a_index=idx_a,
            candidate_b_index=idx_b,
            winner_index=None,
            raw_response="",
        )

    choice = parse_comparison_result(raw_response)
    if choice == _SOLUTION_1:
        winner = idx_a
    elif choice == _SOLUTION_2:
        winner = idx_b
    else:
        LOGGER.warning(
            "Task %s round %s comparison %s vs %s: could not parse judge response %r — no points awarded.",
            task_id,
            round_index,
            idx_a,
            idx_b,
            raw_response,
        )
        winner = None

    LOGGER.info(
        "Task %s round %s: candidate %s vs %s → winner=%s (raw=%r)",
        task_id,
        round_index,
        idx_a,
        idx_b,
        winner,
        raw_response.strip()[:50],
    )

    return MatchResult(
        candidate_a_index=idx_a,
        candidate_b_index=idx_b,
        winner_index=winner,
        raw_response=raw_response,
    )


def build_standings(
    scores: dict[int, int],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build a sorted standings table from current scores.

    Candidates are sorted by score descending.  Ties are broken by
    random shuffle (shuffle first, then stable-sort by score).

    Args:
        scores: Mapping of candidate index to current score.
        rng: Random number generator for tie-breaking.

    Returns:
        List of dicts with ``"rank"``, ``"candidate_index"``, and
        ``"score"`` keys, sorted by score descending.

    """
    entries = [{"candidate_index": idx, "score": s} for idx, s in scores.items()]
    rng.shuffle(entries)
    entries.sort(key=lambda e: e["score"], reverse=True)
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank
    return entries


def run_swiss_tournament(  # noqa: PLR0913
    *,
    task_id: int,
    problem_statement: str,
    candidates: list[dict[str, Any]],
    config: SwissTournamentConfig,
    call_logger: TaskCallLogger,
    rng: random.Random,
) -> list[list[dict[str, Any]]]:
    """Run a full Swiss-system tournament for a single task.

    Runs ``config.n_rounds`` rounds.  In each round, candidates are
    paired using Swiss pairing and each pair plays a comparison match.
    The winner receives 1 point.

    Args:
        task_id: Dataset task identifier (for logging).
        problem_statement: Raw problem text.
        candidates: List of candidate dicts with ``"index"`` and
            ``"solution_text"`` keys.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.
        rng: Random number generator.

    Returns:
        List of standings tables (one per round), where each standings
        table is a list of dicts with ``"rank"``, ``"candidate_index"``,
        and ``"score"`` keys.

    """
    # Build lookup for candidate data by index.
    candidate_by_index = {c["index"]: c for c in candidates}

    # Initialize scores and match history.
    scores: dict[int, int] = {c["index"]: 0 for c in candidates}
    played: dict[int, set[int]] = defaultdict(set)

    all_standings: list[list[dict[str, Any]]] = []

    for round_index in range(config.n_rounds):
        LOGGER.info(
            "Task %s starting Swiss round %s/%s",
            task_id,
            round_index + 1,
            config.n_rounds,
        )

        # Generate pairings.
        pairs = swiss_pair(scores=scores, played=played, rng=rng)
        LOGGER.info(
            "Task %s round %s: %s pairs generated",
            task_id,
            round_index,
            len(pairs),
        )

        # Play all matches in this round.
        for idx_a, idx_b in pairs:
            # Record that they played each other.
            played[idx_a].add(idx_b)
            played[idx_b].add(idx_a)

            match_result = run_comparison_match(
                task_id=task_id,
                round_index=round_index,
                problem_statement=problem_statement,
                candidate_a=candidate_by_index[idx_a],
                candidate_b=candidate_by_index[idx_b],
                config=config,
                call_logger=call_logger,
            )

            if match_result.winner_index is not None:
                scores[match_result.winner_index] += 1

        # Build standings after this round.
        standings = build_standings(scores, rng)
        all_standings.append(standings)

        LOGGER.info(
            "Task %s round %s standings: top 3 = %s",
            task_id,
            round_index,
            standings[:3],
        )

    return all_standings


# ---------------------------------------------------------------------------
# Task processing
# ---------------------------------------------------------------------------


def is_task_done(output_dir: Path, task_id: int) -> bool:
    """Return ``True`` if the task output JSONL already exists and is non-empty.

    A non-empty ``Task_{task_id}_swiss_tournament.jsonl`` indicates that
    the task was successfully processed in a previous run and can be
    skipped.
    """
    output_file = output_dir / f"Task_{task_id}_swiss_tournament.jsonl"
    return output_file.exists() and output_file.stat().st_size > 0


def process_task(
    *,
    task_id: int,
    problem_statement: str,
    candidates_dir: Path,
    config: SwissTournamentConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Process one task through a Swiss-system tournament.

    Loads baseline candidates, filters out those with empty solutions
    (requiring at least two non-empty candidates), and runs
    ``config.n_rounds`` of Swiss-system tournament rounds.  All LLM
    calls are written to a per-task JSONL file.  Standings after each
    round are written to a separate JSON file.

    Args:
        task_id: Dataset task identifier.
        problem_statement: Raw problem text.
        candidates_dir: Directory with baseline LLM output JSONL logs.
        config: Pipeline configuration.
        output_dir: Directory for output files.

    Returns:
        A summary dict with ``task_id``, ``status``, and ``output_path``.

    """
    log_path = output_dir / f"Task_{task_id}_swiss_tournament.jsonl"
    standings_path = output_dir / f"Task_{task_id}_standings.json"
    call_logger = TaskCallLogger(task_id=task_id, log_path=log_path)

    candidates = load_candidates(candidates_dir, task_id)

    # Filter out empty candidates.
    valid_candidates = [c for c in candidates if c["solution_text"]]
    if len(valid_candidates) < _MIN_CANDIDATES_FOR_TOURNAMENT:
        LOGGER.warning(
            "Task %s has fewer than %s non-empty candidates (%s) — skipping.",
            task_id,
            _MIN_CANDIDATES_FOR_TOURNAMENT,
            len(valid_candidates),
        )
        return {
            "task_id": task_id,
            "status": "skipped_too_few_candidates",
            "output_path": str(log_path),
        }

    rng = random.Random(config.seed + task_id if config.seed is not None else None)  # noqa: S311

    LOGGER.info(
        "Task %s starting Swiss tournament with %s candidates over %s rounds",
        task_id,
        len(valid_candidates),
        config.n_rounds,
    )

    all_standings = run_swiss_tournament(
        task_id=task_id,
        problem_statement=problem_statement,
        candidates=valid_candidates,
        config=config,
        call_logger=call_logger,
        rng=rng,
    )

    # Write standings to a separate JSON file.
    standings_output: dict[str, Any] = {
        "task_id": task_id,
        "n_rounds": config.n_rounds,
        "n_candidates": len(valid_candidates),
        "rounds": [{"round_index": i, "standings": s} for i, s in enumerate(all_standings)],
    }
    write_json(standings_path, standings_output)

    LOGGER.info(
        "Task %s complete. LLM log: %s, Standings: %s",
        task_id,
        log_path,
        standings_path,
    )
    return {
        "task_id": task_id,
        "status": "success",
        "output_path": str(log_path),
        "standings_path": str(standings_path),
    }


def _process_task_wrapper(
    task_id: int,
    problem_statement: str,
    candidates_dir: Path,
    config: SwissTournamentConfig,
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
    config: SwissTournamentConfig,
    output_dir: Path,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Run Swiss tournament for multiple tasks using a thread pool.

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
        description=("Swiss-tournament ablation: rank baseline candidates via pairwise comparisons."),
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
        help=("Text file with one task ID per line.  Mutually exclusive with --start/--end."),
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
        help="Judge model name.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Base output directory.",
    )
    parser.add_argument(
        "--judge_max_tokens",
        type=int,
        default=100000,
        help="Maximum judge output tokens.",
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=8,
        help="Number of Swiss tournament rounds per task.",
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
        help=("Random seed for pairing reproducibility (default: non-deterministic)."),
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
    """Run the Swiss-tournament ablation over baseline candidates."""
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

    config = SwissTournamentConfig(
        judge_model=args.model,
        judge_max_tokens=args.judge_max_tokens,
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
