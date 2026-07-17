r"""Count cumulative completion tokens and Swiss-tournament accuracy per round.

Reads baseline candidate-generation JSONL files (round 0) and Swiss-tournament
JSONL files (rounds 1..N), computes per-candidate cumulative completion tokens,
and reports the average across all tasks and candidates for each round.

Accuracy is computed using Swiss tournament standings and baseline candidate
evaluations.  For round 0, the mean population accuracy is reported (across
all candidates and tasks).  For round *i* > 0, the standings after Swiss
round *i* - 1 are loaded, candidates with the highest score are identified,
and their evaluation verdicts from the baseline candidate evaluation are
looked up.  The per-task accuracy is the fraction of those top-ranked
candidates that have a correct verdict; the reported accuracy is the mean
of these per-task accuracies across all tasks.

No standard deviation is reported.

The output is a table::

    round  avg_cumulative_tokens  mean_accuracy
    0                  25000.0          0.650
    1                  42000.0          0.750
    ...

Usage::

    uv run python -m open_deep_think.scripts.ablations.count_tokens_swiss \\
        --candidates_dir data/ablations/subset_baseline_32_candidates_gpt_oss \\
        --rounds_dir data/ablations/ablation_comparison_gpt_oss
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from open_deep_think.scripts.ablations.count_tokens import (
    _extract_task_id,
    _get_completion_tokens,
    compute_baseline_accuracy,
    compute_cumulative_table,
    load_baseline_evaluation_verdicts,
    load_baseline_tokens,
)

# ---------------------------------------------------------------------------
# Swiss-tournament token loading
# ---------------------------------------------------------------------------


def load_swiss_round_tokens(
    rounds_dir: Path,
) -> dict[int, dict[int, dict[int, int]]]:
    """Load per-candidate, per-round completion tokens from Swiss tournament JSONL files.

    Each Swiss tournament match produces a comparison record with
    ``candidate_a_index`` and ``candidate_b_index``.  The completion
    tokens for each match are split evenly between the two participants
    so that the total per-round cost is not double-counted.

    Args:
        rounds_dir: Directory containing Swiss tournament JSONL files
            (e.g. ``Task_{id}_swiss_tournament.jsonl``).

    Returns:
        Mapping ``{task_id: {candidate_index: {round_index: completion_tokens}}}``.

    """
    pattern = re.compile(r"Task_(\d+)_.*\.jsonl$")
    tokens: dict[int, dict[int, dict[int, int]]] = {}

    for path in sorted(rounds_dir.iterdir()):
        task_id = _extract_task_id(path.name, pattern)
        if task_id is None:
            continue

        task_tokens: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            record = json.loads(line)
            if record.get("record_type") != "llm_call":
                continue
            candidate_a = record.get("candidate_a_index")
            candidate_b = record.get("candidate_b_index")
            round_idx = record.get("round_index")
            if candidate_a is None or candidate_b is None or round_idx is None:
                continue
            ct = _get_completion_tokens(record)
            # Split tokens evenly; assign any remainder to candidate_a.
            half = ct // 2
            remainder = ct - 2 * half
            task_tokens[candidate_a][round_idx] += half + remainder
            task_tokens[candidate_b][round_idx] += half

        tokens[task_id] = {cand: dict(rounds) for cand, rounds in task_tokens.items()}

    return tokens


# ---------------------------------------------------------------------------
# Swiss standings loading
# ---------------------------------------------------------------------------


def load_swiss_standings(
    rounds_dir: Path,
) -> dict[int, dict[int, list[dict[str, Any]]]]:
    """Load Swiss tournament standings from JSON files.

    Args:
        rounds_dir: Directory containing ``Task_{id}_standings.json`` files.

    Returns:
        Mapping ``{task_id: {round_index: [standings_entries]}}``,
        where each entry has ``candidate_index``, ``score``, and ``rank``.

    """
    pattern = re.compile(r"Task_(\d+)_standings\.json$")
    standings: dict[int, dict[int, list[dict[str, Any]]]] = {}

    for path in sorted(rounds_dir.iterdir()):
        task_id = _extract_task_id(path.name, pattern)
        if task_id is None:
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        task_standings: dict[int, list[dict[str, Any]]] = {}
        for round_data in data.get("rounds", []):
            ri = round_data["round_index"]
            task_standings[ri] = round_data["standings"]
        standings[task_id] = task_standings

    return standings


# ---------------------------------------------------------------------------
# Swiss-tournament accuracy computation
# ---------------------------------------------------------------------------


def compute_swiss_accuracy_per_round(
    standings: dict[int, dict[int, list[dict[str, Any]]]],
    baseline_verdicts: dict[int, dict[int, bool]],
    n_output_rounds: int,
    baseline_accuracy: float = float("nan"),
) -> list[float]:
    """Compute accuracy per round using Swiss standings and baseline verdicts.

    For round 0, uses the pre-computed *baseline_accuracy*.  For round
    *i* > 0, loads standings for Swiss round *i* - 1, finds the
    candidates with the maximum score, looks up their verdicts from the
    baseline evaluation, and computes the fraction of correct verdicts.
    This per-task accuracy is then averaged across all tasks.

    Args:
        standings: ``{task_id: {round_index: [standings_entries]}}``
            loaded from ``Task_{id}_standings.json`` files.
        baseline_verdicts: ``{task_id: {candidate_index: verdict}}``
            loaded from baseline evaluation JSON files.
        n_output_rounds: Total number of output rounds (including round 0).
        baseline_accuracy: Pre-computed accuracy for round 0 (baseline).
            Defaults to ``NaN`` when no baseline evaluation is available.

    Returns:
        List of accuracies, one per output round.

    """
    accuracies: list[float] = [float("nan")] * n_output_rounds
    accuracies[0] = baseline_accuracy

    # Use tasks present in both standings and baseline verdicts.
    common_tasks = sorted(set(standings) & set(baseline_verdicts))

    for output_round in range(1, n_output_rounds):
        swiss_round = output_round - 1
        task_accuracies: list[float] = []

        for task_id in common_tasks:
            task_standings = standings[task_id]
            if swiss_round not in task_standings:
                continue

            round_standings = task_standings[swiss_round]
            if not round_standings:
                continue

            # Find maximum score.
            max_score = max(entry["score"] for entry in round_standings)

            # Find all candidates with the maximum score.
            top_candidates = [entry["candidate_index"] for entry in round_standings if entry["score"] == max_score]

            # Look up their verdicts from baseline evaluation.
            task_verdicts = baseline_verdicts.get(task_id, {})
            verdicts = [
                1.0 if task_verdicts[cand_idx] else 0.0 for cand_idx in top_candidates if cand_idx in task_verdicts
            ]

            if verdicts:
                task_accuracies.append(sum(verdicts) / len(verdicts))

        if task_accuracies:
            accuracies[output_round] = sum(task_accuracies) / len(task_accuracies)

    return accuracies


# ---------------------------------------------------------------------------
# Accuracy resolution
# ---------------------------------------------------------------------------


def _resolve_accuracies(
    candidates_dir: Path,
    rounds_dir: Path,
    n_rounds: int,
) -> list[float] | None:
    """Resolve per-round accuracy using Swiss standings and baseline evaluation.

    Baseline (round 0) accuracy is read from
    ``candidates_dir/evaluation``.  Rounds 1..N accuracy is computed
    from Swiss tournament standings in *rounds_dir* combined with
    baseline evaluation verdicts.

    Args:
        candidates_dir: Baseline candidates directory (must contain
            ``evaluation/`` subfolder with per-task evaluation JSON files).
        rounds_dir: Directory containing Swiss tournament standings JSON
            files (``Task_{id}_standings.json``).
        n_rounds: Total number of output rounds (including round 0).

    Returns:
        List of accuracies (one per round), or ``None`` if no
        evaluation data is available.

    """
    baseline_eval = candidates_dir / "evaluation"
    if not baseline_eval.is_dir():
        return None

    baseline_verdicts = load_baseline_evaluation_verdicts(baseline_eval)
    if not baseline_verdicts:
        return None

    baseline_acc = compute_baseline_accuracy(baseline_verdicts)

    # Load Swiss tournament standings.
    standings = load_swiss_standings(rounds_dir)
    if not standings:
        # Only baseline accuracy available.
        accuracies = [float("nan")] * n_rounds
        accuracies[0] = baseline_acc
        return accuracies

    return compute_swiss_accuracy_per_round(
        standings=standings,
        baseline_verdicts=baseline_verdicts,
        n_output_rounds=n_rounds,
        baseline_accuracy=baseline_acc,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_table(
    table: list[tuple[int, float, int]],
    accuracies: list[float] | None = None,
) -> None:
    """Print the round / avg_cumulative_tokens / count / mean_accuracy table.

    Args:
        table: List of ``(round_number, avg_cumulative_tokens, pair_count)``
            tuples.
        accuracies: Optional list of mean accuracy values, one per round.
            If ``None``, the accuracy column is omitted.

    """
    if accuracies is not None:
        header = f"{'round':<8}{'avg_cumulative_tokens':>22}{'n_pairs':>10}{'mean_accuracy':>16}"
        print(header)  # noqa: T201
        print("-" * 56)  # noqa: T201
        for i, (round_num, avg_tokens, count) in enumerate(table):
            acc = accuracies[i] if i < len(accuracies) else float("nan")
            acc_str = "NaN" if math.isnan(acc) else f"{acc:.4f}"
            line = f"{round_num:<8}{avg_tokens:>22.1f}{count:>10}{acc_str:>16}"
            print(line)  # noqa: T201
    else:
        print(f"{'round':<8}{'avg_cumulative_tokens':>22}{'n_pairs':>10}")  # noqa: T201
        print("-" * 40)  # noqa: T201
        for round_num, avg_tokens, count in table:
            print(f"{round_num:<8}{avg_tokens:>22.1f}{count:>10}")  # noqa: T201


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Count cumulative completion tokens per round "
            "(baseline + Swiss tournament) and report the average "
            "across tasks and candidates, with per-round accuracy "
            "derived from Swiss tournament standings."
        ),
    )
    parser.add_argument(
        "--candidates_dir",
        type=str,
        required=True,
        help=(
            "Directory with baseline LLM output JSONL files "
            "(e.g. data/ablations/subset_baseline_32_candidates_gpt_oss)."
        ),
    )
    parser.add_argument(
        "--rounds_dir",
        type=str,
        required=True,
        help=(
            "Directory with Swiss tournament JSONL and standings files "
            "(e.g. data/ablations/ablation_comparison_gpt_oss)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Load token data and Swiss tournament standings, then print the cumulative table."""
    args = parse_args()

    candidates_dir = Path(args.candidates_dir)
    rounds_dir = Path(args.rounds_dir)

    if not candidates_dir.is_dir():
        print(f"Error: candidates_dir not found: {candidates_dir}", file=sys.stderr)  # noqa: T201
        sys.exit(1)
    if not rounds_dir.is_dir():
        print(f"Error: rounds_dir not found: {rounds_dir}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    baseline_tokens = load_baseline_tokens(candidates_dir)
    round_tokens = load_swiss_round_tokens(rounds_dir)

    if not baseline_tokens:
        print("Error: no baseline JSONL files found.", file=sys.stderr)  # noqa: T201
        sys.exit(1)
    if not round_tokens:
        print("Error: no Swiss tournament JSONL files found.", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    table = compute_cumulative_table(baseline_tokens, round_tokens)
    if not table:
        print(  # noqa: T201
            "Error: no common tasks found between baseline and rounds.",
            file=sys.stderr,
        )
        sys.exit(1)

    accuracies = _resolve_accuracies(candidates_dir, rounds_dir, len(table))
    print_table(table, accuracies)


if __name__ == "__main__":
    main()
