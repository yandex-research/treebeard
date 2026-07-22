r"""Count cumulative completion tokens and accuracy per round for merge-improve-compare.

Reads baseline candidate-generation JSONL files (round 0) and merge-improve-compare
JSONL files (rounds 1..N), computes total cumulative completion tokens per task
(summed across all candidates) averaged across tasks, and reports accuracy per
round.

Merge-improve-compare data cycles through merge, self-improvement, and
comparison rounds with period 3:

- **Merge rounds** (round indices 0, 3, 6, …): each candidate produces
  a merged solution.  Accuracy is computed directly from evaluation verdicts
  at that round index.

- **Improve rounds** (round indices 1, 4, 7, …): each candidate is
  self-improved.  Accuracy is computed directly from evaluation verdicts
  at that round index.

- **Comparison rounds** (round indices 2, 5, 8, …): candidates are
  paired and a judge selects a winner.  Winners inherit the evaluation
  verdict of their original candidate from the preceding improve round.

For token counts, all completion tokens for every LLM call in a given round
are summed (no per-candidate normalisation) and accumulated across rounds.
The per-task cumulative totals are then averaged across tasks.

The output is a table::

    round  type       avg_cumulative_tokens   n_tasks  mean_accuracy  std_accuracy
    0      baseline              25000.0         20          0.6500          0.4770
    1      merge                 42000.0         20          0.7500          0.4330
    2      improve               55000.0         20          0.7800          0.4100
    3      compare               56500.0         20          0.8000          0.4000
    ...

Usage::

    uv run python -m open_deep_think.scripts.ablations.count_tokens_merge_improve_compare \\
        --candidates_dir data/ablations/subset_baseline_candidates_gpt_oss \\
        --rounds_dir data/ablations/ablation_merge_improve_compare_gpt_oss
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from open_deep_think.scripts.ablations.count_tokens import (
    _extract_task_id,
    _get_completion_tokens,
    _population_std,
    compute_baseline_accuracy,
    compute_baseline_accuracy_std,
    load_baseline_evaluation_verdicts,
    load_baseline_tokens,
    load_evaluation_verdicts,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SOLUTION_1 = 1
_SOLUTION_2 = 2

# Round type constants (cycle with period 3).
_MERGE_PHASE = 0
_IMPROVE_PHASE = 1
_COMPARE_PHASE = 2
_ROUND_CYCLE = 3


# ---------------------------------------------------------------------------
# Token loading
# ---------------------------------------------------------------------------


def load_merge_improve_compare_round_tokens(
    rounds_dir: Path,
) -> dict[int, dict[int, int]]:
    """Load per-round total completion tokens from merge-improve-compare JSONL files.

    For each task, sums all ``completion_tokens`` from LLM-call records
    at each ``round_index``, regardless of phase (merge, improve, or
    comparison).

    Args:
        rounds_dir: Directory containing
            ``Task_{id}_merge_improve_compare.jsonl`` files produced by
            :mod:`merge_improve_compare`.

    Returns:
        Mapping ``{task_id: {round_index: total_completion_tokens}}``.

    """
    pattern = re.compile(r"Task_(\d+)_.*\.jsonl$")
    tokens: dict[int, dict[int, int]] = {}

    for path in sorted(rounds_dir.iterdir()):
        task_id = _extract_task_id(path.name, pattern)
        if task_id is None:
            continue

        task_tokens: dict[int, int] = defaultdict(int)
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            record = json.loads(line)
            if record.get("record_type") != "llm_call":
                continue
            round_idx = record.get("round_index")
            if round_idx is None:
                continue
            ct = _get_completion_tokens(record)
            task_tokens[round_idx] += ct

        tokens[task_id] = dict(task_tokens)

    return tokens


# ---------------------------------------------------------------------------
# Comparison winner loading
# ---------------------------------------------------------------------------


def _parse_comparison_result(response_text: str) -> int | None:
    """Parse judge response to determine which solution won (1 or 2).

    Mirrors the logic in :func:`merge_improve_compare.parse_comparison_result`.

    Args:
        response_text: Raw judge response text.

    Returns:
        ``1`` if Solution 1 won, ``2`` if Solution 2 won, or ``None``
        if the response could not be parsed.

    """
    text = response_text.strip()
    if text in ("1", "2"):
        return int(text)
    match = re.search(r"\b([12])\b", text)
    if match:
        return int(match.group(1))
    return None


def load_comparison_winners(
    rounds_dir: Path,
) -> dict[int, dict[int, list[int]]]:
    """Load comparison round winners from merge-improve-compare JSONL files.

    For each comparison-phase LLM call, parses ``response_text`` to
    determine which candidate won (1 → ``candidate_a``, 2 →
    ``candidate_b``).  If the response is unparseable, candidate A wins
    by default (matching :mod:`merge_improve_compare` behaviour).

    Args:
        rounds_dir: Directory containing
            ``Task_{id}_merge_improve_compare.jsonl`` files.

    Returns:
        Mapping ``{task_id: {round_index: [winner_original_indices]}}``.
        The list preserves match order so that position *k* corresponds
        to the re-indexed winner *k* in the post-comparison population.

    """
    pattern = re.compile(r"Task_(\d+)_.*\.jsonl$")
    winners: dict[int, dict[int, list[int]]] = {}

    for path in sorted(rounds_dir.iterdir()):
        task_id = _extract_task_id(path.name, pattern)
        if task_id is None:
            continue

        task_winners: dict[int, list[int]] = defaultdict(list)
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            record = json.loads(line)
            if record.get("record_type") != "llm_call":
                continue
            if record.get("phase") != "comparison":
                continue

            round_idx = record.get("round_index")
            candidate_a = record.get("candidate_a_index")
            candidate_b = record.get("candidate_b_index")
            response_text = record.get("response_text", "")

            if round_idx is None or candidate_a is None or candidate_b is None:
                continue

            choice = _parse_comparison_result(response_text)
            if choice == _SOLUTION_2:
                task_winners[round_idx].append(candidate_b)
            else:
                # Default to candidate A (same as merge_improve_compare.py).
                task_winners[round_idx].append(candidate_a)

        winners[task_id] = dict(task_winners)

    return winners


# ---------------------------------------------------------------------------
# Cumulative token table
# ---------------------------------------------------------------------------


def compute_cumulative_table(
    baseline_tokens: dict[int, dict[int, int]],
    round_tokens: dict[int, dict[int, int]],
) -> list[tuple[int, float, int]]:
    """Compute total cumulative completion tokens per round averaged across tasks.

    For each task the total token cost is summed across all candidates
    (no per-candidate normalisation).  The per-task totals are then
    averaged across tasks.

    Output row 0 uses the total baseline candidate tokens for a task.
    Output row *r* (*r* >= 1) adds merge-improve-compare round *r - 1*
    tokens to the previous cumulative total.

    Args:
        baseline_tokens: ``{task_id: {candidate_index: tokens}}`` from
            baseline candidate generation.
        round_tokens: ``{task_id: {round_index: total_tokens}}`` from
            merge-improve-compare rounds.

    Returns:
        Sorted list of ``(round_number, avg_total_tokens, n_tasks)``
        tuples.

    """
    common_tasks = sorted(set(baseline_tokens) & set(round_tokens))
    if not common_tasks:
        return []

    # Determine the maximum round_index across all tasks.
    max_round_idx = 0
    for task_id in common_tasks:
        if round_tokens[task_id]:
            max_round_idx = max(max_round_idx, *round_tokens[task_id])

    n_mic_rounds = max_round_idx + 1  # merge-improve-compare rounds (0-indexed)
    n_output_rounds = n_mic_rounds + 1  # +1 for baseline (row 0)

    round_sums: list[float] = [0.0] * n_output_rounds
    round_counts: list[int] = [0] * n_output_rounds

    for task_id in common_tasks:
        baseline_cands = baseline_tokens[task_id]
        if not baseline_cands:
            continue

        # Row 0: total baseline tokens for this task.
        baseline_total = sum(baseline_cands.values())
        round_sums[0] += baseline_total
        round_counts[0] += 1

        # Rows 1..N: cumulative total cost for this task.
        cumulative = baseline_total
        task_rt = round_tokens[task_id]
        for ri in range(n_mic_rounds):
            cumulative += task_rt.get(ri, 0)
            round_sums[ri + 1] += cumulative
            round_counts[ri + 1] += 1

    table: list[tuple[int, float, int]] = []
    for r in range(n_output_rounds):
        avg = round_sums[r] / round_counts[r] if round_counts[r] > 0 else 0.0
        table.append((r, avg, round_counts[r]))

    return table


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------


def _direct_round_accuracy(
    eval_verdicts: dict[int, dict[int, dict[int, bool]]],
    mic_round: int,
) -> float:
    """Compute accuracy for a merge or improve round from evaluation verdicts.

    Merge and improve rounds have direct evaluation verdicts stored at
    the corresponding ``round_index``.

    Args:
        eval_verdicts: ``{task_id: {candidate_index: {round_index: verdict}}}``.
        mic_round: Merge-improve-compare round index (merge or improve).

    Returns:
        Fraction of correct verdicts, or ``NaN`` if no verdicts exist.

    """
    correct_count = 0
    total_count = 0
    for cand_verdicts in eval_verdicts.values():
        for round_verdicts in cand_verdicts.values():
            if mic_round in round_verdicts:
                total_count += 1
                if round_verdicts[mic_round]:
                    correct_count += 1
    if total_count == 0:
        return float("nan")
    return correct_count / total_count


def _comparison_round_accuracy(
    eval_verdicts: dict[int, dict[int, dict[int, bool]]],
    comparison_winners: dict[int, dict[int, list[int]]],
    mic_round: int,
) -> float:
    """Compute accuracy for a comparison round from winner verdicts.

    Winners inherit the evaluation verdict of their original candidate
    from the preceding improve round (``mic_round - 1``).

    Args:
        eval_verdicts: ``{task_id: {candidate_index: {round_index: verdict}}}``.
        comparison_winners: ``{task_id: {round_index: [winner_original_indices]}}``.
        mic_round: Merge-improve-compare round index (comparison phase).

    Returns:
        Fraction of correct verdicts, or ``NaN`` if no verdicts exist.

    """
    prev_improve_round = mic_round - 1
    correct_count = 0
    total_count = 0
    for task_id, task_eval in eval_verdicts.items():
        task_winners = comparison_winners.get(task_id, {}).get(mic_round, [])
        for winner_orig_idx in task_winners:
            winner_round_verdicts = task_eval.get(winner_orig_idx, {})
            if prev_improve_round in winner_round_verdicts:
                total_count += 1
                if winner_round_verdicts[prev_improve_round]:
                    correct_count += 1
    if total_count == 0:
        return float("nan")
    return correct_count / total_count


def compute_accuracy_per_round(
    eval_verdicts: dict[int, dict[int, dict[int, bool]]],
    comparison_winners: dict[int, dict[int, list[int]]],
    n_output_rounds: int,
    baseline_accuracy: float = float("nan"),
) -> list[float]:
    """Compute mean accuracy for each output round.

    Output row 0 uses the pre-computed *baseline_accuracy*.

    For merge rounds (``mic_round % 3 == 0``): accuracy is the fraction
    of ``(task, candidate)`` pairs with a correct evaluation verdict at
    that round index.

    For improve rounds (``mic_round % 3 == 1``): same as merge — direct
    evaluation verdicts at that round index.

    For comparison rounds (``mic_round % 3 == 2``): accuracy is the
    fraction of comparison winners with a correct evaluation verdict
    inherited from the preceding improve round.

    Args:
        eval_verdicts: ``{task_id: {candidate_index: {round_index: verdict}}}``
            loaded from evaluation JSON files.
        comparison_winners: ``{task_id: {round_index: [winner_original_indices]}}``
            loaded from merge-improve-compare JSONL files.
        n_output_rounds: Total number of output rows (including row 0).
        baseline_accuracy: Pre-computed accuracy for row 0 (baseline).

    Returns:
        List of mean accuracies, one per output row.

    """
    accuracies: list[float] = [float("nan")] * n_output_rounds
    accuracies[0] = baseline_accuracy

    for output_round in range(1, n_output_rounds):
        mic_round = output_round - 1
        round_type = mic_round % _ROUND_CYCLE

        if round_type == _COMPARE_PHASE:
            accuracies[output_round] = _comparison_round_accuracy(eval_verdicts, comparison_winners, mic_round)
        else:
            # Both merge and improve rounds have direct evaluation verdicts.
            accuracies[output_round] = _direct_round_accuracy(eval_verdicts, mic_round)

    return accuracies


def compute_accuracy_std_per_round(
    eval_verdicts: dict[int, dict[int, dict[int, bool]]],
    comparison_winners: dict[int, dict[int, list[int]]],
    n_output_rounds: int,
    baseline_std: float = float("nan"),
) -> list[float]:
    """Compute std of per-candidate mean accuracy for each output round.

    For merge and improve rounds, candidates are grouped by their
    ``candidate_index``.  For comparison rounds, winners are grouped
    by their post-comparison position index (0 ... n/2 - 1).

    For each group, the mean accuracy across tasks is computed; the
    returned value is the population standard deviation of those
    per-group means.

    Args:
        eval_verdicts: ``{task_id: {candidate_index: {round_index: verdict}}}``
            loaded from evaluation JSON files.
        comparison_winners: ``{task_id: {round_index: [winner_original_indices]}}``
            loaded from merge-improve-compare JSONL files.
        n_output_rounds: Total number of output rows (including row 0).
        baseline_std: Pre-computed std for row 0 (baseline).

    Returns:
        List of per-candidate accuracy standard deviations, one per
        output row.

    """
    stds: list[float] = [float("nan")] * n_output_rounds
    stds[0] = baseline_std

    for output_round in range(1, n_output_rounds):
        mic_round = output_round - 1
        round_type = mic_round % _ROUND_CYCLE

        cand_task_verdicts: dict[int, list[float]] = defaultdict(list)

        if round_type == _COMPARE_PHASE:
            prev_improve_round = mic_round - 1
            for task_id, task_eval in eval_verdicts.items():
                task_winners = comparison_winners.get(task_id, {}).get(mic_round, [])
                for new_idx, winner_orig_idx in enumerate(task_winners):
                    winner_round_verdicts = task_eval.get(winner_orig_idx, {})
                    if prev_improve_round in winner_round_verdicts:
                        v = 1.0 if winner_round_verdicts[prev_improve_round] else 0.0
                        cand_task_verdicts[new_idx].append(v)
        else:
            # Both merge and improve rounds have direct evaluation verdicts.
            for cand_verdicts in eval_verdicts.values():
                for cand_idx, round_verdicts in cand_verdicts.items():
                    if mic_round in round_verdicts:
                        v = 1.0 if round_verdicts[mic_round] else 0.0
                        cand_task_verdicts[cand_idx].append(v)

        if cand_task_verdicts:
            cand_means = [sum(vs) / len(vs) for vs in cand_task_verdicts.values()]
            stds[output_round] = _population_std(cand_means)

    return stds


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _round_type_label(output_round: int) -> str:
    """Return a human-readable label for the round type.

    Row 0 is ``baseline``.  Subsequent rows cycle through ``merge``
    (``mic_round % 3 == 0``), ``improve`` (``mic_round % 3 == 1``),
    and ``compare`` (``mic_round % 3 == 2``).
    """
    if output_round == 0:
        return "baseline"
    mic_round = output_round - 1
    round_type = mic_round % _ROUND_CYCLE
    if round_type == _MERGE_PHASE:
        return "merge"
    if round_type == _IMPROVE_PHASE:
        return "improve"
    return "compare"


def print_table(
    table: list[tuple[int, float, int]],
    accuracies: list[float] | None = None,
    accuracy_stds: list[float] | None = None,
) -> None:
    """Print the round / type / tokens / n_tasks / accuracy / std table.

    Args:
        table: List of ``(round_number, avg_cumulative_tokens, n_tasks)``
            tuples.
        accuracies: Optional list of mean accuracy values, one per round.
            If ``None``, the accuracy column is omitted.
        accuracy_stds: Optional list of accuracy std values, one per
            round.  If ``None``, the std column is omitted.

    """
    if accuracies is not None:
        has_std = accuracy_stds is not None
        header = f"{'round':<8}{'type':<12}{'avg_cumulative_tokens':>22}{'n_tasks':>10}{'mean_accuracy':>16}"
        sep_len = 68
        if has_std:
            header += f"{'std_accuracy':>16}"
            sep_len += 16
        print(header)  # noqa: T201
        print("-" * sep_len)  # noqa: T201
        for i, (round_num, avg_tokens, count) in enumerate(table):
            acc = accuracies[i] if i < len(accuracies) else float("nan")
            acc_str = "NaN" if math.isnan(acc) else f"{acc:.4f}"
            rtype = _round_type_label(round_num)
            line = f"{round_num:<8}{rtype:<12}{avg_tokens:>22.1f}{count:>10}{acc_str:>16}"
            if has_std:
                std_val = accuracy_stds[i] if i < len(accuracy_stds) else float("nan")
                std_str = "NaN" if math.isnan(std_val) else f"{std_val:.4f}"
                line += f"{std_str:>16}"
            print(line)  # noqa: T201
    else:
        header = f"{'round':<8}{'type':<12}{'avg_cumulative_tokens':>22}{'n_tasks':>10}"
        print(header)  # noqa: T201
        print("-" * 52)  # noqa: T201
        for round_num, avg_tokens, count in table:
            rtype = _round_type_label(round_num)
            print(  # noqa: T201
                f"{round_num:<8}{rtype:<12}{avg_tokens:>22.1f}{count:>10}"
            )


# ---------------------------------------------------------------------------
# Accuracy resolution
# ---------------------------------------------------------------------------


def _resolve_accuracies(
    candidates_dir: Path,
    rounds_eval_dir: Path | None,
    comparison_winners: dict[int, dict[int, list[int]]],
    n_rounds: int,
) -> tuple[list[float] | None, list[float] | None]:
    """Resolve per-round accuracy values and stds from evaluation data.

    Baseline (row 0) accuracy and std are read from
    ``candidates_dir/evaluation``.  Rows 1..N accuracy and std are
    computed from evaluation verdicts in *rounds_eval_dir* and the
    *comparison_winners* mapping.

    Args:
        candidates_dir: Baseline candidates directory (may contain
            ``evaluation/`` subfolder).
        rounds_eval_dir: Directory with evaluation files,
            or ``None`` if unavailable.
        comparison_winners: ``{task_id: {round_index: [winner_indices]}}``
            loaded from merge-improve-compare JSONL files.
        n_rounds: Total number of output rows (including row 0).

    Returns:
        Tuple of ``(accuracies, accuracy_stds)``, each a list of values
        (one per round) or ``None`` if no evaluation data is available.

    """
    # Baseline (row 0) accuracy and std.
    baseline_acc = float("nan")
    baseline_std = float("nan")
    baseline_eval = candidates_dir / "evaluation"
    if baseline_eval.is_dir():
        baseline_verdicts = load_baseline_evaluation_verdicts(baseline_eval)
        if baseline_verdicts:
            baseline_acc = compute_baseline_accuracy(baseline_verdicts)
            baseline_std = compute_baseline_accuracy_std(baseline_verdicts)

    has_any_accuracy = not math.isnan(baseline_acc)
    accuracies: list[float] | None = None
    accuracy_stds: list[float] | None = None

    # Per-round accuracy and std from merge-improve-compare evaluation.
    if rounds_eval_dir is not None and rounds_eval_dir.is_dir():
        eval_verdicts = load_evaluation_verdicts(rounds_eval_dir)
        if eval_verdicts:
            accuracies = compute_accuracy_per_round(
                eval_verdicts,
                comparison_winners,
                n_rounds,
                baseline_accuracy=baseline_acc,
            )
            accuracy_stds = compute_accuracy_std_per_round(
                eval_verdicts,
                comparison_winners,
                n_rounds,
                baseline_std=baseline_std,
            )
            has_any_accuracy = True
        else:
            print(  # noqa: T201
                "Warning: evaluation directory exists but contains no evaluation files.",
                file=sys.stderr,
            )

    # If only baseline accuracy available, still show columns.
    if accuracies is None and has_any_accuracy:
        accuracies = [float("nan")] * n_rounds
        accuracies[0] = baseline_acc
        accuracy_stds = [float("nan")] * n_rounds
        accuracy_stds[0] = baseline_std

    return accuracies, accuracy_stds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Count cumulative completion tokens per round "
            "(baseline + merge-improve-compare) and report the average "
            "across tasks and candidates, optionally with per-round "
            "accuracy and std."
        ),
    )
    parser.add_argument(
        "--candidates_dir",
        type=str,
        required=True,
        help=(
            "Directory with baseline LLM output JSONL files (e.g. data/ablations/subset_baseline_candidates_gpt_oss)."
        ),
    )
    parser.add_argument(
        "--rounds_dir",
        type=str,
        required=True,
        help=(
            "Directory with merge-improve-compare JSONL files "
            "(e.g. data/ablations/"
            "ablation_merge_improve_compare_gpt_oss)."
        ),
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default=None,
        help=(
            "Directory with evaluation JSON files.  "
            "Defaults to ``{rounds_dir}/evaluation``.  If the directory "
            "does not exist, accuracy reporting is skipped.  Baseline "
            "(row 0) accuracy is always read from "
            "``{candidates_dir}/evaluation`` when available."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Load token data and print the cumulative table with accuracy."""
    args = parse_args()

    candidates_dir = Path(args.candidates_dir)
    rounds_dir = Path(args.rounds_dir)

    if not candidates_dir.is_dir():
        print(  # noqa: T201
            f"Error: candidates_dir not found: {candidates_dir}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not rounds_dir.is_dir():
        print(  # noqa: T201
            f"Error: rounds_dir not found: {rounds_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve evaluation directory.
    rounds_eval_dir: Path | None = None
    if args.eval_dir is not None:
        rounds_eval_dir = Path(args.eval_dir)
    else:
        candidate_eval_dir = rounds_dir / "evaluation"
        if candidate_eval_dir.is_dir():
            rounds_eval_dir = candidate_eval_dir

    # Load data.
    baseline_tokens = load_baseline_tokens(candidates_dir)
    round_tokens = load_merge_improve_compare_round_tokens(rounds_dir)
    comparison_winners = load_comparison_winners(rounds_dir)

    if not baseline_tokens:
        print(  # noqa: T201
            "Error: no baseline JSONL files found.", file=sys.stderr
        )
        sys.exit(1)
    if not round_tokens:
        print(  # noqa: T201
            "Error: no merge-improve-compare JSONL files found.",
            file=sys.stderr,
        )
        sys.exit(1)

    table = compute_cumulative_table(baseline_tokens, round_tokens)
    if not table:
        print(  # noqa: T201
            "Error: no common tasks found between baseline and rounds.",
            file=sys.stderr,
        )
        sys.exit(1)

    accuracies, accuracy_stds = _resolve_accuracies(
        candidates_dir,
        rounds_eval_dir,
        comparison_winners,
        len(table),
    )
    print_table(table, accuracies, accuracy_stds)


if __name__ == "__main__":
    main()
