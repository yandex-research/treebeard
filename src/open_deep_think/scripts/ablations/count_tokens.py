"""Count cumulative completion tokens, mean accuracy, and accuracy std per round.

Reads baseline candidate-generation JSONL files (round 0) and self-improvement
JSONL files (rounds 1..N), computes per-candidate cumulative completion tokens,
and reports the average across all tasks and candidates for each round.

When an evaluation directory is available inside the candidates directory
(``candidates_dir/evaluation``), the script reports mean population accuracy
and std of accuracy for round 0 (baseline).  When an evaluation directory is
available inside the rounds directory (``rounds_dir/evaluation``), the script
reports mean accuracy and std for each self-improvement round.  If either
evaluation directory is absent, the corresponding values are reported as
``NaN``.

The accuracy std is computed as follows: for each candidate, compute its mean
accuracy across all tasks (fraction of tasks solved correctly) on a given
round, then report the population standard deviation of these per-candidate
mean accuracies across all candidates.

The output is a table::

    round  avg_cumulative_tokens  mean_accuracy  std_accuracy
    0                  25000.0          0.650          0.4770
    1                  42000.0          0.750          0.4330
    ...

Usage::

    uv run python -m open_deep_think.scripts.ablations.count_tokens \
        --candidates_dir data/ablations/subset_baseline_candidates_gpt_oss \
        --rounds_dir data/ablations/ablation_self_improve_no_verification_gpt_oss
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


def _extract_task_id(filename: str, pattern: re.Pattern[str]) -> int | None:
    """Extract integer task ID from a filename using *pattern*.

    Returns ``None`` if the filename does not match.
    """
    match = pattern.search(filename)
    if match is None:
        return None
    return int(match.group(1))


def _get_completion_tokens(record: dict[str, Any]) -> int:
    """Extract ``completion_tokens`` from an LLM-call record.

    Handles both dict and string forms of the ``completion`` field.
    Returns ``0`` when the field is missing or ``None``.
    """
    completion = record.get("completion")
    if completion is None:
        return 0
    if isinstance(completion, str):
        completion = json.loads(completion)
    usage = completion.get("usage")
    if usage is None:
        return 0
    return usage.get("completion_tokens", 0)


def _population_std(values: list[float]) -> float:
    """Compute the population standard deviation of *values*.

    Returns ``0.0`` for empty lists or single-element lists.
    """
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)


def load_baseline_tokens(
    candidates_dir: Path,
) -> dict[int, dict[int, int]]:
    """Load per-candidate completion tokens from baseline JSONL files.

    Args:
        candidates_dir: Directory containing ``Task_{id}_llm_outputs.jsonl``.

    Returns:
        Mapping ``{task_id: {candidate_index: completion_tokens}}``.

    """
    pattern = re.compile(r"Task_(\d+)_llm_outputs\.jsonl$")
    tokens: dict[int, dict[int, int]] = {}

    for path in sorted(candidates_dir.iterdir()):
        task_id = _extract_task_id(path.name, pattern)
        if task_id is None:
            continue

        task_tokens: dict[int, int] = {}
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            record = json.loads(line)
            # Only count llm_call records (baseline files may lack record_type).
            record_type = record.get("record_type", "llm_call")
            if record_type != "llm_call":
                continue
            candidate_idx = record.get("candidate_index")
            if candidate_idx is None:
                continue
            ct = _get_completion_tokens(record)
            task_tokens[candidate_idx] = task_tokens.get(candidate_idx, 0) + ct

        tokens[task_id] = task_tokens

    return tokens


def load_round_tokens(
    rounds_dir: Path,
) -> dict[int, dict[int, dict[int, int]]]:
    """Load per-candidate, per-round completion tokens from round JSONL files.

    Matches any ``Task_{id}_*.jsonl`` file in the directory, so it works
    with both ``_self_improve.jsonl`` and ``_merge_candidates.jsonl``
    naming conventions.

    Args:
        rounds_dir: Directory containing per-task JSONL files
            (e.g. ``Task_{id}_self_improve.jsonl`` or
            ``Task_{id}_merge_candidates.jsonl``).

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
            candidate_idx = record.get("candidate_index")
            round_idx = record.get("round_index")
            if candidate_idx is None or round_idx is None:
                continue
            ct = _get_completion_tokens(record)
            task_tokens[candidate_idx][round_idx] += ct

        # Convert nested defaultdicts to plain dicts for safety.
        tokens[task_id] = {cand: dict(rounds) for cand, rounds in task_tokens.items()}

    return tokens


def load_evaluation_verdicts(
    eval_dir: Path,
) -> dict[int, dict[int, dict[int, bool]]]:
    """Load per-candidate, per-round evaluation verdicts from JSON files.

    Each evaluation file contains a ``details`` list with entries keyed by
    ``(task_id, candidate_index, round_index)`` and a boolean ``verdict``.

    Args:
        eval_dir: Directory containing ``Task_{id}_evaluation.json`` files.

    Returns:
        Mapping ``{task_id: {candidate_index: {round_index: verdict}}}``.

    """
    pattern = re.compile(r"Task_(\d+)_evaluation\.json$")
    verdicts: dict[int, dict[int, dict[int, bool]]] = {}

    for path in sorted(eval_dir.iterdir()):
        task_id = _extract_task_id(path.name, pattern)
        if task_id is None:
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        task_verdicts: dict[int, dict[int, bool]] = defaultdict(dict)

        for detail in data.get("details", []):
            cand_idx = detail.get("candidate_index")
            round_idx = detail.get("round_index")
            verdict = detail.get("verdict")
            if cand_idx is not None and round_idx is not None and verdict is not None:
                task_verdicts[cand_idx][round_idx] = bool(verdict)

        verdicts[task_id] = dict(task_verdicts)

    return verdicts


def load_baseline_evaluation_verdicts(
    eval_dir: Path,
) -> dict[int, dict[int, bool]]:
    """Load per-candidate baseline evaluation verdicts from JSON files.

    Baseline evaluation files contain ``details`` entries where
    ``round_index`` is ``null`` (baseline candidates have no round).
    Each entry has a ``candidate_index`` and a boolean ``verdict``.

    Args:
        eval_dir: Directory containing ``Task_{id}_evaluation.json`` files
            for baseline candidates.

    Returns:
        Mapping ``{task_id: {candidate_index: verdict}}``.

    """
    pattern = re.compile(r"Task_(\d+)_evaluation\.json$")
    verdicts: dict[int, dict[int, bool]] = {}

    for path in sorted(eval_dir.iterdir()):
        task_id = _extract_task_id(path.name, pattern)
        if task_id is None:
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        task_verdicts: dict[int, bool] = {}

        for detail in data.get("details", []):
            cand_idx = detail.get("candidate_index")
            verdict = detail.get("verdict")
            if cand_idx is not None and verdict is not None:
                task_verdicts[cand_idx] = bool(verdict)

        if task_verdicts:
            verdicts[task_id] = task_verdicts

    return verdicts


def compute_baseline_accuracy(
    baseline_verdicts: dict[int, dict[int, bool]],
) -> float:
    """Compute mean accuracy across all candidates and tasks for baseline.

    Args:
        baseline_verdicts: ``{task_id: {candidate_index: verdict}}``
            loaded from baseline evaluation JSON files.

    Returns:
        Mean accuracy (fraction of correct verdicts) across all
        (task, candidate) pairs.  Returns ``NaN`` if no verdicts are
        available.

    """
    correct_count = 0
    total_count = 0

    for cand_verdicts in baseline_verdicts.values():
        for verdict in cand_verdicts.values():
            total_count += 1
            if verdict:
                correct_count += 1

    if total_count == 0:
        return float("nan")
    return correct_count / total_count


def compute_baseline_accuracy_std(
    baseline_verdicts: dict[int, dict[int, bool]],
) -> float:
    """Compute std of per-candidate mean accuracy across candidates for baseline.

    For each candidate, the mean accuracy across all tasks (fraction of
    tasks solved correctly) is computed.  The returned value is the
    population standard deviation of these per-candidate mean accuracies.

    This treats each candidate as an independent observation (seed) and
    measures how much per-candidate accuracy varies across candidates.

    Args:
        baseline_verdicts: ``{task_id: {candidate_index: verdict}}``
            loaded from baseline evaluation JSON files.

    Returns:
        Population std of per-candidate mean accuracies.  Returns ``NaN``
        if no candidates have verdicts.

    """
    if not baseline_verdicts:
        return float("nan")

    # Collect per-candidate verdicts across tasks: {candidate_index: [verdict, ...]}.
    cand_task_verdicts: dict[int, list[float]] = defaultdict(list)
    for task_verdicts in baseline_verdicts.values():
        for cand_idx, verdict in task_verdicts.items():
            cand_task_verdicts[cand_idx].append(1.0 if verdict else 0.0)

    if not cand_task_verdicts:
        return float("nan")

    # Per-candidate mean accuracy.
    cand_means = [sum(vs) / len(vs) for vs in cand_task_verdicts.values()]
    return _population_std(cand_means)


def compute_cumulative_table(
    baseline_tokens: dict[int, dict[int, int]],
    round_tokens: dict[int, dict[int, dict[int, int]]],
) -> list[tuple[int, float, int]]:
    """Compute average cumulative completion tokens per round.

    Round 0 uses **all** baseline candidates for common tasks, so that
    the baseline cost is identical regardless of which rounds directory
    is provided.  Round *i* (i >= 1) adds self-improvement round
    ``i - 1`` tokens to the previous cumulative total; for these rounds
    only candidates present in both directories are included.

    Args:
        baseline_tokens: ``{task_id: {candidate_index: tokens}}`` from baseline.
        round_tokens: ``{task_id: {candidate_index: {round_index: tokens}}}``
            from self-improvement.

    Returns:
        Sorted list of ``(round_number, avg_cumulative_tokens, pair_count)``
        tuples, where *pair_count* is the number of (task, candidate) pairs
        used for averaging.

    """
    # Find common task IDs.
    common_tasks = sorted(set(baseline_tokens) & set(round_tokens))
    if not common_tasks:
        return []

    # Determine the maximum round_index across all tasks/candidates.
    max_round_idx = 0
    for task_id in common_tasks:
        for cand_rounds in round_tokens[task_id].values():
            if cand_rounds:
                max_round_idx = max(max_round_idx, *cand_rounds)

    n_rounds = max_round_idx + 1  # self-improvement rounds (0-indexed)
    total_output_rounds = n_rounds + 1  # +1 for baseline (round 0)

    # Accumulate per-round sums and counts.
    round_sums: list[float] = [0.0] * total_output_rounds
    round_counts: list[int] = [0] * total_output_rounds

    for task_id in common_tasks:
        baseline_cands = baseline_tokens[task_id]
        round_cands = round_tokens[task_id]

        # Round 0: use ALL baseline candidates (not filtered by rounds dir).
        for cand_idx in sorted(baseline_cands):
            round_sums[0] += baseline_cands[cand_idx]
            round_counts[0] += 1

        # Rounds 1..N: only candidates present in both.
        common_cands = sorted(set(baseline_cands) & set(round_cands))
        for cand_idx in common_cands:
            cumulative = baseline_cands[cand_idx]
            cand_rounds = round_cands[cand_idx]
            for ri in range(n_rounds):
                cumulative += cand_rounds.get(ri, 0)
                round_sums[ri + 1] += cumulative
                round_counts[ri + 1] += 1

    table: list[tuple[int, float, int]] = []
    for r in range(total_output_rounds):
        avg = round_sums[r] / round_counts[r] if round_counts[r] > 0 else 0.0
        table.append((r, avg, round_counts[r]))

    return table


def compute_accuracy_per_round(
    eval_verdicts: dict[int, dict[int, dict[int, bool]]],
    n_output_rounds: int,
    baseline_accuracy: float = float("nan"),
) -> list[float]:
    """Compute mean accuracy for each output round from evaluation verdicts.

    Output round 0 (baseline / initial candidates) uses
    ``baseline_accuracy`` which is computed separately from the
    candidates evaluation directory.  For output round *r* (r >= 1),
    accuracy is the fraction of (task, candidate) pairs with a correct
    verdict at evaluation ``round_index = r - 1``.

    Args:
        eval_verdicts: ``{task_id: {candidate_index: {round_index: verdict}}}``
            loaded from evaluation JSON files.
        n_output_rounds: Total number of output rounds (including round 0).
        baseline_accuracy: Pre-computed accuracy for round 0 (baseline).
            Defaults to ``NaN`` when no baseline evaluation is available.

    Returns:
        List of mean accuracies, one per output round.

    """
    accuracies: list[float] = [float("nan")] * n_output_rounds
    accuracies[0] = baseline_accuracy

    for output_round in range(1, n_output_rounds):
        eval_round = output_round - 1
        correct_count = 0
        total_count = 0

        for cand_verdicts in eval_verdicts.values():
            for round_verdicts in cand_verdicts.values():
                if eval_round in round_verdicts:
                    total_count += 1
                    if round_verdicts[eval_round]:
                        correct_count += 1

        if total_count > 0:
            accuracies[output_round] = correct_count / total_count

    return accuracies


def compute_accuracy_std_per_round(
    eval_verdicts: dict[int, dict[int, dict[int, bool]]],
    n_output_rounds: int,
    baseline_std: float = float("nan"),
) -> list[float]:
    """Compute std of per-candidate mean accuracy for each output round.

    For each output round *r* (r >= 1, corresponding to evaluation
    ``round_index = r - 1``), this function computes per-candidate mean
    accuracy across tasks, then returns the population standard deviation
    of these per-candidate means.

    This treats each candidate as an independent observation (seed) and
    measures how much per-candidate accuracy varies across candidates.

    Output round 0 uses the pre-computed ``baseline_std``.

    Args:
        eval_verdicts: ``{task_id: {candidate_index: {round_index: verdict}}}``
            loaded from evaluation JSON files.
        n_output_rounds: Total number of output rounds (including round 0).
        baseline_std: Pre-computed std for round 0 (baseline).
            Defaults to ``NaN`` when no baseline evaluation is available.

    Returns:
        List of per-candidate accuracy standard deviations, one per
        output round.

    """
    stds: list[float] = [float("nan")] * n_output_rounds
    stds[0] = baseline_std

    for output_round in range(1, n_output_rounds):
        eval_round = output_round - 1

        # Collect per-candidate verdicts across tasks: {cand_idx: [verdict, ...]}.
        cand_task_verdicts: dict[int, list[float]] = defaultdict(list)
        for cand_verdicts in eval_verdicts.values():
            for cand_idx, round_verdicts in cand_verdicts.items():
                if eval_round in round_verdicts:
                    cand_task_verdicts[cand_idx].append(1.0 if round_verdicts[eval_round] else 0.0)

        if cand_task_verdicts:
            cand_means = [sum(vs) / len(vs) for vs in cand_task_verdicts.values()]
            stds[output_round] = _population_std(cand_means)

    return stds


def print_table(
    table: list[tuple[int, float, int]],
    accuracies: list[float] | None = None,
    accuracy_stds: list[float] | None = None,
) -> None:
    """Print the round / avg_cumulative_tokens / count / mean_accuracy / std_accuracy table.

    Args:
        table: List of ``(round_number, avg_cumulative_tokens, pair_count)``
            tuples.
        accuracies: Optional list of mean accuracy values, one per round.
            If ``None``, the accuracy column is omitted.
        accuracy_stds: Optional list of accuracy std values, one per round.
            If ``None``, the std column is omitted.

    """
    if accuracies is not None:
        has_std = accuracy_stds is not None
        header = f"{'round':<8}{'avg_cumulative_tokens':>22}{'n_pairs':>10}{'mean_accuracy':>16}"
        sep_len = 56
        if has_std:
            header += f"{'std_accuracy':>16}"
            sep_len += 16
        print(header)  # noqa: T201
        print("-" * sep_len)  # noqa: T201
        for i, (round_num, avg_tokens, count) in enumerate(table):
            acc = accuracies[i] if i < len(accuracies) else float("nan")
            acc_str = "NaN" if math.isnan(acc) else f"{acc:.4f}"
            line = f"{round_num:<8}{avg_tokens:>22.1f}{count:>10}{acc_str:>16}"
            if has_std:
                std_val = accuracy_stds[i] if i < len(accuracy_stds) else float("nan")
                std_str = "NaN" if math.isnan(std_val) else f"{std_val:.4f}"
                line += f"{std_str:>16}"
            print(line)  # noqa: T201
    else:
        print(f"{'round':<8}{'avg_cumulative_tokens':>22}{'n_pairs':>10}")  # noqa: T201
        print("-" * 40)  # noqa: T201
        for round_num, avg_tokens, count in table:
            print(f"{round_num:<8}{avg_tokens:>22.1f}{count:>10}")  # noqa: T201


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Count cumulative completion tokens per round "
            "(baseline + self-improvement) and report the average "
            "across tasks and candidates, optionally with per-round accuracy."
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
            "Directory with self-improvement JSONL files "
            "(e.g. data/ablations/ablation_debug_self_improve_no_verification_gpt_oss)."
        ),
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default=None,
        help=(
            "Directory with round-level evaluation JSON files. If not "
            "specified, defaults to ``{rounds_dir}/evaluation``. If the "
            "directory does not exist, round accuracy reporting is skipped. "
            "Baseline (round 0) accuracy is always read from "
            "``{candidates_dir}/evaluation`` when available."
        ),
    )
    return parser.parse_args()


def _resolve_accuracies(
    candidates_dir: Path,
    rounds_eval_dir: Path | None,
    n_rounds: int,
) -> tuple[list[float] | None, list[float] | None]:
    """Resolve per-round accuracy values and stds from evaluation directories.

    Baseline (round 0) accuracy and std are read from
    ``candidates_dir/evaluation``.  Rounds 1..N accuracy and std are read
    from *rounds_eval_dir*.  Returns ``(None, None)`` when neither source
    is available, which suppresses the accuracy and std columns in the
    output table.

    Args:
        candidates_dir: Baseline candidates directory (may contain
            ``evaluation/`` subfolder).
        rounds_eval_dir: Directory with round-level evaluation files,
            or ``None`` if unavailable.
        n_rounds: Total number of output rounds (including round 0).

    Returns:
        Tuple of ``(accuracies, accuracy_stds)``, each a list of values
        (one per round) or ``None`` if no evaluation data is available.

    """
    # Compute baseline (round 0) accuracy and std from candidates evaluation.
    baseline_acc = float("nan")
    baseline_std = float("nan")
    baseline_eval_candidate = candidates_dir / "evaluation"
    if baseline_eval_candidate.is_dir():
        baseline_verdicts = load_baseline_evaluation_verdicts(baseline_eval_candidate)
        if baseline_verdicts:
            baseline_acc = compute_baseline_accuracy(baseline_verdicts)
            baseline_std = compute_baseline_accuracy_std(baseline_verdicts)

    has_any_accuracy = not math.isnan(baseline_acc)
    accuracies: list[float] | None = None
    accuracy_stds: list[float] | None = None

    # Compute per-round accuracy and std if evaluation data is available.
    if rounds_eval_dir is not None and rounds_eval_dir.is_dir():
        eval_verdicts = load_evaluation_verdicts(rounds_eval_dir)
        if eval_verdicts:
            accuracies = compute_accuracy_per_round(eval_verdicts, n_rounds, baseline_accuracy=baseline_acc)
            accuracy_stds = compute_accuracy_std_per_round(eval_verdicts, n_rounds, baseline_std=baseline_std)
            has_any_accuracy = True
        else:
            print(  # noqa: T201
                "Warning: evaluation directory exists but contains no evaluation files.",
                file=sys.stderr,
            )

    # If we only have baseline accuracy (no rounds eval), still show columns.
    if accuracies is None and has_any_accuracy:
        accuracies = [float("nan")] * n_rounds
        accuracies[0] = baseline_acc
        accuracy_stds = [float("nan")] * n_rounds
        accuracy_stds[0] = baseline_std

    return accuracies, accuracy_stds


def main() -> None:
    """Load token data from both directories and print the cumulative table."""
    args = parse_args()

    candidates_dir = Path(args.candidates_dir)
    rounds_dir = Path(args.rounds_dir)

    if not candidates_dir.is_dir():
        print(f"Error: candidates_dir not found: {candidates_dir}", file=sys.stderr)  # noqa: T201
        sys.exit(1)
    if not rounds_dir.is_dir():
        print(f"Error: rounds_dir not found: {rounds_dir}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    # Resolve rounds evaluation directory.
    rounds_eval_dir: Path | None = None
    if args.eval_dir is not None:
        rounds_eval_dir = Path(args.eval_dir)
    else:
        candidate_eval_dir = rounds_dir / "evaluation"
        if candidate_eval_dir.is_dir():
            rounds_eval_dir = candidate_eval_dir

    baseline_tokens = load_baseline_tokens(candidates_dir)
    round_tokens = load_round_tokens(rounds_dir)

    if not baseline_tokens:
        print("Error: no baseline JSONL files found.", file=sys.stderr)  # noqa: T201
        sys.exit(1)
    if not round_tokens:
        print("Error: no self-improvement JSONL files found.", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    table = compute_cumulative_table(baseline_tokens, round_tokens)
    if not table:
        print("Error: no common tasks found between baseline and rounds.", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    accuracies, accuracy_stds = _resolve_accuracies(candidates_dir, rounds_eval_dir, len(table))
    print_table(table, accuracies, accuracy_stds)


if __name__ == "__main__":
    main()
