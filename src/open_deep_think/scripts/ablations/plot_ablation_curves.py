"""Plot accuracy-vs-tokens curves for ablation experiments.

Reads ``all_results.md`` from each hardcoded ablation folder, extracts
(avg_cumulative_tokens, mean_accuracy) pairs, and plots all curves on a
single log-scale graph.  The output is saved to ``plots/ablation_curves.png``.

Usage::

    uv run python -m open_deep_think.scripts.ablations.plot_ablation_curves
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# -- Hardcoded ablation registry ------------------------------------------------
# Maps a human-readable legend label to the data folder (relative to repo root).
ABLATIONS: dict[str, str] = {
    # "Self-Improve (no verification)": "data/ablations/ablation_self_improve_no_verification_gpt_oss",
    # "Self-Improve (with verification)": "data/ablations/ablation_self_improve_verification_gpt_oss",
    # "Comparison (select best)": "data/ablations/ablation_comparison_gpt_oss",
    # "Merge": "data/ablations/ablation_merge_gpt_oss",
    "Merge + Compare": "data/ablations/ablation_merge_compare_gpt_oss",
    # "Merge + Compare (knockout)": "data/ablations/ablation_merge_compare_knockout_gpt_oss",
    "Merge + Improve + Compare": "data/ablations/ablation_merge_improve_compare_gpt_oss",
    # "Merge + Improve + Compare (knockout)": "data/ablations/ablation_merge_improve_compare_knockout_gpt_oss",
}

RESULTS_FILENAME = "all_results.md"
OUTPUT_DIR = Path("plots")
OUTPUT_FILE = OUTPUT_DIR / "imp-vs-no-imp.png"


# -- Parsing -------------------------------------------------------------------
def parse_results_md(path: Path) -> tuple[list[float], list[float]]:
    """Parse an ``all_results.md`` file and return (tokens, accuracy) lists.

    Handles two column layouts produced by different ablation scripts:

    * **Simple** - ``round  avg_cumulative_tokens  n_tasks  mean_accuracy  [std_accuracy]``
    * **Typed**  - ``round  type  avg_cumulative_tokens  n_tasks  mean_accuracy  std_accuracy``

    The heuristic is straightforward: the header line is inspected for the
    presence of a ``type`` column to decide which layout to use.

    Returns:
        A ``(tokens, accuracies)`` tuple where each element is a list of
        floats with one entry per data row.

    """
    text = path.read_text()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    if not lines:
        msg = f"Empty results file: {path}"
        raise ValueError(msg)

    header = lines[0]
    has_type_column = "type" in header.split()

    tokens: list[float] = []
    accuracies: list[float] = []

    # Data lines start after the header and the separator (dashes).
    for line in lines[1:]:
        # Skip separator lines (e.g. "--------")
        if re.fullmatch(r"[-]+", line):
            continue

        parts = line.split()
        if has_type_column:
            # round  type  avg_cumulative_tokens  n_tasks  mean_accuracy  ...
            tok = float(parts[2])
            acc = float(parts[4])
        else:
            # round  avg_cumulative_tokens  n_tasks  mean_accuracy  ...
            tok = float(parts[1])
            acc = float(parts[3])

        tokens.append(tok)
        accuracies.append(acc * 100)  # convert 0-1 to 0-100 scale

    return tokens, accuracies


# -- Plotting ------------------------------------------------------------------
def plot_ablation_curves() -> None:
    """Load all ablation results, plot them on one graph, and save to disk."""
    repo_root = Path(__file__).resolve().parents[4]  # .../open-deep-think

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, folder in ABLATIONS.items():
        results_path = repo_root / folder / RESULTS_FILENAME
        if not results_path.exists():
            logger.warning("%s not found - skipping '%s'", results_path, label)
            continue

        tokens, accuracies = parse_results_md(results_path)
        ax.plot(tokens, accuracies, marker="o", linewidth=2, markersize=5, label=label)

    ax.set_xscale("log")
    ax.set_ylim(50, 80)
    ax.set_xlabel("Cumulative tokens (avg per task)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Ablation: accuracy vs. token budget", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(visible=True, which="both", linestyle="--", alpha=0.4)

    output_path = repo_root / OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_ablation_curves()
