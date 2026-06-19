r"""Ablation: compare selection and merge strategies on pre-generated candidates.

This script loads candidates produced by
:mod:`open_deep_think.scripts.ablation_generate_candidates` and evaluates six
pairwise comparison/merge strategies on ordered solution pairs.

For each task with ``2*K`` candidates (indices ``0, 1, …, 2K-1``), the script
forms ``2*K`` ordered pairs:

    ``(0, 1), (1, 0), (2, 3), (3, 2), …``

Each pair is presented in both orders so that positional bias can be measured.

The six methods evaluated on every pair are:

1. **clean_select** — judge picks the better solution (no verification reports).
2. **clean_merge** — merger synthesises a new solution (no verification reports).
3. **verification_based_select** — judge picks the better solution (with
   verification reports).
4. **verification_based_merge** — merger synthesises a new solution (with
   verification reports).
5. **clean_select_improve** — clean selection followed by one round of
   self-improvement using the selected solution's verification report.
6. **verification_based_select_improve** — verification-based selection followed
   by one round of self-improvement using the selected solution's verification
   report.

Concurrency is managed via a :class:`~concurrent.futures.ThreadPoolExecutor`
operating on ``(task_id, solution_pair)`` work items.  All LLM calls are logged
to ``llm_outputs.jsonl``; method results are logged to ``results.jsonl``.

Usage::

    python -m open_deep_think.scripts.ablation_compare_matches \
        --inputs_path data/subset_candidates_debug_gpt_oss \
        --model openai/gpt-oss-120b \
        --concurrency 10 \
        --output_path data/ \
        --run_name ablation_compare_subset_1
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import urllib3

from open_deep_think.imo_answer_bench.templates import (
    CLEAN_COMPARISON_SYSTEM_PROMPT,
    CLEAN_MERGE_SYSTEM_PROMPT,
    IMO25_CORRECTION_PROMPT,
    IMO25_STEP1_SYSTEM_PROMPT,
    TOURNAMENT_COMPARISON_SYSTEM_PROMPT,
    TOURNAMENT_MERGE_SYSTEM_PROMPT,
    build_clean_comparison_prompt,
    build_clean_merge_prompt,
    build_tournament_comparison_prompt,
    build_tournament_merge_prompt,
)
from open_deep_think.scripts.tournament_merge_improve import (
    TaskCallLogger,
    call_model,
    utc_now_iso,
    write_json,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LOGGER = logging.getLogger(__name__)
_PICK_PATTERN = re.compile(r"\b([12])\b")

ALL_METHODS = (
    "clean_select",
    "clean_merge",
    "verification_based_select",
    "verification_based_merge",
    "clean_select_improve",
    "verification_based_select_improve",
)
"""Names of the six comparison/merge methods evaluated by this script."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateData:
    """Pre-generated candidate loaded from ``candidates.jsonl``.

    Attributes:
        task_id: Dataset task identifier.
        candidate_index: Zero-based candidate index within the task.
        solution_text: Full solution text.
        verification: Nested dict with ``is_pass``, ``verifier_output``, etc.

    """

    task_id: int
    candidate_index: int
    solution_text: str
    verification: dict[str, Any]


@dataclass(frozen=True)
class PairWorkItem:
    """A single ``(task_id, ordered_pair)`` unit of work.

    Attributes:
        task_id: Dataset task identifier.
        solution_a: First solution in the pair (presented as Solution 1).
        solution_b: Second solution in the pair (presented as Solution 2).
        problem_statement: The original problem text.

    """

    task_id: int
    solution_a: CandidateData
    solution_b: CandidateData
    problem_statement: str


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for the ablation comparison pipeline.

    Attributes:
        model: Model name used for selection, merging, and self-improvement.
        max_tokens: Maximum output tokens for all model calls.
        temperature: Sampling temperature.
        top_p: Nucleus sampling top_p.

    """

    model: str
    max_tokens: int
    temperature: float | None
    top_p: float | None


# ---------------------------------------------------------------------------
# Thread-safe JSONL writer (reused from ablation_generate_candidates)
# ---------------------------------------------------------------------------


class ThreadSafeJSONLWriter:
    """Append-only JSONL writer safe for concurrent use from multiple threads.

    Each :meth:`append` call serialises the payload to JSON, acquires an
    internal lock, opens the file in append mode, writes one line, and closes
    the file — ensuring atomic per-record writes even under high concurrency.

    Args:
        path: Filesystem path to the JSONL file.

    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Return the underlying file path."""
        return self._path

    def append(self, payload: dict[str, Any]) -> None:
        """Serialise *payload* and append it as a single JSON line."""
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)


# ---------------------------------------------------------------------------
# Loading candidates
# ---------------------------------------------------------------------------


def load_candidates(candidates_jsonl_path: Path) -> dict[int, list[CandidateData]]:
    """Load all candidate records from a ``candidates.jsonl`` file.

    Returns a mapping from ``task_id`` to a list of :class:`CandidateData`
    ordered by ``candidate_index``.

    Args:
        candidates_jsonl_path: Path to the JSONL file produced by
            :mod:`ablation_generate_candidates`.

    Returns:
        Dict mapping task IDs to sorted lists of candidates.

    Raises:
        FileNotFoundError: If the JSONL file does not exist.

    """
    candidates: dict[int, list[CandidateData]] = {}
    with candidates_jsonl_path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            cand = CandidateData(
                task_id=record["task_id"],
                candidate_index=record["candidate_index"],
                solution_text=record["solution_text"],
                verification=record["verification"],
            )
            candidates.setdefault(cand.task_id, []).append(cand)

    # Sort by candidate_index within each task.
    for cand_list in candidates.values():
        cand_list.sort(key=lambda c: c.candidate_index)

    return candidates


def load_problem_statements(inputs_path: Path) -> dict[int, str]:
    """Load problem statements from the ``config.json`` alongside candidates.

    The config written by :mod:`ablation_generate_candidates` contains the
    ``task_ids`` list.  We need the dataset to get problem statements.  However,
    to avoid requiring the HuggingFace dataset at load time (and to keep tests
    simple), the caller is responsible for providing problem statements.

    This function is a thin wrapper that reads task IDs from config and returns
    them as a list so the caller can look them up from the dataset.

    Args:
        inputs_path: Directory containing ``config.json``.

    Returns:
        Dict mapping task_id to an empty string (caller must fill in).

    """
    config_path = inputs_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return dict.fromkeys(config["task_ids"], "")


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------


def build_ordered_pairs(
    candidates: list[CandidateData],
) -> list[tuple[CandidateData, CandidateData]]:
    """Build ordered solution pairs from a list of candidates.

    Given candidates with indices ``[0, 1, 2, 3, …, 2K-1]``, produces pairs::

        (0, 1), (1, 0), (2, 3), (3, 2), …

    Each consecutive pair of candidates is presented in both orders to allow
    measurement of positional bias.

    Args:
        candidates: Sorted list of candidates (by ``candidate_index``).

    Returns:
        List of ``(solution_a, solution_b)`` tuples.

    Raises:
        ValueError: If the number of candidates is not even.

    """
    if len(candidates) % 2 != 0:
        msg = f"Expected an even number of candidates, got {len(candidates)}"
        raise ValueError(msg)

    pairs: list[tuple[CandidateData, CandidateData]] = []
    for i in range(0, len(candidates), 2):
        a, b = candidates[i], candidates[i + 1]
        pairs.append((a, b))  # original order
        pairs.append((b, a))  # reversed order
    return pairs


# ---------------------------------------------------------------------------
# Judge helper
# ---------------------------------------------------------------------------


def parse_pick(text: str) -> int:
    """Parse the judge's pick from its response text.

    Returns 1 or 2 if found, defaulting to 1 if neither is present.

    Args:
        text: Raw judge response text.

    Returns:
        1 or 2 indicating which solution was chosen.

    """
    match = _PICK_PATTERN.search(text.strip())
    if match:
        return int(match.group(1))
    LOGGER.warning("Could not parse judge pick from: %r — defaulting to 1", text[:200])
    return 1


# ---------------------------------------------------------------------------
# Individual method implementations
# ---------------------------------------------------------------------------


def run_clean_select(
    *,
    work_item: PairWorkItem,
    config: AblationConfig,
    call_logger: TaskCallLogger,
) -> dict[str, Any]:
    """Run clean selection (no verification reports).

    The judge sees only the two solutions and picks 1 or 2.

    Args:
        work_item: The pair to evaluate.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.

    Returns:
        Result dict with ``selected_index`` (1 or 2) and ``result_solution_text``.

    """
    prompt = build_clean_comparison_prompt(
        problem=work_item.problem_statement,
        solution_1=work_item.solution_a.solution_text,
        solution_2=work_item.solution_b.solution_text,
    )
    messages = [
        {"role": "system", "content": CLEAN_COMPARISON_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    result = call_model(
        model=config.model,
        messages=messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        phase="clean_select",
        candidate_index=None,
        round_index=None,
        call_logger=call_logger,
    )
    pick = parse_pick(result.text)
    selected = work_item.solution_a if pick == 1 else work_item.solution_b
    return {
        "selected_index": pick,
        "selected_candidate_index": selected.candidate_index,
        "result_solution_text": selected.solution_text,
    }


def run_verification_based_select(
    *,
    work_item: PairWorkItem,
    config: AblationConfig,
    call_logger: TaskCallLogger,
) -> dict[str, Any]:
    """Run verification-based selection (with verification reports).

    The judge sees both solutions and their verification reports.

    Args:
        work_item: The pair to evaluate.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.

    Returns:
        Result dict with ``selected_index`` (1 or 2) and ``result_solution_text``.

    """
    prompt = build_tournament_comparison_prompt(
        problem=work_item.problem_statement,
        solution_1=work_item.solution_a.solution_text,
        verification_1=work_item.solution_a.verification.get("verifier_output", ""),
        solution_2=work_item.solution_b.solution_text,
        verification_2=work_item.solution_b.verification.get("verifier_output", ""),
    )
    messages = [
        {"role": "system", "content": TOURNAMENT_COMPARISON_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    result = call_model(
        model=config.model,
        messages=messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        phase="verification_based_select",
        candidate_index=None,
        round_index=None,
        call_logger=call_logger,
    )
    pick = parse_pick(result.text)
    selected = work_item.solution_a if pick == 1 else work_item.solution_b
    return {
        "selected_index": pick,
        "selected_candidate_index": selected.candidate_index,
        "result_solution_text": selected.solution_text,
    }


def run_clean_merge(
    *,
    work_item: PairWorkItem,
    config: AblationConfig,
    call_logger: TaskCallLogger,
) -> dict[str, Any]:
    """Run clean merge (no verification reports).

    The merger sees only the two solutions and produces a merged solution.

    Args:
        work_item: The pair to evaluate.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.

    Returns:
        Result dict with ``result_solution_text``.

    """
    prompt = build_clean_merge_prompt(
        problem=work_item.problem_statement,
        solution_1=work_item.solution_a.solution_text,
        solution_2=work_item.solution_b.solution_text,
    )
    messages = [
        {"role": "system", "content": CLEAN_MERGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    result = call_model(
        model=config.model,
        messages=messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        phase="clean_merge",
        candidate_index=None,
        round_index=None,
        call_logger=call_logger,
    )
    return {"result_solution_text": result.text}


def run_verification_based_merge(
    *,
    work_item: PairWorkItem,
    config: AblationConfig,
    call_logger: TaskCallLogger,
) -> dict[str, Any]:
    """Run verification-based merge (with verification reports).

    The merger sees both solutions and their verification reports.

    Args:
        work_item: The pair to evaluate.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.

    Returns:
        Result dict with ``result_solution_text``.

    """
    prompt = build_tournament_merge_prompt(
        problem=work_item.problem_statement,
        solution_1=work_item.solution_a.solution_text,
        verification_1=work_item.solution_a.verification.get("verifier_output", ""),
        solution_2=work_item.solution_b.solution_text,
        verification_2=work_item.solution_b.verification.get("verifier_output", ""),
    )
    messages = [
        {"role": "system", "content": TOURNAMENT_MERGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    result = call_model(
        model=config.model,
        messages=messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        phase="verification_based_merge",
        candidate_index=None,
        round_index=None,
        call_logger=call_logger,
    )
    return {"result_solution_text": result.text}


def _run_self_improvement(  # noqa: PLR0913
    *,
    problem_statement: str,
    solution_text: str,
    verification_report: str,
    config: AblationConfig,
    call_logger: TaskCallLogger,
    phase: str,
) -> str:
    """Run one round of self-improvement on a solution using its verification report.

    The solver is presented with the original problem, its current solution, and
    the verification report (via :data:`IMO25_CORRECTION_PROMPT`).  It is asked
    to fix any issues identified in the report.

    Self-improvement is always run (even if verification passed) to enable
    fair comparison with merge methods.

    Args:
        problem_statement: The original problem text.
        solution_text: The solution to improve.
        verification_report: Verifier output for *solution_text*, appended
            after :data:`IMO25_CORRECTION_PROMPT` so the model can address
            the specific issues found.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.
        phase: Phase name for logging (distinguishes clean vs verification-based).

    Returns:
        The improved solution text.

    """
    correction_content = f"{IMO25_CORRECTION_PROMPT}\n\n{verification_report}"
    messages = [
        {"role": "system", "content": IMO25_STEP1_SYSTEM_PROMPT},
        {"role": "user", "content": problem_statement},
        {"role": "assistant", "content": solution_text},
        {"role": "user", "content": correction_content},
    ]
    result = call_model(
        model=config.model,
        messages=messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        phase=phase,
        candidate_index=None,
        round_index=None,
        call_logger=call_logger,
    )
    return result.text


def run_clean_select_improve(
    *,
    work_item: PairWorkItem,
    config: AblationConfig,
    call_logger: TaskCallLogger,
) -> dict[str, Any]:
    """Run clean selection followed by one round of self-improvement.

    Selection uses no verification reports.  Self-improvement always runs
    (even if the selected solution's verification passed).

    Args:
        work_item: The pair to evaluate.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.

    Returns:
        Result dict with selection info and improved ``result_solution_text``.

    """
    select_result = run_clean_select(
        work_item=work_item,
        config=config,
        call_logger=call_logger,
    )
    selected = work_item.solution_a if select_result["selected_index"] == 1 else work_item.solution_b
    improved_text = _run_self_improvement(
        problem_statement=work_item.problem_statement,
        solution_text=select_result["result_solution_text"],
        verification_report=selected.verification.get("verifier_output", ""),
        config=config,
        call_logger=call_logger,
        phase="clean_select_improve_si",
    )
    return {
        "selected_index": select_result["selected_index"],
        "selected_candidate_index": select_result["selected_candidate_index"],
        "result_solution_text": improved_text,
    }


def run_verification_based_select_improve(
    *,
    work_item: PairWorkItem,
    config: AblationConfig,
    call_logger: TaskCallLogger,
) -> dict[str, Any]:
    """Run verification-based selection followed by one round of self-improvement.

    Selection uses verification reports.  Self-improvement always runs
    (even if the selected solution's verification passed).

    Args:
        work_item: The pair to evaluate.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.

    Returns:
        Result dict with selection info and improved ``result_solution_text``.

    """
    select_result = run_verification_based_select(
        work_item=work_item,
        config=config,
        call_logger=call_logger,
    )
    selected = work_item.solution_a if select_result["selected_index"] == 1 else work_item.solution_b
    improved_text = _run_self_improvement(
        problem_statement=work_item.problem_statement,
        solution_text=select_result["result_solution_text"],
        verification_report=selected.verification.get("verifier_output", ""),
        config=config,
        call_logger=call_logger,
        phase="verification_based_select_improve_si",
    )
    return {
        "selected_index": select_result["selected_index"],
        "selected_candidate_index": select_result["selected_candidate_index"],
        "result_solution_text": improved_text,
    }


# ---------------------------------------------------------------------------
# Method dispatch
# ---------------------------------------------------------------------------

_METHOD_RUNNERS = {
    "clean_select": run_clean_select,
    "clean_merge": run_clean_merge,
    "verification_based_select": run_verification_based_select,
    "verification_based_merge": run_verification_based_merge,
    "clean_select_improve": run_clean_select_improve,
    "verification_based_select_improve": run_verification_based_select_improve,
}


def run_method(
    *,
    method: str,
    work_item: PairWorkItem,
    config: AblationConfig,
    call_logger: TaskCallLogger,
) -> dict[str, Any]:
    """Dispatch to the appropriate method runner.

    Args:
        method: One of :data:`ALL_METHODS`.
        work_item: The pair to evaluate.
        config: Pipeline configuration.
        call_logger: Logger for LLM calls.

    Returns:
        Method-specific result dict.

    Raises:
        ValueError: If *method* is not recognised.

    """
    runner = _METHOD_RUNNERS.get(method)
    if runner is None:
        msg = f"Unknown method: {method!r}. Must be one of {ALL_METHODS}"
        raise ValueError(msg)
    return runner(work_item=work_item, config=config, call_logger=call_logger)


# ---------------------------------------------------------------------------
# Single work-item processor
# ---------------------------------------------------------------------------


def process_pair(
    *,
    work_item: PairWorkItem,
    config: AblationConfig,
    run_dir: Path,
    results_writer: ThreadSafeJSONLWriter,
    llm_writer: ThreadSafeJSONLWriter,
) -> list[dict[str, Any]]:
    """Run all six methods on a single ordered pair and log results.

    Creates a per-pair :class:`TaskCallLogger` whose output is forwarded to
    the shared ``llm_outputs.jsonl`` writer for thread-safe appending.

    Args:
        work_item: The ``(task_id, solution_a, solution_b)`` pair to process.
        config: Pipeline configuration.
        run_dir: Output directory for per-pair LLM log files.
        results_writer: Thread-safe writer for method results.
        llm_writer: Thread-safe writer for LLM call logs.

    Returns:
        List of result dicts (one per method).

    """
    task_id = work_item.task_id
    a_idx = work_item.solution_a.candidate_index
    b_idx = work_item.solution_b.candidate_index

    # Per-pair LLM log file avoids conflicts between concurrent workers.
    llm_log_path = run_dir / f"Task_{task_id}_pair_{a_idx}_{b_idx}_llm_outputs.jsonl"
    call_logger = TaskCallLogger(task_id=task_id, task_log_path=llm_log_path)

    records: list[dict[str, Any]] = []
    for method in ALL_METHODS:
        record: dict[str, Any] = {
            "task_id": task_id,
            "solution_a_index": a_idx,
            "solution_b_index": b_idx,
            "method": method,
            "timestamp": utc_now_iso(),
        }
        try:
            method_result = run_method(
                method=method,
                work_item=work_item,
                config=config,
                call_logger=call_logger,
            )
            record.update(method_result)
            record["status"] = "ok"
        except Exception as exc:
            LOGGER.exception(
                "Task %s pair (%s, %s) method %s failed",
                task_id,
                a_idx,
                b_idx,
                method,
            )
            record["status"] = "error"
            record["error"] = str(exc)
            record["result_solution_text"] = ""

        records.append(record)
        results_writer.append(record)
        LOGGER.info(
            "Task %s pair (%s, %s) method %s: status=%s",
            task_id,
            a_idx,
            b_idx,
            method,
            record["status"],
        )

    # Copy per-pair LLM logs to the shared llm_outputs.jsonl.
    if llm_log_path.exists():
        with llm_log_path.open(encoding="utf-8") as fh:
            for raw_line in fh:
                stripped = raw_line.strip()
                if stripped:
                    llm_writer.append(json.loads(stripped))

    return records


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def load_completed_pairs(results_jsonl_path: Path) -> set[tuple[int, int, int, str]]:
    """Return ``(task_id, a_index, b_index, method)`` tuples already in results.

    This provides fine-grained resumability: individual method runs on specific
    pairs that have already completed are skipped on the next invocation.

    Args:
        results_jsonl_path: Path to the ``results.jsonl`` file.

    Returns:
        Set of already-completed ``(task_id, a_index, b_index, method)`` tuples.

    """
    if not results_jsonl_path.exists():
        return set()

    completed: set[tuple[int, int, int, str]] = set()
    with results_jsonl_path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            completed.add(
                (
                    record["task_id"],
                    record["solution_a_index"],
                    record["solution_b_index"],
                    record["method"],
                )
            )
    return completed


def load_completed_pair_keys(results_jsonl_path: Path) -> set[tuple[int, int, int]]:
    """Return ``(task_id, a_index, b_index)`` tuples fully completed in results.

    A pair key is considered fully completed when all six methods have records.

    Args:
        results_jsonl_path: Path to the ``results.jsonl`` file.

    Returns:
        Set of fully-completed ``(task_id, a_index, b_index)`` triples.

    """
    completed_methods = load_completed_pairs(results_jsonl_path)

    # Group by (task_id, a_index, b_index) and count methods.
    pair_method_counts: dict[tuple[int, int, int], int] = {}
    for task_id, a_idx, b_idx, _method in completed_methods:
        key = (task_id, a_idx, b_idx)
        pair_method_counts[key] = pair_method_counts.get(key, 0) + 1

    return {key for key, count in pair_method_counts.items() if count >= len(ALL_METHODS)}


# ---------------------------------------------------------------------------
# Work item builders
# ---------------------------------------------------------------------------


def build_work_items(
    candidates_by_task: dict[int, list[CandidateData]],
    problem_statements: dict[int, str],
    completed_pair_keys: set[tuple[int, int, int]],
) -> list[PairWorkItem]:
    """Build the list of pair work items to process.

    Pairs whose ``(task_id, a_index, b_index)`` key appears in
    *completed_pair_keys* are excluded (resume support).

    Args:
        candidates_by_task: Mapping from task_id to sorted candidate list.
        problem_statements: Mapping from task_id to problem text.
        completed_pair_keys: Already-completed pair keys to skip.

    Returns:
        Ordered list of :class:`PairWorkItem` objects.

    """
    items: list[PairWorkItem] = []
    for task_id in sorted(candidates_by_task):
        cands = candidates_by_task[task_id]
        pairs = build_ordered_pairs(cands)
        for sol_a, sol_b in pairs:
            key = (task_id, sol_a.candidate_index, sol_b.candidate_index)
            if key in completed_pair_keys:
                continue
            items.append(
                PairWorkItem(
                    task_id=task_id,
                    solution_a=sol_a,
                    solution_b=sol_b,
                    problem_statement=problem_statements[task_id],
                )
            )
    return items


# ---------------------------------------------------------------------------
# Concurrent execution
# ---------------------------------------------------------------------------


def run_comparison(  # noqa: PLR0913
    *,
    work_items: list[PairWorkItem],
    config: AblationConfig,
    run_dir: Path,
    results_writer: ThreadSafeJSONLWriter,
    llm_writer: ThreadSafeJSONLWriter,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Execute all work items with up to *concurrency* parallel workers.

    Uses :class:`ThreadPoolExecutor` because the work is I/O-bound (LLM API
    calls).

    Args:
        work_items: Ordered list of pair work items.
        config: Pipeline configuration.
        run_dir: Output directory for per-pair LLM logs.
        results_writer: Thread-safe writer for method results.
        llm_writer: Thread-safe writer for LLM call logs.
        concurrency: Maximum number of parallel workers.

    Returns:
        Flat list of result dicts (six per work item), in completion order.

    """
    all_results: list[dict[str, Any]] = []

    if concurrency <= 1:
        for item in work_items:
            records = process_pair(
                work_item=item,
                config=config,
                run_dir=run_dir,
                results_writer=results_writer,
                llm_writer=llm_writer,
            )
            all_results.extend(records)
        return all_results

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_item = {
            pool.submit(
                process_pair,
                work_item=item,
                config=config,
                run_dir=run_dir,
                results_writer=results_writer,
                llm_writer=llm_writer,
            ): item
            for item in work_items
        }
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                records = future.result()
                all_results.extend(records)
            except Exception:
                LOGGER.exception(
                    "Unexpected worker error for task %s pair (%s, %s)",
                    item.task_id,
                    item.solution_a.candidate_index,
                    item.solution_b.candidate_index,
                )

    return all_results


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ablation comparison."""
    parser = argparse.ArgumentParser(
        description="Ablation: compare selection and merge strategies on pre-generated candidates",
    )
    parser.add_argument(
        "--inputs_path",
        type=str,
        required=True,
        help="Path to directory containing candidates.jsonl (output of ablation_generate_candidates)",
    )
    parser.add_argument("--model", type=str, required=True, help="Model name for selection/merge/improvement calls")
    parser.add_argument("--output_path", type=str, required=True, help="Base output directory")
    parser.add_argument("--run_name", type=str, default="default", help="Run directory name (default: 'default')")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent (task, pair) workers (default: 1)",
    )
    parser.add_argument("--max_tokens", type=int, default=64000, help="Maximum output tokens for all model calls")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top_p")

    # Dataset (needed to look up problem statements).
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Hwilner/imo-answerbench",
        help="Hugging Face dataset name",
    )
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument constraints.

    Args:
        args: Parsed namespace.

    Raises:
        ValueError: On invalid arguments.

    """
    if args.concurrency < 1:
        msg = f"--concurrency must be >= 1, got {args.concurrency}"
        raise ValueError(msg)

    inputs_path = Path(args.inputs_path)
    if not inputs_path.exists():
        msg = f"--inputs_path does not exist: {inputs_path}"
        raise ValueError(msg)

    candidates_path = inputs_path / "candidates.jsonl"
    if not candidates_path.exists():
        msg = f"candidates.jsonl not found in {inputs_path}"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Logging & main
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    """Configure stdout-only logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def main() -> None:
    """Run ablation comparison over pre-generated candidates."""
    args = parse_args()
    validate_args(args)
    configure_logging()

    inputs_path = Path(args.inputs_path)
    candidates_jsonl_path = inputs_path / "candidates.jsonl"

    run_dir = Path(args.output_path) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    results_jsonl_path = run_dir / "results.jsonl"
    llm_outputs_path = run_dir / "llm_outputs.jsonl"

    results_writer = ThreadSafeJSONLWriter(results_jsonl_path)
    llm_writer = ThreadSafeJSONLWriter(llm_outputs_path)

    config = AblationConfig(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Write config (idempotent).
    config_path = run_dir / "config.json"
    config_dict: dict[str, Any] = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "inputs_path": str(inputs_path),
        "script": "ablation_compare_matches",
        "concurrency": args.concurrency,
        "methods": list(ALL_METHODS),
    }
    if not config_path.exists():
        write_json(config_path, config_dict)
        LOGGER.info("Config written to %s", config_path)
    else:
        LOGGER.info("Config already exists at %s — skipping write.", config_path)

    # Load candidates.
    LOGGER.info("Loading candidates from %s", candidates_jsonl_path)
    candidates_by_task = load_candidates(candidates_jsonl_path)
    task_ids = sorted(candidates_by_task.keys())
    LOGGER.info("Loaded candidates for %s tasks: %s", len(task_ids), task_ids)

    # Load problem statements from dataset.
    from datasets import load_dataset  # noqa: PLC0415

    LOGGER.info("Loading dataset: %s (%s)", args.dataset_name, args.dataset_split)
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    LOGGER.info("Loaded %s tasks from dataset", len(dataset))

    problem_statements: dict[int, str] = {}
    for tid in task_ids:
        if tid < 0 or tid >= len(dataset):
            msg = f"Task ID {tid} out of range [0, {len(dataset)})"
            raise ValueError(msg)
        problem_statements[tid] = dataset[tid]["Problem"]

    # Determine already-completed pairs.
    completed_pair_keys = load_completed_pair_keys(results_jsonl_path)
    work_items = build_work_items(candidates_by_task, problem_statements, completed_pair_keys)

    total_pairs = sum(len(build_ordered_pairs(cands)) for cands in candidates_by_task.values())
    LOGGER.info(
        "%s total pairs — %s already complete, %s remaining. Concurrency: %s.",
        total_pairs,
        len(completed_pair_keys),
        len(work_items),
        args.concurrency,
    )

    if not work_items:
        LOGGER.info("Nothing to do — all pairs already processed.")
        return

    results = run_comparison(
        work_items=work_items,
        config=config,
        run_dir=run_dir,
        results_writer=results_writer,
        llm_writer=llm_writer,
        concurrency=args.concurrency,
    )

    n_ok = sum(1 for r in results if r.get("status") == "ok")
    LOGGER.info(
        "Comparison complete. %s/%s method runs succeeded. Results in %s",
        n_ok,
        len(results),
        results_jsonl_path,
    )


if __name__ == "__main__":
    main()
