"""Data-access layer for the tournament viewer.

This module is responsible for turning a directory of tournament logs into
typed Python objects that the Streamlit app and the bracket renderer can
consume.  It deliberately keeps the I/O layer narrow: ``progress.json`` files
are tiny and parsed eagerly to build run summaries, while ``llm_outputs.jsonl``
files (which can reach several megabytes) are streamed on demand and cached
per (run, task).

Two filesystem layouts are supported:

* A *single-run* directory containing ``Task_<id>_progress.json`` files
  directly — the directory's basename is used as the run name.
* A *multi-run* directory whose children are single-run directories — each
  subdirectory becomes a run.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

_TASK_FILE_PATTERN = re.compile(r"Task_(?P<task_id>\d+)_progress\.json$")


@dataclass(frozen=True)
class CandidateNode:
    """A single node in the tournament bracket.

    Initial candidates have non-negative indices in ``[0, num_solutions)``.
    Merged candidates have synthetic negative indices computed as
    ``-(round_index * 1000 + match_index + 1)`` by the pipeline; this module
    accepts the indices as-is and only treats them as opaque identifiers.
    """

    index: int
    kind: str
    """One of ``"initial"`` or ``"merged"``."""
    round_index: int | None
    """Tournament round number for merged nodes; ``None`` for initial ones."""
    match_index: int | None
    """Match number within the round for merged nodes; ``None`` for initial ones."""
    parents: tuple[int, ...]
    """Indices of the two parent candidates that produced this node, when applicable."""
    status: str
    """Pipeline status, e.g. ``"ok"`` or ``"failed"``."""
    verification_pass: bool | None
    """``True`` / ``False`` once a verifier has run; ``None`` if no verifier output is recorded."""
    verifier_output: str
    """Full verifier markdown."""
    classifier_output: str
    """Raw classifier text (typically a yes/no token)."""
    bug_report: str
    """Extracted bug report when verification failed."""
    verifier_call_id: int | None
    """Call id of the verification step that produced the final verdict."""
    classifier_call_id: int | None
    """Call id of the classifier step that produced the final verdict."""


@dataclass(frozen=True)
class Match:
    """One match within a tournament round."""

    round_index: int
    match_index: int
    candidate_a: int
    candidate_b: int
    merged_index: int
    merged_verification_pass: bool | None
    status: str


@dataclass(frozen=True)
class CallRecord:
    """A single LLM call as persisted in ``llm_outputs.jsonl``."""

    call_id: int
    timestamp: str
    phase: str
    candidate_index: int | None
    round_index: int | None
    model: str
    messages: list[dict[str, Any]]
    response_text: str
    reasoning: str | None
    usage: dict[str, Any] | None
    error: str | None


@dataclass(frozen=True)
class TaskSummary:
    """High-level metadata for a task (used in lists and headers)."""

    task_id: int
    status: str
    final_candidate_index: int | None
    final_verification_pass: bool | None
    num_initial: int
    num_initial_passed: int
    problem_preview: str
    solver_model: str
    started_at: str
    completed_at: str | None


@dataclass(frozen=True)
class TaskData:
    """Full parsed task: problem, candidates, bracket, and summary."""

    task_id: int
    run_name: str
    summary: TaskSummary
    pipeline_config: dict[str, Any]
    problem_statement: str
    solution_text: str
    candidates: list[CandidateNode]
    matches: list[Match]
    rounds: list[list[Match]]
    raw_progress: dict[str, Any]


@dataclass(frozen=True)
class RunInfo:
    """Metadata for a single run (directory of Task_*_progress.json files)."""

    name: str
    path: Path
    task_ids: tuple[int, ...]
    task_summaries: dict[int, TaskSummary] = field(default_factory=dict)


def _coerce_pass(value: object) -> bool | None:
    """Best-effort interpretation of a ``is_pass``-style value."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "pass", "1"}:
            return True
        if lowered in {"false", "no", "fail", "0"}:
            return False
    return None


def _candidate_from_payload(
    payload: dict[str, Any],
    *,
    parents: tuple[int, ...] = (),
    round_index: int | None = None,
    match_index: int | None = None,
) -> CandidateNode:
    """Convert a raw candidate payload (from ``progress.json``) into a node.

    Args:
        payload: Raw candidate dict as produced by the pipeline.
        parents: Parent candidate indices for merged nodes; empty for initial ones.
        round_index: Tournament round number for merged nodes; ``None`` for initials.
        match_index: Match index within the round for merged nodes; ``None`` for initials.

    Returns:
        A frozen :class:`CandidateNode`.

    """
    verification = payload.get("verification") or {}
    candidate_index = int(payload["candidate_index"])
    return CandidateNode(
        index=candidate_index,
        kind="initial" if candidate_index >= 0 else "merged",
        round_index=round_index if round_index is not None else payload.get("round_index"),
        match_index=match_index if match_index is not None else payload.get("match_index"),
        parents=parents,
        status=str(payload.get("status", "unknown")),
        verification_pass=_coerce_pass(verification.get("is_pass")),
        verifier_output=str(verification.get("verifier_output", "")),
        classifier_output=str(verification.get("classifier_output", "")),
        bug_report=str(verification.get("bug_report", "")),
        verifier_call_id=verification.get("verifier_call_id"),
        classifier_call_id=verification.get("classifier_call_id"),
    )


def _match_from_payload(payload: dict[str, Any], round_index: int) -> Match:
    """Convert a raw match payload into a :class:`Match`."""
    return Match(
        round_index=round_index,
        match_index=int(payload["match_index"]),
        candidate_a=int(payload["candidate_a"]),
        candidate_b=int(payload["candidate_b"]),
        merged_index=int(payload["merged_index"]),
        merged_verification_pass=_coerce_pass(payload.get("merged_verification_pass")),
        status=str(payload.get("status", "unknown")),
    )


def _summary_from_progress(progress: dict[str, Any]) -> TaskSummary:
    """Build a :class:`TaskSummary` from a parsed ``progress.json``."""
    candidates_payload = progress.get("candidates") or []
    initial_candidates = [c for c in candidates_payload if int(c.get("candidate_index", -1)) >= 0]
    initial_passes = sum(
        1 for c in initial_candidates if _coerce_pass((c.get("verification") or {}).get("is_pass")) is True
    )
    pipeline_config = progress.get("pipeline_config") or {}
    problem = str(progress.get("problem_statement", ""))
    preview = problem.strip().splitlines()[0][:300] if problem.strip() else ""
    final_index = progress.get("final_candidate_index")
    final_pass: bool | None = None
    if final_index is not None:
        final_int = int(final_index)
        for candidate in candidates_payload:
            if int(candidate.get("candidate_index", -1)) == final_int:
                final_pass = _coerce_pass((candidate.get("verification") or {}).get("is_pass"))
                break
        if final_pass is None:
            for round_payload in progress.get("tournament_rounds") or []:
                for match in round_payload.get("matches") or []:
                    if int(match.get("merged_index", 0)) == final_int:
                        final_pass = _coerce_pass(match.get("merged_verification_pass"))
                        break
                if final_pass is not None:
                    break
    return TaskSummary(
        task_id=int(progress["task_id"]),
        status=str(progress.get("status", "unknown")),
        final_candidate_index=final_index if final_index is None else int(final_index),
        final_verification_pass=final_pass,
        num_initial=len(initial_candidates),
        num_initial_passed=initial_passes,
        problem_preview=preview,
        solver_model=str(pipeline_config.get("solver_model", "")),
        started_at=str(progress.get("started_at", "")),
        completed_at=progress.get("completed_at"),
    )


def _build_match_metadata(progress: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Map every merged-candidate index to its origin match metadata.

    Each entry contains ``parents`` (the two source candidate indices),
    ``round_index`` and ``match_index``.  Useful both for populating
    :class:`CandidateNode` fields and for cross-referencing the call stream
    in :func:`calls_for_node`.
    """
    meta: dict[int, dict[str, Any]] = {}
    for round_payload in progress.get("tournament_rounds") or []:
        round_index = int(round_payload["round_index"])
        for match_payload in round_payload.get("matches") or []:
            merged_index = int(match_payload["merged_index"])
            meta[merged_index] = {
                "parents": (int(match_payload["candidate_a"]), int(match_payload["candidate_b"])),
                "round_index": round_index,
                "match_index": int(match_payload["match_index"]),
            }
    return meta


def _read_solution(task_dir: Path, task_id: int) -> str:
    """Read the ``solution.txt`` file for a task, returning empty string if absent."""
    path = task_dir / f"Task_{task_id}_solution.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def parse_progress(progress_path: Path, run_name: str) -> TaskData:
    """Load a ``progress.json`` file and assemble a :class:`TaskData`.

    Args:
        progress_path: Filesystem path to the ``Task_{id}_progress.json`` file.
        run_name: Name of the parent run (used purely for display).

    Returns:
        Fully populated :class:`TaskData`.

    """
    raw = json.loads(progress_path.read_text(encoding="utf-8"))
    match_meta = _build_match_metadata(raw)
    candidates_by_index: dict[int, CandidateNode] = {}
    for candidate_payload in raw.get("candidates") or []:
        idx = int(candidate_payload["candidate_index"])
        meta = match_meta.get(idx, {})
        candidates_by_index[idx] = _candidate_from_payload(
            candidate_payload,
            parents=meta.get("parents", ()),
            round_index=meta.get("round_index"),
            match_index=meta.get("match_index"),
        )

    # Synthesize merged-candidate nodes from the match list — the tournament_merge_improve
    # pipeline does not record them in the `candidates` array, only their pass/fail bit
    # in each match.  Their verifier text lives in the LLM call stream.
    for round_payload in raw.get("tournament_rounds") or []:
        round_index = int(round_payload["round_index"])
        for match_payload in round_payload.get("matches") or []:
            merged_index = int(match_payload["merged_index"])
            if merged_index in candidates_by_index:
                continue
            candidates_by_index[merged_index] = CandidateNode(
                index=merged_index,
                kind="merged",
                round_index=round_index,
                match_index=int(match_payload["match_index"]),
                parents=(int(match_payload["candidate_a"]), int(match_payload["candidate_b"])),
                status=str(match_payload.get("status", "unknown")),
                verification_pass=_coerce_pass(match_payload.get("merged_verification_pass")),
                verifier_output="",
                classifier_output="",
                bug_report="",
                verifier_call_id=None,
                classifier_call_id=None,
            )

    candidates = sorted(
        candidates_by_index.values(),
        key=lambda c: (c.kind != "initial", c.index if c.kind == "initial" else -c.index),
    )

    matches: list[Match] = []
    rounds: list[list[Match]] = []
    for round_payload in raw.get("tournament_rounds") or []:
        round_index = int(round_payload["round_index"])
        round_matches: list[Match] = []
        for match_payload in round_payload.get("matches") or []:
            match = _match_from_payload(match_payload, round_index=round_index)
            matches.append(match)
            round_matches.append(match)
        rounds.append(round_matches)

    summary = _summary_from_progress(raw)
    solution = _read_solution(progress_path.parent, task_id=summary.task_id)
    return TaskData(
        task_id=summary.task_id,
        run_name=run_name,
        summary=summary,
        pipeline_config=raw.get("pipeline_config") or {},
        problem_statement=str(raw.get("problem_statement", "")),
        solution_text=solution,
        candidates=candidates,
        matches=matches,
        rounds=rounds,
        raw_progress=raw,
    )


def _iter_progress_files(directory: Path) -> list[Path]:
    """Return all ``Task_*_progress.json`` files in ``directory`` sorted by task id."""
    files: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = _TASK_FILE_PATTERN.match(path.name)
        if match and path.is_file():
            files.append((int(match.group("task_id")), path))
    files.sort(key=lambda pair: pair[0])
    return [path for _, path in files]


def _is_run_directory(directory: Path) -> bool:
    """Heuristic: a directory is a run if it contains any progress files."""
    if not directory.is_dir():
        return False
    try:
        return any(_TASK_FILE_PATTERN.match(p.name) for p in directory.iterdir() if p.is_file())
    except OSError:
        return False


def discover_runs(logs_dir: Path) -> list[RunInfo]:
    """Discover runs under ``logs_dir``.

    If ``logs_dir`` itself contains progress files, it is treated as a single
    run and returned as one :class:`RunInfo`.  Otherwise, every immediate
    subdirectory that contains progress files becomes its own run.

    Args:
        logs_dir: Root directory to scan.

    Returns:
        A list of :class:`RunInfo` ordered by run name.

    Raises:
        FileNotFoundError: If ``logs_dir`` does not exist.

    """
    if not logs_dir.exists():
        msg = f"Logs directory does not exist: {logs_dir}"
        raise FileNotFoundError(msg)
    if _is_run_directory(logs_dir):
        return [_run_info(logs_dir, logs_dir.name)]
    return [
        _run_info(child, child.name)
        for child in sorted(logs_dir.iterdir(), key=lambda p: p.name)
        if _is_run_directory(child)
    ]


def _run_info(directory: Path, name: str) -> RunInfo:
    """Build a :class:`RunInfo` for a single-run directory."""
    progress_paths = _iter_progress_files(directory)
    summaries: dict[int, TaskSummary] = {}
    task_ids: list[int] = []
    for path in progress_paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Skipping unreadable progress file %s: %s", path, exc)
            continue
        summary = _summary_from_progress(raw)
        summaries[summary.task_id] = summary
        task_ids.append(summary.task_id)
    return RunInfo(name=name, path=directory, task_ids=tuple(task_ids), task_summaries=summaries)


def load_task(run: RunInfo, task_id: int) -> TaskData:
    """Load a single task by id for the given run."""
    progress_path = run.path / f"Task_{task_id}_progress.json"
    return parse_progress(progress_path, run_name=run.name)


def _reasoning_from_completion(completion: dict[str, Any] | None) -> str | None:
    """Best-effort extraction of the assistant ``reasoning`` field, if present."""
    if not completion:
        return None
    choices = completion.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    details = message.get("reasoning_details") or []
    texts = [item.get("text", "") for item in details if isinstance(item, dict)]
    joined = "\n\n".join(t for t in texts if t)
    return joined or None


def _call_from_payload(payload: dict[str, Any]) -> CallRecord:
    """Convert a raw JSONL line into a :class:`CallRecord`."""
    completion = payload.get("completion")
    usage: dict[str, Any] | None = None
    if isinstance(completion, dict):
        usage = completion.get("usage")
    return CallRecord(
        call_id=int(payload["call_id"]),
        timestamp=str(payload.get("timestamp", "")),
        phase=str(payload.get("phase", "")),
        candidate_index=payload.get("candidate_index"),
        round_index=payload.get("round_index"),
        model=str(payload.get("model", "")),
        messages=list(payload.get("messages") or []),
        response_text=str(payload.get("response_text", "")),
        reasoning=_reasoning_from_completion(completion if isinstance(completion, dict) else None),
        usage=usage if isinstance(usage, dict) else None,
        error=payload.get("error"),
    )


@lru_cache(maxsize=64)
def load_calls(run_path_str: str, task_id: int) -> tuple[CallRecord, ...]:
    """Load and cache every LLM call for a task in chronological (call_id) order.

    Args:
        run_path_str: Stringified absolute path to the run directory.  Strings
            are used to make the result hashable for :func:`lru_cache`.
        task_id: Task identifier.

    Returns:
        Immutable tuple of :class:`CallRecord` ordered by ``call_id``.

    """
    path = Path(run_path_str) / f"Task_{task_id}_llm_outputs.jsonl"
    if not path.exists():
        return ()
    calls: list[CallRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()  # noqa: PLW2901
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping malformed JSONL line in %s: %s", path, exc)
                continue
            calls.append(_call_from_payload(payload))
    calls.sort(key=lambda call: call.call_id)
    return tuple(calls)


def enrich_node_from_calls(node: CandidateNode, calls: tuple[CallRecord, ...]) -> CandidateNode:
    """Return ``node`` with verifier / classifier text populated from the call stream.

    Merged candidates are not stored verbatim in ``progress.json``; their verifier
    output and classifier verdict only exist as call-stream entries with
    ``candidate_index == node.index``.  This helper folds the latest such entries
    back onto the node so the UI can render them.  Initial candidates already
    have their text in-band and are returned unchanged.
    """
    if node.kind == "initial":
        return node
    verifier_text = ""
    classifier_text = ""
    verifier_call: int | None = None
    classifier_call: int | None = None
    for call in calls:
        if call.candidate_index != node.index:
            continue
        if call.phase == "verification":
            verifier_text = call.response_text
            verifier_call = call.call_id
        elif call.phase == "verification_check":
            classifier_text = call.response_text
            classifier_call = call.call_id
    if not (verifier_text or classifier_text):
        return node
    return CandidateNode(
        index=node.index,
        kind=node.kind,
        round_index=node.round_index,
        match_index=node.match_index,
        parents=node.parents,
        status=node.status,
        verification_pass=node.verification_pass,
        verifier_output=verifier_text or node.verifier_output,
        classifier_output=classifier_text or node.classifier_output,
        bug_report=node.bug_report,
        verifier_call_id=verifier_call if verifier_call is not None else node.verifier_call_id,
        classifier_call_id=classifier_call if classifier_call is not None else node.classifier_call_id,
    )


def calls_for_node(calls: tuple[CallRecord, ...], node: CandidateNode) -> list[CallRecord]:
    """Filter the call stream to those associated with a candidate node.

    For an initial candidate (``index >= 0``) this includes the initial solve,
    the verification(s) and any self-improvement pass.  For a merged candidate
    (``index < 0``) it includes the merge call (matched on ``round_index`` +
    parent indices) plus the post-merge verification calls.

    Args:
        calls: The full chronological call stream for the task.
        node: The bracket node whose calls we want.

    Returns:
        Calls filtered (and ordered) for the given node.

    """
    if node.kind == "initial":
        return [
            call
            for call in calls
            if call.candidate_index == node.index
            and call.phase in {"initial_solution", "verification", "verification_check", "self_improvement"}
        ]
    related: list[CallRecord] = []
    pending_merge: CallRecord | None = None
    matched_yet = False
    for call in calls:
        if call.phase == "tournament_merge" and call.round_index == node.round_index and not matched_yet:
            pending_merge = call
        elif call.candidate_index == node.index:
            matched_yet = True
            if pending_merge is not None:
                related.append(pending_merge)
                pending_merge = None
            related.append(call)
    return related
