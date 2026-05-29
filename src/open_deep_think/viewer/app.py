"""Streamlit entry point for the tournament viewer.

Run via the ``odt-viewer`` console script (see :mod:`open_deep_think.viewer.cli`)
or directly with ``streamlit run src/open_deep_think/viewer/app.py``.

The application reads the logs directory from the ``ODT_VIEWER_LOGS_DIR``
environment variable so that the CLI wrapper can pass it through cleanly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from open_deep_think.viewer.data import (
    CallRecord,
    CandidateNode,
    RunInfo,
    TaskData,
    calls_for_node,
    discover_runs,
    enrich_node_from_calls,
    load_calls,
    parse_progress,
)
from open_deep_think.viewer.render import (
    collapse_blank_lines,
    format_usage,
    normalise_math,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_LOGS_DIR = Path(os.environ.get("ODT_VIEWER_LOGS_DIR", "tournament_logs"))
"""Logs root resolved from the environment.  CLI sets this before launching Streamlit."""

_PHASE_BADGES = {
    "initial_solution": ("blue", "Initial"),
    "verification": ("orange", "Verify"),
    "verification_check": ("violet", "Classify"),
    "self_improvement": ("blue", "Self-improve"),
    "tournament_merge": ("green", "Merge"),
}


def _format_path(path: Path) -> str:
    """Return a short human-friendly path for display in the sidebar."""
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


@st.cache_resource(show_spinner=False)
def _index_runs(logs_dir: str) -> list[RunInfo]:
    return discover_runs(Path(logs_dir))


@st.cache_data(show_spinner=False)
def _load_task_cached(run_path: str, run_name: str, task_id: int) -> TaskData:
    progress_path = Path(run_path) / f"Task_{task_id}_progress.json"
    return parse_progress(progress_path, run_name=run_name)


def _task_label(run: RunInfo, task_id: int) -> str:
    """Sidebar label for a task: pass/fail glyph + task id + first words of problem."""
    summary = run.task_summaries.get(task_id)
    if summary is None:
        return f"Task {task_id}"
    if summary.final_verification_pass is True:
        glyph = "✓"
    elif summary.final_verification_pass is False:
        glyph = "✗"
    else:
        glyph = "·"
    preview = summary.problem_preview[:60].rstrip()
    return f"{glyph} Task {task_id}  ·  {preview}"


def _select_run_and_task(runs: list[RunInfo]) -> tuple[RunInfo, int]:
    """Render the sidebar and return the selected (run, task_id)."""
    st.sidebar.title("Tournament viewer")
    st.sidebar.caption(f"logs: `{_format_path(DEFAULT_LOGS_DIR)}`")

    run_names = [run.name for run in runs]
    default_run_idx = 0
    cached_run = st.session_state.get("__run_name")
    if cached_run in run_names:
        default_run_idx = run_names.index(cached_run)
    run_name = st.sidebar.selectbox("Run", run_names, index=default_run_idx, key="__run_name")
    run = next(r for r in runs if r.name == run_name)

    if not run.task_ids:
        st.sidebar.warning("This run has no tasks.")
        st.stop()

    default_task_idx = 0
    cached_task = st.session_state.get("__task_id")
    if cached_task in run.task_ids:
        default_task_idx = run.task_ids.index(cached_task)
    task_id = st.sidebar.selectbox(
        "Task",
        run.task_ids,
        index=default_task_idx,
        format_func=lambda tid: _task_label(run, tid),
        key="__task_id",
    )
    return run, task_id


def _summary_pills(task: TaskData) -> None:
    """Render the small metric strip at the top of the task view."""
    cols = st.columns(4)
    final_pass = task.summary.final_verification_pass
    final_glyph = "✓" if final_pass else "✗" if final_pass is False else "·"
    cols[0].metric("Final", f"{final_glyph} {task.summary.status}")
    cols[1].metric("Initial pass", f"{task.summary.num_initial_passed}/{task.summary.num_initial}")
    cols[2].metric("Solver", task.summary.solver_model.split("/")[-1] or "—")
    cols[3].metric("Rounds", str(len(task.rounds)))


def _selected_node_index(task: TaskData) -> int | None:
    """Decide which candidate should be considered "selected" on this rerun."""
    param = st.query_params.get("node")
    if isinstance(param, list):
        param = param[0] if param else None
    if param is not None:
        try:
            return int(param)
        except ValueError:
            pass
    return task.summary.final_candidate_index


def _set_selected_node(index: int | None) -> None:
    """Write the selected node into the URL query params (triggers a rerun)."""
    if index is None:
        st.query_params.clear()
    else:
        st.query_params["node"] = str(index)


def _node_options(task: TaskData) -> list[CandidateNode]:
    """Return candidates ordered for the node selector — initials first, then merges."""
    initials = sorted([c for c in task.candidates if c.kind == "initial"], key=lambda c: c.index)
    merges = sorted([c for c in task.candidates if c.kind == "merged"], key=lambda c: c.index, reverse=True)
    return [*initials, *merges]


def _node_label(node: CandidateNode) -> str:
    """Human label for a candidate node — used in the radio selector."""
    if node.kind == "initial":
        head = f"C{node.index}"
    else:
        head = f"M{node.index}"
        if node.round_index is not None:
            head += f" · round {node.round_index}"
    if node.verification_pass is True:
        badge = "✓"
    elif node.verification_pass is False:
        badge = "✗"
    else:
        badge = "·"
    return f"{badge} {head}"


def _phase_badge(phase: str) -> None:
    """Render a Streamlit badge for a phase string (falls back to a markdown tag)."""
    colour, label = _PHASE_BADGES.get(phase, ("gray", phase))
    try:
        st.badge(label, color=colour)
    except AttributeError:
        st.markdown(f"`{label}`")


def _solution_text_for(node_calls: Iterable[CallRecord]) -> str:
    """Pick the most recent solver/merger response from a node's call stream."""
    relevant = [c for c in node_calls if c.phase in {"initial_solution", "self_improvement", "tournament_merge"}]
    if not relevant:
        return ""
    return relevant[-1].response_text


def _render_messages(messages: list[dict[str, object]]) -> None:
    """Render the (system, user, …) prompt messages of a call."""
    for message in messages:
        role = str(message.get("role", "?")).capitalize()
        content = str(message.get("content", ""))
        with st.expander(f"{role} message  ·  {len(content):,} chars", expanded=False):
            st.markdown(normalise_math(collapse_blank_lines(content)))


def _render_call(call: CallRecord) -> None:
    """Render one LLM call: header, prompt messages, response, reasoning, usage."""
    header_cols = st.columns([1, 2, 3, 3])
    with header_cols[0]:
        st.markdown(f"**#{call.call_id}**")
    with header_cols[1]:
        _phase_badge(call.phase)
    with header_cols[2]:
        st.caption(call.model)
    with header_cols[3]:
        st.caption(call.timestamp)
    if call.error:
        st.error(call.error)
    if call.response_text:
        with st.expander("Response", expanded=True):
            st.markdown(normalise_math(collapse_blank_lines(call.response_text)))
    if call.reasoning:
        with st.expander(f"Reasoning ({len(call.reasoning):,} chars)", expanded=False):
            st.markdown(normalise_math(collapse_blank_lines(call.reasoning)))
    if call.messages:
        with st.expander("Prompt messages", expanded=False):
            _render_messages(call.messages)
    usage = format_usage(call.usage)
    if usage:
        st.caption(f"Tokens: {usage}")
    st.divider()


def _render_node_detail(node: CandidateNode, calls: tuple[CallRecord, ...]) -> None:
    """Render the bottom panel describing a single candidate."""
    node_calls = calls_for_node(calls, node)
    label = _node_label(node)
    st.subheader(f"{label}")

    if node.kind == "merged" and node.parents:
        st.caption(
            "Merged from "
            + " + ".join(f"`{p}`" for p in node.parents)
            + (f" · round {node.round_index}" if node.round_index is not None else ""),
        )

    tab_solution, tab_verification, tab_calls = st.tabs(["Solution", "Verification", f"Calls ({len(node_calls)})"])

    with tab_solution:
        text = _solution_text_for(node_calls)
        if text:
            st.markdown(normalise_math(collapse_blank_lines(text)))
        else:
            st.info("No solver response captured for this node.")

    with tab_verification:
        if node.verification_pass is True:
            st.success("Verification passed.")
        elif node.verification_pass is False:
            st.error("Verification failed.")
        else:
            st.info("Verification status unknown.")
        if node.classifier_output:
            st.markdown(f"**Classifier:** `{node.classifier_output.strip()[:64]}`")
        if node.bug_report:
            with st.expander("Bug report", expanded=False):
                st.markdown(normalise_math(collapse_blank_lines(node.bug_report)))
        if node.verifier_output:
            st.markdown(normalise_math(collapse_blank_lines(node.verifier_output)))
        else:
            st.caption("No verifier output recorded in progress.json.")

    with tab_calls:
        if not node_calls:
            st.info("No LLM calls associated with this node.")
        for call in node_calls:
            _render_call(call)


def _render_node_picker(task: TaskData, selected_index: int | None) -> None:
    """Render the horizontal radio that picks which candidate to inspect."""
    options = _node_options(task)
    labels = [_node_label(n) for n in options]
    values = [n.index for n in options]
    default = 0
    if selected_index is not None and selected_index in values:
        default = values.index(selected_index)
    picked_label = st.radio(
        "Inspect node",
        labels,
        index=default,
        horizontal=True,
        key=f"__node_radio_{task.task_id}",
    )
    picked_index = values[labels.index(picked_label)]
    if picked_index != selected_index:
        _set_selected_node(picked_index)
        st.rerun()


def _render_problem_and_solution(task: TaskData) -> None:
    """Render the problem statement and final solution panels at the top of the page."""
    with st.expander("Problem statement", expanded=True):
        if task.problem_statement.strip():
            st.markdown(normalise_math(collapse_blank_lines(task.problem_statement)))
        else:
            st.info("No problem statement recorded.")

    with st.expander("Final solution", expanded=False):
        if task.solution_text.strip():
            st.markdown(normalise_math(collapse_blank_lines(task.solution_text)))
        else:
            st.info("No final solution.txt was written for this task.")


def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(page_title="Open Deep Think — Tournament viewer", layout="wide")

    try:
        runs = _index_runs(str(DEFAULT_LOGS_DIR))
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if not runs:
        st.error(f"No runs with `Task_*_progress.json` files found under {DEFAULT_LOGS_DIR}.")
        st.stop()

    run, task_id = _select_run_and_task(runs)
    task = _load_task_cached(str(run.path), run.name, task_id)
    calls = load_calls(str(run.path), task_id)

    st.title(f"Task {task.task_id}")
    st.caption(f"run: `{run.name}`  ·  started {task.summary.started_at}")

    _summary_pills(task)
    _render_problem_and_solution(task)

    st.subheader("Candidates")
    selected_index = _selected_node_index(task)
    _render_node_picker(task, selected_index)

    candidates_by_index = {c.index: c for c in task.candidates}
    node = candidates_by_index.get(selected_index) if selected_index is not None else None
    if node is None:
        st.info("Pick a candidate above to inspect its solution, verification, and calls.")
        return
    _render_node_detail(enrich_node_from_calls(node, calls), calls)


if __name__ == "__main__":
    main()
