"""Streamlit ``AppTest`` smoke checks for the tournament viewer.

These tests exercise the actual Streamlit script by spinning it up through
Streamlit's headless ``AppTest`` harness; they catch wiring breakage — wrong
widget keys, broken caches, missing imports — that pure-unit tests would miss.

If Streamlit is not installed (the viewer is an optional extra) the whole
module is skipped.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP_PATH = "src/open_deep_think/viewer/app.py"


@pytest.fixture
def app(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    """Return a freshly-initialised :class:`AppTest` pointing at ``run_dir``."""
    monkeypatch.setenv("ODT_VIEWER_LOGS_DIR", str(run_dir.parent))
    # ``DEFAULT_LOGS_DIR`` is captured at import time, so patch the symbol too.
    from open_deep_think.viewer import app as viewer_app  # noqa: PLC0415

    monkeypatch.setattr(viewer_app, "DEFAULT_LOGS_DIR", run_dir.parent)
    # ``_index_runs`` is wrapped in ``st.cache_resource`` and can hold a stale
    # mapping from previous tests; clear it so each test sees the patched root.
    viewer_app._index_runs.clear()  # type: ignore[attr-defined]  # noqa: SLF001
    viewer_app._load_task_cached.clear()  # type: ignore[attr-defined]  # noqa: SLF001
    return AppTest.from_file(os.fspath(viewer_app.__file__), default_timeout=30)


def test_app_runs_without_exception(app: object) -> None:
    app.run()
    assert app.exception == []
    titles = [t.value for t in app.title]
    assert any(title.startswith("Task ") for title in titles)


def test_app_renders_candidate_picker_and_tabs(app: object) -> None:
    app.run()
    subheaders = [s.value for s in app.subheader]
    assert "Candidates" in subheaders
    tab_labels = [tab.label for tab in app.tabs]
    assert {"Solution", "Verification"} <= set(tab_labels)


def test_app_query_param_selects_initial_candidate(app: object) -> None:
    app.query_params["node"] = "0"
    app.run()
    subheaders = [s.value for s in app.subheader]
    # The detail subheader for the selected initial candidate should mention "C0".
    assert any("C0" in sh for sh in subheaders)


def test_app_metric_strip_shows_initial_pass_rate(app: object) -> None:
    app.run()
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Initial pass"] == "3/4"
    assert metrics["Rounds"] == "2"
