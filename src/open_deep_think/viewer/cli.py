"""Console-script entry point that boots the Streamlit viewer.

Streamlit must be launched via its own CLI so that the dev server, websocket
runtime and hot-reload machinery are wired up correctly.  This wrapper resolves
the user's ``--logs-dir`` argument, exports it to ``ODT_VIEWER_LOGS_DIR`` so
that :mod:`open_deep_think.viewer.app` can pick it up, and then invokes
``streamlit run`` on the bundled app script.

Example::

    odt-viewer --logs-dir tournament_logs --port 8501
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse the wrapper's known arguments; forward everything else to Streamlit."""
    parser = argparse.ArgumentParser(
        prog="odt-viewer",
        description="Launch the Streamlit viewer for tournament-merge-improve logs.",
        add_help=True,
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("tournament_logs"),
        help="Root directory containing tournament logs (single run or many).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind the Streamlit server to (default: Streamlit chooses).",
    )
    parser.add_argument(
        "--address",
        type=str,
        default=None,
        help="Address to bind the Streamlit server to (default: localhost).",
    )
    return parser.parse_known_args(argv)


def _app_script_path() -> Path:
    """Return the path to the bundled ``app.py`` Streamlit script."""
    return Path(__file__).with_name("app.py")


try:
    from streamlit.web import cli as stcli
except ImportError:  # pragma: no cover - viewer extras must be installed
    stcli = None  # type: ignore[assignment]


def main(argv: list[str] | None = None) -> int:
    """Entry point used by the ``odt-viewer`` console script.

    Args:
        argv: Optional argument list.  Defaults to :data:`sys.argv[1:]`.

    Returns:
        Process exit code (forwarded from Streamlit).

    """
    args, extra = _parse_args(argv)
    logs_dir = args.logs_dir.expanduser().resolve()
    if not logs_dir.exists():
        sys.stderr.write(f"Logs directory does not exist: {logs_dir}\n")
        return 2

    os.environ["ODT_VIEWER_LOGS_DIR"] = str(logs_dir)

    streamlit_argv: list[str] = ["streamlit", "run", str(_app_script_path())]
    if args.port is not None:
        streamlit_argv += ["--server.port", str(args.port)]
    if args.address is not None:
        streamlit_argv += ["--server.address", args.address]
    if extra:
        streamlit_argv += ["--", *extra]

    if stcli is None:  # pragma: no cover - depends on environment
        sys.stderr.write(
            "Streamlit is not installed. Install the viewer extras with "
            "`uv sync --extra viewer` (or `pip install streamlit`).\n",
        )
        return 1

    sys.argv = streamlit_argv
    return int(stcli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
