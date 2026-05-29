"""Allow ``python -m open_deep_think.viewer`` to launch the Streamlit viewer."""

from open_deep_think.viewer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
