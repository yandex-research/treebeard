# open-deep-think

## Getting Started with uv

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable Python package management.

### Installing uv

Install uv using pip:

```bash
pip install uv
```

Or use the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setting Up the Project

1. Clone the repository:
```bash
git clone https://github.com/yandex-research/open-deep-think.git
cd open-deep-think
```

2. Install dependencies:
```bash
uv sync
```

This will install all project dependencies specified in `pyproject.toml` and locked in `uv.lock`.

### Common uv Commands

- **Install dependencies**: `uv sync`
- **Add a new dependency**: `uv add <package-name>`
- **Add a dev dependency**: `uv add --group dev <package-name>`
- **Remove a dependency**: `uv remove <package-name>`
- **Update dependencies**: `uv lock --upgrade`
- **Run a Python script**: `uv run python script.py`
- **Run a command in the virtual environment**: `uv run <command>`

### Development

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

```bash
# Check code with ruff
uv run ruff check src/ --fix

# Format code with ruff
uv run ruff format src/
```

### Why uv?

uv is a fast Python package installer and resolver written in Rust. It offers:
- 10-100x faster than pip
- Deterministic dependency resolution
- Compatible with pip and PyPI
- Built-in virtual environment management
