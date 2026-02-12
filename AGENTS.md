# Instructions
- Write good, readable and reusable code
- Write docstrings
- Write good, meaningful tests. Do not write stupid, useless tests
- Always run tests and ruff at the end and fix all issues

```bash
# Check code with ruff
uv run ruff check src/ --fix

# Format code with ruff
uv run ruff format src/
```