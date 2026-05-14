#!/usr/bin/env bash
# The script resolves paths relative to its own location so it can be invoked
# from any working directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m open_deep_think.scripts.parallel_solve \
    --start 0 \
    --end 400 \
    --script tournament_merge_improve \
    --model qwen/qwen3.5-397b-a17b \
    --solver_max_tokens 128000 \
    --verifier_max_tokens 128000 \
    --classifier_max_tokens 128000 \
    --merger_max_tokens 128000 \
    --num_solutions 8 \
    --temperature 0.6 \
    --top_p 0.95 \
    --concurrency 15 \
    --run_name full_tournament_qwen3.5_397_A17 \
    --output_path "${REPO_ROOT}/data/"
