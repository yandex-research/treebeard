#!/usr/bin/env bash
# The script resolves paths relative to its own location so it can be invoked
# from any working directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m open_deep_think.scripts.parallel_solve \
    --task_ids_file "${SCRIPT_DIR}/subset_1.txt" \
    --script imo25 \
    --model qwen/qwen3.5-397b-a17b \
    --solver_max_tokens 128000 \
    --verifier_max_tokens 128000 \
    --classifier_max_tokens 128000 \
    --max_runs 2 \
    --max_iterations 10 \
    --required_consecutive_passes 5 \
    --max_consecutive_failures 10 \
    --temperature 0.6 \
    --top_p 0.95 \
    --concurrency 15 \
    --run_name subset_1_imo25_qwen3.5_397_A17 \
    --output_path "${REPO_ROOT}/data/"
