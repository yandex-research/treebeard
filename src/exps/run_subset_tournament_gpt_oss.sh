#!/usr/bin/env bash
# The script resolves paths relative to its own location so it can be invoked
# from any working directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m open_deep_think.scripts.parallel_solve \
    --task_ids_file "${SCRIPT_DIR}/subset_1.txt" \
    --script tournament_merge_improve \
    --model openai/gpt-oss-120b \
    --solver_max_tokens 64000 \
    --verifier_max_tokens 64000 \
    --classifier_max_tokens 64000 \
    --merger_max_tokens 64000 \
    --num_solutions 8 \
    --temperature 1.0 \
    --top_p 1.0 \
    --concurrency 15 \
    --run_name subset_1_tournament_gpt_oss \
    --output_path "${REPO_ROOT}/data/"
