#!/usr/bin/env bash
# The script resolves paths relative to its own location so it can be invoked
# from any working directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m open_deep_think.scripts.parallel_solve \
    --start 0 \
    --end 400 \
    --script imo25 \
    --model openai/gpt-oss-120b \
    --solver_max_tokens 64000 \
    --verifier_max_tokens 64000 \
    --classifier_max_tokens 64000 \
    --max_runs 2 \
    --max_iterations 10 \
    --required_consecutive_passes 5 \
    --max_consecutive_failures 10 \
    --temperature 1.0 \
    --top_p 1.0 \
    --concurrency 15 \
    --run_name full_imo25_gpt_oss \
    --output_path "${REPO_ROOT}/data/"
