#!/usr/bin/env bash
# Ablation: generate independent candidates (no tournament) for the subset_1
# task list using gpt-oss-120b.
#
# The script resolves paths relative to its own location so it can be invoked
# from any working directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m open_deep_think.scripts.ablation_generate_candidates \
    --task_ids_file "${SCRIPT_DIR}/subset_1.txt" \
    --model openai/gpt-oss-120b \
    --n_solutions 8 \
    --concurrency 15 \
    --solver_max_tokens 64000 \
    --verifier_max_tokens 64000 \
    --classifier_max_tokens 64000 \
    --temperature 1.0 \
    --top_p 1.0 \
    --si_rounds 0 \
    --run_name subset_1_ablation_gpt_oss \
    --output_path "${REPO_ROOT}/data/"
