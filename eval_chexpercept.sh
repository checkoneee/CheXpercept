#!/bin/bash
# Evaluate a VLM on the CheXpercept benchmark, then aggregate the result.
#
# Place the downloaded benchmark at data/chexpercept/ (or override with
# --benchmark-dir; see src/03_eval_vlm_on_chexpercept/00_eval.py --help).
#
# All arguments are forwarded to 00_eval.py; this script just activates the
# right conda env, applies the LD-path workaround, and runs Stage 04
# afterwards.
#
# Usage:
#   bash eval_chexpercept.sh --provider opensource --model medgemma1.5
#   bash eval_chexpercept.sh --provider gemini --model gemini-3.1-pro
#   bash eval_chexpercept.sh --model hulu-med               # uses `hulumed` env
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash eval_chexpercept.sh --model qwen3.6-27b
#   ENV=my-chexpercept-env bash eval_chexpercept.sh ...     # custom conda env name
#
# Defaults: 1 GPU (CUDA_VISIBLE_DEVICES=0), env=chexpercept (hulumed for hulu-med).
# Larger models (>=32B) need more GPUs; set CUDA_VISIBLE_DEVICES accordingly.

set -euo pipefail
cd "$(dirname "$0")"

# Pick env: hulumed for hulu-med, chexpercept otherwise. Override with $ENV.
if [ -z "${ENV:-}" ]; then
    ENV=chexpercept
    for arg in "$@"; do [ "$arg" = "hulu-med" ] && ENV=hulumed; done
fi

# Activate conda + patch LD_LIBRARY_PATH (cusparse/nvjitlink mismatch fix).
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV"
nvjit="$(python -c 'import nvidia.nvjitlink as m; print(list(m.__path__)[0])' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${nvjit:+$nvjit/lib:}${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Stage 03: evaluate.
python src/03_eval_vlm_on_chexpercept/00_eval.py "$@"

# Stage 04: aggregate. Re-use --oracle_setting if the user passed it.
oracle=implicit
prev=""
for arg in "$@"; do
    [ "$prev" = "--oracle_setting" ] && oracle="$arg"
    prev="$arg"
done
python src/04_analyze_eval_result/01_analyze_model_performance.py --oracle_setting "$oracle"
