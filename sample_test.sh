#!/bin/bash
# End-to-end smoke test: 1 positive + 1 negative case through Stages 00 -> 04.
#
# Bundled fixtures under data/sample_test/ remove the need for MIMIC access.
# Two GPU-heavy steps are SKIPPED and their outputs are pre-staged from fixtures:
#   - Stage 00.01  ROSALIA inference (LISA env + 7B GPU model)
#   - Stage 01.00  CXAS / CheXmask-U anatomy mask generation
#
# Doctor annotation `optimal?` is auto-filled with 'y' so the test runs
# unattended; production runs must use real annotations.
#
# Envs:
#   rosalia      Stages 00, 01
#   chexpercept  Stages 02, 03, 04
#
# Usage:
#   bash sample_test.sh                          # default: single GPU (CUDA_VISIBLE_DEVICES=0)
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash sample_test.sh
#   CONDA_BASE=/path/to/conda  bash sample_test.sh
#
# Prereqs (one-time):
#   1. cp api_info/api_keys_example.yaml api_info/api_keys.yaml; fill in HF token + cache paths.
#   2. Conda envs created from envs/{rosalia,chexpercept}.yml.

set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
POS_DICOM="1eec54ad-adf25374-4687fbfc-8ceded62-45be37c4"
POS_PAIR="s58956720_positive_0"
HELPERS="$ROOT/scripts/sample_test"
FIX="$ROOT/data/sample_test"
CFG="cfg/_config_sample_test_rendered.yaml"


# --------------------------------------------------------------------------
# Conda + LD-path helpers
# --------------------------------------------------------------------------
if [ -z "${CONDA_BASE:-}" ]; then
    command -v conda >/dev/null && CONDA_BASE="$(conda info --base 2>/dev/null)"
    for cand in "$HOME/miniconda3" "$HOME/anaconda3" /opt/miniconda3 /opt/conda; do
        [ -n "${CONDA_BASE:-}" ] && break
        [ -f "$cand/etc/profile.d/conda.sh" ] && CONDA_BASE="$cand"
    done
fi
[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] || { echo "conda.sh not found"; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"

_LD_BASE="${LD_LIBRARY_PATH:-}"
activate_env() {
    conda deactivate >/dev/null 2>&1 || true
    conda activate "$1"
    local nvjit
    nvjit="$(python -c 'import nvidia.nvjitlink as m; print(list(m.__path__)[0])' 2>/dev/null || true)"
    export LD_LIBRARY_PATH="${nvjit:+$nvjit/lib:}$_LD_BASE"
    echo "[env] active=$1"
}

banner() { printf '\n========== %s ==========\n' "$*"; }


# --------------------------------------------------------------------------
# Stage 00: source data curation
# --------------------------------------------------------------------------

stage_00_sample_mimic_ils() {
    banner "Stage 00.00  Sample MIMIC-ILS cases"
    rm -rf src/00_source_data_curation/outputs
    python src/00_source_data_curation/00_sample_mimic_ils_case.py --config "$CFG"
}

stage_00_skip_rosalia() {
    banner "Stage 00.01  ROSALIA inference  [SKIP — pre-staging fixture]"
    local out=src/00_source_data_curation/outputs/rosalia_pred
    mkdir -p "$out/cardiomegaly/plots/cardiomegaly"
    cp -r "$FIX/prebuilt/rosalia_pred/cardiomegaly/$POS_PAIR" "$out/cardiomegaly/"
    python "$HELPERS/build_rosalia_fixture.py" \
        --out-dir "$out" --pair-id "$POS_PAIR" --dicom-id "$POS_DICOM"
}

stage_00_merge_positive() {
    banner "Stage 00.02  Merge positive part outputs"
    python src/00_source_data_curation/02_prepare_positive_annotation.py --total-parts 1
    banner "Auto-fill positive 'optimal?'"
    python "$HELPERS/autofill_optimal.py" \
        src/00_source_data_curation/outputs/rosalia_pred/labeling_sheet.csv
}

stage_00_negative_annotation() {
    banner "Stage 00.03  Prepare negative annotation"
    python src/00_source_data_curation/03_prepare_negative_annotation.py \
        --config "$CFG" --samples-per-lesion 1
    banner "Auto-fill negative 'optimal?'"
    python "$HELPERS/autofill_optimal.py" \
        src/00_source_data_curation/outputs/negative/labeling_sheet.csv
}

stage_00_distribute() {
    banner "Stage 00.04  Distribute labeling"
    python src/00_source_data_curation/04_distribute_labeling.py --num-annotators 2 \
        || echo "(distribute is bookkeeping only; ignored on tiny sample)"
}


# --------------------------------------------------------------------------
# Stage 01: mask deformation
# --------------------------------------------------------------------------

stage_01_skip_anatomy() {
    banner "Stage 01.00  Anatomy mask generation  [SKIP — pre-staging fixture]"
    rm -rf src/01_mask_deformation/outputs
    mkdir -p src/01_mask_deformation/outputs/{chexmasku_pred,cxas_pred}
    cp -r "$FIX"/prebuilt/chexmasku_pred/* src/01_mask_deformation/outputs/chexmasku_pred/
    cp -r "$FIX"/prebuilt/cxas_pred/*      src/01_mask_deformation/outputs/cxas_pred/
}

stage_01_deform_mask() {
    banner "Stage 01.01  Mask deformation (SAM3 may auto-download from HF)"
    python src/01_mask_deformation/01_deform_mask.py --config "$CFG" --num-workers 1
}


# --------------------------------------------------------------------------
# Stage 02–04
# --------------------------------------------------------------------------

stage_02_generate_qa() {
    banner "Stage 02  Generate QA"
    rm -rf src/02_qa_generation/outputs
    python src/02_qa_generation/00_generate_qa.py --config "$CFG" --num-workers 2
}

stage_03_eval() {
    banner "Re-stage CheXpercept benchmark layout under data/chexpercept/"
    rm -rf data/chexpercept
    mkdir -p data/chexpercept
    cp src/02_qa_generation/outputs/qa_results/qa_results.json data/chexpercept/chexpercept.json
    cp -rl src/02_qa_generation/outputs/chexpercept data/chexpercept/chexpercept

    banner "Stage 03  Evaluate medgemma1.5 on CUDA $CUDA_VISIBLE_DEVICES"
    python src/03_eval_vlm_on_chexpercept/00_eval.py --provider opensource --model medgemma1.5
}

stage_04_analyze() {
    banner "Stage 04  Analyze evaluation results"
    python src/04_analyze_eval_result/01_analyze_model_performance.py --oracle_setting implicit
}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

# Render config_sample_test.yaml once.
sed "s|{ROOT}|$ROOT|g" cfg/config_sample_test.yaml > "$CFG"

activate_env rosalia
stage_00_sample_mimic_ils
stage_00_skip_rosalia
stage_00_merge_positive
stage_00_negative_annotation
stage_00_distribute
stage_01_skip_anatomy
stage_01_deform_mask

activate_env chexpercept
stage_02_generate_qa
stage_03_eval
stage_04_analyze

banner "sample_test.sh complete"
