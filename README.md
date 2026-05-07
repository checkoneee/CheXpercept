# CheXpercept: A Benchmark for Evaluating Expert-Level Lesion Perception in Chest X-rays

> [!IMPORTANT]
> **Read the [Dataset](#dataset) section before downloading or using anything.**
> CheXpercept is derived from MIMIC-CXR-JPG and inherits PhysioNet credentialing requirements.
> The review-time release is private to OpenReview reviewers — **do not redistribute, mirror, or post publicly**.

*Under review*

Anonymous Authors

Anonymous Institutions

[\[Paper\]](#citation) [\[Dataset\]](#dataset) [\[Code\]](#repository-structure)

---

## Overview

CheXpercept is a sequential, multi-level perception benchmark that evaluates vision-language models (VLMs) on chest X-ray (CXR) lesion perception. It mirrors the cognitive workflow of a radiologist across three levels of perception — coarse, fine, and semantic — through four sequential QA stages.

| Stat | Value |
|------|-------|
| QA items | 10,400 |
| CXR images | 2,100 |
| Lesion types | 7 |
| QA paths | 3 |
| Models benchmarked | 14 |

**Target lesions:** atelectasis, cardiomegaly, consolidation, edema, effusion, opacity, pneumonia

---

## Benchmark Structure

### Evaluation Stages

Each CXR case is evaluated through up to four sequential stages:

| Stage | Perception Level | Task |
|-------|-----------------|------|
| 1. Detection | Coarse | Is the target lesion present? (Yes / No) |
| 2. Contour Evaluation | Fine | Does the overlaid mask need major revision? (Yes / No) |
| 3. Contour Revision | Fine | Which boundary points need expansion / contraction? Which revised mask is correct? |
| 4. Attribute Extraction | Semantic | Distribution, location, severity, and comparison of the lesion |

### Evaluation Paths

The pipeline branches dynamically based on lesion presence and mask quality:

| Path | Condition | Stages |
|------|-----------|--------|
| **Revision-Required (RR)** | Lesion present, mask suboptimal | 1 → 2 → 3 → 4 |
| **Revision-Free (RF)** | Lesion present, mask optimal | 1 → 2 → 4 |
| **Lesion-Free (LF)** | Lesion absent | 1 only |

Cardiomegaly is a structural exception: Stage 4 attributes (lung-based) are not applicable, so its RR path is 1 → 2 → 3 and RF path is 1 → 2.

---

## Repository Structure

```
CheXpercept/
├── setup.sh                        # Clone third-party deps (LISA, SAM3, CheXmask-U)
├── sample_test.sh                  # End-to-end smoke test (Stages 00 → 04 on bundled fixtures)
├── eval_chexpercept.sh             # Stage 03 + 04 on the downloaded benchmark
├── cfg/                            # config_example.yaml, config_sample_test.yaml
├── api_info/                       # api_keys_example.yaml
├── envs/                           # Conda environment files
├── data/
│   ├── sample_test/                # Bundled fixtures used by sample_test.sh
│   └── chexpercept/                # (download here) chexpercept.json + chexpercept/ image folders
├── src/
│   ├── 00_source_data_curation/    # Stage 00: Sample MIMIC-ILS; generate optimal masks via ROSALIA
│   ├── 01_mask_deformation/        # Stage 01: Anatomy masks + SAM3-based mask deformation
│   ├── 02_qa_generation/           # Stage 02: Generate QA pairs and CheXpercept benchmark export
│   ├── 03_eval_vlm_on_chexpercept/ # Stage 03: Evaluate VLMs on CheXpercept
│   └── 04_analyze_eval_result/     # Stage 04: Analyze and visualize results
├── scripts/                        # Helper scripts used by sample_test.sh
└── utils/                          # Shared utilities (config loader, LLM helpers)
```

### Data Flow

```
Stage 00: MIMIC-ILS CXRs → optimal masks + true-normal CXRs
    ↓
Stage 01: optimal masks → deformed (suboptimal) masks → deformation_results.json
    ↓
Stage 02: deformation_results → QA generation → qa_results.json → sampling → chexpercept/
    ↓
Stage 03: chexpercept/ + CXR images → VLM evaluation → outputs/{model}/oracle_{setting}/
    ↓
Stage 04: evaluation outputs → per-stage accuracy tables and visualizations
```

---

## Dataset

> **Distribution during peer review.** The CheXpercept benchmark is being released only as a private link on the OpenReview submission page (see the paper's supplementary material). It is **not** linked from this GitHub repository, and **must not be redistributed** outside of the review context. After publication it will be hosted on PhysioNet with the standard credentialing flow described below.

Because CheXpercept is derived from MIMIC-CXR-JPG, it inherits PhysioNet's restricted-access requirements. **By using these files you commit to:**

- Not redistributing them, in whole or in part, to anyone outside your immediate research group.
- Not posting them publicly (cloud buckets, code-hosting forks, paper supplementary material, etc.).
- Deleting your local copy if you cease to be an authorized user.

We strongly recommend completing the standard PhysioNet credentialing **before** downloading the review copy, even though we cannot enforce it during review. Specifically:

1. Become a [credentialed PhysioNet user](https://physionet.org/settings/credentialing/).
2. Complete the required **CITI "Data or Specimens Only Research"** training and upload the certificate to PhysioNet.
3. Sign the MIMIC-CXR-JPG and MIMIC-ILS data-use agreements:
   - [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
   - [MIMIC-ILS](https://physionet.org/content/mimic-cxr-ext-ils/)

The provisional review-time release is a concession to the venue's reproducibility requirements; the credentialed PhysioNet release is the long-term distribution channel.

---

## Data Requirements

To run the **full pipeline** (Stages 00 → 04 from raw MIMIC), you also need local copies of the upstream datasets, both behind the same PhysioNet credentialing flow above:

1. Download [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) (CXR images + reports + metadata).
2. Download [MIMIC-ILS](https://physionet.org/content/mimic-cxr-ext-ils/) (lesion segmentation masks + instructions).
3. Update the paths in `cfg/config.yaml` to point to your local copies.

To run **only the smoke test** (`sample_test.sh`) or **only Stage 03 + 04** (`eval_chexpercept.sh` against the released benchmark), the upstream MIMIC datasets are **not required** — the benchmark bundle includes the CXR images you need.

---

## Setup

### Environments

Separate conda environments are used across the pipeline:

| Environment | Stages | Purpose |
|-------------|--------|---------|
| `rosalia` | 00, 01 | Optimal mask generation and mask deformation |
| `chexpercept` | 02, 03, 04 | QA generation, VLM evaluation (vLLM + APIs), result analysis |
| `hulumed` | 03 (Hulu-Med only) | Dedicated env for `hulu-med` (incompatible deps with `chexpercept`) |

```bash
conda env create -f envs/rosalia.yml
conda env create -f envs/chexpercept.yml
conda env create -f envs/hulumed.yml      # only if evaluating hulu-med
```

### Third-Party Dependencies (`setup.sh`)

Three external GitHub repos are required and not bundled. `setup.sh` clones whatever is missing:

```bash
bash setup.sh                                  # full pipeline (clones all three)
bash setup.sh --skip-lisa --skip-chexmask-u    # smoke test only (clones SAM3 only)
```

| Repo | Path | Used by |
|------|------|---------|
| [LISA](https://github.com/JIA-Lab-research/LISA) | `src/00_source_data_curation/LISA/` | Stage 00 (ROSALIA inference) |
| [SAM3](https://github.com/facebookresearch/sam3) | `src/01_mask_deformation/sam3/` | Stage 01 (mask deformation) |
| [CheXmask-U](https://github.com/mcosarinsky/CheXmask-U) | `src/01_mask_deformation/CheXmask-U/` | Stage 01 (anatomy masks) |

`sample_test.sh` skips the ROSALIA and anatomy-mask inference steps (bundled fixtures stand in for them), so only SAM3 is strictly required for the smoke test.

### API Keys

Copy the example and fill in your keys:

```bash
cp api_info/api_keys_example.yaml api_info/api_keys.yaml
# Edit api_info/api_keys.yaml with your HuggingFace token and API keys
```

### Configuration

Copy the example config and fill in your local paths:

```bash
cp cfg/config_example.yaml cfg/config.yaml
# Edit cfg/config.yaml with your dataset paths
```

### Troubleshooting

**`ImportError: libcusparse.so.12: undefined symbol: __nvJitLinkGetErrorLogSize_12_9`**

This appears at `import torch` when the `nvidia-cusparse-cu12` and `nvidia-nvjitlink-cu12`
versions installed by pip into a conda env disagree on the CUDA minor version.
Prepend the env's bundled `nvjitlink` library to `LD_LIBRARY_PATH` so the loader picks
the matching one before launching any script:

```bash
# Run after `conda activate <env>` (works for any env using torch + CUDA 12.x wheels).
export LD_LIBRARY_PATH=$(python -c "import nvidia.nvjitlink as m; print(list(m.__path__)[0])")/lib:$LD_LIBRARY_PATH
```

You may want to add the line to your shell profile or to a small wrapper script that
activates the env before running Stage 03 (or any other torch-using stage).

---

## Quick Start

Two top-level shell scripts wrap the whole pipeline. Use these for the common cases; drop down to the per-stage commands in [Usage](#usage) only when you need to customize.

### `sample_test.sh` — end-to-end smoke test

Runs **Stages 00 → 04** on a single positive + single negative case using the bundled fixtures under `data/sample_test/`, so no MIMIC access is required. Use it to verify your environment before running anything serious.

```bash
bash sample_test.sh                            # default: 1 GPU (CUDA_VISIBLE_DEVICES=0)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash sample_test.sh
```

What it does:

1. Activates `rosalia`, runs Stage 00 (lightweight sub-steps) and Stage 01 (mask deformation, exercises SAM3).
2. Switches to `chexpercept`, runs Stage 02 (QA generation), Stage 03 (evaluates `medgemma1.5`), Stage 04 (analysis).
3. Skips two GPU-heavy steps (`01_generate_rosalia_pred.py` and `00_generate_anatomy_mask.py`) and pre-stages their outputs from fixtures.
4. Auto-fills the `optimal?` annotation column with `y`. Production runs must use real annotations.

Total time: ~3–4 minutes.

Prereqs: `rosalia` and `chexpercept` envs created from `envs/*.yml`, `api_info/api_keys.yaml` configured with at least the HF token.

### `eval_chexpercept.sh` — full benchmark evaluation

Runs **Stage 03 + Stage 04** on the downloaded CheXpercept benchmark. All arguments are forwarded to `00_eval.py`, then the per-model summary is aggregated.

```bash
# Place the downloaded benchmark at data/chexpercept/{chexpercept.json, chexpercept/}.

bash eval_chexpercept.sh --provider opensource --model medgemma1.5
bash eval_chexpercept.sh --provider opensource --model qwen3.6-27b --no-thinking
bash eval_chexpercept.sh --provider gemini    --model gemini-3.1-pro
bash eval_chexpercept.sh --model hulu-med                                # auto-switches to hulumed env

CUDA_VISIBLE_DEVICES=0,1,2,3 bash eval_chexpercept.sh --model qwen3.6-27b
ENV=my-chexpercept-env       bash eval_chexpercept.sh --model medgemma1.5  # custom env name
```

Defaults: `CUDA_VISIBLE_DEVICES=0`, `ENV=chexpercept` (auto-switches to `hulumed` for `--model hulu-med`).

Larger models need more GPUs — set `CUDA_VISIBLE_DEVICES` accordingly.

---

## Usage

### Stage 00: Source Data Curation

Samples cases from MIMIC-ILS and generates candidate lesion masks using ROSALIA. Run all commands from the project root.

```bash
conda activate rosalia

# 1. Sample positive and negative cases from MIMIC-ILS
python src/00_source_data_curation/00_sample_mimic_ils_case.py --config cfg/config.yaml

# 2. Generate ROSALIA predictions (multi-GPU via --part)
python src/00_source_data_curation/01_generate_rosalia_pred.py --config cfg/config.yaml --part 0 --total-parts 4
# Run with --part 1, 2, 3 on separate GPUs in parallel

# 3. Merge part outputs and build the positive annotation sheet
python src/00_source_data_curation/02_prepare_positive_annotation.py --total-parts 4

# 4. Build the negative annotation sheet
python src/00_source_data_curation/03_prepare_negative_annotation.py --config cfg/config.yaml

# 5. Distribute the sheets and images across annotators
python src/00_source_data_curation/04_distribute_labeling.py
```

After annotators mark the `optimal?` column in the labeling sheets, proceed to Stage 01.

### Stage 01: Mask Deformation

Generates lung/heart anatomy masks (CheXmask-U + CXAS), then deforms optimal masks using SAM3 to produce suboptimal candidates.

```bash
conda activate rosalia

# 1. Generate anatomy masks for all annotated CXRs
python src/01_mask_deformation/00_generate_anatomy_mask.py --config cfg/config.yaml

# 2. Deform optimal masks (uses the annotated `optimal?` rows from Stage 00)
python src/01_mask_deformation/01_deform_mask.py --config cfg/config.yaml --num-workers 8
```

### Stage 02: QA Generation

Generates QA pairs from Stage 01 deformation results and writes the per-key_id image bundle that Stage 03 evaluates against.

```bash
conda activate chexpercept

python src/02_qa_generation/00_generate_qa.py --config cfg/config.yaml --num-workers 8
```

Outputs land under `src/02_qa_generation/outputs/`:
- `qa_results/qa_results.json` — all QA dicts keyed by key_id
- `chexpercept/<key_id>/...` — image bundle (detection, contour QA images, fakes)

To run Stage 03 against this output, copy or symlink to `data/chexpercept/`:
```bash
mkdir -p data/chexpercept
cp src/02_qa_generation/outputs/qa_results/qa_results.json data/chexpercept/chexpercept.json
ln -sf "$PWD/src/02_qa_generation/outputs/chexpercept" data/chexpercept/chexpercept
```

### Stage 03: VLM Evaluation

Evaluates a VLM on the CheXpercept benchmark. Open-source models use vLLM; proprietary models use their respective APIs. Use `chexpercept` for all models except `hulu-med`, which requires the dedicated `hulumed` env.

```bash
conda activate chexpercept     # use `hulumed` only when --model hulu-med

# Open-source model (vLLM)
python src/03_eval_vlm_on_chexpercept/00_eval.py \
    --provider opensource \
    --model qwen3.6-27b \
    --oracle_setting none

# Proprietary model (e.g., Gemini)
python src/03_eval_vlm_on_chexpercept/00_eval.py \
    --provider gemini \
    --model gemini-3.1-pro \
    --oracle_setting implicit
```

Default `--benchmark-dir` is `data/chexpercept/`. Override with `--benchmark-dir <path>` if your benchmark lives elsewhere.

**Key arguments:**

| Argument | Options | Description |
|----------|---------|-------------|
| `--provider` | `opensource`, `openai`, `azure`, `gemini`, `dummy` | Model provider |
| `--model` | see `model_configs.py` | Model alias |
| `--oracle_setting` | `none`, `implicit`, `explicit` | Oracle injection mode |
| `--no-thinking` | flag | Disable thinking mode (Qwen3 series) |
| `--limit` | int | Limit number of QA items (for testing) |

**Oracle settings:**

- `none` — End-to-End: model's own answers carry forward (strictest, most clinically realistic)
- `implicit` — Oracle-Passed: prior errors are silently corrected to isolate per-stage capability
- `explicit` — Ground truth is prepended explicitly before each stage

### Stage 04: Result Analysis

Aggregates Stage 03 outputs across models into per-stage accuracy CSVs and plots.

```bash
conda activate chexpercept

python src/04_analyze_eval_result/01_analyze_model_performance.py --oracle_setting implicit
```

By default it scans `src/03_eval_vlm_on_chexpercept/outputs/<provider>_<model>/oracle_<setting>/all_results.json` for every model present and writes:
- `all_models_oracle_<setting>/all_models_summary.csv` (per-model, per-stage accuracy)
- `plot_stage_accuracy.png`, `plot_contour_revision_detail.png`, `plot_attribute_extraction_detail.png`, `plot_depth_heatmap.png`

Pass `--results-dir <path>` to point at a different Stage 03 output directory.

---

## Models

We evaluate 14 VLMs: 4 proprietary and 10 open-source (5 general-domain, 5 medical-domain).

| Domain | Model | Alias |
|--------|-------|-------|
| General | Qwen3.6-27B | `qwen3.6-27b` |
| General | Qwen3.5-122B (MoE) | `qwen3.5-moe` |
| General | GLM-4.6V | `glm-4v` |
| General | InternVL3.5-38B | `internvl3.5` |
| General | Gemma4-31B | `gemma4` |
| Medical | MedGemma-27B | `medgemma` |
| Medical | MedGemma1.5-4B | `medgemma1.5` |
| Medical | HuatuoGPT-Vision-7B | `huatuo` |
| Medical | Lingshu-32B | `lingshu` |
| Medical | Hulu-Med-32B | `hulu-med` |
| Proprietary | Gemini-3.1-Pro | via `--provider gemini` |
| Proprietary | Gemini-3.1-Flash | via `--provider gemini` |
| Proprietary | GPT-5.4 | via `--provider azure` |
| Proprietary | GPT-5.4-nano | via `--provider azure` |
