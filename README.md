# CheXpercept: A Benchmark for Evaluating Expert-Level Lesion Perception in Chest X-rays

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
CXReasoning/
├── cfg/                          # Configuration files
│   └── config_nas.yaml           # Main config (paths, hyperparameters)
├── src/
│   ├── 00_optimal_mask_generation/  # Stage 00: Sample MIMIC-ILS; generate optimal masks via ROSALIA
│   ├── 01_mask_deformation/         # Stage 01: Deform optimal masks (SAM3-based)
│   ├── 02_qa_generation/            # Stage 02: Generate and sample QA pairs
│   ├── 03_eval_vlm_on_chexpercept/  # Stage 03: Evaluate VLMs on CheXpercept
│   └── 04_analyze_eval_result/      # Stage 04: Analyze and visualize results
├── utils/
│   ├── config.py                 # Config loader
│   └── llm.py                    # LLM utilities
└── data/                         # Auxiliary metadata files
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

The CheXpercept benchmark dataset is available at: **(link to be added upon publication)**

The dataset requires access to the underlying CXR images from MIMIC-CXR. See [Data Requirements](#data-requirements) below.

---

## Data Requirements

CheXpercept is built on top of [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) and [MIMIC-ILS](https://physionet.org/content/mimic-cxr-ext-ils/). Both datasets require credentialed access on PhysioNet.

1. Request access to [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) on PhysioNet.
2. Request access to [MIMIC-ILS](https://physionet.org/content/mimic-cxr-ext-ils/) on PhysioNet.
3. Update the paths in `cfg/config_nas.yaml` to point to your local copies.

---

## Setup

### Environments

Three separate conda environments are used across the pipeline:

| Environment | Stages | Purpose |
|-------------|--------|---------|
| `rosalia` | 00 | ROSALIA inference for optimal mask generation |
| `cxreasoning` | 01 | Mask deformation |
| `chexpercept` | 02, 04 | QA generation and result analysis |
| `vllm` | 03 | VLM inference with vLLM |

```bash
conda env create -f envs/rosalia.yml
conda env create -f envs/chexpercept.yml
# ... (see envs/ for all environment files)
```

### LISA Setup (required for Stage 00)

Stage 00 uses [LISA](https://github.com/dvlab-research/LISA) as the backbone for ROSALIA. Clone it into the Stage 00 directory:

```bash
cd src/00_source_data_curation
git clone https://github.com/dvlab-research/LISA.git
```

> `use_rosalia.py` (our custom inference wrapper) is already included in the repo and does not need to be copied manually.

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

---

## Usage

### Stage 00: Source Data Curation

Samples cases from MIMIC-ILS and generates candidate lesion masks using ROSALIA.

```bash
cd src/00_source_data_curation

# 1. Sample positive and negative cases from MIMIC-ILS
conda activate chexpercept
python 00_sample_mimic_ils_case.py --config cfg/config.yaml

# 2. Generate ROSALIA predictions (supports multi-GPU via --part)
conda activate rosalia
python 01_generate_rosalia_pred.py --config cfg/config.yaml --part 0 --total-parts 4
# Run with --part 1, 2, 3 on separate GPUs in parallel

# 3. Merge part outputs and prepare positive annotation sheet
conda activate chexpercept
python 02_prepare_positive_annotation.py
```

### Stage 01: Mask Deformation

Generates suboptimal masks by deforming optimal masks using SAM3.

```bash
conda activate cxreasoning
cd src/01_mask_deformation

python 01_deform_mask.py --config cfg/config.yaml
```

### Stage 02: QA Generation

Generates and samples the final CheXpercept benchmark QA set.

```bash
conda activate chexpercept
cd src/02_qa_generation

python generate_qa.py --config cfg/config.yaml
python sample_qa.py --config cfg/config.yaml
python generate_final_chexpercept.py --config cfg/config.yaml
```

### Stage 03: VLM Evaluation

Evaluates a VLM on the CheXpercept benchmark. Open-source models use vLLM; proprietary models use their respective APIs.

```bash
conda activate vllm
cd src/03_eval_vlm_on_chexpercept

# Open-source model (vLLM)
python eval.py \
    --config cfg/config.yaml \
    --provider opensource \
    --model qwen3.6-27b \
    --oracle_setting none

# Proprietary model (e.g., Gemini)
python eval.py \
    --config cfg/config.yaml \
    --provider gemini \
    --model gemini-3.1-pro \
    --oracle_setting implicit
```

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

```bash
conda activate chexpercept
cd src/04_analyze_eval_result

python analyze_model_performance.py --config cfg/config.yaml
python build_per_lesion_table.py --config cfg/config.yaml
python build_per_path_table.py --config cfg/config.yaml
```

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
| Proprietary | GPT-5.4 | via `--provider openai` |
| Proprietary | GPT-5.4-nano | via `--provider openai` |

---

## License

This repository is released under the [MIT License](LICENSE).

The CheXpercept dataset is derived from [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) and [MIMIC-ILS](https://physionet.org/content/mimic-cxr-ext-ils/), which are subject to their respective PhysioNet Data Use Agreements. Users must independently obtain access to the source data through PhysioNet.
