import sys
import os
import argparse

MAX_CPU_THREADS = "4"

import torch
torch.set_num_threads(int(MAX_CPU_THREADS))

import json
from datetime import datetime
import multiprocessing as mp
import numpy as np

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, "../.."))
if _project_root not in sys.path:
    sys.path.append(_project_root)

# sam3 outer directory must be on sys.path so the inner sam3/ package is found correctly,
# not the outer directory treated as a namespace package.
_sam3_dir = os.path.join(_current_dir, "sam3")
if _sam3_dir not in sys.path:
    sys.path.insert(0, _sam3_dir)

from utils.config import open_config
import pandas as pd
from PIL import Image
from select_point import select_points
from process_mask import get_geometrical_mask_info, deform_mask
from qa_generation import deform_mask_for_qa_sequential
from sam_inference import iterative_postprocess_mask
from logger import init_logger, log_print
from tqdm import tqdm

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from scipy.ndimage import label

worker_model = None
worker_processor = None

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def init_worker(log_dir):
    os.environ["OMP_NUM_THREADS"] = MAX_CPU_THREADS
    os.environ["OPENBLAS_NUM_THREADS"] = MAX_CPU_THREADS
    os.environ["MKL_NUM_THREADS"] = MAX_CPU_THREADS
    os.environ["VECLIB_MAXIMUM_THREADS"] = MAX_CPU_THREADS
    os.environ["NUMEXPR_NUM_THREADS"] = MAX_CPU_THREADS

    import torch
    torch.set_num_threads(int(MAX_CPU_THREADS))

    global worker_model, worker_processor

    worker_name = mp.current_process().name
    log_file = os.path.join(log_dir, f'worker_{worker_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    init_logger(log_file)

    bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    worker_model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=True)
    worker_processor = Sam3Processor(worker_model)
    print(f"Worker {worker_name}: SAM model loaded")


def _to_bool(arr):
    a = np.array(arr) if not isinstance(arr, np.ndarray) else arr
    if a.ndim == 3:
        a = a[0] if a.shape[0] == 3 else a[:, :, 0]
    return a.astype(bool)


def _load_mask(path):
    img = Image.open(path).convert('L')
    if img.size != (1024, 1024):
        img = img.resize((1024, 1024), resample=Image.NEAREST)
    return np.array(img) > 0


def prepare_lung_masks(cxas_pred_path, dicom_id, chex_left, chex_right, refined_mask):
    left  = _to_bool(chex_left)  & _load_mask(os.path.join(cxas_pred_path, dicom_id, "left lung.png"))
    right = _to_bool(chex_right) & _load_mask(os.path.join(cxas_pred_path, dicom_id, "right lung.png"))

    if refined_mask is None:
        return left, right

    labeled, n = label(_to_bool(refined_mask))
    for cid in range(1, n + 1):
        comp = labeled == cid
        if np.any(comp & left):
            left  |= comp
        if np.any(comp & right):
            right |= comp

    return left, right


def load_data(row, mimic_cxr_path, rosalia_pred_path, chexmasku_pred_path):
    target   = row['target']
    key_id   = row['key_id']
    dicom_id = row['dicom_id']
    pair_id  = row['pair_id']

    image = Image.open(os.path.join(mimic_cxr_path, dicom_id + '.png'))
    mask  = Image.open(os.path.join(rosalia_pred_path, target, pair_id, 'pred_mask.png'))
    if mask.size != (1024, 1024):
        mask = mask.resize((1024, 1024), resample=Image.NEAREST)

    chex_left  = np.array(Image.open(os.path.join(chexmasku_pred_path, dicom_id, 'left_lung.png')))
    chex_right = np.array(Image.open(os.path.join(chexmasku_pred_path, dicom_id, 'right_lung.png')))

    return target, key_id, dicom_id, image, mask, chex_left, chex_right


def process_single_sample(args):
    global worker_model, worker_processor

    row_dict, config, rosalia_pred_path, chexmasku_pred_path, cxas_pred_path, output_dir = args
    row = pd.Series(row_dict)

    try:
        mimic_cxr_path = config['path']['mimic_cxr_path']
        save_visualization = config.get('mask_deformation', {}).get('save_visualization', True)

        target, key_id, dicom_id, image, mask, chex_left, chex_right = load_data(
            row, mimic_cxr_path, rosalia_pred_path, chexmasku_pred_path
        )

        output_path = os.path.join(output_dir, key_id)
        os.makedirs(output_path, exist_ok=True)

        log_print(f"<<< Start processing {key_id} >>>")

        log_print(f"  1. Selecting grid points ({key_id})")
        pos_pts, _ = select_points(mask)

        log_print(f"  2. Iterative postprocessing ({key_id})")
        mask_component_infos, refined_mask = iterative_postprocess_mask(
            pos_pts, mask, worker_model, worker_processor, output_path, use_iterative=True
        )

        log_print(f"  3. Computing geometrical mask info ({key_id})")
        chex_left, chex_right = prepare_lung_masks(
            cxas_pred_path, dicom_id, chex_left, chex_right, refined_mask
        )
        geometrical_mask_infos, region_masks_cache, cxas_masks_cache = get_geometrical_mask_info(
            config, target, key_id, dicom_id, mask_component_infos, chex_left, chex_right
        )

        log_print(f"  4. Deforming mask ({key_id})")
        deformation_results = deform_mask(
            target, image, geometrical_mask_infos, mask_component_infos,
            worker_model, worker_processor, output_path, chex_left, chex_right,
            region_masks_cache, cxas_masks_cache, key_id=key_id,
            save_visualization=save_visualization
        )

        log_print(f"  5. Deforming mask for QA ({key_id})")
        qa_results = deform_mask_for_qa_sequential(
            config, target, image, dicom_id, geometrical_mask_infos, mask_component_infos, deformation_results,
            worker_model, worker_processor, chex_left, chex_right,
            region_masks_cache, cxas_masks_cache, output_path, key_id=key_id,
            save_visualization=save_visualization
        )

        log_print(f"  Done: {key_id}")
        return key_id, qa_results

    except Exception as e:
        print(f"Error processing {row_dict.get('key_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main(config, num_workers):
    stage00_out = os.path.join(_project_root, "src", "00_source_data_curation", "outputs")
    meta_csv_path     = os.path.join(stage00_out, "rosalia_pred", "labeling_sheet.csv")
    rosalia_pred_path = os.path.join(stage00_out, "rosalia_pred")
    chexmasku_pred_path = os.path.join(_current_dir, "outputs", "chexmasku_pred")
    cxas_pred_path      = os.path.join(_current_dir, "outputs", "cxas_pred")

    base_output_dir      = os.path.join(_current_dir, "outputs")
    output_dir           = os.path.join(base_output_dir, "deformed_masks")
    output_dir_results   = os.path.join(base_output_dir, "deformation_results")
    log_dir              = os.path.join(base_output_dir, "logs")

    for d in [output_dir, output_dir_results, log_dir]:
        os.makedirs(d, exist_ok=True)

    meta = pd.read_csv(meta_csv_path)
    meta = meta[meta['good?'].notna() & (meta['good?'].astype(str).str.strip() != '')]
    print(f"Samples to process: {len(meta)} (filtered to annotated rows)")

    args_list = [
        (row.to_dict(), config, rosalia_pred_path, chexmasku_pred_path, cxas_pred_path, output_dir)
        for _, row in meta.iterrows()
    ]

    results = {}
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(log_dir,)) as pool:
        for key_id, qa_result in tqdm(
            pool.imap_unordered(process_single_sample, args_list),
            total=len(args_list),
            desc="Processing samples"
        ):
            if key_id is not None:
                results[key_id] = qa_result

    results_converted = convert_numpy_types(results)
    output_path = os.path.join(output_dir_results, 'deformation_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_converted, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    print(f"Successfully processed {len(results)} samples")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='cfg/config.yaml')
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()

    config = open_config(args.config)
    main(config, args.num_workers)
