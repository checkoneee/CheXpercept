import sys
import os
import argparse
import yaml

MAX_CPU_THREADS = "8"
os.environ["OMP_NUM_THREADS"] = MAX_CPU_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = MAX_CPU_THREADS
os.environ["MKL_NUM_THREADS"] = MAX_CPU_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = MAX_CPU_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = MAX_CPU_THREADS

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

with open(os.path.join(project_root, "api_info/api_keys.yaml")) as _f:
    os.environ['HF_TOKEN'] = yaml.safe_load(_f)["hf_token"]
_cfg_path = os.path.join(project_root, "cfg/config.yaml")
_hf_cfg = (yaml.safe_load(open(_cfg_path)) if os.path.exists(_cfg_path) else {}).get("huggingface", {})
if _hf_cfg.get("hf_home"):
    os.environ['HF_HOME'] = _hf_cfg["hf_home"]
if _hf_cfg.get("hf_hub_cache"):
    os.environ['HF_HUB_CACHE'] = _hf_cfg["hf_hub_cache"]

for _path in [project_root, current_dir, os.path.join(current_dir, "LISA")]:
    if _path not in sys.path:
        sys.path.append(_path)

import json
import shutil
import csv
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils.config import open_config
from use_rosalia import initialize_model, prepare_inputs, run_inference, save_results


def generate_location_text(location):
    location = list(set(location))
    if len(location) == 0:
        return "right lung and left lung"
    if len(location) == 1:
        return location[0]
    if len(location) == 2:
        return f"{location[0]} and {location[1]}"
    return ", ".join(location[:-1]) + f", and {location[-1]}"


def create_review_plot(img_path, mask_path, key_id, mapped_location, save_path, segmentation_source=None):
    img = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    if mask.shape[:2] != img.shape[:2]:
        mask = np.array(Image.fromarray(mask).resize((img.shape[1], img.shape[0]), resample=Image.NEAREST))

    mask_binary = (mask > 127).astype(np.float32)
    overlay = img.copy().astype(np.float32)
    for c, color_val in enumerate([255, 0, 0]):
        overlay[:, :, c] = img[:, :, c] * (1 - 0.33 * mask_binary) + color_val * 0.33 * mask_binary
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    title = f"key_id: {key_id}\nmapped_location: {mapped_location}"
    if segmentation_source:
        title += f"\nsegmentation_source: {segmentation_source}"
    fig.suptitle(title, fontsize=10, wrap=True)
    axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(overlay); axes[1].set_title("Mask Overlay (33%)"); axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_work_list(global_info, basic_info, max_samples_per_lesion=3000):
    """Flatten positive cases into a single work list (global first, basic as fallback)."""
    all_items = []
    for target, items in global_info.items():
        num_global = min(len(items), max_samples_per_lesion)
        pool = items[:num_global] + basic_info.get(target, [])[:max_samples_per_lesion - num_global]
        for idx, item in enumerate(pool):
            source = "global" if idx < num_global else "basic"
            all_items.append({'target': target, 'segmentation_source': source, **item})
    return all_items


def run(config, part=None, total_parts=4, max_samples_per_lesion=3000):
    input_dir = os.path.join(current_dir, 'outputs')
    global_info = json.load(open(os.path.join(input_dir, 'global_segmentation_info.json')))
    basic_info = json.load(open(os.path.join(input_dir, 'basic_segmentation_info.json')))

    base_out = os.path.join(current_dir, 'outputs', 'rosalia_pred')
    os.makedirs(base_out, exist_ok=True)

    all_items = build_work_list(global_info, basic_info, max_samples_per_lesion)

    if part is not None:
        chunk_size = (len(all_items) + total_parts - 1) // total_parts
        start = part * chunk_size
        end = min(start + chunk_size, len(all_items))
        all_items = all_items[start:end]
        print(f"[Part {part}/{total_parts}] items {start}~{end - 1} ({len(all_items)} total)")

    model, tokenizer, clip_image_processor, transform = initialize_model()

    results = defaultdict(list)
    labeling_rows = []

    for target in {item['target'] for item in all_items}:
        os.makedirs(os.path.join(base_out, 'plots', target), exist_ok=True)

    for item_data in tqdm(all_items, desc="Generating ROSALIA predictions"):
        target = item_data['target']
        segmentation_source = item_data['segmentation_source']
        item = {k: v for k, v in item_data.items() if k not in ('target', 'segmentation_source')}

        pair_id = item['pair_id']
        key_id = f"{target}_{pair_id}"
        image_path = os.path.join(config['path']['mimic_cxr_path'], f"{item['dicom_id']}.png")

        case_dir = os.path.join(base_out, target, pair_id)
        os.makedirs(case_dir, exist_ok=True)
        save_img_path = os.path.join(case_dir, "pred_img.png")
        shutil.copy(image_path, save_img_path)

        mapped_location = 'heart' if target == 'cardiomegaly' else generate_location_text(item['reported_location'])
        pred_mask_path = os.path.join(case_dir, "pred_mask.png")

        bilateral = mapped_location in ('left lung and right lung', 'right lung and left lung')
        instruction = f"Segment the {target}." if target == 'cardiomegaly' or bilateral \
            else f"Segment the {target} in the {mapped_location}."
        image_clip, image, input_ids, resize_list, original_size_list = prepare_inputs(
            instruction, image_path, tokenizer, clip_image_processor, transform
        )
        pred_masks, output_text = run_inference(
            model, tokenizer, image_clip, image, input_ids, resize_list, original_size_list
        )
        save_results(pred_masks, pred_mask_path)

        results[pair_id].append({
            'dicom_id': item['dicom_id'],
            'target': target,
            'pred_mask_path': pred_mask_path,
            'output_text': output_text,
            'mapped_location': mapped_location,
            'instruction': instruction,
            'segmentation_source': segmentation_source,
        })

        plot_path = os.path.join(base_out, 'plots', target, f"{key_id}.png")
        create_review_plot(save_img_path, pred_mask_path, key_id, mapped_location, plot_path, segmentation_source)

        labeling_rows.append({
            'key_id': key_id,
            'dicom_id': item['dicom_id'],
            'target': target,
            'pair_id': pair_id,
            'mapped_location': mapped_location,
            'segmentation_source': segmentation_source,
            'plot_path': plot_path,
            'good?': '',
        })

    suffix = f"_part{part}" if part is not None else ""

    with open(os.path.join(base_out, f'results{suffix}.json'), 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    sheet_fields = ['key_id', 'dicom_id', 'target', 'pair_id', 'mapped_location', 'segmentation_source', 'good?']
    with open(os.path.join(base_out, f'labeling_sheet{suffix}.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sheet_fields + ['plot_path'])
        writer.writeheader()
        writer.writerows(labeling_rows)

    print(f"Done. Results saved to {base_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ROSALIA to generate candidate lesion masks")
    parser.add_argument('--config', type=str, default='cfg/config.yaml')
    parser.add_argument('--part', type=int, default=None,
                        help='Part index (0-based) for parallel execution across GPUs')
    parser.add_argument('--total-parts', type=int, default=4,
                        help='Total number of parts (default: 4)')
    parser.add_argument('--max-samples', type=int, default=3000,
                        help='Max cases to process per lesion type (default: 3000)')
    args = parser.parse_args()

    config = open_config(args.config)
    run(config, part=args.part, total_parts=args.total_parts, max_samples_per_lesion=args.max_samples)
