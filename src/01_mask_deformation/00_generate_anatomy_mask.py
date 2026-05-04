"""Generate lung/heart masks: CXAS + CheXmask-U."""

import os
import sys
import shutil
import argparse
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import yaml

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, "../.."))

# HF_HOME / HF_HUB_CACHE come from cfg/config.yaml (path config), not api_keys.yaml.
_cfg_path = os.path.join(_project_root, "cfg/config.yaml")
_hf_cfg = (yaml.safe_load(open(_cfg_path)) if os.path.exists(_cfg_path) else {}).get("huggingface", {})
if _hf_cfg.get("hf_home"):
    os.environ["HF_HOME"] = _hf_cfg["hf_home"]
if _hf_cfg.get("hf_hub_cache"):
    os.environ["HF_HUB_CACHE"] = _hf_cfg["hf_hub_cache"]

for _path in [
    _project_root,
    os.path.join(_current_dir, "CheXmask-U", "HybridGNet"),
]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from models.HybridGNet2IGSC import HybridGNetHF
from utils.config import open_config

warnings.filterwarnings("ignore", message=".*num_features.*affine=False")


def get_dense_mask(RL, LL, H, size=1024):
    RL = RL.reshape(-1, 1, 2).astype("int")
    LL = LL.reshape(-1, 1, 2).astype("int")
    H = H.reshape(-1, 1, 2).astype("int")
    RL_mask = cv2.drawContours(np.zeros([size, size], dtype="uint8"), [RL], -1, 1, -1)
    LL_mask = cv2.drawContours(np.zeros([size, size], dtype="uint8"), [LL], -1, 1, -1)
    H_mask  = cv2.drawContours(np.zeros([size, size], dtype="uint8"), [H],  -1, 1, -1)
    return RL_mask, LL_mask, H_mask


def get_dicom_ids_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return list(dict.fromkeys(df["dicom_id"].tolist()))


def copy_images_for_cxas(dicom_ids, image_dir, cxas_input_dir):
    os.makedirs(cxas_input_dir, exist_ok=True)
    copied = 0
    for dicom_id in dicom_ids:
        src = os.path.join(image_dir, f"{dicom_id}.png")
        if not os.path.isfile(src):
            print(f"  [Skip] not found: {src}")
            continue
        shutil.copy(src, os.path.join(cxas_input_dir, f"{dicom_id}.png"))
        copied += 1
    return copied


def run_cxas(cxas_input_dir, cxas_output_dir, batch_size=48):
    from cxas import CXAS
    os.makedirs(cxas_output_dir, exist_ok=True)
    model = CXAS(model_name="UNet_ResNet50_default", gpus="0")
    model.process_folder(
        input_directory_name=cxas_input_dir,
        output_directory=cxas_output_dir,
        storage_type="png",
        create=True,
        batch_size=batch_size,
    )


def run_chexmasku(meta_csv_path, image_dir, output_dir, device="cuda"):
    meta = pd.read_csv(meta_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    model = HybridGNetHF.from_pretrained("mcosarinsky/CheXmask-U", subfolder="v1_skip", device=device)

    for _, row in meta.iterrows():
        dicom_id = row['dicom_id']
        image_path = os.path.join(image_dir, f"{dicom_id}.png")
        if not os.path.isfile(image_path):
            print(f"  [Skip] not found: {image_path}")
            continue

        image_np = np.array(Image.open(image_path))
        if image_np.ndim == 3:
            image_np = image_np[:, :, 0]

        image_tensor = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output, _, _ = model(image_tensor)
            output = (output.cpu().numpy().reshape(-1, 2) * 1024).round().astype("int")

        RL_mask, LL_mask, H_mask = get_dense_mask(output[:44], output[44:94], output[94:])

        dicom_out = os.path.join(output_dir, dicom_id)
        os.makedirs(dicom_out, exist_ok=True)
        for mask, name in [(RL_mask, "right_lung.png"), (LL_mask, "left_lung.png"), (H_mask, "heart.png")]:
            Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").convert("1").save(
                os.path.join(dicom_out, name)
            )


def main(args):
    config = open_config(args.config)
    image_dir = config['path']['mimic_cxr_path']

    stage00_out = os.path.join(_project_root, "src", "00_source_data_curation", "outputs", "rosalia_pred")
    meta_csv_path = args.meta_csv or os.path.join(stage00_out, "labeling_sheet.csv")

    base_out = os.path.join(_current_dir, "outputs")
    cxas_input_dir   = os.path.join(base_out, "cxas_input")
    cxas_output_dir  = os.path.join(base_out, "cxas_pred")
    chexmasku_out    = os.path.join(base_out, "chexmasku_pred")

    dicom_ids = get_dicom_ids_from_csv(meta_csv_path)
    print(f"{len(dicom_ids)} unique dicom_ids from {meta_csv_path}")

    n = copy_images_for_cxas(dicom_ids, image_dir, cxas_input_dir)
    print(f"Copied {n} images -> {cxas_input_dir}")

    if n > 0:
        print("Running CXAS...")
        run_cxas(cxas_input_dir, cxas_output_dir)
        print(f"CXAS output: {cxas_output_dir}")
    else:
        print("CXAS skipped (no images).")

    print("Running CheXmask-U...")
    run_chexmasku(meta_csv_path, image_dir, chexmasku_out)
    print(f"CheXmask-U output: {chexmasku_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate lung/heart masks via CXAS + CheXmask-U")
    parser.add_argument('--config', type=str, default='cfg/config.yaml')
    parser.add_argument('--meta-csv', type=str, default=None,
                        help="Labeling sheet CSV with dicom_id column "
                             "(default: src/00_source_data_curation/outputs/rosalia_pred/labeling_sheet.csv)")
    args = parser.parse_args()
    main(args)
