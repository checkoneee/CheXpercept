"""
Helpers for exporting CheXpercept benchmark assets.

Used by `generate_qa.py` to populate the `chexpercept/` output structure:
    chexpercept/{key_id}/
        detection_qa/xray.png
        contour_qa/contour_eval_qa/xray_with_mask.png
        contour_qa/contour_revision_qa/xray_with_mask_and_points.png
        contour_qa/contour_revision_qa/option_*.png

Doctor-review visualization PNGs (used during dataset curation) live in
`_internal/visualize_qa.py` instead.
"""

import os
import shutil

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, "../.."))


def load_mask_image(mask_path):
    """Load a grayscale mask PNG into a numpy array. Returns None if missing."""
    if not os.path.exists(mask_path):
        return None
    mask = Image.open(mask_path)
    if mask.mode != "L":
        mask = mask.convert("L")
    return np.array(mask)


def overlay_mask_on_image(image, mask, alpha=0.5, color="red"):
    """Overlay a binary mask on a CXR image with the given color/alpha."""
    if mask is None:
        return image

    if len(image.shape) == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image.copy()

    mask_normalized = (mask > 0).astype(float)

    color_rgb = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "sky_blue": (0.53, 0.81, 0.92),
        "cyan": (0.0, 1.0, 1.0),
    }.get(color, (1.0, 1.0, 0.0))  # default: yellow-ish
    r, g, b = color_rgb
    mask_color = np.stack(
        [mask_normalized * r, mask_normalized * g, mask_normalized * b],
        axis=-1,
    )

    weight = alpha * mask_normalized[..., np.newaxis]
    overlay = image_rgb * (1 - weight) + mask_color * weight
    return np.clip(overlay, 0, 1)


def get_dicom_id_from_key_id(
    key_id, config, labeling_sheet_positive_path, labeling_sheet_negative_path
):
    """Look up the dicom_id for a key_id from Stage 00 labeling sheets."""
    pos = pd.read_csv(labeling_sheet_positive_path)
    neg = pd.read_csv(labeling_sheet_negative_path)

    if key_id in pos["key_id"].values:
        return pos.loc[pos["key_id"] == key_id, "dicom_id"].values[0]
    if key_id in neg["key_id"].values:
        return neg.loc[neg["key_id"] == key_id, "dicom_id"].values[0]
    return None


def copy_masks_to_chexpercept(
    key_id,
    qa,
    config,
    deformed_mask_path,
    chexpercept_base_path,
    labeling_sheet_positive_path,
    labeling_sheet_negative_path,
):
    """
    Populate the per-case CheXpercept folder for one key_id.

    Layout:
        chexpercept/{key_id}/
            detection_qa/xray.png
            contour_qa/contour_eval_qa/xray_with_mask.png
            contour_qa/contour_revision_qa/xray_with_mask_and_points.png
            contour_qa/contour_revision_qa/<option_*.png>
    """
    dicom_id = get_dicom_id_from_key_id(
        key_id, config, labeling_sheet_positive_path, labeling_sheet_negative_path
    )
    if dicom_id is None:
        print(f"Warning: Could not find dicom_id for {key_id}")
        return

    key_id_folder = os.path.join(chexpercept_base_path, key_id)
    os.makedirs(key_id_folder, exist_ok=True)

    xray_path = os.path.join(config["path"]["mimic_cxr_path"], f"{dicom_id}.png")

    # Negative case: only the detection_qa xray is needed.
    if deformed_mask_path == "negative":
        if "detection_qa" in qa:
            detection_qa_folder = os.path.join(key_id_folder, "detection_qa")
            os.makedirs(detection_qa_folder, exist_ok=True)
            if os.path.exists(xray_path):
                shutil.copy2(xray_path, os.path.join(detection_qa_folder, "xray.png"))
        return

    mask_dir = os.path.join(deformed_mask_path, key_id)

    xray_image = None
    if os.path.exists(xray_path):
        xray_image = np.array(Image.open(xray_path)) / 255.0

    if "detection_qa" in qa:
        detection_qa_folder = os.path.join(key_id_folder, "detection_qa")
        os.makedirs(detection_qa_folder, exist_ok=True)
        if os.path.exists(xray_path):
            shutil.copy2(xray_path, os.path.join(detection_qa_folder, "xray.png"))

    if "contour_qa" not in qa or xray_image is None:
        return

    contour_qa = qa["contour_qa"]
    if not contour_qa:
        return

    contour_qa_folder = os.path.join(key_id_folder, "contour_qa")
    os.makedirs(contour_qa_folder, exist_ok=True)

    no_deformation = contour_qa.get("no_deformation", False)
    needs_revision = not no_deformation

    revision_before_masks = contour_qa.get("revision_before_masks", [])
    revision_after_masks = contour_qa.get("revision_after_masks", [])

    combined_before_mask = _combine_named_masks(mask_dir, revision_before_masks)
    combined_after_mask = _combine_named_masks(mask_dir, revision_after_masks)

    # Contour-eval overlay
    contour_eval_qa_folder = os.path.join(contour_qa_folder, "contour_eval_qa")
    os.makedirs(contour_eval_qa_folder, exist_ok=True)

    if no_deformation and combined_after_mask is not None:
        base_mask = combined_after_mask
    elif combined_before_mask is not None:
        base_mask = combined_before_mask
    else:
        base_mask = combined_after_mask

    if base_mask is not None:
        overlay = overlay_mask_on_image(xray_image, base_mask, alpha=0.5, color="cyan")
        Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(contour_eval_qa_folder, "xray_with_mask.png")
        )

    if not needs_revision:
        return

    revision_folder = os.path.join(contour_qa_folder, "contour_revision_qa")
    os.makedirs(revision_folder, exist_ok=True)

    expansion_qa = contour_qa.get("contour_revision_qa_expansion", {})
    answer_options = expansion_qa.get("answer_options", [])

    color_map = {
        "red": (1.0, 0.0, 0.0),
        "orange": (1.0, 0.5, 0.0),
        "yellow": (1.0, 1.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "purple": (0.5, 0.0, 0.5),
        "pink": (1.0, 0.75, 0.8),
        "brown": (0.6, 0.3, 0.0),
        "sky_blue": (0.53, 0.81, 0.92),
        "cyan": (0.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0),
        "None": (0.5, 0.5, 0.5),
    }

    if combined_before_mask is not None:
        _save_xray_with_mask_and_points(
            xray_image,
            combined_before_mask,
            answer_options,
            color_map,
            os.path.join(revision_folder, "xray_with_mask_and_points.png"),
        )

    revision_result_qa = contour_qa.get("contour_revision_qa_revision_result", {})
    result_options = revision_result_qa.get("answer_options", [])

    chexmask_path = config.get("path", {}).get(
        "chexmask_path",
        os.path.join(_project_root, "src/01_mask_deformation/outputs/chexmasku_pred"),
    )
    cxas_mask_path = config.get("path", {}).get(
        "cxas_mask_path",
        os.path.join(_project_root, "src/01_mask_deformation/outputs/cxas_pred"),
    )

    for option_data in result_options:
        relative_path = option_data.get("relative_path")
        if not relative_path:
            continue

        combined_mask = _resolve_option_mask(
            option_data.get("masks", []),
            mask_dir,
            chexmask_path,
            cxas_mask_path,
            dicom_id,
            combined_before_mask,
        )
        if combined_mask is None:
            continue

        combined_uint8 = (combined_mask.astype(np.uint8) * 255)
        overlay = overlay_mask_on_image(
            xray_image, combined_uint8, alpha=0.5, color="cyan"
        )
        Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(revision_folder, relative_path)
        )


def _combine_named_masks(mask_dir, mask_names):
    """OR-combine all named PNG masks under mask_dir; return uint8 [0,255] or None."""
    combined = None
    for name in mask_names:
        mask = load_mask_image(os.path.join(mask_dir, f"{name}.png"))
        if mask is None:
            continue
        binary = mask > 0
        combined = binary if combined is None else np.logical_or(combined, binary)
    if combined is None:
        return None
    return (combined.astype(np.uint8) * 255)


def _resolve_option_mask(
    mask_names,
    mask_dir,
    chexmask_path,
    cxas_mask_path,
    dicom_id,
    combined_before_mask,
):
    """Resolve a list of option mask names (including special tokens) into a binary mask."""
    combined = None
    for name in mask_names:
        mask = None
        if name == "default_empty":
            continue
        elif name == "default_keep":
            mask = combined_before_mask
        elif name == "default_right_lung_chex":
            mask = load_mask_image(os.path.join(chexmask_path, dicom_id, "right_lung.png"))
        elif name == "default_left_lung_chex":
            mask = load_mask_image(os.path.join(chexmask_path, dicom_id, "left_lung.png"))
        elif name == "default_right_lung_cxas":
            mask = load_mask_image(os.path.join(cxas_mask_path, dicom_id, "right lung.png"))
        elif name == "default_left_lung_cxas":
            mask = load_mask_image(os.path.join(cxas_mask_path, dicom_id, "left lung.png"))
        elif name == "default_dilated" and combined_before_mask is not None:
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(combined_before_mask, kernel, iterations=3)
        elif name == "default_eroded" and combined_before_mask is not None:
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(combined_before_mask, kernel, iterations=3)
        else:
            mask = load_mask_image(os.path.join(mask_dir, f"{name}.png"))

        if mask is None:
            continue
        binary = mask > 0
        combined = binary if combined is None else np.logical_or(combined, binary)
    return combined


def _save_xray_with_mask_and_points(
    xray_image, combined_before_mask, answer_options, color_map, save_path
):
    """Render xray + before-mask overlay + colored answer-option points; save PNG."""
    overlay = overlay_mask_on_image(
        xray_image, combined_before_mask, alpha=0.5, color="cyan"
    )
    if len(overlay.shape) == 2:
        overlay = np.stack([overlay] * 3, axis=-1)

    fig, ax = plt.subplots(
        1, 1,
        figsize=(xray_image.shape[1] / 100, xray_image.shape[0] / 100),
        dpi=100,
    )
    ax.imshow(overlay)
    ax.axis("off")

    for option in answer_options:
        point = option.get("point")
        color_name = option.get("color")
        if point is None or color_name not in color_map:
            continue
        x, y = point
        ax.add_patch(Circle((x, y), radius=15, color=color_map[color_name], fill=True, alpha=1.0))
        ax.add_patch(Circle((x, y), radius=15, color="white", fill=False, linewidth=2))

    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
