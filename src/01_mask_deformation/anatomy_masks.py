"""Lung anatomy region masks (zones, axial regions, CPA, radial regions).

Pure geometry over CheXmask-U lung masks. No SAM / no deformation logic.
"""

import os
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # noqa: F401  (kept for parity with old import context)
from skimage.morphology import erosion


def _make_vertical_thirds_zones(lung_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Split a lung mask into upper / mid / base zones by dividing the y-extent into thirds."""
    lung_mask_bool = lung_mask.astype(bool)
    ys = np.where(lung_mask_bool)[0]
    if ys.size == 0:
        empty = np.zeros_like(lung_mask_bool, dtype=bool)
        return {"upper": empty, "mid": empty, "base": empty}

    y_min, y_max = int(ys.min()), int(ys.max())
    height = max(1, y_max - y_min + 1)
    b1 = y_min + height // 3
    b2 = y_min + (2 * height) // 3

    yy = np.arange(lung_mask_bool.shape[0])[:, None]
    return {
        "upper": np.logical_and(lung_mask_bool, yy <= b1),
        "mid":   np.logical_and(lung_mask_bool, np.logical_and(yy > b1, yy <= b2)),
        "base":  np.logical_and(lung_mask_bool, yy > b2),
    }


def get_mid_line_x(axial_mask):
    _, x_coords = np.where(axial_mask > 0)
    return int((np.max(x_coords) + np.min(x_coords)) / 2)


def get_all_axial_masks(chex_ll, chex_rl, debug_save_path=None):
    """Generate medial / lateral masks for both lungs from CheXmask-U lung masks."""
    axial_masks = {}

    left_mid_line_x = get_mid_line_x(chex_ll)
    left_medial = chex_ll.copy();  left_medial[:, left_mid_line_x:] = 0
    left_lateral = chex_ll.copy(); left_lateral[:, :left_mid_line_x] = 0
    axial_masks['left medial lung']  = left_medial
    axial_masks['left lateral lung'] = left_lateral

    right_mid_line_x = get_mid_line_x(chex_rl)
    right_medial = chex_rl.copy();  right_medial[:, :right_mid_line_x] = 0
    right_lateral = chex_rl.copy(); right_lateral[:, right_mid_line_x:] = 0
    axial_masks['right medial lung']  = right_medial
    axial_masks['right lateral lung'] = right_lateral

    if debug_save_path is not None:
        os.makedirs(debug_save_path, exist_ok=True)
        for region_name, axial_mask in axial_masks.items():
            side = 'left' if 'left' in region_name else 'right'
            region_type = 'medial' if 'medial' in region_name else 'lateral'
            mid_line_x = left_mid_line_x if side == 'left' else right_mid_line_x

            _, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(chex_ll, cmap='gray');  axes[0].set_title('Left Lung Mask');  axes[0].axis('off')
            axes[1].imshow(chex_rl, cmap='gray');  axes[1].set_title('Right Lung Mask'); axes[1].axis('off')
            base_mask = chex_ll if side == 'left' else chex_rl
            axes[2].imshow(base_mask, cmap='gray')
            axes[2].imshow(axial_mask, cmap='spring', alpha=0.5)
            axes[2].axvline(x=mid_line_x, color='red', linestyle='--', linewidth=2,
                            label=f'Mid line (x={mid_line_x})')
            axes[2].set_title(f'Axial Mask ({side} {region_type})')
            axes[2].legend(); axes[2].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(debug_save_path, f"debug_{region_name}_overlay.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()

    return axial_masks


def get_axial_mask(region, chex_ll, chex_rl, debug_save_path=None):
    return get_all_axial_masks(chex_ll, chex_rl, debug_save_path=debug_save_path)[region]


def process_peripheral_mask(right_lung_mask_bin, left_lung_mask_bin, side, shared_data=None):
    """Compute the peripheral region of one lung side.

    `shared_data` lets callers reuse the eroded combined mask across sides.
    """
    if shared_data is not None:
        eroded_mask   = shared_data['eroded_mask']
        right_overlap = shared_data['right_overlap']
        left_overlap  = shared_data['left_overlap']
    else:
        right_points = np.where(right_lung_mask_bin > 0)
        if len(right_points[0]) > 0:
            min_y, max_y = np.min(right_points[0]), np.max(right_points[0])
            one_third_y  = min_y + (max_y - min_y) // 3
            two_thirds_y = min_y + 2 * (max_y - min_y) // 3
            avg_width = (np.sum(right_lung_mask_bin[one_third_y] > 0) +
                         np.sum(right_lung_mask_bin[two_thirds_y] > 0)) / 2
            erosion_size = int(avg_width / 2)
        else:
            erosion_size = 50

        footprint     = np.ones((erosion_size, erosion_size))
        combined_mask = np.maximum(right_lung_mask_bin, left_lung_mask_bin)
        eroded_mask   = erosion(combined_mask, footprint=footprint, mode='reflect')
        right_overlap = np.logical_and(eroded_mask, right_lung_mask_bin)
        left_overlap  = np.logical_and(eroded_mask, left_lung_mask_bin)

        for ov_name in ('right', 'left'):
            ov = right_overlap if ov_name == 'right' else left_overlap
            num_labels, labels = cv2.connectedComponents(ov.astype(np.uint8))
            if num_labels > 1:
                largest = max(((i, np.sum(labels == i)) for i in range(1, num_labels)),
                              key=lambda x: x[1])[0]
                ov = (labels == largest).astype(np.uint8)
                if ov_name == 'right':
                    right_overlap = ov
                else:
                    left_overlap = ov

    def get_overlap_points(overlap):
        points_y, points_x = np.where(overlap)
        if len(points_x) == 0:
            return None, None
        top_idx    = np.lexsort((points_x, points_y))[0]
        bottom_idx = np.lexsort((points_x, -points_y))[0]
        return (points_x[top_idx], points_y[top_idx]), (points_x[bottom_idx], points_y[bottom_idx])

    def process_lung_side(lung_array, overlap, is_right=True):
        lung_points = np.where(lung_array > 0)
        eroded_points = np.where(overlap > 0)
        if len(lung_points[0]) == 0:
            return lung_array

        top_edge_y    = np.min(lung_points[0])
        bottom_edge_y = np.max(lung_points[0])

        if is_right:
            top_point, bottom_point = get_overlap_points(overlap)
        else:
            points_y, points_x = np.where(overlap)
            if len(points_x) == 0:
                return lung_array
            top_idx    = np.lexsort((-points_x, points_y))[0]
            bottom_idx = np.lexsort((points_x, -points_y))[0]
            top_point    = (points_x[top_idx], points_y[top_idx])
            bottom_point = (points_x[bottom_idx], points_y[bottom_idx])

        if top_point is None or len(eroded_points[0]) == 0:
            return lung_array

        if is_right:
            lung_array[top_edge_y:top_point[1] + 1, top_point[0] + 1:] = 0
            middle_y_range = range(top_point[1] + 1, bottom_point[1])
            if len(middle_y_range) > 0:
                y_to_min_x = {}
                for y, x in zip(eroded_points[0], eroded_points[1]):
                    if y in middle_y_range:
                        y_to_min_x[y] = min(y_to_min_x.get(y, x), x)
                for y, min_x in y_to_min_x.items():
                    lung_array[y, min_x:] = 0
            lung_array[bottom_point[1]:bottom_edge_y + 1, bottom_point[0] + 1:] = 0
        else:
            lung_array[top_edge_y:top_point[1] + 1, :top_point[0]] = 0
            middle_y_range = range(top_point[1] + 1, bottom_point[1])
            if len(middle_y_range) > 0:
                y_to_max_x = {}
                for y, x in zip(eroded_points[0], eroded_points[1]):
                    if y in middle_y_range:
                        y_to_max_x[y] = max(y_to_max_x.get(y, x), x)
                for y, max_x in y_to_max_x.items():
                    lung_array[y, :max_x + 1] = 0
            lung_array[bottom_point[1]:bottom_edge_y + 1, :bottom_point[0]] = 0

        return lung_array

    if side == 'right':
        return process_lung_side(right_lung_mask_bin.copy(), right_overlap, True)
    if side == 'left':
        return process_lung_side(left_lung_mask_bin.copy(), left_overlap, False)


def get_all_cpa_radial_masks(chex_ll, chex_rl, debug_save_path=None):
    """Generate CPA (costophrenic angle) and peripheral lung region masks for both sides."""
    cpa_radial_masks = {}

    right_points = np.where(chex_rl > 0)
    if len(right_points[0]) > 0:
        min_y, max_y = np.min(right_points[0]), np.max(right_points[0])
        one_third_y  = min_y + (max_y - min_y) // 3
        two_thirds_y = min_y + 2 * (max_y - min_y) // 3
        avg_width = (np.sum(chex_rl[one_third_y] > 0) + np.sum(chex_rl[two_thirds_y] > 0)) / 2
        erosion_size = int(avg_width / 2)
    else:
        erosion_size = 50

    footprint     = np.ones((erosion_size, erosion_size))
    combined_mask = np.maximum(chex_rl, chex_ll)
    eroded_mask   = erosion(combined_mask, footprint=footprint, mode='reflect')
    right_overlap = np.logical_and(eroded_mask, chex_rl)
    left_overlap  = np.logical_and(eroded_mask, chex_ll)

    for ov_name in ('right', 'left'):
        ov = right_overlap if ov_name == 'right' else left_overlap
        num_labels, labels = cv2.connectedComponents(ov.astype(np.uint8))
        if num_labels > 1:
            largest = max(((i, np.sum(labels == i)) for i in range(1, num_labels)),
                          key=lambda x: x[1])[0]
            ov = (labels == largest).astype(np.uint8)
            if ov_name == 'right':
                right_overlap = ov
            else:
                left_overlap = ov

    shared_data = {
        'eroded_mask':   eroded_mask,
        'right_overlap': right_overlap,
        'left_overlap':  left_overlap,
        'erosion_size':  erosion_size,
    }

    for side, lung_mask in (('left', chex_ll), ('right', chex_rl)):
        peripheral = process_peripheral_mask(chex_rl, chex_ll, side, shared_data=shared_data)
        cpa_radial_masks[f'{side} peripheral lung'] = peripheral

        cpa = peripheral.copy()
        lung_points = np.where(lung_mask > 0)
        if len(lung_points[0]) > 0:
            lung_min_y = np.min(lung_points[0])
            lung_max_y = np.max(lung_points[0])
            cut_y = lung_min_y + int((lung_max_y - lung_min_y) * 3 / 4)
            cpa[:cut_y, :] = 0
        cpa_radial_masks[f'{side} costophrenic angle'] = cpa

    if debug_save_path is not None:
        os.makedirs(debug_save_path, exist_ok=True)
        combined_lungs = np.logical_or(chex_ll, chex_rl)
        for region_name, mask in cpa_radial_masks.items():
            _, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(chex_ll, cmap='gray'); axes[0].set_title('Left Lung Mask'); axes[0].axis('off')
            axes[1].imshow(chex_rl, cmap='gray'); axes[1].set_title('Right Lung Mask'); axes[1].axis('off')
            axes[2].imshow(combined_lungs, cmap='gray')
            axes[2].imshow(mask, cmap='spring', alpha=0.5)
            axes[2].set_title(region_name); axes[2].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(debug_save_path, f"debug_{region_name}_overlay.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()

    return cpa_radial_masks


def get_cpa_radial_mask(region, chex_ll, chex_rl, debug_save_path=None):
    return get_all_cpa_radial_masks(chex_ll, chex_rl, debug_save_path=debug_save_path)[region]


def get_lung_region_mask(region, chex_ll, chex_rl, lung_region_dict, debug_save_path=None):
    if region in lung_region_dict['costophrenic angles'] + lung_region_dict['radial regions']:
        return get_cpa_radial_mask(region, chex_ll, chex_rl, debug_save_path=debug_save_path)
    if region in lung_region_dict['axial regions']:
        return get_axial_mask(region, chex_ll, chex_rl, debug_save_path=debug_save_path)
    raise ValueError(f"Invalid region: {region}")
