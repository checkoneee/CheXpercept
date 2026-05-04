"""Generic mask utilities: PIL/numpy conversion, connected-component filtering, centroid lookup."""

import numpy as np
from PIL import Image
from scipy.ndimage import label

from logger import log_print


def pil_to_numpy(pil):
    arr = np.array(pil)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def get_component_center_point(mask_component_np, pos_pts=None):
    """Return the centroid of a mask component (or the closest pos_pt if the centroid lies outside)."""
    y_coords, x_coords = np.where(mask_component_np == 255)
    if len(y_coords) == 0:
        return None

    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    center_pt  = (int(centroid_x), int(centroid_y))

    if mask_component_np[center_pt[1], center_pt[0]] == 255:
        return center_pt

    if pos_pts:
        closest_pt   = None
        min_distance = float('inf')
        for x, y in pos_pts:
            if mask_component_np[y, x] != 255:
                continue
            d = np.sqrt((x - centroid_x) ** 2 + (y - centroid_y) ** 2)
            if d < min_distance:
                min_distance = d
                closest_pt   = (x, y)
        if closest_pt is not None:
            log_print(f"  Centroid {center_pt} not in mask, using closest pos_pt {closest_pt}")
            return closest_pt

    return center_pt


def component_filtering(mask, min_size=1000):
    """Return a list of single-component PIL masks (>= min_size pixels), sorted largest-first."""
    if hasattr(mask, 'mode'):
        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
    else:
        mask_np = mask.copy()

    binary_mask = (mask_np == 255).astype(np.uint8)
    labeled_mask, num_labels = label(binary_mask)

    components = []
    for i in range(1, num_labels + 1):
        size = np.sum(labeled_mask == i)
        if size >= min_size:
            comp = np.zeros_like(mask_np)
            comp[labeled_mask == i] = 255
            components.append((size, Image.fromarray(comp.astype(np.uint8))))

    components.sort(key=lambda x: x[0], reverse=True)
    out = [m for _, m in components]
    log_print(f"    Component filtering: {num_labels} components found, kept {len(out)} (>= {min_size} px)")
    return out
