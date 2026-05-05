import numpy as np
import os
from scipy.ndimage import label
from PIL import Image
import matplotlib.pyplot as plt
import random
import cv2

from select_point import select_points
from logger import log_print
from anatomy_masks import (
    _make_vertical_thirds_zones,
    get_all_axial_masks,
    get_all_cpa_radial_masks,
)
from mask_utils import pil_to_numpy, component_filtering, get_component_center_point
from sam_inference import (
    iterative_postprocess_mask,
    iterative_postprocess_mask_single,
    postprocess_mask_using_sam3,
)

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, "../.."))

cxas_mask_dict = {
    "zones": [
        "right upper zone lung",
        "right mid zone lung",
        "right lung base",
        "left upper zone lung",
        "left mid zone lung",
        "left lung base",
    ],
}

lung_region_dict = {
    'costophrenic angles': [
        'right costophrenic angle',
        'left costophrenic angle',
    ],
    'axial regions':[
        'right medial lung',
        'left medial lung',
        'right lateral lung',
        'left lateral lung',
    ],
    'radial regions':[
        'right peripheral lung',
        'left peripheral lung',
    ]
}


def _overlap_ratio(mask_np, ref_mask):
    """Return fraction of ref_mask pixels covered by mask_np."""
    ref = ref_mask.astype(bool)
    total = np.sum(ref)
    return float(np.sum(mask_np & ref) / total) if total > 0 else 0.0


def _overlap_entry(mask_np, ref_mask, threshold=0.2):
    ratio = _overlap_ratio(mask_np, ref_mask)
    return {"has_overlap": ratio >= threshold, "overlap_ratio": ratio}


def _ensure_best_overlap(entries):
    """If all entries have has_overlap=False, promote the highest-ratio entry to True."""
    if not entries or any(v['has_overlap'] for v in entries.values()):
        return
    best_key = max(entries, key=lambda k: entries[k]['overlap_ratio'])
    if entries[best_key]['overlap_ratio'] > 0:
        entries[best_key]['has_overlap'] = True


def _build_cxas_cache(config, dicom_id, target, chex_mask_left_lung, chex_mask_right_lung, use_chex_zones):
    cxas_mask_path = os.path.join(config['path']['cxas_mask_path'], dicom_id)
    cache = {}

    if use_chex_zones:
        left_zones  = _make_vertical_thirds_zones(chex_mask_left_lung)
        right_zones = _make_vertical_thirds_zones(chex_mask_right_lung)
        cache.update({
            'left upper zone lung':  left_zones['upper'],
            'left mid zone lung':    left_zones['mid'],
            'left lung base':        left_zones['base'],
            'right upper zone lung': right_zones['upper'],
            'right mid zone lung':   right_zones['mid'],
            'right lung base':       right_zones['base'],
        })

    lung_union = np.logical_or(chex_mask_left_lung, chex_mask_right_lung)
    for key, class_names in cxas_mask_dict.items():
        if use_chex_zones and key == "zones":
            continue
        for class_name in class_names:
            arr = np.array(Image.open(os.path.join(cxas_mask_path, class_name + '.png')).resize((1024, 1024)))
            if target != 'cardiomegaly':
                arr = np.logical_and(arr, lung_union)
            cache[class_name] = arr

    return cache


def _build_region_cache(chex_mask_left_lung, chex_mask_right_lung):
    cache = {}
    cache.update(get_all_axial_masks(chex_mask_left_lung, chex_mask_right_lung, debug_save_path=None))
    cache.update(get_all_cpa_radial_masks(chex_mask_left_lung, chex_mask_right_lung, debug_save_path=None))
    return cache


def _debug_lung_overlap(key_id, mask_component_id, mask_np,
                         chex_mask_left_lung, chex_mask_right_lung,
                         left_ratio, right_ratio, debug_dir):
    left_ov  = mask_np & chex_mask_left_lung.astype(bool)
    right_ov = mask_np & chex_mask_right_lung.astype(bool)
    both     = np.logical_or(chex_mask_left_lung, chex_mask_right_lung)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes[0, 0].imshow(mask_np,            cmap='Reds',   alpha=0.7); axes[0, 0].set_title(f'Mask Component\nSize: {int(np.sum(mask_np))} pixels')
    axes[0, 1].imshow(chex_mask_left_lung, cmap='Blues',  alpha=0.7); axes[0, 1].set_title(f'Left Lung Mask\nSize: {int(np.sum(chex_mask_left_lung))} pixels')
    axes[0, 2].imshow(chex_mask_right_lung,cmap='Greens', alpha=0.7); axes[0, 2].set_title(f'Right Lung Mask\nSize: {int(np.sum(chex_mask_right_lung))} pixels')
    axes[1, 0].imshow(both,   cmap='gray', alpha=0.5); axes[1, 0].imshow(mask_np, cmap='Reds', alpha=0.5); axes[1, 0].set_title('Mask Component over Both Lungs')
    axes[1, 1].imshow(chex_mask_left_lung,  cmap='gray', alpha=0.5); axes[1, 1].imshow(left_ov,  cmap='Reds', alpha=0.7)
    axes[1, 1].set_title(f'Left Lung Overlap\nSize: {int(np.sum(left_ov))} pixels\nRatio: {left_ratio:.4f} ({left_ratio*100:.2f}%)')
    axes[1, 2].imshow(chex_mask_right_lung, cmap='gray', alpha=0.5); axes[1, 2].imshow(right_ov, cmap='Reds', alpha=0.7)
    axes[1, 2].set_title(f'Right Lung Overlap\nSize: {int(np.sum(right_ov))} pixels\nRatio: {right_ratio:.4f} ({right_ratio*100:.2f}%)')
    for ax in axes.flatten():
        ax.axis('off')

    plt.suptitle(f'Overlap Debug Visualization\nKey ID: {key_id}, Component ID: {mask_component_id}', fontsize=14, y=0.995)
    plt.tight_layout()
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, f"{key_id}_component_{mask_component_id}_overlap_debug.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    log_print(f"    Debug visualization saved: {path}")
    log_print(f"      - Mask: {int(np.sum(mask_np))} px  |  Left: {int(np.sum(left_ov))} px ({left_ratio:.4f})  |  Right: {int(np.sum(right_ov))} px ({right_ratio:.4f})")


def _debug_zone_overlap(key_id, mask_component_id, mask_np, cxas_masks_cache, debug_dir):
    zone_names = [
        'left upper zone lung', 'left mid zone lung',  'left lung base',
        'right upper zone lung','right mid zone lung', 'right lung base',
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, zone_name in zip(axes.flatten(), zone_names):
        zone = (cxas_masks_cache.get(zone_name, 0) > 0)
        ov   = mask_np & zone
        total = int(np.sum(zone))
        ratio = float(np.sum(ov) / total) if total > 0 else 0.0
        ax.imshow(zone, cmap='gray', alpha=0.5)
        ax.imshow(ov,   cmap='Reds', alpha=0.75)
        ax.set_title(f"{zone_name}\nOverlap: {int(np.sum(ov))} px\nRatio: {ratio:.4f} ({ratio*100:.2f}%)")
        ax.axis('off')

    plt.suptitle(f'Zone Overlap Debug\nKey ID: {key_id}, Component ID: {mask_component_id}', fontsize=14, y=0.995)
    plt.tight_layout()
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, f"{key_id}_component_{mask_component_id}_zone_overlap_debug.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log_print(f"    Zone overlap visualization saved: {path}")


def get_geometrical_mask_info(config, target, key_id, dicom_id, mask_component_infos,
                               chex_mask_left_lung, chex_mask_right_lung):

    zone_source    = config.get('mask_deformation', {}).get('zone_mask_source') if isinstance(config, dict) else None
    use_chex_zones = (zone_source == 'chex_vertical_thirds')
    save_visualization = config.get('mask_deformation', {}).get('save_visualization', False)
    debug_dir = os.path.join(_current_dir, 'output', 'debug_overlap_visualizations')

    cxas_masks_cache   = _build_cxas_cache(config, dicom_id, target, chex_mask_left_lung, chex_mask_right_lung, use_chex_zones)
    region_masks_cache = _build_region_cache(chex_mask_left_lung, chex_mask_right_lung) if target != 'cardiomegaly' else {}

    all_geometrical_mask_info = []

    for info in mask_component_infos:
        component_id = info['mask_component_id']
        mask_np      = (info['best_mask'] == 255)

        left_ratio  = _overlap_ratio(mask_np, chex_mask_left_lung)
        right_ratio = _overlap_ratio(mask_np, chex_mask_right_lung)

        if save_visualization:
            _debug_lung_overlap(key_id, component_id, mask_np,
                                chex_mask_left_lung, chex_mask_right_lung,
                                left_ratio, right_ratio, debug_dir)
            try:
                _debug_zone_overlap(key_id, component_id, mask_np, cxas_masks_cache, debug_dir)
            except Exception as e:
                log_print(f"    [WARN] zone overlap visualization failed: {e}")

        geometrical_mask_info = {
            "mask_component_id": component_id,
            "overlap": {
                'left lung':  {"has_overlap": left_ratio  >= 0.01, "overlap_ratio": left_ratio,
                               "size": int(np.sum(mask_np & chex_mask_left_lung.astype(bool)))  if left_ratio  >= 0.01 else 0},
                'right lung': {"has_overlap": right_ratio >= 0.01, "overlap_ratio": right_ratio,
                               "size": int(np.sum(mask_np & chex_mask_right_lung.astype(bool))) if right_ratio >= 0.01 else 0},
            },
        }

        for key, class_names in cxas_mask_dict.items():
            geometrical_mask_info["overlap"].setdefault(key, {})
            for class_name in class_names:
                geometrical_mask_info["overlap"][key][class_name] = _overlap_entry(mask_np, cxas_masks_cache[class_name])
            _ensure_best_overlap(geometrical_mask_info["overlap"][key])

        if target != 'cardiomegaly':
            geometrical_mask_info["overlap"].setdefault('lung_regions', {})
            for key, region_names in lung_region_dict.items():
                for region_name in region_names:
                    region_mask = region_masks_cache[region_name]
                    if key == 'costophrenic angles':
                        geometrical_mask_info["overlap"]['lung_regions'][region_name] = _overlap_entry(mask_np, region_mask)
                    else:
                        side = 'left' if 'left' in region_name else 'right'
                        for zone_name in [f'{side} upper zone lung', f'{side} mid zone lung', f'{side} lung base']:
                            combined      = region_mask.astype(bool) & (cxas_masks_cache[zone_name] > 0)
                            combined_name = f"{region_name} & {zone_name}"
                            geometrical_mask_info["overlap"]['lung_regions'][combined_name] = _overlap_entry(mask_np, combined)
                            region_masks_cache[combined_name] = combined
            _ensure_best_overlap(geometrical_mask_info["overlap"]['lung_regions'])

        all_geometrical_mask_info.append(geometrical_mask_info)

    return all_geometrical_mask_info, region_masks_cache, cxas_masks_cache

def flip_positive_to_negative(accumulated_points, accumulated_labels, num_flips=3):
    """
    Randomly flip several existing positive points to negative.

    Args:
        accumulated_points: existing point list
        accumulated_labels: existing label list
        num_flips: number of points to flip (default: 3)

    Returns:
        flipped_points: list of flipped points [(x, y), ...]; empty if no points
        updated_points: updated point list
        updated_labels: updated label list
    """
    flipped_points = []
    updated_points = accumulated_points.copy()
    updated_labels = accumulated_labels.copy()

    # Find indices of positive points
    positive_indices = [i for i, label in enumerate(updated_labels) if label == 1]

    if len(positive_indices) > 0:
        # Actual number to flip is min(num_flips, number of positive points)
        actual_num_flips = min(num_flips, len(positive_indices))

        # Pick indices at random (without replacement)
        flip_indices = random.sample(positive_indices, actual_num_flips)
        
        for flip_idx in flip_indices:
            flipped_point = updated_points[flip_idx]
            updated_labels[flip_idx] = 0  # positive -> negative
            flipped_points.append(flipped_point)
            log_print(f"  Flipped point {flipped_point} from positive to negative (index {flip_idx})")
        
        log_print(f"  Total flipped: {len(flipped_points)} points")
    
    return flipped_points, updated_points, updated_labels

def add_negative_as_positive(neg_pts, cxas_mask_np, accumulated_points, accumulated_labels):
    """
    Among the negative points overlapping the CXAS mask, randomly add one as a positive prompt.

    Args:
        neg_pts: negative points list
        cxas_mask_np: CXAS mask numpy array
        accumulated_points: existing point list
        accumulated_labels: existing label list

    Returns:
        selected_point: added point (x, y) tuple, or None
        updated_points: updated point list
        updated_labels: updated label list
    """
    selected_point = None
    updated_points = accumulated_points.copy()
    updated_labels = accumulated_labels.copy()

    # Find points in neg_pts that overlap with cxas_mask_np
    neg_pts_in_cxas_mask = []
    for pt in neg_pts:
        x, y = pt
        if cxas_mask_np[y, x] > 0:
            neg_pts_in_cxas_mask.append(pt)

    # Randomly pick one and add it as a positive prompt
    if len(neg_pts_in_cxas_mask) > 0:
        selected_point = random.choice(neg_pts_in_cxas_mask)
        updated_points.append(selected_point)
        updated_labels.append(1)  # positive prompt
        log_print(f"  Added positive prompt at {selected_point} from {len(neg_pts_in_cxas_mask)} candidates")
    
    return selected_point, updated_points, updated_labels

def get_expansion_points(previous_best_mask, cxas_mask_np, num_points=5, kernel_size=32, contraction_points=None, min_distance=100, depth_level=1, iterations_per_depth=2):
    """
    Find points on the contour of the dilated mask that overlap cxas_mask.
    Only points that are at least min_distance away from contraction_points are picked.
    (Positive prompt points used to expand the region.)

    Args:
        previous_best_mask: previous best mask (1024x1024 numpy array, range 0~1)
        cxas_mask_np: CXAS mask numpy array (1024x1024, range 0~255)
        num_points: number of points to return (default: 5, ignored if width_level > 1)
        kernel_size: kernel size used for the dilation operation (default: 32)
        contraction_points: contraction points list [(x, y), ...]; None disables distance checks
        min_distance: minimum distance from contraction points (in pixels, default: 100)
        depth_level: depth deformation level (1=1 iter, 2=2 iters, 3=3 iters, ...);
                     additional points are placed near the base points at each iteration
        iterations_per_depth: number of dilation iterations performed per depth step (default: 2)

    Returns:
        expansion_points: contour points of the dilated mask [(x, y), ...]; empty if none
        dilated_mask: dilated mask (1024x1024 numpy array, range 0~1)
        depth_width_levels: list of width levels selected at each depth
        dilated_masks_by_depth: list of dilated masks per depth [(mask, depth_iter), ...]
    """
    expansion_points = []
    previous_depth_points = []  # points added in the previous depth iteration (base for the next depth)
    depth_width_levels = []  # record of width levels selected at each depth
    dilated_masks_by_depth = []  # store dilated masks per depth

    # Convert previous_best_mask to binary (threshold 0.5)
    prev_mask_binary = (previous_best_mask > 0.5).astype(np.uint8)
    prev_mask_binary = prev_mask_binary * 255  # convert to 0-255 range

    # Convert cxas_mask to binary
    cxas_mask_binary = (cxas_mask_np > 0).astype(np.uint8)
    cxas_mask_binary = cxas_mask_binary * 255  # convert to 0-255 range

    # Create kernel of size kernel_size x kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # ===== DEPTH LEVEL LOOP: perform multiple dilation iterations =====
    log_print(f"    [Expansion] Depth Level={depth_level}, Iterations per depth={iterations_per_depth}")

    # Final dilated_mask (for visualization, the result of the last iteration of depth_level)
    final_dilated_mask = None

    for depth_iter in range(1, depth_level + 1):
        # Perform dilation per depth (cumulative: depth_iter * iterations_per_depth)
        total_iterations = depth_iter * iterations_per_depth
        dilated_mask = cv2.dilate(prev_mask_binary, kernel, iterations=total_iterations)
        final_dilated_mask = dilated_mask  # save the last one

        # Save the per-depth mask (normalized to 0-1)
        dilated_masks_by_depth.append((dilated_mask / 255.0, depth_iter))

        # Extract the contour (outline) of dilated_mask
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Convert contour to binary mask (draw outline only)
        contour_mask = np.zeros_like(dilated_mask)
        cv2.drawContours(contour_mask, contours, -1, 255, thickness=1)

        # Keep only contour parts overlapping with cxas_mask
        valid_contour = cv2.bitwise_and(contour_mask, cxas_mask_binary)

        # Extract point coordinates from valid_contour
        y_coords, x_coords = np.where(valid_contour > 0)


        # ===== WIDTH LEVEL handling =====
        if depth_iter == 1:
            # Depth 1: pick width level
            current_depth_width_level = np.random.choice([2, 3, 4])
        else:
            # Depth > 1: pick a value smaller than the previous depth's width
            if len(depth_width_levels) == 0:
                # Previous depth had no candidates and was skipped
                log_print(f"    [Depth {depth_iter}/{depth_level}] No previous depth width levels, skipping")
                continue
            previous_width = depth_width_levels[-1]
            if previous_width > 2:
                # If previous width > 2, choose from 2 up to (previous_width - 1)
                available_widths = list(range(2, previous_width))
                current_depth_width_level = np.random.choice(available_widths)
            else:
                # If previous width is 2, cannot decrease further, keep at 2
                current_depth_width_level = 2

        # Record the selected width level
        depth_width_levels.append(current_depth_width_level)

        if len(y_coords) == 0:
            log_print(f"    [Depth {depth_iter}/{depth_level}] No valid contour points found")
            continue

        # Convert coordinates to a list
        candidate_points = list(zip(x_coords, y_coords))

        # Filter by distance if contraction_points is provided
        if contraction_points is not None and len(contraction_points) > 0:
            valid_candidates = []
            for candidate_pt in candidate_points:
                cx, cy = candidate_pt
                is_far_enough = True

                for con_pt in contraction_points:
                    ex, ey = con_pt
                    distance = np.sqrt((cx - ex)**2 + (cy - ey)**2)

                    if distance < min_distance:
                        is_far_enough = False
                        break

                if is_far_enough:
                    valid_candidates.append(candidate_pt)

            #if depth_iter == 1:  # log only on the first iteration
            #    log_print(f"  Filtered candidates by distance: {len(valid_candidates)}/{len(candidate_points)} points are far enough from contraction points (min_distance={min_distance})")
            candidate_points = valid_candidates

        if len(candidate_points) == 0:
            log_print(f"    [Depth {depth_iter}/{depth_level}] No valid candidates after filtering")
            continue

        # Points to be added in this depth iteration
        current_depth_points = []

        if depth_iter == 1:
            # Depth 1: pick base points (according to a random width level)
            first_point = random.choice(candidate_points)
            current_depth_points.append(first_point)
            expansion_points.append(first_point)

            # If width level > 1, pick additional points
            if current_depth_width_level > 1:
                points_to_add = current_depth_width_level - 1
                min_spacing = 30

                first_x, first_y = first_point
                selected_points = [first_point]

                for _ in range(points_to_add):
                    valid_candidates = []

                    for cand_pt in candidate_points:
                        if cand_pt in selected_points:
                            continue

                        cand_x, cand_y = cand_pt

                        # Check minimum distance against already-selected points
                        min_dist_to_selected = float('inf')
                        for sel_pt in selected_points:
                            sel_x, sel_y = sel_pt
                            dist = np.sqrt((cand_x - sel_x)**2 + (cand_y - sel_y)**2)
                            min_dist_to_selected = min(min_dist_to_selected, dist)

                        if min_dist_to_selected >= min_spacing:
                            dist_to_first = np.sqrt((cand_x - first_x)**2 + (cand_y - first_y)**2)
                            valid_candidates.append((dist_to_first, cand_pt))

                    if len(valid_candidates) == 0:
                        break

                    valid_candidates.sort(key=lambda x: x[0])
                    next_pt = valid_candidates[0][1]
                    current_depth_points.append(next_pt)
                    expansion_points.append(next_pt)
                    selected_points.append(next_pt)

                log_print(f"    [Depth 1/{depth_level}, Width {current_depth_width_level}] Selected {len(current_depth_points)} point(s)")
            else:
                log_print(f"    [Depth 1/{depth_level}, Width 1] Selected 1 point")

        else:
            # Depth > 1: pick current_depth_width_level points that are close to any previous-depth point
            if len(previous_depth_points) == 0:
                log_print(f"    [Depth {depth_iter}/{depth_level}] No previous depth points to extend from, skipping")
                continue

            # Randomly pick current_depth_width_level points from the previous depth
            num_prev_points_to_sample = min(current_depth_width_level, len(previous_depth_points))
            sampled_prev_points = random.sample(previous_depth_points, num_prev_points_to_sample)

            # For each chosen prev_pt, find the closest candidate
            for prev_pt in sampled_prev_points:
                prev_x, prev_y = prev_pt

                # Find the closest candidate among those not yet added
                min_dist = float('inf')
                closest_cand = None

                for cand_pt in candidate_points:
                    # Skip already-added points
                    if cand_pt in expansion_points:
                        continue

                    cand_x, cand_y = cand_pt
                    dist = np.sqrt((cand_x - prev_x)**2 + (cand_y - prev_y)**2)

                    if dist < min_dist:
                        min_dist = dist
                        closest_cand = cand_pt

                # If a closest candidate was found, add it
                if closest_cand is not None:
                    current_depth_points.append(closest_cand)
                    expansion_points.append(closest_cand)

            if len(current_depth_points) == 0:
                log_print(f"    [Depth {depth_iter}/{depth_level}] No valid candidates found")
                continue

            log_print(f"    [Depth {depth_iter}/{depth_level}, Width {current_depth_width_level}] Added {len(current_depth_points)} point(s) near previous depth points")

        # Save the points added at this depth so the next depth iteration can use them
        previous_depth_points = current_depth_points

    # Final result log
    if len(expansion_points) > 0:
        width_levels_str = ", ".join([f"D{i+1}:W{w}" for i, w in enumerate(depth_width_levels)])
        log_print(f"    [Expansion TOTAL] {len(expansion_points)} points (Depth={depth_level}, Widths=[{width_levels_str}])")
    else:
        log_print(f"    [Expansion] No points collected")

    # Convert final_dilated_mask to 0-1 range and return (for visualization)
    if final_dilated_mask is not None:
        dilated_mask_normalized = (final_dilated_mask / 255.0).astype(np.float32)
    else:
        # If all depth iterations failed, return the original mask
        dilated_mask_normalized = (prev_mask_binary / 255.0).astype(np.float32)

    return expansion_points, dilated_mask_normalized, depth_width_levels, dilated_masks_by_depth


def get_expansion_points_for_fake_mask(previous_best_mask, cxas_mask_np, num_points=5, kernel_size=32, contraction_points=None, min_distance=100, depth_level=1, iterations_per_depth=2):
    """
    Find points on the contour of the dilated mask that overlap cxas_mask.
    Only points that are at least min_distance away from contraction_points are picked.
    (Positive prompt points used to expand the region.)

    Args:
        previous_best_mask: previous best mask (1024x1024 numpy array, range 0~1)
        cxas_mask_np: CXAS mask numpy array (1024x1024, range 0~255)
        num_points: number of points to return (default: 5, ignored if width_level > 1)
        kernel_size: kernel size used for the dilation operation (default: 32)
        contraction_points: contraction points list [(x, y), ...]; None disables distance checks
        min_distance: minimum distance from contraction points (in pixels, default: 100)
        depth_level: depth deformation level (1=1 iter, 2=2 iters, 3=3 iters, ...);
                     additional points are placed near the base points at each iteration
        iterations_per_depth: number of dilation iterations performed per depth step (default: 2)

    Returns:
        expansion_points: contour points of the dilated mask [(x, y), ...]; empty if none
        dilated_mask: dilated mask (1024x1024 numpy array, range 0~1)
        depth_width_levels: list of width levels selected at each depth
        dilated_masks_by_depth: list of dilated masks per depth [(mask, depth_iter), ...]
    """
    expansion_points = []
    previous_depth_points = []  # points added in the previous depth iteration (base for the next depth)
    depth_width_levels = []  # record of width levels selected at each depth
    dilated_masks_by_depth = []  # store dilated masks per depth

    # Convert previous_best_mask to binary (threshold 0.5)
    prev_mask_binary = (previous_best_mask > 0.5).astype(np.uint8)
    prev_mask_binary = prev_mask_binary * 255  # convert to 0-255 range

    # Create kernel of size kernel_size x kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Convert cxas_mask_np to binary (threshold 0.5)
    cxas_mask_binary = (cxas_mask_np > 0.5).astype(np.uint8)
    cxas_mask_binary = cxas_mask_binary * 255  # convert to 0-255 range

    # ===== DEPTH LEVEL LOOP: perform multiple dilation iterations =====
    log_print(f"    [Expansion] Depth Level={depth_level}, Iterations per depth={iterations_per_depth}")

    # Final dilated_mask (for visualization, the result of the last iteration of depth_level)
    final_dilated_mask = None
    current_mask = prev_mask_binary.copy()  # current mask (for cumulative dilation)

    for depth_iter in range(1, depth_level + 1):
        # Perform a fixed iterations_per_depth dilations per depth (cumulative on previous result)
        dilated_mask = cv2.dilate(current_mask, kernel, iterations=iterations_per_depth)
        current_mask = dilated_mask  # update for the next depth_iter
        final_dilated_mask = dilated_mask  # save the last one

        # Save the per-depth mask (normalized to 0-1)
        dilated_masks_by_depth.append((dilated_mask / 255.0, depth_iter))

        # Extract the contour (outline) of dilated_mask
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Convert contour to binary mask (draw outline only)
        contour_mask = np.zeros_like(dilated_mask)
        cv2.drawContours(contour_mask, contours, -1, 255, thickness=1)

        # Keep only contour parts overlapping with cxas_mask
        valid_contour = cv2.bitwise_and(contour_mask, cxas_mask_binary)

        # Extract point coordinates from valid_contour
        y_coords, x_coords = np.where(valid_contour > 0)


        # ===== WIDTH LEVEL handling =====
        if depth_iter == 1:
            # Depth 1: pick width level
            if depth_level > 2:
                current_depth_width_level = np.random.choice([3, 4, 5])
            else:
                current_depth_width_level = np.random.choice([2, 3, 4])
        else:
            # Depth > 1: pick a value smaller than the previous depth's width
            if len(depth_width_levels) == 0:
                # Previous depth had no candidates and was skipped
                log_print(f"    [Depth {depth_iter}/{depth_level}] No previous depth width levels, skipping")
                continue
            previous_width = depth_width_levels[-1]
            if previous_width > 2:
                # If previous width > 2, choose from 2 up to (previous_width - 1)
                available_widths = list(range(2, previous_width))
                current_depth_width_level = np.random.choice(available_widths)
            else:
                # If previous width is 2, cannot decrease further, keep at 2
                current_depth_width_level = 2

        # Record the selected width level
        depth_width_levels.append(current_depth_width_level)

        if len(y_coords) == 0:
            log_print(f"    [Depth {depth_iter}/{depth_level}] No valid contour points found")
            continue

        # Convert coordinates to a list
        candidate_points = list(zip(x_coords, y_coords))

        # Filter by distance if contraction_points is provided
        if contraction_points is not None and len(contraction_points) > 0:
            valid_candidates = []
            for candidate_pt in candidate_points:
                cx, cy = candidate_pt
                is_far_enough = True

                for con_pt in contraction_points:
                    ex, ey = con_pt
                    distance = np.sqrt((cx - ex)**2 + (cy - ey)**2)

                    if distance < min_distance:
                        is_far_enough = False
                        break

                if is_far_enough:
                    valid_candidates.append(candidate_pt)

            #if depth_iter == 1:  # log only on the first iteration
            #    log_print(f"  Filtered candidates by distance: {len(valid_candidates)}/{len(candidate_points)} points are far enough from contraction points (min_distance={min_distance})")
            candidate_points = valid_candidates

        if len(candidate_points) == 0:
            log_print(f"    [Depth {depth_iter}/{depth_level}] No valid candidates after filtering")
            continue

        # Points to be added in this depth iteration
        current_depth_points = []

        if depth_iter == 1:
            # Depth 1: pick base points (according to a random width level)
            first_point = random.choice(candidate_points)
            current_depth_points.append(first_point)
            expansion_points.append(first_point)

            # If width level > 1, pick additional points
            if current_depth_width_level > 1:
                points_to_add = current_depth_width_level - 1
                min_spacing = 30

                first_x, first_y = first_point
                selected_points = [first_point]

                for _ in range(points_to_add):
                    valid_candidates = []

                    for cand_pt in candidate_points:
                        if cand_pt in selected_points:
                            continue

                        cand_x, cand_y = cand_pt

                        # Check minimum distance against already-selected points
                        min_dist_to_selected = float('inf')
                        for sel_pt in selected_points:
                            sel_x, sel_y = sel_pt
                            dist = np.sqrt((cand_x - sel_x)**2 + (cand_y - sel_y)**2)
                            min_dist_to_selected = min(min_dist_to_selected, dist)

                        if min_dist_to_selected >= min_spacing:
                            dist_to_first = np.sqrt((cand_x - first_x)**2 + (cand_y - first_y)**2)
                            valid_candidates.append((dist_to_first, cand_pt))

                    if len(valid_candidates) == 0:
                        break

                    valid_candidates.sort(key=lambda x: x[0])
                    next_pt = valid_candidates[0][1]
                    current_depth_points.append(next_pt)
                    expansion_points.append(next_pt)
                    selected_points.append(next_pt)

                log_print(f"    [Depth 1/{depth_level}, Width {current_depth_width_level}] Selected {len(current_depth_points)} point(s)")
            else:
                log_print(f"    [Depth 1/{depth_level}, Width 1] Selected 1 point")

        else:
            # Depth > 1: pick current_depth_width_level points that are close to any previous-depth point
            if len(previous_depth_points) == 0:
                log_print(f"    [Depth {depth_iter}/{depth_level}] No previous depth points to extend from, skipping")
                continue

            # Randomly pick current_depth_width_level points from the previous depth
            num_prev_points_to_sample = min(current_depth_width_level, len(previous_depth_points))
            sampled_prev_points = random.sample(previous_depth_points, num_prev_points_to_sample)

            # For each chosen prev_pt, find the closest candidate
            for prev_pt in sampled_prev_points:
                prev_x, prev_y = prev_pt

                # Find the closest candidate among those not yet added
                min_dist = float('inf')
                closest_cand = None

                for cand_pt in candidate_points:
                    # Skip already-added points
                    if cand_pt in expansion_points:
                        continue

                    cand_x, cand_y = cand_pt
                    dist = np.sqrt((cand_x - prev_x)**2 + (cand_y - prev_y)**2)

                    if dist < min_dist:
                        min_dist = dist
                        closest_cand = cand_pt

                # If a closest candidate was found, add it
                if closest_cand is not None:
                    current_depth_points.append(closest_cand)
                    expansion_points.append(closest_cand)

            if len(current_depth_points) == 0:
                log_print(f"    [Depth {depth_iter}/{depth_level}] No valid candidates found")
                continue

            log_print(f"    [Depth {depth_iter}/{depth_level}, Width {current_depth_width_level}] Added {len(current_depth_points)} point(s) near previous depth points")

        # Save the points added at this depth so the next depth iteration can use them
        previous_depth_points = current_depth_points

    # Final result log
    if len(expansion_points) > 0:
        width_levels_str = ", ".join([f"D{i+1}:W{w}" for i, w in enumerate(depth_width_levels)])
        log_print(f"    [Expansion TOTAL] {len(expansion_points)} points (Depth={depth_level}, Widths=[{width_levels_str}])")
    else:
        log_print(f"    [Expansion] No points collected")

    # Convert final_dilated_mask to 0-1 range and return (for visualization)
    if final_dilated_mask is not None:
        dilated_mask_normalized = (final_dilated_mask / 255.0).astype(np.float32)
    else:
        # If all depth iterations failed, return the original mask
        dilated_mask_normalized = (prev_mask_binary / 255.0).astype(np.float32)

    return expansion_points, dilated_mask_normalized, depth_width_levels, dilated_masks_by_depth

def get_contraction_points(previous_best_mask, cxas_mask_np, num_points=5, kernel_size=32, expansion_points=None, min_distance=100, depth_level=1, iterations_per_depth=2):
    """
    Find points on the contour of the eroded mask that overlap cxas_mask.
    Only points that are at least min_distance away from expansion_points are picked.
    (Negative prompt points used to contract the region.)

    Args:
        previous_best_mask: previous best mask (1024x1024 numpy array, range 0~1)
        cxas_mask_np: CXAS mask numpy array (1024x1024, range 0~255)
        num_points: number of points to return (default: 5, ignored if width_level > 1)
        kernel_size: kernel size used for the erosion operation (default: 32)
        expansion_points: expansion points list [(x, y), ...]; None disables distance checks
        min_distance: minimum distance from expansion points (in pixels, default: 100)
        depth_level: depth deformation level (1=1 iter, 2=2 iters, 3=3 iters, ...);
                     additional points are placed near the base points at each iteration
        iterations_per_depth: number of erosion iterations performed per depth step (default: 2)

    Returns:
        contraction_points: contour points of the eroded mask [(x, y), ...]; empty if none
        eroded_mask: eroded mask (1024x1024 numpy array, range 0~1)
        depth_width_levels: list of width levels selected at each depth
        eroded_masks_by_depth: list of eroded masks per depth [(mask, depth_iter), ...]
    """
    contraction_points = []
    previous_depth_points = []  # points added in the previous depth iteration (base for the next depth)
    depth_width_levels = []  # record of width levels selected at each depth
    eroded_masks_by_depth = []  # store eroded masks per depth

    # Convert previous_best_mask to binary (threshold 0.5)
    prev_mask_binary = (previous_best_mask > 0.5).astype(np.uint8)
    prev_mask_binary = prev_mask_binary * 255  # convert to 0-255 range

    # Convert cxas_mask to binary
    cxas_mask_binary = (cxas_mask_np > 0).astype(np.uint8)
    cxas_mask_binary = cxas_mask_binary * 255  # convert to 0-255 range

    # Create kernel of size kernel_size x kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Final eroded_mask (for visualization, the result of the last iteration of depth_level)
    final_eroded_mask = None
    current_mask = prev_mask_binary.copy()  # current mask (for cumulative erosion)

    for depth_iter in range(1, depth_level + 1):
        # Perform a fixed iterations_per_depth erosions per depth (cumulative on previous result)
        eroded_mask = cv2.erode(current_mask, kernel, iterations=iterations_per_depth)
        current_mask = eroded_mask  # update for the next depth_iter
        final_eroded_mask = eroded_mask  # save the last one

        # Save the per-depth mask (normalized to 0-1)
        eroded_masks_by_depth.append((eroded_mask / 255.0, depth_iter))

        # Extract the contour (outline) of eroded_mask
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Convert contour to binary mask (draw outline only)
        contour_mask = np.zeros_like(eroded_mask)
        cv2.drawContours(contour_mask, contours, -1, 255, thickness=1)

        # Keep only contour parts overlapping with cxas_mask
        valid_contour = cv2.bitwise_and(contour_mask, cxas_mask_binary)

        # Extract point coordinates from valid_contour
        y_coords, x_coords = np.where(valid_contour > 0)


        # ===== WIDTH LEVEL handling =====
        if depth_iter == 1:
            # Depth 1: pick width level
            current_depth_width_level = np.random.choice([2, 3, 4])
        else:
            # Depth > 1: pick a value smaller than the previous depth's width
            if len(depth_width_levels) == 0:
                # Previous depth had no candidates and was skipped
                log_print(f"    [Depth {depth_iter}/{depth_level}] No previous depth width levels, skipping")
                continue
            previous_width = depth_width_levels[-1]
            if previous_width > 2:
                # If previous width > 2, choose from 2 up to (previous_width - 1)
                available_widths = list(range(2, previous_width))
                current_depth_width_level = np.random.choice(available_widths)
            else:
                # If previous width is 2, cannot decrease further, keep at 2
                current_depth_width_level = 2

        # Record the selected width level
        depth_width_levels.append(current_depth_width_level)

        if len(y_coords) == 0:
            log_print(f"    [Depth {depth_iter}/{depth_level}] No valid contour points found")
            continue

        # Convert coordinates to a list
        candidate_points = list(zip(x_coords, y_coords))

        # Filter by distance if expansion_points is provided
        if expansion_points is not None and len(expansion_points) > 0:
            valid_candidates = []
            for candidate_pt in candidate_points:
                cx, cy = candidate_pt
                is_far_enough = True

                for exp_pt in expansion_points:
                    ex, ey = exp_pt
                    distance = np.sqrt((cx - ex)**2 + (cy - ey)**2)

                    if distance < min_distance:
                        is_far_enough = False
                        break

                if is_far_enough:
                    valid_candidates.append(candidate_pt)

            #if depth_iter == 1:  # log only on the first iteration
            #    log_print(f"  Filtered candidates by distance: {len(valid_candidates)}/{len(candidate_points)} points are far enough from expansion points (min_distance={min_distance})")
            candidate_points = valid_candidates



        if len(candidate_points) == 0:
            log_print(f"    [Depth {depth_iter}/{depth_level}] No valid candidates after filtering")
            continue

        # Points to be added in this depth iteration
        current_depth_points = []

        if depth_iter == 1:
            # Depth 1: pick base points (according to a random width level)
            first_point = random.choice(candidate_points)
            current_depth_points.append(first_point)
            contraction_points.append(first_point)

            # If width level > 1, pick additional points
            if current_depth_width_level > 1:
                points_to_add = current_depth_width_level - 1
                min_spacing = 30

                first_x, first_y = first_point
                selected_points = [first_point]

                for _ in range(points_to_add):
                    valid_candidates = []

                    for cand_pt in candidate_points:
                        if cand_pt in selected_points:
                            continue

                        cand_x, cand_y = cand_pt

                        # Check minimum distance against already-selected points
                        min_dist_to_selected = float('inf')
                        for sel_pt in selected_points:
                            sel_x, sel_y = sel_pt
                            dist = np.sqrt((cand_x - sel_x)**2 + (cand_y - sel_y)**2)
                            min_dist_to_selected = min(min_dist_to_selected, dist)

                        if min_dist_to_selected >= min_spacing:
                            dist_to_first = np.sqrt((cand_x - first_x)**2 + (cand_y - first_y)**2)
                            valid_candidates.append((dist_to_first, cand_pt))

                    if len(valid_candidates) == 0:
                        break

                    valid_candidates.sort(key=lambda x: x[0])
                    next_pt = valid_candidates[0][1]
                    current_depth_points.append(next_pt)
                    contraction_points.append(next_pt)
                    selected_points.append(next_pt)

                log_print(f"    [Depth 1/{depth_level}, Width {current_depth_width_level}] Selected {len(current_depth_points)} point(s)")
            else:
                log_print(f"    [Depth 1/{depth_level}, Width 1] Selected 1 point")

        else:
            # Depth > 1: pick current_depth_width_level points that are close to any previous-depth point
            if len(previous_depth_points) == 0:
                log_print(f"    [Depth {depth_iter}/{depth_level}] No previous depth points to extend from, skipping")
                continue

            # Randomly pick current_depth_width_level points from the previous depth
            num_prev_points_to_sample = min(current_depth_width_level, len(previous_depth_points))
            sampled_prev_points = random.sample(previous_depth_points, num_prev_points_to_sample)

            # For each chosen prev_pt, find the closest candidate
            for prev_pt in sampled_prev_points:
                prev_x, prev_y = prev_pt

                # Find the closest candidate among those not yet added
                min_dist = float('inf')
                closest_cand = None

                for cand_pt in candidate_points:
                    # Skip already-added points
                    if cand_pt in contraction_points:
                        continue

                    cand_x, cand_y = cand_pt
                    dist = np.sqrt((cand_x - prev_x)**2 + (cand_y - prev_y)**2)

                    if dist < min_dist:
                        min_dist = dist
                        closest_cand = cand_pt

                # If a closest candidate was found, add it
                if closest_cand is not None:
                    current_depth_points.append(closest_cand)
                    contraction_points.append(closest_cand)

            if len(current_depth_points) == 0:
                log_print(f"    [Depth {depth_iter}/{depth_level}] No valid candidates found")
                continue

            log_print(f"    [Depth {depth_iter}/{depth_level}, Width {current_depth_width_level}] Added {len(current_depth_points)} point(s) near previous depth points")

        # Save the points added at this depth so the next depth iteration can use them
        previous_depth_points = current_depth_points

    # Final result log
    if len(contraction_points) > 0:
        width_levels_str = ", ".join([f"D{i+1}:W{w}" for i, w in enumerate(depth_width_levels)])
        log_print(f"    [Contraction TOTAL] {len(contraction_points)} points (Depth={depth_level}, Widths=[{width_levels_str}])")
    else:
        log_print(f"    [Contraction] No points collected")

    # Convert final_eroded_mask to 0-1 range and return (for visualization)
    if final_eroded_mask is not None:
        eroded_mask_normalized = (final_eroded_mask / 255.0).astype(np.float32)
    else:
        # If all depth iterations failed, return the original mask
        eroded_mask_normalized = (prev_mask_binary / 255.0).astype(np.float32)

    return contraction_points, eroded_mask_normalized, depth_width_levels, eroded_masks_by_depth


def get_contraction_points_for_fake_mask(previous_best_mask, cxas_mask_np, num_points=5, kernel_size=32, expansion_points=None, min_distance=100, depth_level=1, iterations_per_depth=2):
    """
    Find points on the contour of the eroded mask that overlap cxas_mask.
    Only points that are at least min_distance away from expansion_points are picked.
    (Negative prompt points used to contract the region.)

    Args:
        previous_best_mask: previous best mask (1024x1024 numpy array, range 0~1)
        cxas_mask_np: CXAS mask numpy array (1024x1024, range 0~255)
        num_points: number of points to return (default: 5, ignored if width_level > 1)
        kernel_size: kernel size used for the erosion operation (default: 32)
        expansion_points: expansion points list [(x, y), ...]; None disables distance checks
        min_distance: minimum distance from expansion points (in pixels, default: 100)
        depth_level: depth deformation level (1=1 iter, 2=2 iters, 3=3 iters, ...);
                     additional points are placed near the base points at each iteration
        iterations_per_depth: number of erosion iterations performed per depth step (default: 2)

    Returns:
        contraction_points: contour points of the eroded mask [(x, y), ...]; empty if none
        eroded_mask: eroded mask (1024x1024 numpy array, range 0~1)
        depth_width_levels: list of width levels selected at each depth
        eroded_masks_by_depth: list of eroded masks per depth [(mask, depth_iter), ...]
    """
    contraction_points = []
    previous_depth_points = []  # points added in the previous depth iteration (base for the next depth)
    depth_width_levels = []  # record of width levels selected at each depth
    eroded_masks_by_depth = []  # store eroded masks per depth

    # Convert previous_best_mask to binary (threshold 0.5)
    prev_mask_binary = (previous_best_mask > 0.5).astype(np.uint8)
    prev_mask_binary = prev_mask_binary * 255  # convert to 0-255 range

    # Create kernel of size kernel_size x kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Convert cxas_mask_np to binary (threshold 0.5)
    cxas_mask_binary = (cxas_mask_np > 0.5).astype(np.uint8)
    cxas_mask_binary = cxas_mask_binary * 255  # convert to 0-255 range

    # Final eroded_mask (for visualization, the result of the last iteration of depth_level)
    final_eroded_mask = None
    current_mask = prev_mask_binary.copy()  # current mask (for cumulative erosion)

    for depth_iter in range(1, depth_level + 1):
        # Perform a fixed iterations_per_depth erosions per depth (cumulative on previous result)
        eroded_mask = cv2.erode(current_mask, kernel, iterations=iterations_per_depth)
        current_mask = eroded_mask  # update for the next depth_iter
        final_eroded_mask = eroded_mask  # save the last one

        # Save the per-depth mask (normalized to 0-1)
        eroded_masks_by_depth.append((eroded_mask / 255.0, depth_iter))

        # Extract the contour (outline) of eroded_mask
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Convert contour to binary mask (draw outline only)
        contour_mask = np.zeros_like(eroded_mask)
        cv2.drawContours(contour_mask, contours, -1, 255, thickness=1)

        # Keep only contour parts overlapping with cxas_mask
        valid_contour = cv2.bitwise_and(contour_mask, cxas_mask_binary)

        # Extract point coordinates from valid_contour
        y_coords, x_coords = np.where(valid_contour > 0)


        # ===== WIDTH LEVEL handling =====
        if depth_iter == 1:
            # Depth 1: pick width level
            if depth_level > 2:
                current_depth_width_level = np.random.choice([3, 4, 5])
            else:
                current_depth_width_level = np.random.choice([2, 3, 4])
        else:
            # Depth > 1: pick a value smaller than the previous depth's width
            if len(depth_width_levels) == 0:
                # Previous depth had no candidates and was skipped
                log_print(f"    [Depth {depth_iter}/{depth_level}] No previous depth width levels, skipping")
                continue
            previous_width = depth_width_levels[-1]
            if previous_width > 2:
                # If previous width > 2, choose from 2 up to (previous_width - 1)
                available_widths = list(range(2, previous_width))
                current_depth_width_level = np.random.choice(available_widths)
            else:
                # If previous width is 2, cannot decrease further, keep at 2
                current_depth_width_level = 2

        # Record the selected width level
        depth_width_levels.append(current_depth_width_level)

        if len(y_coords) == 0:
            log_print(f"    [Depth {depth_iter}/{depth_level}] No valid contour points found")
            continue

        # Convert coordinates to a list
        candidate_points = list(zip(x_coords, y_coords))

        # Filter by distance if expansion_points is provided
        if expansion_points is not None and len(expansion_points) > 0:
            valid_candidates = []
            for candidate_pt in candidate_points:
                cx, cy = candidate_pt
                is_far_enough = True

                for exp_pt in expansion_points:
                    ex, ey = exp_pt
                    distance = np.sqrt((cx - ex)**2 + (cy - ey)**2)

                    if distance < min_distance:
                        is_far_enough = False
                        break

                if is_far_enough:
                    valid_candidates.append(candidate_pt)

            #if depth_iter == 1:  # log only on the first iteration
            #    log_print(f"  Filtered candidates by distance: {len(valid_candidates)}/{len(candidate_points)} points are far enough from expansion points (min_distance={min_distance})")
            candidate_points = valid_candidates



        if len(candidate_points) == 0:
            log_print(f"    [Depth {depth_iter}/{depth_level}] No valid candidates after filtering")
            continue

        # Points to be added in this depth iteration
        current_depth_points = []

        if depth_iter == 1:
            # Depth 1: pick base points (according to a random width level)
            first_point = random.choice(candidate_points)
            current_depth_points.append(first_point)
            contraction_points.append(first_point)

            # If width level > 1, pick additional points
            if current_depth_width_level > 1:
                points_to_add = current_depth_width_level - 1
                min_spacing = 30

                first_x, first_y = first_point
                selected_points = [first_point]

                for _ in range(points_to_add):
                    valid_candidates = []

                    for cand_pt in candidate_points:
                        if cand_pt in selected_points:
                            continue

                        cand_x, cand_y = cand_pt

                        # Check minimum distance against already-selected points
                        min_dist_to_selected = float('inf')
                        for sel_pt in selected_points:
                            sel_x, sel_y = sel_pt
                            dist = np.sqrt((cand_x - sel_x)**2 + (cand_y - sel_y)**2)
                            min_dist_to_selected = min(min_dist_to_selected, dist)

                        if min_dist_to_selected >= min_spacing:
                            dist_to_first = np.sqrt((cand_x - first_x)**2 + (cand_y - first_y)**2)
                            valid_candidates.append((dist_to_first, cand_pt))

                    if len(valid_candidates) == 0:
                        break

                    valid_candidates.sort(key=lambda x: x[0])
                    next_pt = valid_candidates[0][1]
                    current_depth_points.append(next_pt)
                    contraction_points.append(next_pt)
                    selected_points.append(next_pt)

                log_print(f"    [Depth 1/{depth_level}, Width {current_depth_width_level}] Selected {len(current_depth_points)} point(s)")
            else:
                log_print(f"    [Depth 1/{depth_level}, Width 1] Selected 1 point")

        else:
            # Depth > 1: pick current_depth_width_level points that are close to any previous-depth point
            if len(previous_depth_points) == 0:
                log_print(f"    [Depth {depth_iter}/{depth_level}] No previous depth points to extend from, skipping")
                continue

            # Randomly pick current_depth_width_level points from the previous depth
            num_prev_points_to_sample = min(current_depth_width_level, len(previous_depth_points))
            sampled_prev_points = random.sample(previous_depth_points, num_prev_points_to_sample)

            # For each chosen prev_pt, find the closest candidate
            for prev_pt in sampled_prev_points:
                prev_x, prev_y = prev_pt

                # Find the closest candidate among those not yet added
                min_dist = float('inf')
                closest_cand = None

                for cand_pt in candidate_points:
                    # Skip already-added points
                    if cand_pt in contraction_points:
                        continue

                    cand_x, cand_y = cand_pt
                    dist = np.sqrt((cand_x - prev_x)**2 + (cand_y - prev_y)**2)

                    if dist < min_dist:
                        min_dist = dist
                        closest_cand = cand_pt

                # If a closest candidate was found, add it
                if closest_cand is not None:
                    current_depth_points.append(closest_cand)
                    contraction_points.append(closest_cand)

            if len(current_depth_points) == 0:
                log_print(f"    [Depth {depth_iter}/{depth_level}] No valid candidates found")
                continue

            log_print(f"    [Depth {depth_iter}/{depth_level}, Width {current_depth_width_level}] Added {len(current_depth_points)} point(s) near previous depth points")

        # Save the points added at this depth so the next depth iteration can use them
        previous_depth_points = current_depth_points

    # Final result log
    if len(contraction_points) > 0:
        width_levels_str = ", ".join([f"D{i+1}:W{w}" for i, w in enumerate(depth_width_levels)])
        log_print(f"    [Contraction TOTAL] {len(contraction_points)} points (Depth={depth_level}, Widths=[{width_levels_str}])")
    else:
        log_print(f"    [Contraction] No points collected")

    # Convert final_eroded_mask to 0-1 range and return (for visualization)
    if final_eroded_mask is not None:
        eroded_mask_normalized = (final_eroded_mask / 255.0).astype(np.float32)
    else:
        # If all depth iterations failed, return the original mask
        eroded_mask_normalized = (prev_mask_binary / 255.0).astype(np.float32)

    return contraction_points, eroded_mask_normalized, depth_width_levels, eroded_masks_by_depth

# Resize CXAS mask to match mask_input dimensions
from scipy.ndimage import zoom

def adjust_mask_input_with_points(mask_input, pos_points, neg_points, cxas_mask_np, radius=30, neg_strength=2.0, fake=False):
    """
    Adjust the probability values around positive/negative points in mask_input.
    - Positive points: only adjusted inside the CXAS mask
    - Negative points: adjusted strongly without CXAS mask restriction
    Uses a complex noise pattern.

    Args:
        mask_input: SAM mask input (numpy array, e.g. 288x288)
        pos_points: positive points list [(x, y), ...]
        neg_points: negative points list [(x, y), ...]
        cxas_mask_np: CXAS mask (1024x1024), used only when adjusting positive points
        radius: influence radius (in pixels, default: 30)
        neg_strength: strength of negative adjustment (default: 2.0, larger means stronger suppression)

    Returns:
        adjusted_mask_input: adjusted mask input
    """
    adjusted_mask_input = mask_input.copy()
    mask_size = mask_input.shape[0]  # 288

    # Min and max of the original mask_input
    mask_min = mask_input.min()
    mask_max = mask_input.max()

    # Scale factor for converting 1024 coords to mask_input size
    scale_factor = mask_size / 1024.0

    cxas_mask_resized = zoom(cxas_mask_np, scale_factor, order=1)
    cxas_mask_binary = (cxas_mask_resized > 0).astype(np.uint8)

    # Function to create a complex noise pattern
    def create_organic_kernel(radius):
        """Generate a Perlin-noise-style organic pattern (random ellipse)."""
        size = 2 * radius + 1
        kernel = np.zeros((size, size))

        # Combine noise at several frequencies
        for freq in [1, 2, 4]:
            # Generate random noise
            noise = np.random.rand(size // freq + 2, size // freq + 2)

            # Upsample to smooth it
            from scipy.ndimage import zoom
            noise_upsampled = zoom(noise, freq, order=3)

            # Match the size
            if noise_upsampled.shape[0] > size:
                noise_upsampled = noise_upsampled[:size, :size]
            else:
                pad_y = (size - noise_upsampled.shape[0]) // 2
                pad_x = (size - noise_upsampled.shape[1]) // 2
                noise_padded = np.zeros((size, size))
                noise_padded[pad_y:pad_y+noise_upsampled.shape[0],
                           pad_x:pad_x+noise_upsampled.shape[1]] = noise_upsampled
                noise_upsampled = noise_padded

            kernel += noise_upsampled / freq

        # Normalize
        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)

        # Apply random ellipse weighting
        # Random ratio of major and minor axes (0.5 ~ 1.5)
        ellipse_ratio_x = np.random.uniform(0.5, 1.5)
        ellipse_ratio_y = np.random.uniform(0.5, 1.5)

        # Random rotation angle of the ellipse (0 ~ 180 degrees)
        rotation_angle = np.random.uniform(0, np.pi)

        y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]

        # Apply rotation transform
        cos_angle = np.cos(rotation_angle)
        sin_angle = np.sin(rotation_angle)
        x_rot = cos_angle * x_grid - sin_angle * y_grid
        y_rot = sin_angle * x_grid + cos_angle * y_grid

        # Compute elliptical distance (different ratio per axis)
        ellipse_distance = np.sqrt((x_rot / ellipse_ratio_x)**2 + (y_rot / ellipse_ratio_y)**2)
        radial_weight = np.clip(1 - ellipse_distance / radius, 0, 1)

        kernel = kernel * radial_weight

        return kernel

    # Set values around positive points to the maximum (only inside CXAS mask)
    if pos_points is not None and len(pos_points) > 0:
        for pt in pos_points:
            x_1024, y_1024 = pt
            # Convert 1024 coords to mask_input coords
            x = int(x_1024 * scale_factor)
            y = int(y_1024 * scale_factor)

            # Range check
            if 0 <= x < mask_size and 0 <= y < mask_size:
                # Create the organic noise kernel
                organic_kernel = create_organic_kernel(radius)

                y_start = max(0, y - radius)
                y_end = min(mask_size, y + radius + 1)
                x_start = max(0, x - radius)
                x_end = min(mask_size, x + radius + 1)

                # Extract the corresponding kernel slice
                kernel_y_start = radius - (y - y_start)
                kernel_y_end = kernel_y_start + (y_end - y_start)
                kernel_x_start = radius - (x - x_start)
                kernel_x_end = kernel_x_start + (x_end - x_start)

                kernel_slice = organic_kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end]

                # Apply only inside the CXAS mask
                cxas_region = cxas_mask_binary[y_start:y_end, x_start:x_end]

                # Apply only where the kernel exceeds a threshold (removes square boundaries)
                kernel_threshold = 0.1  # ignore values <= 10%
                kernel_mask = kernel_slice > kernel_threshold

                # Set to max value (only inside CXAS mask & where kernel >= threshold)
                new_values = mask_max * kernel_slice

                if not fake:
                    apply_mask = np.logical_and(cxas_region > 0, kernel_mask)
                else:
                    apply_mask = kernel_mask

                # Compare against current values and keep the larger one
                adjusted_mask_input[y_start:y_end, x_start:x_end] = np.where(
                    apply_mask,
                    np.maximum(adjusted_mask_input[y_start:y_end, x_start:x_end], new_values),
                    adjusted_mask_input[y_start:y_end, x_start:x_end]
                )

    # Set values around negative points to the minimum (no CXAS mask restriction)
    if neg_points is not None and len(neg_points) > 0:
        for pt in neg_points:
            x_1024, y_1024 = pt
            # Convert 1024 coords to mask_input coords
            x = int(x_1024 * scale_factor)
            y = int(y_1024 * scale_factor)

            # Range check
            if 0 <= x < mask_size and 0 <= y < mask_size:
                # Create the organic noise kernel
                organic_kernel = create_organic_kernel(radius)

                y_start = max(0, y - radius)
                y_end = min(mask_size, y + radius + 1)
                x_start = max(0, x - radius)
                x_end = min(mask_size, x + radius + 1)

                # Extract the corresponding kernel slice
                kernel_y_start = radius - (y - y_start)
                kernel_y_end = kernel_y_start + (y_end - y_start)
                kernel_x_start = radius - (x - x_start)
                kernel_x_end = kernel_x_start + (x_end - x_start)

                kernel_slice = organic_kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end]

                # Apply only where the kernel exceeds a threshold (removes square boundaries)
                kernel_threshold = 0.1  # ignore values <= 10%
                kernel_mask = kernel_slice > kernel_threshold

                # Set to a value strongly below the minimum (boost the negative effect)
                # Lower mask_min by (mask_max - mask_min) * neg_strength to suppress strongly
                very_low_value = mask_min - (mask_max - mask_min) * neg_strength
                new_values = very_low_value + (mask_max - very_low_value) * (1 - kernel_slice)

                # Compare against current values and keep the smaller one (only where kernel >= threshold)
                adjusted_mask_input[y_start:y_end, x_start:x_end] = np.where(
                    kernel_mask,
                    np.minimum(adjusted_mask_input[y_start:y_end, x_start:x_end], new_values),
                    adjusted_mask_input[y_start:y_end, x_start:x_end]
                )

    return adjusted_mask_input

def save_mask_visualization(image, mask_input, model_output, previous_best_mask=None, pos_points=None, neg_points=None, dilated_mask=None, eroded_mask=None, save_path=None, all_expansion_points=None, all_contraction_points=None, anatomy_operations=None, anatomy_width_levels=None, anatomy_depth_levels=None, anatomy_depth_width_levels=None, all_dilated_masks_by_depth=None, all_eroded_masks_by_depth=None, key_id=None, suboptimal_component_id=None):
    """
    Visualize and save the original image, previous_best_mask overlay, dilated/eroded mask overlay, mask_input overlay, and model output overlay.

    Args:
        config: configuration dictionary
        dicom_id: DICOM ID
        image: original image (PIL Image or numpy array)
        mask_input: SAM mask input (logits)
        model_output: model output mask
        idx: component index
        key: geometrical mask key (e.g., 'zones', 'ribs')
        class_name: class name (e.g., 'right upper zone lung')
        added_point: added positive point (x, y) tuple, not shown if None
        previous_best_mask: previous best mask (numpy array), not shown if None
        pos_points: list of positive points [(x, y), ...], not shown if None
        neg_points: list of negative points [(x, y), ...], not shown if None
        dilated_mask: dilated mask (numpy array), not shown if None
        eroded_mask: eroded mask (numpy array), not shown if None
        all_expansion_points: expansion points dict {class_name: [(x, y), ...]}
        all_contraction_points: contraction points dict {class_name: [(x, y), ...]}
        anatomy_operations: anatomy operations dict {class_name: bool (True=expansion, False=contraction)}
        anatomy_width_levels: anatomy width levels dict {class_name: int (1, 2, or 3)} - number of points on the contour
        anatomy_depth_levels: anatomy depth levels dict {class_name: int (1, 2, or 3)} - number of iterations
        anatomy_depth_width_levels: anatomy depth width levels dict {class_name: [w1, w2, ...]} - widths actually chosen at each depth
        key_id: key ID (e.g., 'edema_s50083821_positive_1')
        suboptimal_component_id: suboptimal component ID (e.g., 'suboptimal_component_0_0')
    """
    # Create output directory
    # Convert the image to a numpy array
    if hasattr(image, 'mode'):  # PIL Image
        image_np = np.array(image)
    else:
        image_np = image

    # Check image size
    img_height, img_width = image_np.shape[:2]

    # Convert grayscale image to RGB (for overlay)
    if len(image_np.shape) == 2:  # grayscale
        image_rgb = np.stack([image_np, image_np, image_np], axis=-1)
    else:
        image_rgb = image_np

    # Normalize mask_input to 0-1 range
    mask_input_vis = mask_input.copy()
    mask_input_vis = (mask_input_vis - mask_input_vis.min()) / (mask_input_vis.max() - mask_input_vis.min() + 1e-8)

    # Set extent (aligned with image coordinate system)
    extent = [0, img_width, img_height, 0]

    # Create five subplots
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))

    # Add key_id, suboptimal_component, and per-prompt info at the top
    title_lines = []

    # Add key_id and suboptimal_component info
    if key_id is not None:
        title_lines.append(f"Key ID: {key_id}")
    if suboptimal_component_id is not None:
        title_lines.append(f"Component: {suboptimal_component_id}")

    # Add a blank line (as a separator)
    if title_lines:
        title_lines.append("")

    # Add per-anatomy operation info
    if all_expansion_points is not None and anatomy_operations is not None and anatomy_depth_levels is not None:
        for class_name, points in all_expansion_points.items():
            operation = "EXPANSION"
            depth = anatomy_depth_levels.get(class_name, "N/A")

            # Show width levels per depth
            if anatomy_depth_width_levels is not None and class_name in anatomy_depth_width_levels:
                depth_widths = anatomy_depth_width_levels[class_name]
                widths_str = ", ".join([f"D{i+1}:W{w}" for i, w in enumerate(depth_widths)])
                title_lines.append(f"{class_name}: {operation} (Depth={depth}, Widths=[{widths_str}]) - {len(points)} point(s)")
            else:
                # fallback to old format
                width = anatomy_width_levels.get(class_name, "N/A") if anatomy_width_levels else "N/A"
                title_lines.append(f"{class_name}: {operation} (W={width}, D={depth}) - {len(points)} point(s)")

    if all_contraction_points is not None and anatomy_operations is not None and anatomy_depth_levels is not None:
        for class_name, points in all_contraction_points.items():
            operation = "CONTRACTION"
            depth = anatomy_depth_levels.get(class_name, "N/A")

            # Show width levels per depth
            if anatomy_depth_width_levels is not None and class_name in anatomy_depth_width_levels:
                depth_widths = anatomy_depth_width_levels[class_name]
                widths_str = ", ".join([f"D{i+1}:W{w}" for i, w in enumerate(depth_widths)])
                title_lines.append(f"{class_name}: {operation} (Depth={depth}, Widths=[{widths_str}]) - {len(points)} point(s)")
            else:
                # fallback to old format
                width = anatomy_width_levels.get(class_name, "N/A") if anatomy_width_levels else "N/A"
                title_lines.append(f"{class_name}: {operation} (W={width}, D={depth}) - {len(points)} point(s)")
    
    if title_lines:
        fig.suptitle("\n".join(title_lines), fontsize=12, y=1.02)
    
    # 1. Original image (no prompts)
    axes[0].imshow(image_np, cmap='gray', extent=extent)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 2. Image + previous_best_mask overlay (no prompts)
    axes[1].imshow(image_np, cmap='gray', extent=extent)
    if previous_best_mask is not None:
        # Color only the parts where the mask is present
        masked_data = np.ma.masked_where(previous_best_mask < 0.5, previous_best_mask)
        axes[1].imshow(masked_data, cmap='jet', alpha=0.6, extent=extent, interpolation='bilinear', vmin=0, vmax=1)
    axes[1].set_title('Image + Previous Best Mask Overlay')
    axes[1].axis('off')

    # 3. Image + dilated/eroded mask overlay (with prompts), different color per depth
    axes[2].imshow(image_np, cmap='gray', extent=extent)

    # Show dilated masks per depth (different color per anatomy and per depth)
    if all_dilated_masks_by_depth is not None:
        # List of colormaps to use (different color per depth)
        dilated_cmaps = ['spring', 'summer', 'autumn']  # depth 1, 2, 3

        for class_name, masks_by_depth in all_dilated_masks_by_depth.items():
            for mask, depth_iter in masks_by_depth:
                # Use a different colormap depending on depth
                cmap_idx = (depth_iter - 1) % len(dilated_cmaps)
                current_cmap = dilated_cmaps[cmap_idx]

                masked_dilated = np.ma.masked_where(mask < 0.5, mask)
                axes[2].imshow(masked_dilated, cmap=current_cmap, alpha=0.4, extent=extent, interpolation='bilinear', vmin=0, vmax=1)
    elif dilated_mask is not None:
        # fallback: show a single dilated mask
        masked_dilated = np.ma.masked_where(dilated_mask < 0.5, dilated_mask)
        axes[2].imshow(masked_dilated, cmap='spring', alpha=0.6, extent=extent, interpolation='bilinear', vmin=0, vmax=1)

    # Show eroded masks per depth (different color per anatomy and per depth)
    if all_eroded_masks_by_depth is not None:
        # List of colormaps to use (different color per depth)
        eroded_cmaps = ['winter', 'cool', 'Wistia']  # depth 1, 2, 3

        for class_name, masks_by_depth in all_eroded_masks_by_depth.items():
            for mask, depth_iter in masks_by_depth:
                # Use a different colormap depending on depth
                cmap_idx = (depth_iter - 1) % len(eroded_cmaps)
                current_cmap = eroded_cmaps[cmap_idx]

                masked_eroded = np.ma.masked_where(mask < 0.5, mask)
                axes[2].imshow(masked_eroded, cmap=current_cmap, alpha=0.4, extent=extent, interpolation='bilinear', vmin=0, vmax=1)
    elif eroded_mask is not None:
        # fallback: show a single eroded mask
        masked_eroded = np.ma.masked_where(eroded_mask < 0.5, eroded_mask)
        axes[2].imshow(masked_eroded, cmap='winter', alpha=0.6, extent=extent, interpolation='bilinear', vmin=0, vmax=1)
    if pos_points is not None and len(pos_points) > 0:
        for i, (x, y) in enumerate(pos_points):
            axes[2].plot(x, y, '*', color='green', markersize=4, markeredgewidth=1)
            #if i == 0:  # show label only on the first point
            #    axes[2].text(x, y - 40, f'Pos Points ({len(pos_points)})', color='green', fontsize=10,
            #                ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    if neg_points is not None and len(neg_points) > 0:
        for i, (x, y) in enumerate(neg_points):
            axes[2].plot(x, y, 'x', color='red', markersize=4, markeredgewidth=1)
            #if i == 0:  # show label only on the first point
            #    axes[2].text(x, y - 20, f'Neg Points ({len(neg_points)})', color='red', fontsize=10,
            #        ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Show centroids of expansion points per deformation/anatomy (blue star)
    if all_expansion_points is not None:
        for anatomy_name, expansion_points in all_expansion_points.items():
            if len(expansion_points) > 0:
                centroid_x = np.mean([pt[0] for pt in expansion_points])
                centroid_y = np.mean([pt[1] for pt in expansion_points])
                axes[2].plot(centroid_x, centroid_y, '*', color='blue', markersize=4, markeredgewidth=1,
                           markeredgecolor='white', label=f'{anatomy_name} exp centroid')

    # Show centroids of contraction points per deformation/anatomy (blue X)
    if all_contraction_points is not None:
        for anatomy_name, contraction_points in all_contraction_points.items():
            if len(contraction_points) > 0:
                centroid_x = np.mean([pt[0] for pt in contraction_points])
                centroid_y = np.mean([pt[1] for pt in contraction_points])
                axes[2].plot(centroid_x, centroid_y, 'x', color='blue', markersize=4, markeredgewidth=1,
                           markeredgecolor='white', label=f'{anatomy_name} con centroid')

    axes[2].set_title('Image + Dilated/Eroded Mask Overlay')
    axes[2].axis('off')

    # 4. Image + mask_input overlay (with prompts)
    axes[3].imshow(image_np, cmap='gray', extent=extent)
    # mask_input has continuous values, so it is shown without thresholding
    axes[3].imshow(mask_input_vis, cmap='jet', alpha=0.4, extent=extent, interpolation='bilinear')
    if pos_points is not None and len(pos_points) > 0:
        for i, (x, y) in enumerate(pos_points):
            axes[3].plot(x, y, '*', color='green', markersize=4, markeredgewidth=1)
            #if i == 0:  # show label only on the first point
            #    axes[3].text(x, y - 40, f'Pos Points ({len(pos_points)})', color='green', fontsize=10,
            #                ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    if neg_points is not None and len(neg_points) > 0:
        for i, (x, y) in enumerate(neg_points):
            axes[3].plot(x, y, 'x', color='red', markersize=4, markeredgewidth=1)
            #if i == 0:  # show label only on the first point
            #    axes[3].text(x, y - 20, f'Neg Points ({len(neg_points)})', color='red', fontsize=10,
            #        ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Show centroids of expansion points per deformation/anatomy (blue star)
    if all_expansion_points is not None:
        for anatomy_name, expansion_points in all_expansion_points.items():
            if len(expansion_points) > 0:
                centroid_x = np.mean([pt[0] for pt in expansion_points])
                centroid_y = np.mean([pt[1] for pt in expansion_points])
                axes[3].plot(centroid_x, centroid_y, '*', color='blue', markersize=4, markeredgewidth=1,
                           markeredgecolor='white')

    # Show centroids of contraction points per deformation/anatomy (blue X)
    if all_contraction_points is not None:
        for anatomy_name, contraction_points in all_contraction_points.items():
            if len(contraction_points) > 0:
                centroid_x = np.mean([pt[0] for pt in contraction_points])
                centroid_y = np.mean([pt[1] for pt in contraction_points])
                axes[3].plot(centroid_x, centroid_y, 'x', color='blue', markersize=4, markeredgewidth=1,
                           markeredgecolor='white')

    axes[3].set_title('Image + Mask Input Overlay')
    axes[3].axis('off')

    # 5. Image + model output overlay (no prompts)
    axes[4].imshow(image_np, cmap='gray', extent=extent)
    # Color only the parts where the mask is present
    masked_output = np.ma.masked_where(model_output < 0.5, model_output)
    axes[4].imshow(masked_output, cmap='jet', alpha=0.6, extent=extent, interpolation='bilinear', vmin=0, vmax=1)
    axes[4].set_title('Image + Model Output Overlay')
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log_print(f"    Saved visualization to {save_path}")

from scipy.ndimage import zoom

def get_all_expansion_points(pts_dict):
    return [pt for pts in pts_dict.values() for pt in pts]


def get_all_contraction_points(pts_dict):
    return [pt for pts in pts_dict.values() for pt in pts]


def extract_overlap_anatomy(geometrical_mask_info):
    overlap_anatomy = {'zones': {}, 'lung_regions': {}}
    for key, value in geometrical_mask_info['overlap'].items():
        if key in ['left lung', 'right lung']:
            continue
        for class_name, overlap_info in value.items():
            if overlap_info['has_overlap']:
                overlap_anatomy[key][class_name] = {
                    'overlap_ratio': overlap_info['overlap_ratio'],
                    'key': key,
                }
    return overlap_anatomy

def select_anatomies_and_operations(overlap_anatomy, geometrical_mask_info):
    zone_names        = list(overlap_anatomy['zones'].keys())
    lung_region_names = list(overlap_anatomy['lung_regions'].keys())

    anatomy_operations = {}
    anatomy_depth_levels = {}
    selected_anatomies = []

    for zone_name in zone_names:
        candidates = [
            r for r in lung_region_names
            if zone_name in r or ('lung base' in zone_name and 'costophrenic angle' in r)
        ]
        if not candidates:
            continue

        do_expansion_list = []
        for region in candidates:
            do_expansion  = bool(np.random.choice([True, False]))
            overlap_ratio = overlap_anatomy['lung_regions'][region]['overlap_ratio']
            if overlap_ratio >= 0.75:
                do_expansion = False
            elif overlap_ratio <= 0.5:
                do_expansion = True
            side = 'left' if 'left' in region else 'right'
            if geometrical_mask_info['overlap'][f'{side} lung']['overlap_ratio'] >= 0.7:
                do_expansion = False
            do_expansion_list.append(do_expansion)

        idx = np.random.choice(len(candidates))
        anatomy_operations[candidates[idx]]  = do_expansion_list[idx]
        anatomy_depth_levels[candidates[idx]] = 2
        selected_anatomies.append(candidates[idx])

    return selected_anatomies, anatomy_operations, anatomy_depth_levels


def select_anatomies_and_operations_for_cardiomegaly(overlap_anatomy, max_anatomies=3):
    zone_names    = list(overlap_anatomy['zones'].keys())
    num_anatomies = random.randint(1, max_anatomies)
    if len(zone_names) > num_anatomies:
        zone_names = list(np.random.choice(zone_names, num_anatomies, replace=False))

    anatomy_operations   = {}
    anatomy_depth_levels = {}

    for zone_name in zone_names:
        anatomy_operations[zone_name]   = bool(np.random.choice([True, False]))
        anatomy_depth_levels[zone_name] = 1

    return list(anatomy_operations.keys()), anatomy_operations, anatomy_depth_levels

def verify_deformation(best_mask, expansion_points, contraction_points):
    """Check that expansion points fall inside the mask and contraction points outside."""
    mask_uint8    = (best_mask * 255).astype(np.uint8)
    failed_points = []

    for x, y in (expansion_points or []):
        if mask_uint8[y, x] < 128:
            failed_points.append(('expansion', x, y))

    for x, y in (contraction_points or []):
        if mask_uint8[y, x] >= 128:
            failed_points.append(('contraction', x, y))

    return len(failed_points) == 0, failed_points

def _collect_points_for_anatomy(class_name, do_expansion, depth_level,
                                 cxas_mask_np, previous_best_mask,
                                 all_expansion_points, all_contraction_points,
                                 min_point_distance, iterations_per_depth, max_retries):
    """Collect expansion or contraction points for one anatomy. Returns True on success."""
    for _ in range(max_retries):
        if do_expansion:
            pts, dilated_mask, depth_widths, masks_by_depth = get_expansion_points(
                previous_best_mask, cxas_mask_np, num_points=1, kernel_size=32,
                contraction_points=get_all_contraction_points(all_contraction_points),
                min_distance=min_point_distance, depth_level=depth_level,
                iterations_per_depth=iterations_per_depth * 2
            )
            if len(pts) == sum(depth_widths) and len(pts) > 0:
                all_expansion_points[class_name] = pts
                return True, dilated_mask, depth_widths, masks_by_depth, 'expansion'
        else:
            pts, eroded_mask, depth_widths, masks_by_depth = get_contraction_points(
                previous_best_mask, cxas_mask_np, num_points=1, kernel_size=32,
                expansion_points=get_all_expansion_points(all_expansion_points),
                min_distance=min_point_distance, depth_level=depth_level,
                iterations_per_depth=iterations_per_depth
            )
            if len(pts) == sum(depth_widths) and len(pts) > 0:
                all_contraction_points[class_name] = pts
                return True, eroded_mask, depth_widths, masks_by_depth, 'contraction'
    return False, None, None, None, None


def collect_deformation_points(selected_anatomies, anatomy_operations,
                                anatomy_depth_levels, previous_best_mask, region_masks_cache, chex_rl, chex_ll,
                                min_point_distance, iterations_per_depth=2, max_retries=3):
    all_expansion_points    = {}
    all_contraction_points  = {}
    all_dilated_masks       = []
    all_eroded_masks        = []
    anatomy_depth_width_levels = {}
    all_dilated_masks_by_depth = {}
    all_eroded_masks_by_depth  = {}
    any_success = False

    for class_name in selected_anatomies:
        do_expansion  = anatomy_operations[class_name]
        depth_level   = anatomy_depth_levels[class_name]
        cxas_mask_np  = region_masks_cache[class_name]

        ok, mask, depth_widths, masks_by_depth, kind = _collect_points_for_anatomy(
            class_name, do_expansion, depth_level, cxas_mask_np, previous_best_mask,
            all_expansion_points, all_contraction_points,
            min_point_distance, iterations_per_depth, max_retries
        )
        if ok:
            anatomy_depth_width_levels[class_name] = depth_widths
            if kind == 'expansion':
                all_dilated_masks.append(mask)
                all_dilated_masks_by_depth[class_name] = masks_by_depth
            else:
                all_eroded_masks.append(mask)
                all_eroded_masks_by_depth[class_name] = masks_by_depth
            any_success = True

    return (all_expansion_points, all_contraction_points,
            dict(all_expansion_points), dict(all_contraction_points),
            all_dilated_masks, all_eroded_masks, [],
            anatomy_depth_width_levels, all_dilated_masks_by_depth,
            all_eroded_masks_by_depth, not any_success)

def collect_deformation_points_for_cardiomegaly(selected_anatomies, anatomy_operations,
                                anatomy_depth_levels, previous_best_mask, cxas_masks_cache, chex_rl, chex_ll,
                                min_point_distance, iterations_per_depth=2, max_retries=3):
    all_expansion_points    = {}
    all_contraction_points  = {}
    all_dilated_masks       = []
    all_eroded_masks        = []
    anatomy_depth_width_levels = {}
    all_dilated_masks_by_depth = {}
    all_eroded_masks_by_depth  = {}
    any_success = False

    for class_name in selected_anatomies:
        do_expansion  = anatomy_operations[class_name]
        depth_level   = anatomy_depth_levels[class_name]
        cxas_mask_np  = cxas_masks_cache[class_name]

        ok, mask, depth_widths, masks_by_depth, kind = _collect_points_for_anatomy(
            class_name, do_expansion, depth_level, cxas_mask_np, previous_best_mask,
            all_expansion_points, all_contraction_points,
            min_point_distance, iterations_per_depth, max_retries
        )
        if ok:
            anatomy_depth_width_levels[class_name] = depth_widths
            if kind == 'expansion':
                all_dilated_masks.append(mask)
                all_dilated_masks_by_depth[class_name] = masks_by_depth
            else:
                all_eroded_masks.append(mask)
                all_eroded_masks_by_depth[class_name] = masks_by_depth
            any_success = True

    return (all_expansion_points, all_contraction_points,
            dict(all_expansion_points), dict(all_contraction_points),
            all_dilated_masks, all_eroded_masks, [],
            anatomy_depth_width_levels, all_dilated_masks_by_depth,
            all_eroded_masks_by_depth, not any_success)

def collect_deformation_points_for_fake_mask(previous_best_mask, previous_all_expansion_points, previous_all_contraction_points,
                                min_point_distance, anatomy_operations, anatomy_depth_levels, chex_rl, chex_ll, iterations_per_depth=2, max_retries=3):
    """
    Collect expansion/contraction points for the selected anatomies, retrying on failure.

    Args:
        iterations_per_depth: number of dilation/erosion iterations per depth step (default: 2)
        max_retries: maximum retries per anatomy (default: 3)

    Returns:
        all_expansion_points: {class_name: [(x, y), ...]}
        all_contraction_points: {class_name: [(x, y), ...]}
        all_dilated_masks: [mask1, mask2, ...]
        all_eroded_masks: [mask1, mask2, ...]
        all_cxas_masks: [mask1, mask2, ...]
        all_dilated_masks_by_depth: {class_name: [(mask, depth_iter), ...]} - per-depth dilated masks for each anatomy
        all_eroded_masks_by_depth: {class_name: [(mask, depth_iter), ...]} - per-depth eroded masks for each anatomy
    """
    all_expansion_points = {}
    all_contraction_points = {}
    all_dilated_masks = []
    all_eroded_masks = []
    anatomy_depth_width_levels = {}  # actual per-depth width levels for each anatomy
    all_dilated_masks_by_depth = {}  # per-depth dilated masks for each anatomy
    all_eroded_masks_by_depth = {}  # per-depth eroded masks for each anatomy


    log_print(f"    Collecting deformation points for fake mask...")

    do_expansion = anatomy_operations['fake_mask']
    current_depth_level = anatomy_depth_levels['fake_mask']

    # Retry logic: repeat until points are successfully extracted
    retry_count = 0
    points_collected = False

    cxas_mask_np = np.logical_or(chex_rl, chex_ll)

    log_print(f"    Starting to collect deformation points for fake mask... {'EXPANSION' if do_expansion else 'CONTRACTION'}, Max Depth={current_depth_level}")

    while not points_collected and retry_count < max_retries: # Begin point extraction for the previously fixed Width of First Depth and Max Depth
        if retry_count > 0:
            log_print(f"    Retry {retry_count}/{max_retries}")



        # Perform expansion or contraction
        if do_expansion:
            expansion_points, dilated_mask, depth_widths, dilated_masks_by_depth = get_expansion_points_for_fake_mask(
                previous_best_mask,
                cxas_mask_np,
                num_points=1,
                kernel_size=32,
                contraction_points=get_all_contraction_points(previous_all_contraction_points)+get_all_expansion_points(previous_all_expansion_points),
                min_distance=min_point_distance,
                depth_level=current_depth_level,
                iterations_per_depth=iterations_per_depth * 2
            )
            
            if len(expansion_points) != sum(depth_widths):
                log_print(f"    Expansion points length mismatch: {len(expansion_points)} != {sum(depth_widths)}, retrying...")
                retry_count += 1
            elif len(expansion_points) > 0:
                all_expansion_points['fake_mask'] = expansion_points
                all_dilated_masks.append(dilated_mask)
                anatomy_depth_width_levels['fake_mask'] = depth_widths
                all_dilated_masks_by_depth['fake_mask'] = dilated_masks_by_depth
                points_collected = True
                log_print(f"    Added {len(expansion_points)} expansion point(s)")
            else:
                log_print(f"    No expansion points found, retrying...")
                retry_count += 1
        else:
            contraction_points, eroded_mask, depth_widths, eroded_masks_by_depth = get_contraction_points_for_fake_mask(
                previous_best_mask,
                cxas_mask_np,
                num_points=1,
                kernel_size=32,
                expansion_points=get_all_expansion_points(previous_all_expansion_points)+get_all_contraction_points(previous_all_contraction_points),
                min_distance=min_point_distance,
                depth_level=current_depth_level,
                iterations_per_depth=iterations_per_depth
            )
            
            if len(contraction_points) != sum(depth_widths):
                log_print(f"    Contraction points length mismatch: {len(contraction_points)} != {sum(depth_widths)}, retrying...")
                retry_count += 1
            elif len(contraction_points) > 0:
                all_contraction_points['fake_mask'] = contraction_points
                all_eroded_masks.append(eroded_mask)
                anatomy_depth_width_levels['fake_mask'] = depth_widths
                all_eroded_masks_by_depth['fake_mask'] = eroded_masks_by_depth
                points_collected = True
                log_print(f"    Added {len(contraction_points)} contraction point(s)")
            else:
                log_print(f"    No contraction points found, retrying...")
                retry_count += 1
    
    if not points_collected:
        log_print(f"    WARNING: Failed to collect points after {max_retries} retries")
        no_points_collected = True
    else:
        log_print(f"    Successfully collected points for fake mask")
        no_points_collected = False
    
    return all_expansion_points, all_contraction_points, all_dilated_masks, all_eroded_masks, anatomy_depth_width_levels, all_dilated_masks_by_depth, all_eroded_masks_by_depth, no_points_collected

def prepare_accumulated_points(mask_component_info, all_expansion_points, all_contraction_points):
    points = mask_component_info['accumulated_points'].copy()
    labels = mask_component_info['accumulated_labels'].copy()
    for pt in get_all_expansion_points(all_expansion_points):
        points.append(pt); labels.append(1)
    for pt in get_all_contraction_points(all_contraction_points):
        points.append(pt); labels.append(0)
    return points, labels


def create_mask_input(mask_component_info, previous_best_mask, all_expansion_points,
                      all_contraction_points, chex_rl, chex_ll, fake=False):
    original_mask_input = mask_component_info['mask_input']
    mask_input_size     = original_mask_input.shape[0]
    scale_factor        = mask_input_size / 1024

    mask_input = original_mask_input.copy()
    resized    = zoom(previous_best_mask, scale_factor, order=1)
    epsilon    = 1e-8
    mask_input[resized > 0] = np.log(1 / (1 - 1 + epsilon))

    combined_lung = np.logical_or(chex_rl, chex_ll)
    mask_input = adjust_mask_input_with_points(
        mask_input,
        pos_points=get_all_expansion_points(all_expansion_points),
        neg_points=get_all_contraction_points(all_contraction_points),
        cxas_mask_np=combined_lung,
        radius=15, neg_strength=2.0, fake=fake
    )
    return mask_input, combined_lung


def predict_and_postprocess_mask(model, processor, inference_state, current_accumulated_points,
                                  current_accumulated_labels, mask_input,
                                  all_expansion_points, all_contraction_points, mask_component_id):
    epsilon         = 1e-8
    mask_input_size = mask_input.shape[0]
    scale_factor    = mask_input_size / 1024.0
    logit_threshold = np.log(0.5 / (0.5 + epsilon))
    logit_mask      = mask_input - logit_threshold > 0

    filtered_points, filtered_labels = [], []
    for pt, label in zip(current_accumulated_points, current_accumulated_labels):
        if label == 1:
            px, py = int(pt[0] * scale_factor), int(pt[1] * scale_factor)
            if 0 <= px < mask_input_size and 0 <= py < mask_input_size and logit_mask[py, px]:
                filtered_points.append(pt); filtered_labels.append(label)
        else:
            filtered_points.append(pt); filtered_labels.append(label)

    if not filtered_points:
        filtered_points, filtered_labels = current_accumulated_points, current_accumulated_labels

    masks, scores, logits = model.predict_inst(
        inference_state,
        point_coords=np.array(filtered_points),
        point_labels=np.array(filtered_labels),
        multimask_output=True,
        mask_input=mask_input[None, :, :]
    )
    best_mask = masks[np.argsort(scores)[::-1][0]]

    best_mask_pil = Image.fromarray((best_mask * 255).astype(np.uint8))
    postprocessed = postprocess_mask_using_sam3(best_mask_pil, model, processor)
    if postprocessed is None:
        postprocessed = best_mask_pil

    best_mask_np = np.array(postprocessed).squeeze() / 255.0

    is_valid, _ = verify_deformation(
        best_mask_np,
        get_all_expansion_points(all_expansion_points),
        get_all_contraction_points(all_contraction_points)
    )
    return best_mask_np, is_valid, logits[0, :, :]

def _select_anatomies(target, overlap_anatomy, geometrical_mask_info):
    if target != 'cardiomegaly':
        return select_anatomies_and_operations(overlap_anatomy, geometrical_mask_info)
    return select_anatomies_and_operations_for_cardiomegaly(overlap_anatomy)


def _collect_points(target, selected_anatomies, anatomy_operations, anatomy_depth_levels,
                    previous_best_mask, region_masks_cache, cxas_masks_cache,
                    chex_rl, chex_ll, min_point_distance, iterations_per_depth):
    if target != 'cardiomegaly':
        return collect_deformation_points(
            selected_anatomies, anatomy_operations, anatomy_depth_levels,
            previous_best_mask, region_masks_cache, chex_rl, chex_ll,
            min_point_distance, iterations_per_depth
        )
    return collect_deformation_points_for_cardiomegaly(
        selected_anatomies, anatomy_operations, anatomy_depth_levels,
        previous_best_mask, cxas_masks_cache, chex_rl, chex_ll,
        min_point_distance, iterations_per_depth
    )


def _record_selected_points(deformation_result, deformation_results, points_dict, operation):
    for class_name, pts in points_dict.items():
        deformation_result['deformation'][class_name] = {'operation': operation, 'points': pts}
        deformation_results['possible_deformation_anatomy'].append(class_name)
        if 'left' in class_name:
            deformation_result['left'] = True
        elif 'right' in class_name:
            deformation_result['right'] = True


def deform_mask(target, image, geometrical_mask_infos, mask_component_infos,
                model, processor, output_path, chex_ll, chex_rl, region_masks_cache, cxas_masks_cache,
                min_point_distance=50, iterations_per_depth=1,
                max_deformation_retries=10, key_id=None, save_visualization=True):
    deformation_results = {
        'possible_deformation_anatomy': [],
        'deformation_results': {}
    }

    for idx, mask_component_info in enumerate(mask_component_infos):
        previous_best_mask    = mask_component_info['best_mask']
        geometrical_mask_info = geometrical_mask_infos[idx]
        overlap_anatomy       = extract_overlap_anatomy(geometrical_mask_info)
        inference_state       = processor.set_image(image)

        for attempt in range(max_deformation_retries):
            selected_anatomies, anatomy_operations, anatomy_depth_levels = _select_anatomies(
                target, overlap_anatomy, geometrical_mask_info
            )
            if not selected_anatomies:
                continue

            (all_expansion_points, all_contraction_points,
             selected_expansion_points, selected_contraction_points,
             all_dilated_masks, all_eroded_masks, _,
             anatomy_depth_width_levels, all_dilated_masks_by_depth,
             all_eroded_masks_by_depth, _) = _collect_points(
                target, selected_anatomies, anatomy_operations, anatomy_depth_levels,
                previous_best_mask, region_masks_cache, cxas_masks_cache,
                chex_rl, chex_ll, min_point_distance, iterations_per_depth
            )
            if not all_expansion_points and not all_contraction_points:
                continue

            current_accumulated_points, current_accumulated_labels = prepare_accumulated_points(
                mask_component_info, selected_expansion_points, selected_contraction_points
            )
            mask_input, _ = create_mask_input(
                mask_component_info, previous_best_mask,
                selected_expansion_points, selected_contraction_points, chex_rl, chex_ll
            )
            best_mask, is_valid, _ = predict_and_postprocess_mask(
                model, processor, inference_state, current_accumulated_points,
                current_accumulated_labels, mask_input,
                selected_expansion_points, selected_contraction_points,
                mask_component_info['mask_component_id']
            )

            if is_valid:
                Image.fromarray((best_mask * 255).astype(np.uint8)).save(
                    os.path.join(output_path, f"suboptimal_component_{mask_component_info['mask_component_id']}.png")
                )
                break
        else:
            continue  # all retries failed — skip this component

        first_anatomy  = selected_anatomies[0]
        first_key      = overlap_anatomy['lung_regions' if target != 'cardiomegaly' else 'zones'][first_anatomy]['key']
        first_dilated  = all_dilated_masks[0] if all_dilated_masks else None
        first_eroded   = all_eroded_masks[0]  if all_eroded_masks  else None
        suboptimal_component_id = f"suboptimal_component_{mask_component_info['mask_component_id']}"

        if save_visualization:
            save_mask_visualization(
                image=image, mask_input=mask_input,
                model_output=best_mask,
                previous_best_mask=previous_best_mask,
                pos_points=get_all_expansion_points(selected_expansion_points),
                neg_points=get_all_contraction_points(selected_contraction_points),
                dilated_mask=first_dilated, eroded_mask=first_eroded,
                save_path=os.path.join(output_path, f"deformation_process_{suboptimal_component_id}.png"),
                all_expansion_points=selected_expansion_points,
                all_contraction_points=selected_contraction_points,
                anatomy_operations=anatomy_operations, anatomy_width_levels={},
                anatomy_depth_levels=anatomy_depth_levels, anatomy_depth_width_levels=anatomy_depth_width_levels,
                all_dilated_masks_by_depth=all_dilated_masks_by_depth,
                all_eroded_masks_by_depth=all_eroded_masks_by_depth,
                key_id=key_id, suboptimal_component_id=suboptimal_component_id
            )

        deformation_result = {'left': False, 'right': False, 'deformation': {}}
        _record_selected_points(deformation_result, deformation_results, selected_expansion_points,  'expansion')
        _record_selected_points(deformation_result, deformation_results, selected_contraction_points, 'contraction')

        deformation_results['deformation_results'][suboptimal_component_id] = {
            "deformation_result":         deformation_result,
            "anatomy_depth_width_levels":  anatomy_depth_width_levels,
            "deformation_success":         True,
            "retry_count":                 attempt,
            "lesion_overlap":              overlap_anatomy,
            "anatomy_operations":          anatomy_operations,
            "anatomy_width_levels":        {},
            "anatomy_depth_levels":        anatomy_depth_levels,
            "all_dilated_masks_by_depth":  all_dilated_masks_by_depth,
            "all_eroded_masks_by_depth":   all_eroded_masks_by_depth,
            "first_key":                   first_key,
            "first_anatomy":               first_anatomy,
            "first_dilated":               first_dilated,
            "first_eroded":                first_eroded,
        }

    return deformation_results

def deform_mask_for_qa(config, target, image, dicom_id, geometrical_mask_infos, mask_component_infos, deformation_results,
                model, processor, chex_ll, chex_rl, output_path, max_deformation_retries=10, key_id=None,
                save_visualization=True):
    """
    QA result generation function.
    """
    
    possible_deformation_anatomy = deformation_results['possible_deformation_anatomy']
    retry_count = 0
    
    all_deformation_success = False
    
    org_deformation_results = {
        'possible_deformation_anatomy': deformation_results['possible_deformation_anatomy'],
        'deformation_results': {},
    }
    
    for suboptimal_component_id, suboptimal_component_result in deformation_results['deformation_results'].items():
        org_deformation_results['deformation_results'][suboptimal_component_id] = {
            'deformation_result': suboptimal_component_result['deformation_result'],
            'anatomy_depth_width_levels': suboptimal_component_result['anatomy_depth_width_levels'],
            'deformation_success': suboptimal_component_result['deformation_success'],
            'retry_count': suboptimal_component_result['retry_count'],
            'lesion_overlap': suboptimal_component_result['lesion_overlap'],
            'anatomy_operations': suboptimal_component_result['anatomy_operations'],
        }
    
    while all_deformation_success == False and retry_count < max_deformation_retries:
        
        qa_results = {
            'org_deformation_results': org_deformation_results,
            'qa_deformation_results': {},
            'geometrical_mask_infos': geometrical_mask_infos,
        }
        
        deformation_number = random.randint(0, 3)# probability tuning, leave for later
        selected_deformation_anatomy = set(random.sample(possible_deformation_anatomy, min(deformation_number, len(possible_deformation_anatomy))))
        deformation_success_list = []
        
        for idx, mask_component_info in enumerate(mask_component_infos):
            
            previous_best_mask = mask_component_info['best_mask']
            suboptimal_component_id = f"suboptimal_component_{mask_component_info['mask_component_id']}"
            
            log_print(f"    [QA] Checking deformation result for {suboptimal_component_id}... ({idx+1}/{len(mask_component_infos)})")
            
            if suboptimal_component_id not in deformation_results['deformation_results']:
                log_print(f"    [QA] No deformation result for {suboptimal_component_id}, skipping (failed in previous step or mask too small)")
                continue
            
            # 2. Set up SAM inference state
            inference_state = processor.set_image(image)

            deformation_success = False

            # Initialize variables (in case the while loop is not entered)
            mask_input = None
            best_mask = None
            all_expansion_points = {}
            all_contraction_points = {}
            
            deformation_result = deformation_results['deformation_results'][suboptimal_component_id]['deformation_result']['deformation']
            target_anatomy = list(selected_deformation_anatomy.intersection(set(deformation_result.keys())))
            
            all_expansion_points = {}
            all_contraction_points = {}
            
            for anatomy in target_anatomy:
                if deformation_result[anatomy]['operation'] == 'expansion':
                    all_expansion_points[anatomy] = deformation_result[anatomy]['points']
                elif deformation_result[anatomy]['operation'] == 'contraction':
                    all_contraction_points[anatomy] = deformation_result[anatomy]['points']
            
            log_print(f"    Preparing accumulated points...")
            current_accumulated_points, current_accumulated_labels = prepare_accumulated_points(
                mask_component_info, all_expansion_points, all_contraction_points
            )
            
            # 6. Create and adjust mask input
            log_print(f"    Creating mask input...")
            mask_input, combined_cxas_mask = create_mask_input(
                mask_component_info, previous_best_mask, all_expansion_points,
                all_contraction_points, chex_rl, chex_ll
            )

            if len(target_anatomy) != 0:

                inference_state = processor.set_image(image)

                # 7. Predict and postprocess
                log_print(f"    Predicting and postprocessing mask...")
                best_mask, is_valid, logits = predict_and_postprocess_mask(
                    model, processor, inference_state, current_accumulated_points,
                    current_accumulated_labels, mask_input,
                    all_expansion_points, all_contraction_points,
                    mask_component_info['mask_component_id']
                )
                
                if is_valid:
                    deformation_success = True
                    best_mask_uint8 = (best_mask * 255).astype(np.uint8)
                    best_mask_pil = Image.fromarray(best_mask_uint8)
                    best_mask_pil.save(os.path.join(output_path, f"qa_{suboptimal_component_id}.png"))
                    log_print(f"    [QA SUCCESS] Deformation verified successfully!")
                else:
                    deformation_success = False
                    log_print(f"    [QA FAILED] Deformation verification failed, retrying...")
            else:
                previous_pil = Image.fromarray(previous_best_mask)
                previous_pil.save(os.path.join(output_path, f"qa_{suboptimal_component_id}.png"))
                deformation_success = True   
                    
            # Skip if selected_anatomies is empty or mask_input is None
            if mask_input is None:
                log_print(f"    [QA ERROR] No valid deformation result, skipping...")
                deformation_success = False

            deformation_success_list.append(deformation_success)

            # 9. Save visualization
            if deformation_success:
                first_key = deformation_results['deformation_results'][suboptimal_component_id]['first_key']
                first_dilated = deformation_results['deformation_results'][suboptimal_component_id]['first_dilated']
                first_eroded = deformation_results['deformation_results'][suboptimal_component_id]['first_eroded']
                    
                if save_visualization:
                    save_mask_visualization(
                        config=config,
                        dicom_id=dicom_id,
                        image=image,
                        mask_input=mask_input,
                        model_output=best_mask if len(target_anatomy) > 0 else previous_best_mask,
                        idx=idx,
                        key=first_key,
                        class_name=f"combined_{len(target_anatomy)}anatomies",
                        added_point=None,
                        previous_best_mask=previous_best_mask,
                        pos_points=get_all_expansion_points(all_expansion_points),
                        neg_points=get_all_contraction_points(all_contraction_points),
                        dilated_mask=first_dilated,
                        eroded_mask=first_eroded,
                        save_path=os.path.join(output_path, f"qa_deformation_process_{suboptimal_component_id}.png"),
                        all_expansion_points=all_expansion_points,
                        all_contraction_points=all_contraction_points,
                        anatomy_operations=deformation_results['deformation_results'][suboptimal_component_id]['anatomy_operations'],
                        anatomy_width_levels=deformation_results['deformation_results'][suboptimal_component_id]['anatomy_width_levels'],
                        anatomy_depth_levels=deformation_results['deformation_results'][suboptimal_component_id]['anatomy_depth_levels'],
                        anatomy_depth_width_levels=deformation_results['deformation_results'][suboptimal_component_id]['anatomy_depth_width_levels'],
                        all_dilated_masks_by_depth=deformation_results['deformation_results'][suboptimal_component_id]['all_dilated_masks_by_depth'],
                        all_eroded_masks_by_depth=deformation_results['deformation_results'][suboptimal_component_id]['all_eroded_masks_by_depth'],
                        key_id=key_id,
                        suboptimal_component_id=suboptimal_component_id
                    )
                
                # 10. Save results
                deformation_result = {
                    'left': False,
                    'right': False,
                    'deformation': {}
                    
                }
                    
                for class_name, pt in all_expansion_points.items():
                    deformation_result['deformation'][class_name] = {
                        'operation': 'expansion',
                        'points': pt
                    }
                    deformation_results['possible_deformation_anatomy'].append(class_name)
                    
                    if 'left' in class_name:
                        deformation_result['left'] = True
                    elif 'right' in class_name:
                        deformation_result['right'] = True
                    
                for class_name, pt in all_contraction_points.items():
                    deformation_result['deformation'][class_name] = {
                        'operation': 'contraction',
                        'points': pt
                    }
                    deformation_results['possible_deformation_anatomy'].append(class_name)
                    
                    if 'left' in class_name:
                        deformation_result['left'] = True
                    elif 'right' in class_name:
                        deformation_result['right'] = True

                qa_results['qa_deformation_results'][suboptimal_component_id] = {
                    "deformation_result": deformation_result,
                    "deformation_success": deformation_success,
                } # include all information needed to build the benchmark
                
            else:
                qa_results['qa_deformation_results'][suboptimal_component_id] = {
                    "deformation_result": None,
                    "deformation_success": False,
                }
            
        if all(deformation_success_list):
            all_deformation_success = True
        else:
            all_deformation_success = False
            
        retry_count += 1
    
    if all_deformation_success:
        if len(deformation_success_list) > 0:
            log_print(f"    [QA SUCCESS] All deformation success after {retry_count} retries (all succeeded)")
        else:
            log_print(f"    [QA FAILED] No deformation success after {retry_count} retries (failed in previous step)")
    else:
        log_print(f"    [QA FAILED] All deformation failed after {retry_count} retries (QA mask deformation failed)")
        
    return qa_results


