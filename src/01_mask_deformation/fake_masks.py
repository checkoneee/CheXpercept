"""Fake mask candidate generation for QA: dilation/erosion-based contour points and full mask synthesis."""

import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from logger import log_print
from select_point import select_points
from process_mask import (
    collect_deformation_points_for_fake_mask,
    create_mask_input,
    get_all_contraction_points,
    get_all_expansion_points,
    predict_and_postprocess_mask,
    prepare_accumulated_points,
    save_mask_visualization,
)


def generate_fake_masks(
    model,
    processor,
    inference_state,
    logits,
    target,
    chex_ll,
    chex_rl,
    cxas_masks_cache,
    region_masks_cache,
    suboptimal_component_id,
    anatomy,
    best_mask,
    output_path,
    min_point_distance=50,
    iterations_per_depth=2,
    operation='contraction',
    num_option=4,
    fake_mask=False,
    config=None,
    dicom_id=None,
    image=None,
    key_id=None,
    idx=None,
    org_mask_path=None,
):
    save_visualization = (config or {}).get('mask_deformation', {}).get('save_visualization', True)

    best_mask = (best_mask * 255).astype(np.uint8)
    
    fake_masks = []

    overlapped_regions = []
    
    if target != 'cardiomegaly':
        for region_name, region_mask_np in region_masks_cache.items():

            if region_name in ['left medial lung', 'right medial lung', 'right peripheral lung', 'left peripheral lung', 'left lateral lung', 'right lateral lung']:
                continue
            
            overlap = bool(np.any(np.logical_and(best_mask > 0, region_mask_np > 0)))
            overlap_ratio = 0.0
            
            if overlap:
                region_mask_binary = region_mask_np > 0
                region_total_pixels = np.sum(region_mask_binary)
                if region_total_pixels > 0:
                    overlap_pixels = np.sum(np.logical_and(best_mask > 0, region_mask_binary > 0))
                    overlap_ratio = float(overlap_pixels / region_total_pixels)

            if overlap_ratio >= 0.05:
                overlap = True
            else:
                overlap_ratio = 0.0
                overlap = False
            
            if overlap:
                overlapped_regions.append(region_name)
    else:
        for region_name in cxas_masks_cache.keys():

            overlap = bool(np.any(np.logical_and(best_mask > 0, cxas_masks_cache[region_name] > 0)))
            overlap_ratio = 0.0
            
            if overlap:
                region_mask_binary = cxas_masks_cache[region_name] > 0
                region_total_pixels = np.sum(region_mask_binary)
                if region_total_pixels > 0:
                    overlap_pixels = np.sum(np.logical_and(best_mask > 0, region_mask_binary > 0))
                    overlap_ratio = float(overlap_pixels / region_total_pixels)

            if overlap_ratio >= 0.05:
                overlap = True
            else:
                overlap_ratio = 0.0
                overlap = False
            
            if overlap:
                overlapped_regions.append(region_name)
    
    overlapped_regions = list(set(overlapped_regions) - set([anatomy]))

    # Similar anatomies must also be excluded.
    def get_zone_from_anatomy(anatomy):
        """Extract zone info from anatomy (upper zone, mid zone, lung base)."""
        if 'costophrenic angle' in anatomy:
            return 'lung base'
        elif 'upper zone' in anatomy:
            return 'upper zone'
        elif 'mid zone' in anatomy:
            return 'mid zone'
        elif 'lung base' in anatomy:
            return 'lung base'
        return None

    def get_side_from_anatomy(anatomy):
        """Extract side info from anatomy (left, right)."""
        if 'left' in anatomy:
            return 'left'
        elif 'right' in anatomy:
            return 'right'
        return None

    filtered_regions = []

    # zone and side info for anatomy_name
    anatomy_zone = get_zone_from_anatomy(anatomy)
    anatomy_side = get_side_from_anatomy(anatomy)
    # check whether anatomy_name contains 'peripheral'
    anatomy_has_peripheral = 'peripheral' in anatomy
    # check whether anatomy_name contains 'lateral'
    anatomy_has_lateral = 'lateral' in anatomy
    # check whether anatomy_name is a costophrenic angle
    anatomy_is_costophrenic = 'costophrenic angle' in anatomy

    for loc_name in overlapped_regions:
        loc_zone = get_zone_from_anatomy(loc_name)
        loc_side = get_side_from_anatomy(loc_name)

        # Only check for conflicts when zone and side match
        if anatomy_zone and loc_zone and anatomy_zone == loc_zone:
            if anatomy_side and loc_side and anatomy_side == loc_side:
                # Filter out peripheral and lateral pairs in the same zone and side so they don't appear together
                if anatomy_has_peripheral and 'lateral' in loc_name:
                    continue
                if anatomy_has_lateral and 'peripheral' in loc_name:
                    continue

                # Filter so that costophrenic angle and lung base peripheral/lateral don't appear together
                if anatomy_is_costophrenic:
                    if 'peripheral' in loc_name or 'lateral' in loc_name:
                        continue
                if anatomy_has_peripheral or anatomy_has_lateral:
                    if 'costophrenic angle' in loc_name:
                        continue

        filtered_regions.append(loc_name)
    
    for region_name in filtered_regions:
        
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            log_print(f"    Collecting deformation points for {region_name}...")
            
            anatomy_operations = {
                region_name: operation == 'expansion'
            }
            
            if target != 'cardiomegaly':
                anatomy_depth_levels = {
                    region_name: 2
                }
            else:
                anatomy_depth_levels = {
                    region_name: 1
                }
            
            if target != 'cardiomegaly':
                all_expansion_points, all_contraction_points, all_dilated_masks, all_eroded_masks, all_cxas_masks, anatomy_depth_width_levels, all_dilated_masks_by_depth, all_eroded_masks_by_depth, no_points_collected = collect_deformation_points(
                    [region_name], anatomy_operations, 
                    anatomy_depth_levels, best_mask, region_masks_cache, chex_rl, chex_ll, min_point_distance, iterations_per_depth
                )
            else:
                all_expansion_points, all_contraction_points, all_dilated_masks, all_eroded_masks, all_cxas_masks, anatomy_depth_width_levels, all_dilated_masks_by_depth, all_eroded_masks_by_depth, no_points_collected = collect_deformation_points_for_cardiomegaly(
                    [region_name], anatomy_operations, 
                    anatomy_depth_levels, best_mask, cxas_masks_cache, chex_rl, chex_ll, min_point_distance, iterations_per_depth
                )
            
            if no_points_collected:
                break

            # 5. Prepare accumulated points
            log_print(f"    Preparing accumulated points...")

            new_mask_component_info = {
                'mask_component_id': 'test',
                'mask_input': logits,
                'accumulated_points': [],
                'accumulated_labels': [],
            }

            pos_pts, neg_pts = select_points(best_mask)

            for pt in pos_pts:
                x, y = pt
                if best_mask[y, x] == 255:
                    # only add if not the center point
                    new_mask_component_info['accumulated_points'].append(pt)
                    new_mask_component_info['accumulated_labels'].append(1)

            current_accumulated_points, current_accumulated_labels = prepare_accumulated_points(
                new_mask_component_info, all_expansion_points, all_contraction_points
            )

            # 6. Create and adjust mask input
            log_print(f"    Creating mask input...")
            mask_input, combined_cxas_mask = create_mask_input(
                new_mask_component_info, best_mask, all_expansion_points,
                all_contraction_points, chex_rl, chex_ll
            )

            # 7. Predict and postprocess
            log_print(f"    Predicting and postprocessing mask...")
            mask, is_valid, logits = predict_and_postprocess_mask(
                model, processor, inference_state, current_accumulated_points,
                current_accumulated_labels, mask_input,
                all_expansion_points, all_contraction_points,
                new_mask_component_info['mask_component_id']
            )

            if is_valid:
                mask_uint8 = (mask * 255).astype(np.uint8)
                
                mask_pil = Image.fromarray(mask_uint8)
                mask_pil.save(os.path.join(output_path, f"{org_mask_path}_fake_{region_name}_{operation}.png"))

                # Only call when all required visualization arguments are provided
                if config is not None and dicom_id is not None and image is not None:
                    save_path = os.path.join(
                        output_path,
                        f"qa_fake_deformation_{org_mask_path}_{region_name}_{operation}.png",
                    )
                    
                    fake_masks.append({
                        'region_name': region_name,
                        'org_mask': org_mask_path,
                        'fake_mask': f"{org_mask_path}_fake_{region_name}_{operation}",
                    })

                    if save_visualization:
                        save_mask_visualization(
                            config=config,
                            dicom_id=dicom_id,
                            image=image,
                            mask_input=mask_input,
                            model_output=mask,
                            idx=idx if idx is not None else 0,
                            key='fake_revision',
                            class_name=anatomy,
                            added_point=None,
                            previous_best_mask=best_mask,
                            pos_points=get_all_expansion_points(all_expansion_points),
                            neg_points=get_all_contraction_points(all_contraction_points),
                            dilated_mask=None,
                            eroded_mask=None,
                            save_path=save_path,
                            all_expansion_points=all_expansion_points,
                            all_contraction_points=all_contraction_points,
                            anatomy_operations=anatomy_operations,
                            anatomy_width_levels=None,
                            anatomy_depth_levels=anatomy_depth_levels,
                            anatomy_depth_width_levels=anatomy_depth_width_levels,
                            all_dilated_masks_by_depth=all_dilated_masks_by_depth,
                            all_eroded_masks_by_depth=all_eroded_masks_by_depth,
                            key_id=key_id,
                            suboptimal_component_id=suboptimal_component_id,
                        )
                
                break
            else:
                retry_count += 1
    
    return fake_masks

def visualize_fake_points(best_mask, fake_points2, fake_points3, center_point,
                          suboptimal_component_id, org_mask_path, output_path):
    """
    Visualize fake points on top of best_mask and save as an image.

    Args:
        best_mask: best mask (numpy array, range 0~255 or 0~1)
        fake_points2: list of fake points 2 [(x, y), ...]
        fake_points3: list of fake points 3 [(x, y), ...]
        center_point: center point (x, y) tuple or None
        suboptimal_component_id: suboptimal component ID
        org_mask_path: original mask path (used in the filename)
        output_path: output directory
    """
    if len(fake_points2) == 0 and len(fake_points3) == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Show best_mask (grayscale)
    if best_mask.max() > 1.0:
        best_mask_vis = best_mask / 255.0
    else:
        best_mask_vis = best_mask.copy()

    ax.imshow(best_mask_vis, cmap='gray', alpha=0.5)

    # Show fake_points2 (red)
    if len(fake_points2) > 0:
        fake_points2_x = [pt[0] for pt in fake_points2]
        fake_points2_y = [pt[1] for pt in fake_points2]
        ax.scatter(fake_points2_x, fake_points2_y, c='red', s=50, marker='o',
                  label=f'Fake Points 2 ({len(fake_points2)} points)',
                  edgecolors='darkred', linewidths=1.5)

    # Show fake_points3 (blue)
    if len(fake_points3) > 0:
        fake_points3_x = [pt[0] for pt in fake_points3]
        fake_points3_y = [pt[1] for pt in fake_points3]
        ax.scatter(fake_points3_x, fake_points3_y, c='blue', s=50, marker='s',
                  label=f'Fake Points 3 ({len(fake_points3)} points)',
                  edgecolors='darkblue', linewidths=1.5)

    # Show center_point (green)
    if center_point is not None:
        center_x, center_y = center_point
        ax.scatter(center_x, center_y, c='green', s=100, marker='*',
                  label='Center Point', edgecolors='darkgreen', linewidths=2)

    ax.set_title(f'Fake Points Visualization\n{org_mask_path}')
    ax.legend(loc='upper right')
    ax.axis('off')

    # Save image
    debug_save_path = os.path.join(output_path, f"{org_mask_path}_fake_points_debug.png")
    plt.savefig(debug_save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log_print(f"    [Debug] Saved fake points visualization to {debug_save_path}")

def _generate_fake_points_from_dilated_mask(
    previous_best_mask,
    cxas_mask_np,
    center_point,
    depth_level=2,
    iterations_per_depth=2,
    kernel_size=32,
    min_distance_from_center=50,
    previous_points=None,
    min_point_distance=100
):
    """
    Generate fake points on the contour of the dilated previous_best_mask while keeping a distance from the center point.

    Args:
        previous_best_mask: previous best mask (1024x1024 numpy array, range 0~1)
        cxas_mask_np: CXAS mask numpy array (1024x1024, range 0~255)
        center_point: center point (y, x) tuple
        depth_level: depth deformation level
        iterations_per_depth: number of dilation iterations performed each time depth increases by 1
        kernel_size: kernel size used for the dilation operation
        min_distance_from_center: minimum distance from the center point (in pixels)
        previous_points: previously used points [(x, y), ...]
        min_point_distance: minimum distance from previous_points (in pixels)

    Returns:
        fake_points: generated fake points [(x, y), ...]
        dilated_mask: dilated mask (1024x1024 numpy array, range 0~1)
        depth_width_levels: list of width levels selected at each depth (returned empty)
        dilated_masks_by_depth: list of dilated masks per depth [(mask, depth_iter), ...]
    """
    fake_points = []
    dilated_masks_by_depth = []

    # Convert previous_best_mask to binary (threshold 0.5)
    prev_mask_binary = (previous_best_mask > 0.5).astype(np.uint8)
    prev_mask_binary = prev_mask_binary * 255  # convert to 0-255 range

    # Create kernel of size kernel_size x kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Convert cxas_mask_np to binary (threshold 0.5)
    cxas_mask_binary = (cxas_mask_np > 0.5).astype(np.uint8)
    cxas_mask_binary = cxas_mask_binary * 255  # convert to 0-255 range

    # Extract center_point coordinates (y, x) -> convert to (x, y)
    center_x, center_y = center_point

    # Repeat depth_level times, each performing iterations_per_depth dilations (accumulating)
    current_mask = prev_mask_binary.copy()
    for depth_iter in range(1, depth_level + 1):
        dilated_mask = cv2.dilate(current_mask, kernel, iterations=iterations_per_depth)
        current_mask = dilated_mask  # update for the next iteration
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

    if len(y_coords) == 0:
        log_print(f"    [Fake Dilation] No valid contour points found")
        dilated_mask_normalized = (dilated_mask / 255.0).astype(np.float32)
        return fake_points, dilated_mask_normalized, [], dilated_masks_by_depth

    # Convert coordinates to a list (x, y)
    candidate_points = list(zip(x_coords, y_coords))

    # Filter by distance from the center point
    valid_candidates = []
    for candidate_pt in candidate_points:
        cx, cy = candidate_pt
        # Compute distance to center point
        distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)

        if distance_from_center >= min_distance_from_center:
            valid_candidates.append(candidate_pt)

    candidate_points = valid_candidates

    # Filter by distance from previous_points
    if previous_points is not None and len(previous_points) > 0:
        valid_candidates = []
        for candidate_pt in candidate_points:
            cx, cy = candidate_pt
            is_far_enough = True
            
            for prev_pt in previous_points:
                px, py = prev_pt
                distance = np.sqrt((cx - px)**2 + (cy - py)**2)
                
                if distance < min_point_distance:
                    is_far_enough = False
                    break
            
            if is_far_enough:
                valid_candidates.append(candidate_pt)
        
        candidate_points = valid_candidates
    
    if len(candidate_points) == 0:
        log_print(f"    [Fake Dilation] No valid candidates after filtering")
        dilated_mask_normalized = (dilated_mask / 255.0).astype(np.float32)
        return fake_points, dilated_mask_normalized, [], dilated_masks_by_depth

    # Pick at most 5 points (each separated by at least the minimum spacing)
    max_points = 5
    min_spacing = min_point_distance  # minimum distance between points
    selected_points = []

    # Randomly pick the first point
    first_point = random.choice(candidate_points)
    selected_points.append(first_point)
    fake_points.append(first_point)

    # Pick the remaining points (up to 4 more)
    remaining_candidates = [pt for pt in candidate_points if pt != first_point]

    for _ in range(max_points - 1):
        if len(remaining_candidates) == 0:
            break

        valid_candidates = []
        for cand_pt in remaining_candidates:
            cand_x, cand_y = cand_pt
            # Check minimum distance against already-selected points
            min_dist_to_selected = float('inf')
            for sel_pt in selected_points:
                sel_x, sel_y = sel_pt
                dist = np.sqrt((cand_x - sel_x)**2 + (cand_y - sel_y)**2)
                min_dist_to_selected = min(min_dist_to_selected, dist)

            if min_dist_to_selected >= min_spacing:
                valid_candidates.append(cand_pt)

        if len(valid_candidates) == 0:
            break

        # Pick one at random
        next_point = random.choice(valid_candidates)
        selected_points.append(next_point)
        fake_points.append(next_point)
        remaining_candidates.remove(next_point)

    log_print(f"    [Fake Dilation] Selected {len(fake_points)} point(s) from depth {depth_level} contour")

    # Convert dilated_mask to 0-1 range and return
    dilated_mask_normalized = (dilated_mask / 255.0).astype(np.float32)
    
    return fake_points, dilated_mask_normalized, [], dilated_masks_by_depth

def _generate_fake_points_from_eroded_mask(
    previous_best_mask,
    cxas_mask_np,
    center_point,
    depth_level=2,
    iterations_per_depth=2,
    kernel_size=32,
    min_distance_from_center=50,
    previous_points=None,
    min_point_distance=100
):
    """
    Generate fake points on the contour of the eroded previous_best_mask while keeping a distance from the center point.

    Args:
        previous_best_mask: previous best mask (1024x1024 numpy array, range 0~1)
        cxas_mask_np: CXAS mask numpy array (1024x1024, range 0~255)
        center_point: center point (y, x) tuple
        depth_level: depth deformation level
        iterations_per_depth: number of erosion iterations performed each time depth increases by 1
        kernel_size: kernel size used for the erosion operation
        min_distance_from_center: minimum distance from the center point (in pixels)
        previous_points: previously used points [(x, y), ...]
        min_point_distance: minimum distance from previous_points (in pixels)

    Returns:
        fake_points: generated fake points [(x, y), ...]
        eroded_mask: eroded mask (1024x1024 numpy array, range 0~1)
        depth_width_levels: list of width levels selected at each depth (returned empty)
        eroded_masks_by_depth: list of eroded masks per depth [(mask, depth_iter), ...]
    """
    fake_points = []
    eroded_masks_by_depth = []

    # Convert previous_best_mask to binary (threshold 0.5)
    prev_mask_binary = (previous_best_mask > 0.5).astype(np.uint8)
    prev_mask_binary = prev_mask_binary * 255  # convert to 0-255 range

    # Create kernel of size kernel_size x kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Convert cxas_mask_np to binary (threshold 0.5)
    cxas_mask_binary = (cxas_mask_np > 0.5).astype(np.uint8)
    cxas_mask_binary = cxas_mask_binary * 255  # convert to 0-255 range

    # Extract center_point coordinates (y, x) -> convert to (x, y)
    center_x, center_y = center_point

    # Repeat depth_level times, each performing iterations_per_depth erosions (accumulating)
    current_mask = prev_mask_binary.copy()
    for depth_iter in range(1, depth_level + 1):
        eroded_mask = cv2.erode(current_mask, kernel, iterations=iterations_per_depth)
        current_mask = eroded_mask  # update for the next iteration
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

    if len(y_coords) == 0:
        log_print(f"    [Fake Erosion] No valid contour points found")
        eroded_mask_normalized = (eroded_mask / 255.0).astype(np.float32)
        return fake_points, eroded_mask_normalized, [], eroded_masks_by_depth

    # Convert coordinates to a list (x, y)
    candidate_points = list(zip(x_coords, y_coords))

    # Filter by distance from the center point
    valid_candidates = []
    for candidate_pt in candidate_points:
        cx, cy = candidate_pt
        # Compute distance to center point
        distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)

        if distance_from_center >= min_distance_from_center:
            valid_candidates.append(candidate_pt)

    candidate_points = valid_candidates

    # Filter by distance from previous_points
    if previous_points is not None and len(previous_points) > 0:
        valid_candidates = []
        for candidate_pt in candidate_points:
            cx, cy = candidate_pt
            is_far_enough = True
            
            for prev_pt in previous_points:
                px, py = prev_pt
                distance = np.sqrt((cx - px)**2 + (cy - py)**2)
                
                if distance < min_point_distance:
                    is_far_enough = False
                    break
            
            if is_far_enough:
                valid_candidates.append(candidate_pt)
        
        candidate_points = valid_candidates
    
    if len(candidate_points) == 0:
        log_print(f"    [Fake Erosion] No valid candidates after filtering")
        eroded_mask_normalized = (eroded_mask / 255.0).astype(np.float32)
        return fake_points, eroded_mask_normalized, [], eroded_masks_by_depth

    # Pick at most 5 points (each separated by at least the minimum spacing)
    max_points = 5
    min_spacing = min_point_distance  # minimum distance between points
    selected_points = []

    # Randomly pick the first point
    first_point = random.choice(candidate_points)
    selected_points.append(first_point)
    fake_points.append(first_point)

    # Pick the remaining points (up to 4 more)
    remaining_candidates = [pt for pt in candidate_points if pt != first_point]

    for _ in range(max_points - 1):
        if len(remaining_candidates) == 0:
            break

        valid_candidates = []
        for cand_pt in remaining_candidates:
            cand_x, cand_y = cand_pt
            # Check minimum distance against already-selected points
            min_dist_to_selected = float('inf')
            for sel_pt in selected_points:
                sel_x, sel_y = sel_pt
                dist = np.sqrt((cand_x - sel_x)**2 + (cand_y - sel_y)**2)
                min_dist_to_selected = min(min_dist_to_selected, dist)

            if min_dist_to_selected >= min_spacing:
                valid_candidates.append(cand_pt)

        if len(valid_candidates) == 0:
            break

        # Pick one at random
        next_point = random.choice(valid_candidates)
        selected_points.append(next_point)
        fake_points.append(next_point)
        remaining_candidates.remove(next_point)

    log_print(f"    [Fake Erosion] Selected {len(fake_points)} point(s) from depth {depth_level} contour")

    # Convert eroded_mask to 0-1 range and return
    eroded_mask_normalized = (eroded_mask / 255.0).astype(np.float32)
    
    return fake_points, eroded_mask_normalized, [], eroded_masks_by_depth

def generate_fake_masks2(
    model,
    processor,
    inference_state,
    logits,
    target,
    chex_ll,
    chex_rl,
    suboptimal_component_id,
    best_mask,
    output_path,
    previous_all_expansion_points,
    previous_all_contraction_points,
    all_expansion_points_for_fake,
    all_contraction_points_for_fake,
    center_point=None,
    previous_best_mask=None,
    min_point_distance=None,
    iterations_per_depth=2,
    operation='contraction',
    num_option=4,
    config=None,
    dicom_id=None,
    image=None,
    key_id=None,
    idx=None,
    org_mask_path=None,
):
    save_visualization = (config or {}).get('mask_deformation', {}).get('save_visualization', True)

    best_mask = (best_mask * 255).astype(np.uint8)
    
    fake_masks = []
    all_fake_points = []  # store fake points (up to 5)

    max_retries = 10
    max_fake = min(num_option, 5)

    # Accumulate the points used in previous deformations together with newly
    # generated fake points so contour points selected across rounds stay far apart
    prev_exp_points = {k: v.copy() for k, v in previous_all_expansion_points.items()}
    prev_cont_points = {k: v.copy() for k, v in previous_all_contraction_points.items()}

    # Fake points must consider the entire sequence
    fake_expansion_points = {k: v.copy() for k, v in all_expansion_points_for_fake.items()}
    fake_contraction_points = {k: v.copy() for k, v in all_contraction_points_for_fake.items()}

    # Use previous_best_mask to generate fake points (only once, up to 5)
    if center_point is not None and previous_best_mask is not None:
        # Normalize previous_best_mask to 0-1 range (if needed)
        if previous_best_mask.max() > 1.0:
            previous_best_mask_normalized = previous_best_mask / 255.0
        else:
            previous_best_mask_normalized = previous_best_mask.copy()

        # Create cxas_mask
        if target == 'cardiomegaly':
            cxas_mask_np = np.ones_like(previous_best_mask_normalized, dtype=np.uint8) * 255
        else:
            cxas_mask_np = np.logical_or(chex_rl, chex_ll).astype(np.uint8) * 255

        # Generate fake points (if not enough, retry with increased depth_level/iterations_per_depth)
        min_fake_points = 2
        max_fake_points = 5
        initial_depth_level = 2 if target != 'cardiomegaly' else 1
        current_depth_level = initial_depth_level
        current_iterations_per_depth = iterations_per_depth
        max_retry_attempts = 3
        
        fake_points = []
        retry_attempt = 0
        
        while len(all_fake_points) < min_fake_points and retry_attempt < max_retry_attempts:
            if retry_attempt > 0:
                # increase depth_level or iterations_per_depth
                if retry_attempt == 1:
                    current_depth_level += 1
                    log_print(f"    [Fake Points] Retry {retry_attempt}: Increasing depth_level to {current_depth_level}")
                else:
                    current_iterations_per_depth += 1
                    log_print(f"    [Fake Points] Retry {retry_attempt}: Increasing iterations_per_depth to {current_iterations_per_depth}")

            if operation == 'contraction':
                # Expansion: pick fake points from the contour of the dilated previous_best_mask
                fake_points, _, _, _ = _generate_fake_points_from_dilated_mask(
                    previous_best_mask_normalized,
                    cxas_mask_np,
                    center_point,
                    depth_level=current_depth_level,
                    iterations_per_depth=current_iterations_per_depth,
                    kernel_size=32,
                    min_distance_from_center=100,
                    previous_points=get_all_contraction_points(fake_contraction_points),
                    min_point_distance=min_point_distance
                )
                
            elif operation == 'expansion':
                # Contraction: pick fake points from the contour of the eroded previous_best_mask
                fake_points, _, _, _ = _generate_fake_points_from_eroded_mask(
                    previous_best_mask_normalized,
                    cxas_mask_np,
                    center_point,
                    depth_level=current_depth_level,
                    iterations_per_depth=current_iterations_per_depth,
                    kernel_size=32,
                    min_distance_from_center=100,
                    previous_points=get_all_expansion_points(fake_expansion_points),
                    min_point_distance=min_point_distance
                )
            
            # Among the newly generated points, only keep those that stay far enough from existing points
            if len(fake_points) > 0:
                valid_new_points = []
                for new_pt in fake_points:
                    is_valid = True
                    # check distance against existing fake points
                    for existing_pt in all_fake_points:
                        ex, ey = existing_pt
                        nx, ny = new_pt
                        distance = np.sqrt((nx - ex)**2 + (ny - ey)**2)
                        if distance < min_point_distance:
                            is_valid = False
                            break

                    if is_valid:
                        valid_new_points.append(new_pt)

                if len(valid_new_points) > 0:
                    all_fake_points.extend(valid_new_points)
                    log_print(f"    [Fake Points] Generated {len(valid_new_points)} fake points (total: {len(all_fake_points)}, depth_level={current_depth_level}, iterations_per_depth={current_iterations_per_depth})")

            retry_attempt += 1

        # If fewer than 2 fake points were created, generate additional ones from inside chex_mask
        min_fake_points = 2
        if len(all_fake_points) < min_fake_points and center_point is not None:
            log_print(f"    [Fake Points] Only {len(all_fake_points)} points generated, generating additional points from chex_mask...")

            # Find all points inside chex_mask
            chex_mask_np = np.logical_or(chex_rl, chex_ll).astype(np.uint8)
            y_coords, x_coords = np.where(chex_mask_np > 0)

            if len(y_coords) > 0:
                # Build all candidate points
                candidate_points = list(zip(x_coords, y_coords))

                # Filter by distance from center_point (min 100, max 300)
                center_x, center_y = center_point
                max_distance_from_center = 300
                valid_candidates = []
                for candidate_pt in candidate_points:
                    cx, cy = candidate_pt
                    distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)

                    if 100 <= distance_from_center <= max_distance_from_center:  # min_distance_from_center ~ max_distance_from_center
                        valid_candidates.append(candidate_pt)

                # Filter by distance from already-generated fake points
                if len(all_fake_points) > 0:
                    filtered_candidates = []
                    for candidate_pt in valid_candidates:
                        cx, cy = candidate_pt
                        is_far_enough = True

                        for existing_pt in all_fake_points + get_all_expansion_points(fake_expansion_points) + get_all_contraction_points(fake_contraction_points):
                            ex, ey = existing_pt
                            distance = np.sqrt((cx - ex)**2 + (cy - ey)**2)

                            if distance < min_point_distance:
                                is_far_enough = False
                                break

                        if is_far_enough:
                            filtered_candidates.append(candidate_pt)

                    valid_candidates = filtered_candidates

                # Pick as many additional points as needed (up to 5, guaranteeing at least 2)
                num_needed = min(min_fake_points - len(all_fake_points), 5 - len(all_fake_points))

                if len(valid_candidates) > 0 and num_needed > 0:
                    # Pick at random
                    selected_additional = random.sample(valid_candidates, min(num_needed, len(valid_candidates)))
                    all_fake_points.extend(selected_additional)
                    log_print(f"    [Fake Points] Generated {len(selected_additional)} additional fake points from chex_mask (total: {len(all_fake_points)})")
                else:
                    log_print(f"    [Fake Points] WARNING: Could not generate additional points from chex_mask (no valid candidates)")
            else:
                log_print(f"    [Fake Points] WARNING: Could not generate additional points from chex_mask (empty mask)")

    retry_count = 0

    # Set initial values so depth_level and iterations_per_depth can grow dynamically during fake_mask generation
    initial_fake_mask_depth_level = 2 if target != 'cardiomegaly' else 1
    current_fake_mask_depth_level = initial_fake_mask_depth_level
    current_fake_mask_iterations_per_depth = iterations_per_depth
    fake_mask_retry_attempt = 0
    max_fake_mask_retry_attempts = 3

    while retry_count < max_retries and len(fake_masks) < max_fake:

        anatomy_operations = {
            'fake_mask': operation == 'expansion'
        }

        # Set depth_level dynamically
        if fake_mask_retry_attempt > 0:
            current_fake_mask_depth_level += 1
            log_print(f"    [Fake Mask] Retry {fake_mask_retry_attempt}: Increasing depth_level to {current_fake_mask_depth_level}")

        anatomy_depth_levels = {
            'fake_mask': current_fake_mask_depth_level
        }

        # Collect points for the existing fake mask generation (run independently)
        all_expansion_points, all_contraction_points, all_dilated_masks, all_eroded_masks, anatomy_depth_width_levels, all_dilated_masks_by_depth, all_eroded_masks_by_depth, no_points_collected = collect_deformation_points_for_fake_mask(
            best_mask, prev_exp_points, prev_cont_points, min_point_distance, anatomy_operations, anatomy_depth_levels, chex_rl, chex_ll, current_fake_mask_iterations_per_depth
        )

        if no_points_collected:
            # If no points were collected, retry with increased depth_level/iterations_per_depth
            if fake_mask_retry_attempt < max_fake_mask_retry_attempts:
                fake_mask_retry_attempt += 1
                continue
            else:
                break

        # 5. Prepare accumulated points (use only the points collected for fake mask generation)
        log_print(f"    Preparing accumulated points...")
        
        new_mask_component_info = {
            'mask_component_id': 'test',
            'mask_input': logits,
            'accumulated_points': [],
            'accumulated_labels': [],
        }
        pos_pts, neg_pts = select_points(best_mask)
        
        for pt in pos_pts:
            x, y = pt
            if best_mask[y, x] == 255:
                # only add if not the center point
                new_mask_component_info['accumulated_points'].append(pt)
                new_mask_component_info['accumulated_labels'].append(1)

        current_accumulated_points, current_accumulated_labels = prepare_accumulated_points(
            new_mask_component_info, all_expansion_points, all_contraction_points
        )

        # 6. Create and adjust mask input
        log_print(f"    Creating mask input...")
        mask_input, combined_cxas_mask = create_mask_input(
            new_mask_component_info, best_mask, all_expansion_points,
            all_contraction_points, chex_rl, chex_ll, fake=True
        )

        # 7. Predict and postprocess
        log_print(f"    Predicting and postprocessing mask...")
        mask, is_valid, logits = predict_and_postprocess_mask(
            model, processor, inference_state, current_accumulated_points,
            current_accumulated_labels, mask_input,
            all_expansion_points, all_contraction_points,
            new_mask_component_info['mask_component_id']
        )
        
        if is_valid:
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            mask_pil = Image.fromarray(mask_uint8)
            mask_pil.save(os.path.join(output_path, f"{org_mask_path}_fake_{operation}_{len(fake_masks)+1}.png"))
            # Only call when all required visualization arguments are provided
            if config is not None and dicom_id is not None and image is not None:
                save_path = os.path.join(
                    output_path,
                    f"qa_fake_deformation_{org_mask_path}_{operation}_{len(fake_masks)+1}.png",
                )
                
                fake_info = {
                    'org_mask': org_mask_path,
                    'fake_mask': f"{org_mask_path}_fake_{operation}_{len(fake_masks)+1}",
                }
                fake_masks.append(fake_info)
                
                if save_visualization:
                    save_mask_visualization(
                        image=image,
                        mask_input=mask_input,
                        model_output=mask,
                        previous_best_mask=best_mask,
                        pos_points=get_all_expansion_points(all_expansion_points),
                        neg_points=get_all_contraction_points(all_contraction_points),
                        dilated_mask=None,
                        eroded_mask=None,
                        save_path=save_path,
                        all_expansion_points=all_expansion_points,
                        all_contraction_points=all_contraction_points,
                        anatomy_operations=anatomy_operations,
                        anatomy_width_levels=None,
                        anatomy_depth_levels=anatomy_depth_levels,
                        anatomy_depth_width_levels=anatomy_depth_width_levels,
                        all_dilated_masks_by_depth=all_dilated_masks_by_depth,
                        all_eroded_masks_by_depth=all_eroded_masks_by_depth,
                        key_id=key_id,
                        suboptimal_component_id=suboptimal_component_id,
                    )
            
            # Accumulate the fake points generated this round so future rounds avoid them
            if 'fake_mask' in all_expansion_points:
                prev_exp_points.setdefault('fake_mask', [])
                prev_exp_points['fake_mask'].extend(all_expansion_points['fake_mask'])
            if 'fake_mask' in all_contraction_points:
                prev_cont_points.setdefault('fake_mask', [])
                prev_cont_points['fake_mask'].extend(all_contraction_points['fake_mask'])

            # On success, reset the retry counter
            retry_count = 0
            fake_mask_retry_attempt = 0  # also reset the fake_mask retry counter
            current_fake_mask_depth_level = initial_fake_mask_depth_level  # also reset depth_level
            current_fake_mask_iterations_per_depth = iterations_per_depth  # also reset iterations_per_depth
        else:
            retry_count += 1
            # On failure, also try the fake_mask retry path
            if fake_mask_retry_attempt < max_fake_mask_retry_attempts:
                fake_mask_retry_attempt += 1
            else:
                fake_mask_retry_attempt = 0  # reset when max retries reached
                current_fake_mask_depth_level = initial_fake_mask_depth_level
                current_fake_mask_iterations_per_depth = iterations_per_depth
    
    if len(fake_masks) < 3:
        log_print(f"    [Fake Mask] WARNING: Failed to generate enough fake masks (only {len(fake_masks)} generated)")
    if len(all_fake_points) < 2:
        log_print(f"    [Fake Points] WARNING: Failed to generate enough fake points (only {len(all_fake_points)} generated)")
        
    return fake_masks, all_fake_points


