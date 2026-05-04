"""SAM3 inference helpers: per-component prediction and iterative mask postprocessing."""

import os

import numpy as np
from PIL import Image

from logger import log_print
from mask_utils import component_filtering, get_component_center_point, pil_to_numpy
from select_point import select_points


def _sam_predict_best(model, inference_state, points, labels, mask_input=None):
    """Run SAM prediction and return (best_mask, best_logit) sorted by score."""
    masks, scores, logits = model.predict_inst(
        inference_state,
        point_coords=np.array(points),
        point_labels=np.array(labels),
        multimask_output=True,
        mask_input=mask_input,
    )
    top = np.argsort(scores)[::-1]
    return masks[top][0], logits[top][0]


def _run_sam_iteratively(mask_component, pos_pts, model, processor, use_iterative=True):
    """Run SAM on one mask component, iteratively adding grid points.

    Returns (best_mask, mask_input, accumulated_points, accumulated_labels) or None.
    """
    mask_np         = pil_to_numpy(mask_component)
    inference_state = processor.set_image(mask_component)

    grid_pts  = [(x, y) for x, y in pos_pts if mask_np[y, x] == 255]
    center_pt = get_component_center_point(mask_np, pos_pts=grid_pts)
    if center_pt is None:
        return None

    accumulated_points = [center_pt]
    accumulated_labels = [1]
    best_mask, mask_input = _sam_predict_best(model, inference_state, accumulated_points, accumulated_labels)

    if use_iterative:
        for pt in [p for p in grid_pts if p != center_pt]:
            accumulated_points.append(pt)
            accumulated_labels.append(1)
            best_mask, mask_input = _sam_predict_best(
                model, inference_state, accumulated_points, accumulated_labels,
                mask_input=mask_input[None, :, :]
            )

    return best_mask, mask_input, accumulated_points, accumulated_labels


def iterative_postprocess_mask(pos_pts, mask, model, processor, output_path, use_iterative=True):
    mask_components = component_filtering(mask, min_size=1000)
    mask_component_infos = []

    for idx, component in enumerate(mask_components):
        result = _run_sam_iteratively(component, pos_pts, model, processor, use_iterative)
        if result is None:
            continue
        best_mask, mask_input, accumulated_points, accumulated_labels = result

        filtered = component_filtering(Image.fromarray((best_mask * 255).astype(np.uint8)), min_size=1000)
        if not filtered:
            continue

        merged = np.zeros(pil_to_numpy(component).shape, dtype=bool)
        for fc in filtered:
            merged |= np.array(fc).astype(bool)
        merged_pil = Image.fromarray((merged.astype(np.uint8) * 255))
        merged_pil.save(os.path.join(output_path, f"optimal_component_{idx}.png"))
        merged_np = np.array(merged_pil)

        mask_component_infos.append({
            'mask_component_id':              str(idx),
            'postprocessed_mask_components':  [merged_np],
            'accumulated_points':             accumulated_points,
            'accumulated_labels':             accumulated_labels,
            'mask_input':                     mask_input,
            'best_mask':                      merged_np,
        })

    if not mask_component_infos:
        return mask_component_infos, None

    refined_mask = np.zeros((1024, 1024), dtype=bool)
    for info in mask_component_infos:
        for comp in info['postprocessed_mask_components']:
            refined_mask |= comp > 0
    refined_mask = (refined_mask.astype(np.uint8) * 255)

    Image.fromarray(refined_mask).save(os.path.join(output_path, "optimal_combined_mask.png"))
    return mask_component_infos, refined_mask


def iterative_postprocess_mask_single(pos_pts, mask, model, processor, use_iterative=True):
    mask_components = component_filtering(mask, min_size=1000)
    filtered_components = []

    for component in mask_components:
        result = _run_sam_iteratively(component, pos_pts, model, processor, use_iterative)
        if result is None:
            return None
        best_mask, *_ = result
        filtered_components.extend(
            component_filtering(Image.fromarray((best_mask * 255).astype(np.uint8)), min_size=1000)
        )

    return filtered_components


def postprocess_mask_using_sam3(mask, model, processor):
    pos_pts, _ = select_points(mask)
    components = iterative_postprocess_mask_single(pos_pts, mask, model, processor, use_iterative=True)

    if components is None or len(components) == 0:
        log_print("    Warning: No mask components found")
        return None

    if len(components) == 1:
        return components[0]

    # OR-merge multiple components
    merged = np.array(components[0])
    for comp in components[1:]:
        merged = np.logical_or(merged > 0, np.array(comp) > 0).astype(np.uint8) * 255
    return Image.fromarray(merged)
