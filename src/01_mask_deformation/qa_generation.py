"""Sequential QA mask deformation: applies recorded expansion/contraction edits one-by-one
and produces fake-mask candidates for each step."""

import os
import random

import numpy as np
from PIL import Image

from process_mask import (
    create_mask_input,
    get_all_contraction_points,
    get_all_expansion_points,
    predict_and_postprocess_mask,
    prepare_accumulated_points,
    save_mask_visualization,
)
from fake_masks import generate_fake_masks2


def _center_of_points(points):
    if not points:
        return None
    xs, ys = zip(*points)
    return (int(round(np.mean(xs))), int(round(np.mean(ys))))


def _apply_qa_step(model, processor, inference_state, mask_component_info,
                   previous_best_mask, all_expansion_points, all_contraction_points,
                   chex_rl, chex_ll, output_path, suboptimal_component_id, step_label):
    """Run one QA deformation step. Returns (best_mask, is_valid, logits, mask_input)."""
    acc_points, acc_labels = prepare_accumulated_points(
        mask_component_info, all_expansion_points, all_contraction_points
    )
    mask_input, _ = create_mask_input(
        mask_component_info, previous_best_mask,
        all_expansion_points, all_contraction_points, chex_rl, chex_ll
    )
    best_mask, is_valid, logits = predict_and_postprocess_mask(
        model, processor, inference_state, acc_points, acc_labels,
        mask_input, all_expansion_points, all_contraction_points,
        mask_component_info['mask_component_id']
    )
    if mask_input is None:
        is_valid = False
    if is_valid:
        Image.fromarray((best_mask * 255).astype(np.uint8)).save(
            os.path.join(output_path, f"{suboptimal_component_id}_{step_label}.png")
        )
    return best_mask, is_valid, logits, mask_input


def _save_qa_visualization(image, mask_input, best_mask,
                            all_expansion_points, all_contraction_points, previous_best_mask_temp,
                            deformation_results, suboptimal_component_id, step_label, output_path, key_id,
                            save_visualization=True):
    if not save_visualization:
        return
    dr = deformation_results['deformation_results'][suboptimal_component_id]
    save_mask_visualization(
        image=image, mask_input=mask_input,
        model_output=best_mask,
        previous_best_mask=previous_best_mask_temp,
        pos_points=get_all_expansion_points(all_expansion_points),
        neg_points=get_all_contraction_points(all_contraction_points),
        dilated_mask=dr['first_dilated'], eroded_mask=dr['first_eroded'],
        save_path=os.path.join(output_path, f"qa_sequential_deformation_process_{suboptimal_component_id}_{step_label}.png"),
        all_expansion_points=all_expansion_points, all_contraction_points=all_contraction_points,
        anatomy_operations=dr['anatomy_operations'], anatomy_width_levels=dr['anatomy_width_levels'],
        anatomy_depth_levels=dr['anatomy_depth_levels'], anatomy_depth_width_levels=dr['anatomy_depth_width_levels'],
        all_dilated_masks_by_depth=dr['all_dilated_masks_by_depth'],
        all_eroded_masks_by_depth=dr['all_eroded_masks_by_depth'],
        key_id=key_id, suboptimal_component_id=suboptimal_component_id
    )


def _generate_fake_masks_pair(model, processor, inference_state, logits, target,
                               chex_ll, chex_rl, cxas_masks_cache, region_masks_cache,
                               suboptimal_component_id, anatomy, best_mask, output_path,
                               all_expansion_points, all_contraction_points,
                               all_expansion_points_for_fake, all_contraction_points_for_fake,
                               center_point, iterations_per_depth, op1, op2,
                               config, dicom_id, image, key_id, idx, step_label):
    """Call generate_fake_masks2 twice with op1/op2. Returns (combined_masks, pts_op1, pts_op2)."""
    common = dict(
        model=model, processor=processor, inference_state=inference_state, logits=logits,
        target=target, chex_ll=chex_ll, chex_rl=chex_rl,
        suboptimal_component_id=suboptimal_component_id,
        best_mask=best_mask, output_path=output_path,
        previous_all_expansion_points=all_expansion_points, previous_all_contraction_points=all_contraction_points,
        all_expansion_points_for_fake=all_expansion_points_for_fake,
        all_contraction_points_for_fake=all_contraction_points_for_fake,
        center_point=center_point, previous_best_mask=best_mask,
        min_point_distance=100, iterations_per_depth=iterations_per_depth,
        num_option=4, config=config, dicom_id=dicom_id,
        image=image, key_id=key_id, idx=idx, org_mask_path=step_label,
    )
    fm1, fp1 = generate_fake_masks2(**common, operation=op1)
    fm2, fp2 = generate_fake_masks2(**common, operation=op2)
    return fm1 + fm2, fp1, fp2


_OP_META = {
    'expansion':   {'prefix': 'exp',  'fake_op1': 'contraction', 'fake_op2': 'expansion',
                    'revision': 'contraction', 'fp_keys': ('fake_points_expansion', 'fake_points_contraction'),
                    'idx_key': 'exp_idx'},
    'contraction': {'prefix': 'cont', 'fake_op1': 'expansion',   'fake_op2': 'contraction',
                    'revision': 'expansion',  'fp_keys': ('fake_points_contraction', 'fake_points_expansion'),
                    'idx_key': 'cont_idx'},
}


def _apply_qa_anatomy_step(operation, anatomy, op_idx, pts_for_anatomy,
                            mask_component_info, previous_best_mask,
                            suboptimal_component_id, deformation_flow,
                            all_expansion_points, all_contraction_points,
                            all_expansion_points_for_fake, all_contraction_points_for_fake,
                            model, processor, inference_state, image, target, dicom_id, key_id,
                            chex_ll, chex_rl, cxas_masks_cache, region_masks_cache,
                            output_path, idx, iterations_per_depth, save_visualization,
                            deformation_results, config):
    """Apply one expansion or contraction QA step. Returns (is_valid, entry_or_None, new_deformation_flow)."""
    meta = _OP_META[operation]
    if operation == 'expansion':
        all_expansion_points[anatomy] = pts_for_anatomy
    else:
        all_contraction_points[anatomy] = pts_for_anatomy

    step_label   = f"{meta['prefix']}{op_idx + 1}"
    center_point = _center_of_points(pts_for_anatomy)

    best_mask, is_valid, logits, mask_input = _apply_qa_step(
        model, processor, inference_state, mask_component_info,
        previous_best_mask, all_expansion_points, all_contraction_points,
        chex_rl, chex_ll, output_path, suboptimal_component_id, step_label
    )
    mask_component_info['mask_input'] = logits

    if not is_valid:
        return False, None, deformation_flow

    _save_qa_visualization(
        image, mask_input, best_mask,
        all_expansion_points, all_contraction_points, previous_best_mask,
        deformation_results, suboptimal_component_id, step_label, output_path, key_id,
        save_visualization=save_visualization
    )
    fake_masks, fp1, fp2 = _generate_fake_masks_pair(
        model, processor, inference_state, logits, target,
        chex_ll, chex_rl, cxas_masks_cache, region_masks_cache,
        suboptimal_component_id, anatomy, best_mask, output_path,
        all_expansion_points, all_contraction_points,
        all_expansion_points_for_fake, all_contraction_points_for_fake,
        center_point, iterations_per_depth,
        op1=meta['fake_op1'], op2=meta['fake_op2'],
        config=config, dicom_id=dicom_id, image=image, key_id=key_id,
        idx=idx, step_label=f"{suboptimal_component_id}_{step_label}"
    )

    new_flow = f"{suboptimal_component_id}_{step_label}"
    fp_key1, fp_key2 = meta['fp_keys']
    entry = {
        'suboptimal_component_id': suboptimal_component_id,
        meta['idx_key']: op_idx + 1,
        'anatomy':       anatomy,
        'points':        pts_for_anatomy,
        'mask_path':     new_flow,
        'center_point':  center_point,
        'fake_masks':    fake_masks,
        fp_key1:         fp1,
        fp_key2:         fp2,
        'revision':      meta['revision'],
        'revision_flow': {
            'before revision': new_flow,
            'after revision':  deformation_flow,
        },
    }
    return True, entry, new_flow


def deform_mask_for_qa_sequential(config, target, image, dicom_id, geometrical_mask_infos, mask_component_infos, deformation_results,
                model, processor, chex_ll, chex_rl, region_masks_cache, cxas_masks_cache, output_path,
                max_deformation_retries=10, key_id=None, iterations_per_depth=1, min_point_distance=50,
                save_visualization=True):
    possible_deformation_anatomy = deformation_results['possible_deformation_anatomy']

    _keep_keys = ('deformation_result', 'anatomy_depth_width_levels', 'deformation_success',
                  'retry_count', 'lesion_overlap', 'anatomy_operations')
    org_deformation_results = {
        'possible_deformation_anatomy': possible_deformation_anatomy,
        'deformation_results': {
            sid: {k: v for k, v in res.items() if k in _keep_keys}
            for sid, res in deformation_results['deformation_results'].items()
        }
    }

    qa_results = {
        'org_deformation_results': org_deformation_results,
        'qa_deformation_results':  None,
        'geometrical_mask_infos':  geometrical_mask_infos,
        'qa_deformation_success':  False,
    }

    for retry_count in range(max_deformation_retries):
        selected_deformation_anatomy = set(random.sample(
            possible_deformation_anatomy, min(random.randint(1, 3), len(possible_deformation_anatomy))
        ))
        deformation_success_list     = []
        deformation_sequence_results = {'expansion': {}, 'contraction': {}, 'no_deformation': {}}
        skipped_components = 0

        for idx, mask_component_info in enumerate(mask_component_infos):
            previous_best_mask      = mask_component_info['best_mask']
            suboptimal_component_id = f"suboptimal_component_{mask_component_info['mask_component_id']}"
            deformation_flow        = f"optimal_component_{mask_component_info['mask_component_id']}"

            if suboptimal_component_id not in deformation_results['deformation_results']:
                skipped_components += 1
                continue

            deformation = deformation_results['deformation_results'][suboptimal_component_id]['deformation_result']['deformation']
            shuffled    = list(deformation.keys())
            random.shuffle(shuffled)
            op_pts = {
                'expansion':   {a: deformation[a]['points'] for a in shuffled if deformation[a]['operation'] == 'expansion'},
                'contraction': {a: deformation[a]['points'] for a in shuffled if deformation[a]['operation'] == 'contraction'},
            }

            inference_state = processor.set_image(image)
            all_expansion_points   = {}
            all_contraction_points = {}
            all_expansion_points_for_fake   = {a: op_pts['expansion'][a]   for a in selected_deformation_anatomy if a in op_pts['expansion']}
            all_contraction_points_for_fake = {a: op_pts['contraction'][a] for a in selected_deformation_anatomy if a in op_pts['contraction']}

            targets_per_op = {
                op: list(selected_deformation_anatomy & set(pts)) for op, pts in op_pts.items()
            }

            for operation in ('expansion', 'contraction'):
                targets = targets_per_op[operation]
                if not targets:
                    deformation_success_list.append(True)
                    Image.fromarray(previous_best_mask).save(
                        os.path.join(output_path, f"{suboptimal_component_id}_no_{operation}.png")
                    )
                    continue

                for op_idx, anatomy in enumerate(targets):
                    is_valid, entry, deformation_flow = _apply_qa_anatomy_step(
                        operation, anatomy, op_idx, op_pts[operation][anatomy],
                        mask_component_info, previous_best_mask,
                        suboptimal_component_id, deformation_flow,
                        all_expansion_points, all_contraction_points,
                        all_expansion_points_for_fake, all_contraction_points_for_fake,
                        model, processor, inference_state, image, target, dicom_id, key_id,
                        chex_ll, chex_rl, cxas_masks_cache, region_masks_cache,
                        output_path, idx, iterations_per_depth, save_visualization,
                        deformation_results, config,
                    )
                    deformation_success_list.append(is_valid)
                    if is_valid:
                        deformation_sequence_results[operation].setdefault(suboptimal_component_id, []).append(entry)
                        selected_deformation_anatomy.discard(anatomy)

            if not targets_per_op['expansion'] and not targets_per_op['contraction']:
                deformation_sequence_results['no_deformation'].setdefault(suboptimal_component_id, []).append({
                    'suboptimal_component_id': suboptimal_component_id,
                    'mask_path': f"{suboptimal_component_id}_no_contraction",
                })

        qa_results['qa_deformation_results'] = deformation_sequence_results
        if deformation_success_list and all(deformation_success_list):
            qa_results['qa_deformation_success'] = (skipped_components == 0)
            break

    return qa_results


def generate_parallel_qa(simple_deformation_results):
    return simple_deformation_results['deformation_results']
