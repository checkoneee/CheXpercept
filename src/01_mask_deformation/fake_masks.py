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
    
    # 비슷한 anatomy도 빼야 한다.
    def get_zone_from_anatomy(anatomy):
        """anatomy에서 zone 정보 추출 (upper zone, mid zone, lung base)"""
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
        """anatomy에서 side 정보 추출 (left, right)"""
        if 'left' in anatomy:
            return 'left'
        elif 'right' in anatomy:
            return 'right'
        return None
    
    filtered_regions = []
    
    # anatomy_name의 zone과 side 정보
    anatomy_zone = get_zone_from_anatomy(anatomy)
    anatomy_side = get_side_from_anatomy(anatomy)
    # anatomy_name이 peripheral을 포함하는지 확인
    anatomy_has_peripheral = 'peripheral' in anatomy
    # anatomy_name이 lateral을 포함하는지 확인
    anatomy_has_lateral = 'lateral' in anatomy
    # anatomy_name이 costophrenic angle인지 확인
    anatomy_is_costophrenic = 'costophrenic angle' in anatomy
    
    for loc_name in overlapped_regions:
        loc_zone = get_zone_from_anatomy(loc_name)
        loc_side = get_side_from_anatomy(loc_name)
        
        # 같은 zone이고 같은 side일 때만 충돌 체크
        if anatomy_zone and loc_zone and anatomy_zone == loc_zone:
            if anatomy_side and loc_side and anatomy_side == loc_side:
                # peripheral과 lateral이 같은 zone, 같은 side에 있으면 함께 나오지 않도록 필터링
                if anatomy_has_peripheral and 'lateral' in loc_name:
                    continue
                if anatomy_has_lateral and 'peripheral' in loc_name:
                    continue
                
                # costophrenic angle과 lung base의 peripheral/lateral이 함께 나오지 않도록 필터링
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
            
            # 5. Accumulated points 준비
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
                    # center point가 아닌 경우만 추가
                    new_mask_component_info['accumulated_points'].append(pt)
                    new_mask_component_info['accumulated_labels'].append(1)
                    
            current_accumulated_points, current_accumulated_labels = prepare_accumulated_points(
                new_mask_component_info, all_expansion_points, all_contraction_points
            )
            
            # 6. Mask input 생성 및 조정
            log_print(f"    Creating mask input...")
            mask_input, combined_cxas_mask = create_mask_input(
                new_mask_component_info, best_mask, all_expansion_points, 
                all_contraction_points, chex_rl, chex_ll
            )
            
            # 7. 예측 및 후처리
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

                # 시각화를 위한 필수 인자가 모두 있을 때만 호출
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
    Fake points를 best_mask 위에 시각화하여 이미지로 저장
    
    Args:
        best_mask: best mask (numpy array, 0~255 또는 0~1 범위)
        fake_points2: fake points 2 리스트 [(x, y), ...]
        fake_points3: fake points 3 리스트 [(x, y), ...]
        center_point: 중심점 (x, y) 튜플 또는 None
        suboptimal_component_id: suboptimal component ID
        org_mask_path: 원본 mask path (파일명에 사용)
        output_path: 출력 경로
    """
    if len(fake_points2) == 0 and len(fake_points3) == 0:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # best_mask 표시 (grayscale)
    if best_mask.max() > 1.0:
        best_mask_vis = best_mask / 255.0
    else:
        best_mask_vis = best_mask.copy()
    
    ax.imshow(best_mask_vis, cmap='gray', alpha=0.5)
    
    # fake_points2 표시 (빨간색)
    if len(fake_points2) > 0:
        fake_points2_x = [pt[0] for pt in fake_points2]
        fake_points2_y = [pt[1] for pt in fake_points2]
        ax.scatter(fake_points2_x, fake_points2_y, c='red', s=50, marker='o', 
                  label=f'Fake Points 2 ({len(fake_points2)} points)', 
                  edgecolors='darkred', linewidths=1.5)
    
    # fake_points3 표시 (파란색)
    if len(fake_points3) > 0:
        fake_points3_x = [pt[0] for pt in fake_points3]
        fake_points3_y = [pt[1] for pt in fake_points3]
        ax.scatter(fake_points3_x, fake_points3_y, c='blue', s=50, marker='s', 
                  label=f'Fake Points 3 ({len(fake_points3)} points)', 
                  edgecolors='darkblue', linewidths=1.5)
    
    # center_point 표시 (녹색)
    if center_point is not None:
        center_x, center_y = center_point
        ax.scatter(center_x, center_y, c='green', s=100, marker='*', 
                  label='Center Point', edgecolors='darkgreen', linewidths=2)
    
    ax.set_title(f'Fake Points Visualization\n{org_mask_path}')
    ax.legend(loc='upper right')
    ax.axis('off')
    
    # 이미지 저장
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
    previous_best_mask를 dilation 시킨 mask의 contour에서 center point에서 거리를 유지하면서 fake point 생성
    
    Args:
        previous_best_mask: 이전 best mask (1024x1024 numpy array, 0~1 범위)
        cxas_mask_np: CXAS mask numpy array (1024x1024, 0~255 범위)
        center_point: 중심점 (y, x) 튜플
        depth_level: Depth 변형 레벨
        iterations_per_depth: depth가 1 증가할 때마다 수행할 dilation iteration 횟수
        kernel_size: 팽창 연산에 사용할 커널 크기
        min_distance_from_center: center point와의 최소 거리 (픽셀 단위)
        previous_points: 이전에 사용된 포인트들 [(x, y), ...]
        min_point_distance: previous_points와의 최소 거리 (픽셀 단위)
    
    Returns:
        fake_points: 생성된 fake point들 [(x, y), ...]
        dilated_mask: 팽창된 마스크 (1024x1024 numpy array, 0~1 범위)
        depth_width_levels: 각 depth에서 선택된 width level 리스트 (빈 리스트 반환)
        dilated_masks_by_depth: 각 depth별 dilated mask 리스트 [(mask, depth_iter), ...]
    """
    fake_points = []
    dilated_masks_by_depth = []
    
    # previous_best_mask를 binary로 변환 (threshold 0.5)
    prev_mask_binary = (previous_best_mask > 0.5).astype(np.uint8)
    prev_mask_binary = prev_mask_binary * 255  # 0-255 범위로 변환
    
    # kernel_size x kernel_size 크기의 커널 생성
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # cxas_mask_np를 binary로 변환 (threshold 0.5)
    cxas_mask_binary = (cxas_mask_np > 0.5).astype(np.uint8)
    cxas_mask_binary = cxas_mask_binary * 255  # 0-255 범위로 변환
    
    # center_point 좌표 추출 (y, x) -> (x, y)로 변환
    center_x, center_y = center_point
    
    # depth_level만큼 반복하면서 각각 iterations_per_depth만큼 dilation 수행 (누적)
    current_mask = prev_mask_binary.copy()
    for depth_iter in range(1, depth_level + 1):
        dilated_mask = cv2.dilate(current_mask, kernel, iterations=iterations_per_depth)
        current_mask = dilated_mask  # 다음 iteration을 위해 업데이트
        # depth별 mask 저장 (0-1 범위로 정규화)
        dilated_masks_by_depth.append((dilated_mask / 255.0, depth_iter))
    
    # dilated_mask의 contour(외곽선) 추출
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # contour를 binary mask로 변환 (외곽선만 그리기)
    contour_mask = np.zeros_like(dilated_mask)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=1)
    
    # contour 중에서 cxas_mask와 겹치는 부분만 추출
    valid_contour = cv2.bitwise_and(contour_mask, cxas_mask_binary)
    
    # valid_contour에서 점들의 좌표 추출
    y_coords, x_coords = np.where(valid_contour > 0)
    
    if len(y_coords) == 0:
        log_print(f"    [Fake Dilation] No valid contour points found")
        dilated_mask_normalized = (dilated_mask / 255.0).astype(np.float32)
        return fake_points, dilated_mask_normalized, [], dilated_masks_by_depth
    
    # 좌표들을 리스트로 변환 (x, y)
    candidate_points = list(zip(x_coords, y_coords))
    
    # center point에서 거리 필터링
    valid_candidates = []
    for candidate_pt in candidate_points:
        cx, cy = candidate_pt
        # center point와의 거리 계산
        distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        
        if distance_from_center >= min_distance_from_center:
            valid_candidates.append(candidate_pt)
    
    candidate_points = valid_candidates
    
    # previous_points와의 거리 필터링
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
    
    # 최대 5개까지 점 선택 (서로 일정 거리 이상 떨어지도록)
    max_points = 5
    min_spacing = min_point_distance  # 점들 사이의 최소 거리
    selected_points = []
    
    # 첫 번째 점 랜덤 선택
    first_point = random.choice(candidate_points)
    selected_points.append(first_point)
    fake_points.append(first_point)
    
    # 나머지 점들 선택 (최대 4개 더)
    remaining_candidates = [pt for pt in candidate_points if pt != first_point]
    
    for _ in range(max_points - 1):
        if len(remaining_candidates) == 0:
            break
        
        valid_candidates = []
        for cand_pt in remaining_candidates:
            cand_x, cand_y = cand_pt
            # 이미 선택된 점들과의 최소 거리 확인
            min_dist_to_selected = float('inf')
            for sel_pt in selected_points:
                sel_x, sel_y = sel_pt
                dist = np.sqrt((cand_x - sel_x)**2 + (cand_y - sel_y)**2)
                min_dist_to_selected = min(min_dist_to_selected, dist)
            
            if min_dist_to_selected >= min_spacing:
                valid_candidates.append(cand_pt)
        
        if len(valid_candidates) == 0:
            break
        
        # 랜덤으로 하나 선택
        next_point = random.choice(valid_candidates)
        selected_points.append(next_point)
        fake_points.append(next_point)
        remaining_candidates.remove(next_point)
    
    log_print(f"    [Fake Dilation] Selected {len(fake_points)} point(s) from depth {depth_level} contour")
    
    # dilated_mask를 0-1 범위로 변환하여 반환
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
    previous_best_mask를 erosion 시킨 mask의 contour에서 center point에서 거리를 유지하면서 fake point 생성
    
    Args:
        previous_best_mask: 이전 best mask (1024x1024 numpy array, 0~1 범위)
        cxas_mask_np: CXAS mask numpy array (1024x1024, 0~255 범위)
        center_point: 중심점 (y, x) 튜플
        depth_level: Depth 변형 레벨
        iterations_per_depth: depth가 1 증가할 때마다 수행할 erosion iteration 횟수
        kernel_size: 침식 연산에 사용할 커널 크기
        min_distance_from_center: center point와의 최소 거리 (픽셀 단위)
        previous_points: 이전에 사용된 포인트들 [(x, y), ...]
        min_point_distance: previous_points와의 최소 거리 (픽셀 단위)
    
    Returns:
        fake_points: 생성된 fake point들 [(x, y), ...]
        eroded_mask: 침식된 마스크 (1024x1024 numpy array, 0~1 범위)
        depth_width_levels: 각 depth에서 선택된 width level 리스트 (빈 리스트 반환)
        eroded_masks_by_depth: 각 depth별 eroded mask 리스트 [(mask, depth_iter), ...]
    """
    fake_points = []
    eroded_masks_by_depth = []
    
    # previous_best_mask를 binary로 변환 (threshold 0.5)
    prev_mask_binary = (previous_best_mask > 0.5).astype(np.uint8)
    prev_mask_binary = prev_mask_binary * 255  # 0-255 범위로 변환
    
    # kernel_size x kernel_size 크기의 커널 생성
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # cxas_mask_np를 binary로 변환 (threshold 0.5)
    cxas_mask_binary = (cxas_mask_np > 0.5).astype(np.uint8)
    cxas_mask_binary = cxas_mask_binary * 255  # 0-255 범위로 변환
    
    # center_point 좌표 추출 (y, x) -> (x, y)로 변환
    center_x, center_y = center_point
    
    # depth_level만큼 반복하면서 각각 iterations_per_depth만큼 erosion 수행 (누적)
    current_mask = prev_mask_binary.copy()
    for depth_iter in range(1, depth_level + 1):
        eroded_mask = cv2.erode(current_mask, kernel, iterations=iterations_per_depth)
        current_mask = eroded_mask  # 다음 iteration을 위해 업데이트
        # depth별 mask 저장 (0-1 범위로 정규화)
        eroded_masks_by_depth.append((eroded_mask / 255.0, depth_iter))
    
    # eroded_mask의 contour(외곽선) 추출
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # contour를 binary mask로 변환 (외곽선만 그리기)
    contour_mask = np.zeros_like(eroded_mask)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=1)
    
    # contour 중에서 cxas_mask와 겹치는 부분만 추출
    valid_contour = cv2.bitwise_and(contour_mask, cxas_mask_binary)
    
    # valid_contour에서 점들의 좌표 추출
    y_coords, x_coords = np.where(valid_contour > 0)
    
    if len(y_coords) == 0:
        log_print(f"    [Fake Erosion] No valid contour points found")
        eroded_mask_normalized = (eroded_mask / 255.0).astype(np.float32)
        return fake_points, eroded_mask_normalized, [], eroded_masks_by_depth
    
    # 좌표들을 리스트로 변환 (x, y)
    candidate_points = list(zip(x_coords, y_coords))
    
    # center point에서 거리 필터링
    valid_candidates = []
    for candidate_pt in candidate_points:
        cx, cy = candidate_pt
        # center point와의 거리 계산
        distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        
        if distance_from_center >= min_distance_from_center:
            valid_candidates.append(candidate_pt)
    
    candidate_points = valid_candidates
    
    # previous_points와의 거리 필터링
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
    
    # 최대 5개까지 점 선택 (서로 일정 거리 이상 떨어지도록)
    max_points = 5
    min_spacing = min_point_distance  # 점들 사이의 최소 거리
    selected_points = []
    
    # 첫 번째 점 랜덤 선택
    first_point = random.choice(candidate_points)
    selected_points.append(first_point)
    fake_points.append(first_point)
    
    # 나머지 점들 선택 (최대 4개 더)
    remaining_candidates = [pt for pt in candidate_points if pt != first_point]
    
    for _ in range(max_points - 1):
        if len(remaining_candidates) == 0:
            break
        
        valid_candidates = []
        for cand_pt in remaining_candidates:
            cand_x, cand_y = cand_pt
            # 이미 선택된 점들과의 최소 거리 확인
            min_dist_to_selected = float('inf')
            for sel_pt in selected_points:
                sel_x, sel_y = sel_pt
                dist = np.sqrt((cand_x - sel_x)**2 + (cand_y - sel_y)**2)
                min_dist_to_selected = min(min_dist_to_selected, dist)
            
            if min_dist_to_selected >= min_spacing:
                valid_candidates.append(cand_pt)
        
        if len(valid_candidates) == 0:
            break
        
        # 랜덤으로 하나 선택
        next_point = random.choice(valid_candidates)
        selected_points.append(next_point)
        fake_points.append(next_point)
        remaining_candidates.remove(next_point)
    
    log_print(f"    [Fake Erosion] Selected {len(fake_points)} point(s) from depth {depth_level} contour")
    
    # eroded_mask를 0-1 범위로 변환하여 반환
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
    all_fake_points = []  # fake points 저장 (최대 5개)
    
    max_retries = 10
    max_fake = min(num_option, 5)
    
    # 이전 deformation에서 사용된 포인트 + 새 fake 포인트들을 점점 누적해서
    # 서로 멀리 떨어진 contour 포인트가 선택되도록 함
    prev_exp_points = {k: v.copy() for k, v in previous_all_expansion_points.items()}
    prev_cont_points = {k: v.copy() for k, v in previous_all_contraction_points.items()}
    
    # fake points는 모든 sequnece를 다 고려해야함
    fake_expansion_points = {k: v.copy() for k, v in all_expansion_points_for_fake.items()}
    fake_contraction_points = {k: v.copy() for k, v in all_contraction_points_for_fake.items()}
    
    # previous_best_mask를 사용해서 fake point 생성 (한 번만 수행, 최대 5개)
    if center_point is not None and previous_best_mask is not None:
        # previous_best_mask를 0-1 범위로 정규화 (필요한 경우)
        if previous_best_mask.max() > 1.0:
            previous_best_mask_normalized = previous_best_mask / 255.0
        else:
            previous_best_mask_normalized = previous_best_mask.copy()
        
        # cxas_mask 생성
        if target == 'cardiomegaly':
            cxas_mask_np = np.ones_like(previous_best_mask_normalized, dtype=np.uint8) * 255
        else:
            cxas_mask_np = np.logical_or(chex_rl, chex_ll).astype(np.uint8) * 255
        
        # Fake point 생성 (부족하면 depth_level/iterations_per_depth 증가하여 재시도)
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
                # depth_level 증가 또는 iterations_per_depth 증가
                if retry_attempt == 1:
                    current_depth_level += 1
                    log_print(f"    [Fake Points] Retry {retry_attempt}: Increasing depth_level to {current_depth_level}")
                else:
                    current_iterations_per_depth += 1
                    log_print(f"    [Fake Points] Retry {retry_attempt}: Increasing iterations_per_depth to {current_iterations_per_depth}")
            
            if operation == 'contraction':
                # Expansion: previous_best_mask를 dilation 시킨 mask의 contour에서 fake point 뽑기
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
                # Contraction: previous_best_mask를 erosion 시킨 mask의 contour에서 fake point 뽑기
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
            
            # 새로 생성된 점들 중에서 기존 점들과 거리를 유지하는 점만 추가
            if len(fake_points) > 0:
                valid_new_points = []
                for new_pt in fake_points:
                    is_valid = True
                    # 기존 fake points와의 거리 확인
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
        
        # Fake point가 2개 미만이면 chex_mask 내부에서 추가로 생성
        min_fake_points = 2
        if len(all_fake_points) < min_fake_points and center_point is not None:
            log_print(f"    [Fake Points] Only {len(all_fake_points)} points generated, generating additional points from chex_mask...")
            
            # chex_mask 내부의 모든 점 찾기
            chex_mask_np = np.logical_or(chex_rl, chex_ll).astype(np.uint8)
            y_coords, x_coords = np.where(chex_mask_np > 0)
            
            if len(y_coords) > 0:
                # 모든 candidate points 생성
                candidate_points = list(zip(x_coords, y_coords))
                
                # center_point와의 거리 필터링 (최소 100, 최대 300)
                center_x, center_y = center_point
                max_distance_from_center = 300
                valid_candidates = []
                for candidate_pt in candidate_points:
                    cx, cy = candidate_pt
                    distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                    
                    if 100 <= distance_from_center <= max_distance_from_center:  # min_distance_from_center ~ max_distance_from_center
                        valid_candidates.append(candidate_pt)
                
                # 이미 생성된 fake points와의 거리 필터링
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
                
                # 필요한 만큼 추가 점 선택 (최대 5개까지, 최소 2개는 보장)
                num_needed = min(min_fake_points - len(all_fake_points), 5 - len(all_fake_points))
                
                if len(valid_candidates) > 0 and num_needed > 0:
                    # 랜덤으로 선택
                    selected_additional = random.sample(valid_candidates, min(num_needed, len(valid_candidates)))
                    all_fake_points.extend(selected_additional)
                    log_print(f"    [Fake Points] Generated {len(selected_additional)} additional fake points from chex_mask (total: {len(all_fake_points)})")
                else:
                    log_print(f"    [Fake Points] WARNING: Could not generate additional points from chex_mask (no valid candidates)")
            else:
                log_print(f"    [Fake Points] WARNING: Could not generate additional points from chex_mask (empty mask)")
    
    retry_count = 0
    
    # fake_mask 생성 시 depth_level과 iterations_per_depth를 동적으로 증가시킬 수 있도록 초기값 설정
    initial_fake_mask_depth_level = 2 if target != 'cardiomegaly' else 1
    current_fake_mask_depth_level = initial_fake_mask_depth_level
    current_fake_mask_iterations_per_depth = iterations_per_depth
    fake_mask_retry_attempt = 0
    max_fake_mask_retry_attempts = 3
    
    while retry_count < max_retries and len(fake_masks) < max_fake:
        
        anatomy_operations = {
            'fake_mask': operation == 'expansion'
        }
    
        # depth_level을 동적으로 설정
        if fake_mask_retry_attempt > 0:
            current_fake_mask_depth_level += 1
            log_print(f"    [Fake Mask] Retry {fake_mask_retry_attempt}: Increasing depth_level to {current_fake_mask_depth_level}")
        
        anatomy_depth_levels = {
            'fake_mask': current_fake_mask_depth_level
        }

        # 기존 fake mask 생성용 points 수집 (독립적으로 수행)
        all_expansion_points, all_contraction_points, all_dilated_masks, all_eroded_masks, anatomy_depth_width_levels, all_dilated_masks_by_depth, all_eroded_masks_by_depth, no_points_collected = collect_deformation_points_for_fake_mask(
            best_mask, prev_exp_points, prev_cont_points, min_point_distance, anatomy_operations, anatomy_depth_levels, chex_rl, chex_ll, current_fake_mask_iterations_per_depth
        )
  
        if no_points_collected:
            # points가 수집되지 않았으면 depth_level/iterations_per_depth 증가하여 재시도
            if fake_mask_retry_attempt < max_fake_mask_retry_attempts:
                fake_mask_retry_attempt += 1
                continue
            else:
                break
        
        # 5. Accumulated points 준비 (기존 fake mask 생성용 points만 사용)
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
                # center point가 아닌 경우만 추가
                new_mask_component_info['accumulated_points'].append(pt)
                new_mask_component_info['accumulated_labels'].append(1)
                
        current_accumulated_points, current_accumulated_labels = prepare_accumulated_points(
            new_mask_component_info, all_expansion_points, all_contraction_points
        )
        
        # 6. Mask input 생성 및 조정
        log_print(f"    Creating mask input...")
        mask_input, combined_cxas_mask = create_mask_input(
            new_mask_component_info, best_mask, all_expansion_points, 
            all_contraction_points, chex_rl, chex_ll, fake=True
        )
        
        # 7. 예측 및 후처리
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
            # 시각화를 위한 필수 인자가 모두 있을 때만 호출
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
            
            # 이번에 생성된 fake 포인트들을 이후 생성 시에도 피하도록 누적
            if 'fake_mask' in all_expansion_points:
                prev_exp_points.setdefault('fake_mask', [])
                prev_exp_points['fake_mask'].extend(all_expansion_points['fake_mask'])
            if 'fake_mask' in all_contraction_points:
                prev_cont_points.setdefault('fake_mask', [])
                prev_cont_points['fake_mask'].extend(all_contraction_points['fake_mask'])
            
            # 성공했으면 retry 카운터 리셋
            retry_count = 0
            fake_mask_retry_attempt = 0  # fake_mask 재시도 카운터도 리셋
            current_fake_mask_depth_level = initial_fake_mask_depth_level  # depth_level도 초기화
            current_fake_mask_iterations_per_depth = iterations_per_depth  # iterations_per_depth도 초기화
        else:
            retry_count += 1
            # 실패했을 때도 fake_mask 재시도 시도
            if fake_mask_retry_attempt < max_fake_mask_retry_attempts:
                fake_mask_retry_attempt += 1
            else:
                fake_mask_retry_attempt = 0  # 최대 재시도 횟수 도달 시 리셋
                current_fake_mask_depth_level = initial_fake_mask_depth_level
                current_fake_mask_iterations_per_depth = iterations_per_depth
    
    if len(fake_masks) < 3:
        log_print(f"    [Fake Mask] WARNING: Failed to generate enough fake masks (only {len(fake_masks)} generated)")
    if len(all_fake_points) < 2:
        log_print(f"    [Fake Points] WARNING: Failed to generate enough fake points (only {len(all_fake_points)} generated)")
        
    return fake_masks, all_fake_points


