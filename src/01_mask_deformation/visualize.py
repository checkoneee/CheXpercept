import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from logger import log_print

def visualize_points(image, mask, pos_pts, neg_pts, mask_component_infos, save_path):
    """
    이미지, 마스크, SAM 결과들 위에 positive/negative points를 시각화
    
    Args:
        image: 원본 이미지 (PIL Image)
        mask: segmentation mask (PIL Image)
        pos_pts: positive points list [(x, y), ...]
        neg_pts: negative points list [(x, y), ...]
        sam_masks: SAM prediction masks list [numpy array, ...]
        save_path: 저장할 경로
    """
    # PIL Image를 numpy array로 변환
    image_np = np.array(image)
    mask_np = np.array(mask)
    
    # RGB 이미지인 경우 첫 번째 채널만 사용
    if len(mask_np.shape) == 3:
        mask_np = mask_np[:, :, 0]
    
    # subplot 개수: 원본 이미지 + 원본 마스크 + SAM 결과들

    sam_masks = []
    for mask_component_info in mask_component_infos:
        sam_masks.extend(mask_component_info['postprocessed_mask_components'])

    num_plots = 2 + len(sam_masks)
    fig, axes = plt.subplots(1, num_plots, figsize=(10 * num_plots, 10))
    
    # 단일 subplot인 경우 리스트로 변환
    if num_plots == 1:
        axes = [axes]
    
    # 첫 번째: 원본 이미지 위에 points
    axes[0].imshow(image_np, cmap='gray')
    if len(pos_pts) > 0:
        pos_x = [pt[0] for pt in pos_pts]
        pos_y = [pt[1] for pt in pos_pts]
        axes[0].scatter(pos_x, pos_y, c='red', s=50, marker='o', alpha=0.8, label=f'Positive ({len(pos_pts)})')
    if len(neg_pts) > 0:
        neg_x = [pt[0] for pt in neg_pts]
        neg_y = [pt[1] for pt in neg_pts]
        axes[0].scatter(neg_x, neg_y, c='blue', s=50, marker='x', alpha=0.8, label=f'Negative ({len(neg_pts)})')
    axes[0].set_title('Points on Original Image')
    axes[0].legend()
    axes[0].axis('off')
    
    # 두 번째: 마스크 위에 points
    axes[1].imshow(mask_np, cmap='gray')
    if len(pos_pts) > 0:
        pos_x = [pt[0] for pt in pos_pts]
        pos_y = [pt[1] for pt in pos_pts]
        axes[1].scatter(pos_x, pos_y, c='red', s=50, marker='o', alpha=0.8, label=f'Positive ({len(pos_pts)})')
    if len(neg_pts) > 0:
        neg_x = [pt[0] for pt in neg_pts]
        neg_y = [pt[1] for pt in neg_pts]
        axes[1].scatter(neg_x, neg_y, c='blue', s=50, marker='x', alpha=0.8, label=f'Negative ({len(neg_pts)})')
    axes[1].set_title('Points on Original Mask')
    axes[1].legend()
    axes[1].axis('off')
    
    # 나머지: 각 SAM 결과 위에 points
    for idx, sam_mask in enumerate(sam_masks):
        axes[2 + idx].imshow(sam_mask, cmap='gray')
        if len(pos_pts) > 0:
            pos_x = [pt[0] for pt in pos_pts]
            pos_y = [pt[1] for pt in pos_pts]
            axes[2 + idx].scatter(pos_x, pos_y, c='red', s=50, marker='o', alpha=0.8, label=f'Positive ({len(pos_pts)})')
        if len(neg_pts) > 0:
            neg_x = [pt[0] for pt in neg_pts]
            neg_y = [pt[1] for pt in neg_pts]
            axes[2 + idx].scatter(neg_x, neg_y, c='blue', s=50, marker='x', alpha=0.8, label=f'Negative ({len(neg_pts)})')
        axes[2 + idx].set_title(f'SAM Component {idx}')
        axes[2 + idx].legend()
        axes[2 + idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log_print(f"\n\nVisualization saved to: {save_path}\n\n")

def visualize_overlap_with_cxas(config, image, dicom_id, mask_components, geometrical_mask_info, save_dir):
    """
    각 mask component와 겹치는 CXAS mask를 시각화
    
    Args:
        config: 설정
        image: 원본 이미지 (PIL Image)
        dicom_id: DICOM ID
        mask_components: mask component 리스트
        geometrical_mask_info: overlap 정보 리스트
        save_dir: 저장 디렉토리
    """
    image_np = np.array(image)
    cxas_mask_path = os.path.join(config['path']['cxas_mask_path'], dicom_id)
    
    for idx, (mask_component, geo_info) in enumerate(zip(mask_components, geometrical_mask_info)):
        # 모든 CXAS mask 수집 (겹치는 것과 안 겹치는 것 분리)
        overlapping_masks = []
        overlapping_names = []
        overlapping_ratios = []
        non_overlapping_masks = []
        non_overlapping_names = []
        
        for key in geo_info["overlap"]:
            for class_name, overlap_info in geo_info["overlap"][key].items():
                class_mask_path = os.path.join(cxas_mask_path, class_name + '.png')
                if os.path.exists(class_mask_path):
                    class_mask = Image.open(class_mask_path)
                    class_mask = class_mask.resize((1024, 1024))
                    class_mask_np = np.array(class_mask)
                    
                    has_overlap = overlap_info["has_overlap"]
                    overlap_ratio = overlap_info["overlap_ratio"]
                    
                    if has_overlap:
                        overlapping_masks.append(class_mask_np)
                        overlapping_names.append(f"{key}/{class_name}")
                        overlapping_ratios.append(overlap_ratio)
                    else:
                        non_overlapping_masks.append(class_mask_np)
                        non_overlapping_names.append(f"{key}/{class_name}")
        
        total_masks = len(overlapping_masks) + len(non_overlapping_masks)
        if total_masks == 0:
            log_print(f"  Component {idx}: No CXAS masks found")
            continue
        
        log_print(f"  Component {idx}: {len(overlapping_masks)} overlapping, {len(non_overlapping_masks)} non-overlapping CXAS masks")
        
        # 시각화: 원본 이미지 + mask component + 겹치는 CXAS + 안 겹치는 CXAS
        num_plots = 2 + total_masks
        fig, axes = plt.subplots(1, num_plots, figsize=(10 * num_plots, 10))
        
        if num_plots == 1:
            axes = [axes]
        
        # 첫 번째: 원본 이미지
        axes[0].imshow(image_np, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 두 번째: mask component
        mask_component_np = np.array(mask_component) if hasattr(mask_component, 'shape') else mask_component
        axes[1].imshow(mask_component_np, cmap='gray')
        axes[1].set_title(f'Mask Component {idx}')
        axes[1].axis('off')
        
        plot_idx = 2
        
        # 겹치는 CXAS masks (빨간색)
        for i, (cxas_mask, cxas_name, ratio) in enumerate(zip(overlapping_masks, overlapping_names, overlapping_ratios)):
            axes[plot_idx].imshow(image_np, cmap='gray', alpha=0.5)
            axes[plot_idx].imshow(cxas_mask, cmap='Reds', alpha=0.5)
            axes[plot_idx].set_title(f'✓ OVERLAP: {cxas_name}\n({ratio*100:.1f}% of CXAS)', color='red', fontsize=10)
            axes[plot_idx].axis('off')
            plot_idx += 1
        
        # 안 겹치는 CXAS masks (파란색)
        for i, (cxas_mask, cxas_name) in enumerate(zip(non_overlapping_masks, non_overlapping_names)):
            axes[plot_idx].imshow(image_np, cmap='gray', alpha=0.5)
            axes[plot_idx].imshow(cxas_mask, cmap='Blues', alpha=0.5)
            axes[plot_idx].set_title(f'✗ NO OVERLAP: {cxas_name}', color='blue')
            axes[plot_idx].axis('off')
            plot_idx += 1
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{geo_info['mask_component_id']}_cxas_all.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        log_print(f"  CXAS visualization saved: {save_path}")
