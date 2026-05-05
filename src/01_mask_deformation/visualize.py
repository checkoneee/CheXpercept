import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from logger import log_print

def visualize_points(image, mask, pos_pts, neg_pts, mask_component_infos, save_path):
    """
    Visualize positive/negative points on top of the image, mask, and SAM outputs.

    Args:
        image: original image (PIL Image)
        mask: segmentation mask (PIL Image)
        pos_pts: positive points list [(x, y), ...]
        neg_pts: negative points list [(x, y), ...]
        sam_masks: SAM prediction masks list [numpy array, ...]
        save_path: output path
    """
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    mask_np = np.array(mask)

    # Use only the first channel for RGB images
    if len(mask_np.shape) == 3:
        mask_np = mask_np[:, :, 0]

    # subplot count: original image + original mask + SAM outputs

    sam_masks = []
    for mask_component_info in mask_component_infos:
        sam_masks.extend(mask_component_info['postprocessed_mask_components'])

    num_plots = 2 + len(sam_masks)
    fig, axes = plt.subplots(1, num_plots, figsize=(10 * num_plots, 10))

    # Wrap a single subplot in a list
    if num_plots == 1:
        axes = [axes]

    # First: points on original image
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
    
    # Second: points on the mask
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
    
    # Remaining: points on each SAM output
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
    Visualize the CXAS masks overlapping with each mask component.

    Args:
        config: configuration
        image: original image (PIL Image)
        dicom_id: DICOM ID
        mask_components: list of mask components
        geometrical_mask_info: list of overlap info
        save_dir: output directory
    """
    image_np = np.array(image)
    cxas_mask_path = os.path.join(config['path']['cxas_mask_path'], dicom_id)

    for idx, (mask_component, geo_info) in enumerate(zip(mask_components, geometrical_mask_info)):
        # Collect all CXAS masks (split into overlapping vs non-overlapping)
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
        
        # Visualize: original image + mask component + overlapping CXAS + non-overlapping CXAS
        num_plots = 2 + total_masks
        fig, axes = plt.subplots(1, num_plots, figsize=(10 * num_plots, 10))

        if num_plots == 1:
            axes = [axes]

        # First: original image
        axes[0].imshow(image_np, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Second: mask component
        mask_component_np = np.array(mask_component) if hasattr(mask_component, 'shape') else mask_component
        axes[1].imshow(mask_component_np, cmap='gray')
        axes[1].set_title(f'Mask Component {idx}')
        axes[1].axis('off')

        plot_idx = 2

        # Overlapping CXAS masks (red)
        for i, (cxas_mask, cxas_name, ratio) in enumerate(zip(overlapping_masks, overlapping_names, overlapping_ratios)):
            axes[plot_idx].imshow(image_np, cmap='gray', alpha=0.5)
            axes[plot_idx].imshow(cxas_mask, cmap='Reds', alpha=0.5)
            axes[plot_idx].set_title(f'✓ OVERLAP: {cxas_name}\n({ratio*100:.1f}% of CXAS)', color='red', fontsize=10)
            axes[plot_idx].axis('off')
            plot_idx += 1

        # Non-overlapping CXAS masks (blue)
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
