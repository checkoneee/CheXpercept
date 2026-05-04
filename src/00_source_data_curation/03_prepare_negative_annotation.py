import sys
import os
import csv
import json
import shutil
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config import open_config


def prepare_negative_annotation(config, args):
    negative_json_path = os.path.join(current_dir, 'outputs', 'global_segmentation_negative_info.json')
    output_dir = os.path.join(current_dir, 'outputs', 'negative')

    if not os.path.exists(negative_json_path):
        raise FileNotFoundError(
            f"{negative_json_path} not found. Run 00_sample_mimic_ils_case.py first."
        )

    with open(negative_json_path) as f:
        global_segmentation_info = json.load(f)

    # Exclude dicom_ids already used in positive cases (to avoid overlap)
    used_dicom_ids = set()
    if args.positive_csv and os.path.exists(args.positive_csv):
        with open(args.positive_csv) as f:
            for row in csv.DictReader(f):
                did = row.get('dicom_id', '').strip()
                if did:
                    used_dicom_ids.add(did)
        print(f"Excluding {len(used_dicom_ids)} dicom_ids from positive sheet")

    mimic_image_base = config['path']['mimic_cxr_path']
    all_rows = []

    for lesion, items in global_segmentation_info.items():
        available = [item for item in items if item['dicom_id'] not in used_dicom_ids]
        sampled = available[:args.samples_per_lesion]

        for item in sampled:
            used_dicom_ids.add(item['dicom_id'])

        print(f"{lesion}: {len(sampled)} sampled (available={len(available)})")

        lesion_img_dir = os.path.join(output_dir, lesion)
        os.makedirs(lesion_img_dir, exist_ok=True)
        for item in sampled:
            key_id = f"{lesion}_{item['pair_id']}"
            src = os.path.join(mimic_image_base, f"{item['dicom_id']}.png")
            dst = os.path.join(lesion_img_dir, f"{key_id}.png")
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"  [WARN] image not found: {src}")

            all_rows.append({
                'lesion': lesion,
                'key_id': key_id,
                'dicom_id': item['dicom_id'],
                'good?': '',
            })

    os.makedirs(output_dir, exist_ok=True)
    sheet_path = os.path.join(output_dir, 'labeling_sheet.csv')
    with open(sheet_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['lesion', 'key_id', 'dicom_id', 'good?'])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{len(all_rows)} cases -> {sheet_path}")
    print("Run 04_distribute_labeling.py to distribute to annotators.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample negative cases from MIMIC-ILS for annotation")
    parser.add_argument('--config', type=str, default='cfg/config.yaml')
    parser.add_argument('--samples-per-lesion', type=int, default=100,
                        help="Number of cases to sample per lesion (default: 100)")
    parser.add_argument('--positive-csv', type=str, default=None,
                        help="Positive labeling sheet to exclude overlapping dicom_ids")
    args = parser.parse_args()

    config = open_config(args.config)
    prepare_negative_annotation(config, args)
