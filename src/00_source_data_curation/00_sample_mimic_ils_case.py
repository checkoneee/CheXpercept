import sys
import os
import json
import argparse
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config import open_config

LESIONS = ['cardiomegaly', 'consolidation', 'edema', 'atelectasis', 'pneumonia', 'opacity', 'effusion']


def sample_mimic_ils_cases(config):
    output_dir = os.path.join(current_dir, 'outputs')
    global_output_path = os.path.join(output_dir, 'global_segmentation_info.json')
    basic_output_path = os.path.join(output_dir, 'basic_segmentation_info.json')
    negative_output_path = os.path.join(output_dir, 'global_segmentation_negative_info.json')

    if all(os.path.exists(p) for p in [global_output_path, basic_output_path, negative_output_path]):
        print("Output already exists, skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(config['path']['mimic_ils_path'], 'r') as f:
        dataset = json.load(f)

    global_info = {lesion: [] for lesion in LESIONS}
    basic_info = {lesion: [] for lesion in LESIONS}
    negative_info = {lesion: [] for lesion in LESIONS}

    for study_id, data in tqdm(dataset['train'].items(), total=len(dataset['train'])):
        pairs_data = data['instruction_answer_pairs']
        common_fields = {
            'report_path': os.path.join(
                config['path']['mimic_report_path'],
                data['subject_id'][:3], data['subject_id'], f"{study_id}.txt"
            ),
            'study_id': study_id,
            'dicom_id': data['dicom_id'],
            'image_path': data['image_path'],
        }

        positive_pairs = pairs_data.get('positive_pairs', [])
        has_global = {lesion: False for lesion in LESIONS}
        for pair in positive_pairs:
            if pair['type'] == 'global':
                global_info[pair['target']].append({**pair, **common_fields})
                has_global[pair['target']] = True
        for pair in positive_pairs:
            if pair['type'] == 'basic' and not has_global[pair['target']]:
                basic_info[pair['target']].append({**pair, **common_fields})
                has_global[pair['target']] = True

        for pair in pairs_data.get('negative_pairs', []):
            if pair['type'] == 'global':
                negative_info[pair['target']].append({**pair, **common_fields})

    print("\nPositive (global):")
    for lesion in LESIONS:
        print(f"  {lesion}: {len(global_info[lesion])}")
    print("Positive (basic fallback):")
    for lesion in LESIONS:
        print(f"  {lesion}: {len(basic_info[lesion])}")
    print("Negative (global):")
    for lesion in LESIONS:
        print(f"  {lesion}: {len(negative_info[lesion])}")

    with open(global_output_path, 'w') as f:
        json.dump(global_info, f, indent=4)
    with open(basic_output_path, 'w') as f:
        json.dump(basic_info, f, indent=4)
    with open(negative_output_path, 'w') as f:
        json.dump(negative_info, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cfg/config.yaml')
    args = parser.parse_args()

    config = open_config(args.config)
    sample_mimic_ils_cases(config)
