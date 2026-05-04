import os
import json
import csv
import argparse
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))


def merge_results(base_out, total_parts):
    merged_results = defaultdict(list)
    merged_rows = []

    for part in range(total_parts):
        suffix = f"_part{part}"

        results_path = os.path.join(base_out, f'results{suffix}.json')
        if not os.path.exists(results_path):
            print(f"[Warning] Missing: {results_path}, skipping part {part}")
        else:
            with open(results_path) as f:
                part_results = json.load(f)
            for pair_id, items in part_results.items():
                merged_results[pair_id].extend(items)
            print(f"[Part {part}] {len(part_results)} pair_ids from results{suffix}.json")

        csv_path = os.path.join(base_out, f'labeling_sheet{suffix}.csv')
        if not os.path.exists(csv_path):
            print(f"[Warning] Missing: {csv_path}, skipping part {part}")
        else:
            with open(csv_path, newline='', encoding='utf-8') as f:
                merged_rows.extend(list(csv.DictReader(f)))
            print(f"[Part {part}] {len(merged_rows)} rows from labeling_sheet{suffix}.csv")

    with open(os.path.join(base_out, 'results.json'), 'w') as f:
        json.dump(merged_results, f, indent=4, ensure_ascii=False)
    print(f"\nMerged: {len(merged_results)} pair_ids -> {base_out}/results.json")

    fieldnames = ['key_id', 'dicom_id', 'target', 'pair_id', 'mapped_location',
                  'segmentation_source', 'plot_path', 'good?']
    with open(os.path.join(base_out, 'labeling_sheet.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    print(f"Merged: {len(merged_rows)} rows -> {base_out}/labeling_sheet.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge part outputs from 01_generate_rosalia_pred.py")
    parser.add_argument('--base-out', type=str,
                        default=os.path.join(current_dir, 'outputs', 'rosalia_pred'))
    parser.add_argument('--total-parts', type=int, default=4)
    args = parser.parse_args()

    merge_results(args.base_out, args.total_parts)
