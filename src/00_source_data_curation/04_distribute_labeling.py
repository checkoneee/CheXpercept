import os
import csv
import shutil
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))


def split_by_lesion(rows, lesion_col, num_annotators):
    """Split rows evenly per lesion across annotators."""
    from collections import defaultdict
    by_lesion = defaultdict(list)
    for row in rows:
        by_lesion[row[lesion_col]].append(row)

    annotator_rows = {i: [] for i in range(num_annotators)}
    for lesion, group in by_lesion.items():
        for i, row in enumerate(group):
            annotator_rows[i % num_annotators].append(row)
    return annotator_rows


def distribute(sheet_path, lesion_col, img_src_dir, task_name, out_base, num_annotators):
    if not os.path.exists(sheet_path):
        print(f"[Skip] {sheet_path} not found")
        return

    with open(sheet_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    annotator_rows = split_by_lesion(rows, lesion_col, num_annotators)
    annotator_labels = [chr(ord('A') + i) for i in range(num_annotators)]
    fieldnames = list(rows[0].keys()) if rows else []

    for i, label in enumerate(annotator_labels):
        ann_dir = os.path.join(out_base, f"annotator_{label}", task_name)
        os.makedirs(ann_dir, exist_ok=True)

        chunk = annotator_rows[i]

        # Copy images
        for row in chunk:
            key_id = row['key_id']
            lesion = row[lesion_col]
            src = os.path.join(img_src_dir, lesion, f"{key_id}.png")
            dst_dir = os.path.join(ann_dir, lesion)
            os.makedirs(dst_dir, exist_ok=True)
            if os.path.exists(src):
                shutil.copy2(src, dst_dir)
            else:
                print(f"  [WARN] image not found: {src}")

        # Save labeling sheet
        sheet_out = os.path.join(ann_dir, 'labeling_sheet.csv')
        with open(sheet_out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(chunk)

        print(f"  Annotator {label}: {len(chunk)} {task_name} cases -> {ann_dir}")


def run(args):
    out_base = os.path.join(current_dir, 'outputs', 'distribute')

    positive_sheet = args.positive_csv or os.path.join(current_dir, 'outputs', 'rosalia_pred', 'labeling_sheet.csv')
    negative_sheet = args.negative_csv or os.path.join(current_dir, 'outputs', 'negative', 'labeling_sheet.csv')

    positive_img_dir = os.path.join(current_dir, 'outputs', 'rosalia_pred', 'plots')
    negative_img_dir = os.path.join(current_dir, 'outputs', 'negative')

    print(f"=== Positive cases ===")
    distribute(positive_sheet, lesion_col='target', img_src_dir=positive_img_dir,
               task_name='positive', out_base=out_base, num_annotators=args.num_annotators)

    print(f"\n=== Negative cases ===")
    distribute(negative_sheet, lesion_col='lesion', img_src_dir=negative_img_dir,
               task_name='negative', out_base=out_base, num_annotators=args.num_annotators)

    print(f"\nDone. Distributed to {args.num_annotators} annotators -> {out_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distribute positive and negative labeling sheets to annotators")
    parser.add_argument('--positive-csv', type=str, default=None,
                        help="Positive labeling sheet (default: output/rosalia_pred/labeling_sheet.csv)")
    parser.add_argument('--negative-csv', type=str, default=None,
                        help="Negative labeling sheet (default: output/negative/labeling_sheet.csv)")
    parser.add_argument('--num-annotators', type=int, default=6,
                        help="Number of annotators (default: 6)")
    args = parser.parse_args()

    run(args)
