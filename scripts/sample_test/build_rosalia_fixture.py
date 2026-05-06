"""Stage the ROSALIA-inference output that sample_test.sh skips.

Writes the two files Stage 00.02 (`02_prepare_positive_annotation.py`)
expects to merge:
    - results_part0.json
    - labeling_sheet_part0.csv

so the rest of the pipeline runs as if ROSALIA had inferred a single
cardiomegaly positive case.
"""
import argparse
import csv
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True,
                        help="src/00_source_data_curation/outputs/rosalia_pred")
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--dicom-id", required=True)
    parser.add_argument("--target", default="cardiomegaly")
    parser.add_argument("--mapped-location", default="heart")
    args = parser.parse_args()

    key_id = f"{args.target}_{args.pair_id}"
    pred_mask_path = os.path.abspath(
        os.path.join(args.out_dir, args.target, args.pair_id, "pred_mask.png")
    )

    results = {
        args.pair_id: [{
            "dicom_id": args.dicom_id,
            "target": args.target,
            "pred_mask_path": pred_mask_path,
            "output_text": "[SEG]",
            "mapped_location": args.mapped_location,
            "instruction": f"Segment the {args.target}.",
            "segmentation_source": "global",
        }],
    }
    with open(os.path.join(args.out_dir, "results_part0.json"), "w") as f:
        json.dump(results, f, indent=2)

    fields = [
        "key_id", "dicom_id", "target", "pair_id",
        "mapped_location", "segmentation_source", "optimal?", "plot_path",
    ]
    with open(os.path.join(args.out_dir, "labeling_sheet_part0.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "key_id": key_id,
            "dicom_id": args.dicom_id,
            "target": args.target,
            "pair_id": args.pair_id,
            "mapped_location": args.mapped_location,
            "segmentation_source": "global",
            "optimal?": "",
            "plot_path": "",
        })

    print(f"staged ROSALIA fixture for {key_id}")


if __name__ == "__main__":
    main()
