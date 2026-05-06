"""
Generate QA from Stage 01 deformation results.

For each annotated case, builds:
    - detection QA (Yes/No)
    - contour evaluation + revision QA (positive cases)
    - attribute extraction QA (positive non-cardiomegaly cases)

Inputs:
    deformation_results.json   - Stage 01 output (positive cases with deformation outcome)
    positive labeling_sheet    - Stage 00 rosalia_pred labeling sheet (key_id -> dicom_id)
    negative labeling_sheet    - Stage 00 negative labeling sheet (key_id -> dicom_id)
    [optional] positive annotation CSV - if provided, only key_ids with non-empty `optimal`
                                          column are included; otherwise all key_ids in
                                          deformation_results with qa_deformation_success.
    [optional] negative annotation CSV - same convention for negatives; otherwise all
                                          key_ids in the negative labeling sheet.

Outputs (under <output-dir>):
    qa_visualizations/         - per-case QA visualization PNGs
    qa_results/qa_results.json - all QA dicts keyed by key_id
    qa_results/qa_labeling.csv - labeling metadata
    chexpercept/               - per-key_id mask copies for the benchmark
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config import open_config
from qa_utils import build_qa
from chexpercept_export import (
    copy_masks_to_chexpercept,
    get_dicom_id_from_key_id,
)


def load_annotated_key_ids(csv_path):
    """Return key_ids whose `optimal` (or `optimal?`) column is non-empty.

    Works on both the Stage 00 labeling sheet (`optimal?` column) and the combined
    doctor annotation CSV (`optimal` column) since both share the same convention:
    a non-empty value means "annotated as optimal".
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    optimal_col = "optimal" if "optimal" in rows[0] else "optimal?"
    return [
        row["key_id"].strip()
        for row in rows
        if row.get(optimal_col, "").strip()
    ]


def get_no_deformation_ratio(config, lesion_name):
    """Look up the lesion-specific no-deformation ratio (default 0.5)."""
    ratio_cfg = config.get("qa_generation", {}).get("no_deformation_ratio", {})
    return ratio_cfg.get(lesion_name, ratio_cfg.get("default", 0.5))


def assign_no_deformation_flags(key_ids, config):
    """Pre-assign no-deformation (revision-free) flags by exact lesion ratio.

    Returns:
        dict[str, bool]: True means revision is not required for that key_id.
    """
    groups = defaultdict(list)
    for key_id in key_ids:
        groups[key_id.split("_")[0]].append(key_id)

    assignment = {}
    for lesion, ids in groups.items():
        ratio = get_no_deformation_ratio(config, lesion)
        n_no_deform = round(len(ids) * ratio)
        for i, key_id in enumerate(ids):
            assignment[key_id] = i < n_no_deform
        actual = n_no_deform / len(ids) if ids else 0
        print(
            f"[no_deformation assignment] {lesion}: "
            f"{n_no_deform}/{len(ids)} no-deformation "
            f"(target ratio={ratio:.2f}, actual={actual:.2f})"
        )
    return assignment


def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    return obj


def process_single_key_id(args):
    """Build QA and copy masks into the chexpercept folder for a single key_id."""
    (
        key_id,
        deformation_result,
        config_path,
        deformed_mask_path,
        chexpercept_path,
        generate_sequential_qa,
        no_deformation_override,
        labeling_sheet_positive_path,
        labeling_sheet_negative_path,
    ) = args

    config = open_config(config_path)

    # Skip positive cases whose Stage 01 deformation failed.
    if deformation_result != "negative" and not deformation_result["qa_deformation_success"]:
        return None, True

    try:
        lesion_name = key_id.split("_")[0]
        parts = key_id.split("_", 1)
        pair_id = parts[1] if len(parts) > 1 else None

        qa = build_qa(
            key_id,
            deformation_result,
            lesion_name,
            generate_sequential_qa=generate_sequential_qa,
            no_deformation_override=no_deformation_override,
        )

        dicom_id = get_dicom_id_from_key_id(
            key_id,
            config,
            labeling_sheet_positive_path,
            labeling_sheet_negative_path,
        )
        qa["pair_id"] = pair_id
        qa["dicom_id"] = dicom_id

        copy_masks_to_chexpercept(
            key_id,
            qa,
            config,
            deformed_mask_path,
            chexpercept_path,
            labeling_sheet_positive_path=labeling_sheet_positive_path,
            labeling_sheet_negative_path=labeling_sheet_negative_path,
        )
        return (key_id, qa), False
    except Exception as e:
        print(f"Error processing {key_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, True


def write_qa_labeling_csv(csv_path, all_qa_results):
    """Write the per-plot labeling CSV listing each generated visualization."""
    with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["plot_name", "qa_type", "key_id", "lesion_name", "pair_id", "dicom_id"]
        )
        for key_id, qa in all_qa_results.items():
            lesion_name = key_id.split("_")[0]
            pair_id = qa.get("pair_id", "")
            dicom_id = qa.get("dicom_id", "")
            writer.writerow(
                [
                    f"{key_id}_attribute_extraction_qa.png",
                    "attribute_extraction",
                    key_id,
                    lesion_name,
                    pair_id,
                    dicom_id,
                ]
            )
            if qa.get("deformation"):
                writer.writerow(
                    [
                        f"{key_id}_deformation_qa.png",
                        "deformation",
                        key_id,
                        lesion_name,
                        pair_id,
                        dicom_id,
                    ]
                )


def main(args):
    config = open_config(args.config)

    qa_output_path = os.path.join(args.output_dir, "qa_results")
    chexpercept_path = os.path.join(args.output_dir, "chexpercept")
    for d in (qa_output_path, chexpercept_path):
        os.makedirs(d, exist_ok=True)

    with open(args.deformation_results_json, "r") as f:
        deformation_results = json.load(f)

    # --- Positive selection: annotated optimal ∩ Stage 01 deformation success ---
    deformation_success_keys = {
        k
        for k, v in deformation_results.items()
        if isinstance(v, dict) and v.get("qa_deformation_success")
    }
    positive_csv = args.positive_annotation_csv or args.positive_labeling_sheet
    annotated_positive = load_annotated_key_ids(positive_csv)
    positive_key_ids = [k for k in annotated_positive if k in deformation_success_keys]
    print(
        f"[positive] {len(positive_key_ids)} key_ids "
        f"(annotated optimal in {os.path.basename(positive_csv)} ∩ qa_deformation_success)"
    )

    # --- Negative selection: annotated optimal in negative sheet ---
    negative_csv = args.negative_annotation_csv or args.negative_labeling_sheet
    negative_key_ids = load_annotated_key_ids(negative_csv)
    print(
        f"[negative] {len(negative_key_ids)} key_ids "
        f"(annotated optimal in {os.path.basename(negative_csv)})"
    )

    no_deformation_flags = assign_no_deformation_flags(positive_key_ids, config)

    args_list = []
    args_list.extend(
        (
            key_id,
            "negative",
            args.config,
            "negative",
            chexpercept_path,
            args.with_sequential_qa,
            None,
            args.positive_labeling_sheet,
            args.negative_labeling_sheet,
        )
        for key_id in negative_key_ids
    )
    args_list.extend(
        (
            key_id,
            deformation_results[key_id],
            args.config,
            args.deformed_mask_dir,
            chexpercept_path,
            args.with_sequential_qa,
            no_deformation_flags[key_id],
            args.positive_labeling_sheet,
            args.negative_labeling_sheet,
        )
        for key_id in positive_key_ids
    )

    num_workers = min(args.num_workers, cpu_count())
    print(f"Using {num_workers} workers; sequential QA={args.with_sequential_qa}")

    all_qa_results = {}
    false_count = 0
    with Pool(processes=num_workers) as pool:
        for result, is_false in tqdm(
            pool.imap(process_single_key_id, args_list),
            total=len(args_list),
            desc="Generating QA",
        ):
            if is_false:
                false_count += 1
            elif result is not None:
                key_id, qa = result
                all_qa_results[key_id] = qa

    qa_json_path = os.path.join(qa_output_path, "qa_results.json")
    with open(qa_json_path, "w", encoding="utf-8") as f:
        json.dump(convert_to_json_serializable(all_qa_results), f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(qa_output_path, "qa_labeling.csv")
    write_qa_labeling_csv(csv_path, all_qa_results)

    print(f"Saved QA results: {qa_json_path}")
    print(f"Saved labeling CSV: {csv_path}")
    print(f"Failed entries: {false_count}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate QA from Stage 01 deformation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=os.path.join(project_root, "cfg/config.yaml"),
        help="YAML config (paths to MIMIC-CXR / chexmask / cxas, no_deformation ratios).",
    )
    parser.add_argument(
        "--deformation-results-json",
        default=os.path.join(
            project_root,
            "src/01_mask_deformation/outputs/deformation_results/deformation_results.json",
        ),
        help="Stage 01 output JSON.",
    )
    parser.add_argument(
        "--positive-labeling-sheet",
        default=os.path.join(
            project_root,
            "src/00_source_data_curation/outputs/rosalia_pred/labeling_sheet.csv",
        ),
        help=(
            "Stage 00 positive labeling sheet. Only rows with a non-empty "
            "`optimal` (or `optimal?`) column are used."
        ),
    )
    parser.add_argument(
        "--negative-labeling-sheet",
        default=os.path.join(
            project_root,
            "src/00_source_data_curation/outputs/negative/labeling_sheet.csv",
        ),
        help=(
            "Stage 00 negative labeling sheet. Only rows with a non-empty "
            "`optimal` (or `optimal?`) column are used."
        ),
    )
    parser.add_argument(
        "--positive-annotation-csv",
        default=None,
        help=(
            "Optional alternate positive source. If provided, it overrides "
            "--positive-labeling-sheet (e.g., a combined annotation CSV from "
            "multiple annotators). Same `optimal` column convention."
        ),
    )
    parser.add_argument(
        "--negative-annotation-csv",
        default=None,
        help=(
            "Optional alternate negative source. If provided, it overrides "
            "--negative-labeling-sheet. Same `optimal` column convention."
        ),
    )
    parser.add_argument(
        "--deformed-mask-dir",
        default=os.path.join(project_root, "src/01_mask_deformation/outputs/deformed_masks"),
        help="Stage 01 deformed_masks directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_root, "src/02_qa_generation/outputs"),
        help="Stage 02 output base directory.",
    )
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--with-sequential-qa",
        action="store_true",
        help=(
            "Also build sequential-QA fields (parallel QA is always built). "
            "Disabled by default; the CheXpercept benchmark uses parallel QA only."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
