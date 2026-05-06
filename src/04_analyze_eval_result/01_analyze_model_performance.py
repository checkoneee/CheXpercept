"""Aggregate per-model evaluation results into summary CSVs and plots.

Reads ``all_results.json`` produced by Stage 03 for each model under
``<results-dir>/<provider>_<model>/oracle_<setting>/`` and writes per-stage
accuracy summaries plus visualizations under
``<this dir>/all_models_oracle_<setting>/``.
"""

import os
import sys
import argparse
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import json
from typing import List, Dict, Any, Tuple, Optional

from visualize import (
    plot_stage_accuracy_table,
    plot_contour_revision_detail,
    plot_attribute_extraction_detail,
    plot_depth_heatmap,
)

DEFAULT_RESULTS_DIR = os.path.join(
    project_root, "src/03_eval_vlm_on_chexpercept/outputs"
)


def compute_average(values: List[Any]) -> float:
    """
    Compute the average of a list.
    - For a bool list, this is the accuracy (mean accuracy).
    - For an int/float list, this is the regular mean.
    - Returns 0.0 if empty.
    """
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def init_performance_metrics() -> Dict[str, Any]:
    """Build the default structure used to accumulate performance metrics per qa_path."""
    return {
        'revision_required': {
            'depth': {'list': [], 'average': 0},
            'detection_qa': {'list': [], 'average': 0},
            'contour_evaluation_qa': {
                'list (end-to-end)': [],
                'list (oracle-passed)': [],
                'average (end-to-end)': 0,
                'average (oracle-passed)': 0,
            },
            'contour_revision_qa': {
                'list (end-to-end)': [],
                'list (oracle-passed)': [],
                'average (end-to-end)': 0,
                'average (oracle-passed)': 0,
            },
            'attribute_extraction_qa': {
                'list (end-to-end)': [],
                'list (oracle-passed)': [],
                'average (end-to-end)': 0,
                'average (oracle-passed)': 0,
            },
            # Detailed correctness (oracle-passed view)
            'contour_revision_qa_detail': {
                'expansion': [],
                'contraction': [],
                'revision_result': [],
            },
            'attribute_extraction_qa_detail': {
                'distribution': [],
                'location': [],
                'severity/measurement': [],
                'comparison': [],
            },
        },
        'revision_free': {
            'depth': {'list': [], 'average': 0},
            'detection_qa': {'list': [], 'average': 0},
            'contour_evaluation_qa': {
                'list (end-to-end)': [],
                'list (oracle-passed)': [],
                'average (end-to-end)': 0,
                'average (oracle-passed)': 0,
            },
            'attribute_extraction_qa': {
                'list (end-to-end)': [],
                'list (oracle-passed)': [],
                'average (end-to-end)': 0,
                'average (oracle-passed)': 0,
            },
            'attribute_extraction_qa_detail': {
                'distribution': [],
                'location': [],
                'severity/measurement': [],
                'comparison': [],
            },
        },
        'lesion_free': {
            'depth': {'list': [], 'average': 0},
            'detection_qa': {'list': [], 'average': 0},
        },
    }


def accumulate_result_for_case(performance_metrics: Dict[str, Any], result: Dict[str, Any], key_id: str = "") -> None:
    """
    For a single case (result), compute end-to-end / oracle-passed correctness
    and update the per-qa_path lists.
    ``key_id`` (e.g. ``cardiomegaly_s5...``) is used to split depth into
    cardiomegaly vs non-cardiomegaly buckets, since cardiomegaly cases have
    a smaller maximum depth (no attribute_extraction stage).
    """
    qa_path = result['qa_path']
    is_cardio = key_id.startswith("cardiomegaly_")

    # Correctness flags per stage
    end_to_end_correct = {
        'detection_qa': False,
        'contour_evaluation_qa': False,
        'contour_revision_qa': False,
        'attribute_extraction_qa': False,
    }
    oracle_passed_correct = {
        'detection_qa': False,
        'contour_evaluation_qa': False,
        'contour_revision_qa': False,
        'attribute_extraction_qa': False,
    }

    # end-to-end: once any earlier stage is wrong, every subsequent stage is False
    end_to_end_still_valid = True
    stage_order = ['detection_qa', 'contour_evaluation_qa', 'contour_revision_qa', 'attribute_extraction_qa']

    for qa_type in stage_order:
        qa_result = result.get(qa_type)
        if not qa_result:
            continue

        is_correct = bool(qa_result.get('correct'))

        if is_correct and end_to_end_still_valid:
            end_to_end_correct[qa_type] = True
        else:
            end_to_end_still_valid = False

        if is_correct:
            oracle_passed_correct[qa_type] = True

    # Reflect only the stages that actually exist for each qa_path.
    # For cardiomegaly cases, even when qa_path is revision_required/revision_free,
    # the attribute_extraction_qa stage is absent, so empty stages in the result dict are skipped.
    path_to_stages = {
        'revision_required': ['detection_qa', 'contour_evaluation_qa', 'contour_revision_qa', 'attribute_extraction_qa'],
        'revision_free': ['detection_qa', 'contour_evaluation_qa', 'attribute_extraction_qa'],
        'lesion_free': ['detection_qa'],
    }
    qa_type_list = [
        qa_type for qa_type in path_to_stages[qa_path]
        if qa_type == 'detection_qa' or result.get(qa_type)
    ]

    for qa_type in qa_type_list:
        if qa_type == 'detection_qa':
            performance_metrics[qa_path]['detection_qa']['list'].append(end_to_end_correct[qa_type])
        else:
            performance_metrics[qa_path][qa_type]['list (end-to-end)'].append(end_to_end_correct[qa_type])
            performance_metrics[qa_path][qa_type]['list (oracle-passed)'].append(oracle_passed_correct[qa_type])

    # depth: number of stages correct in a row under end-to-end (only stages actually asked).
    # Cardiomegaly cases lack attribute_extraction_qa by construction; clearing the last
    # asked stage is treated as also clearing the missing stage so the max depth is
    # uniform across lesions within a path (RR=4, RF=3, LF=1).
    depth = sum(1 for qa_type in qa_type_list if end_to_end_correct[qa_type])
    if is_cardio and qa_path in ('revision_required', 'revision_free') and qa_type_list:
        if end_to_end_correct[qa_type_list[-1]]:
            depth += 1
    performance_metrics[qa_path]['depth']['list'].append(depth)

    # ----- Detailed correctness (oracle_passed view, no end-to-end gating) -----
    # contour_revision_qa detailed stats: only meaningful for revision_required cases
    if qa_path == 'revision_required':
        cr = result.get('contour_revision_qa')
        if isinstance(cr, dict):
            for sub_key in ['expansion', 'contraction', 'revision_result']:
                sub = cr.get(sub_key)
                if isinstance(sub, dict) and 'correct' in sub:
                    performance_metrics['revision_required']['contour_revision_qa_detail'][sub_key].append(bool(sub['correct']))

    # attribute_extraction_qa detailed stats: collected for every path where attribute_extraction_qa exists
    if qa_path in ['revision_required', 'revision_free']:
        ae = result.get('attribute_extraction_qa')
        if isinstance(ae, dict):
            for sub_key in ['distribution', 'location', 'severity/measurement', 'comparison']:
                sub = ae.get(sub_key)
                if isinstance(sub, dict) and 'correct' in sub:
                    performance_metrics[qa_path]['attribute_extraction_qa_detail'][sub_key].append(bool(sub['correct']))


def finalize_stage_averages(performance_metrics: Dict[str, Any]) -> None:
    """Fill in average values from the lists for each qa_path / stage."""
    for qa_path, qa_dict in performance_metrics.items():
        for qa_type, stats in qa_dict.items():
            if 'list (end-to-end)' in stats:
                stats['average (end-to-end)'] = compute_average(stats['list (end-to-end)'])
                stats['average (oracle-passed)'] = compute_average(stats['list (oracle-passed)'])
            if 'list' in stats:
                stats['average'] = compute_average(stats['list'])


def build_summary_rows(performance_metrics: Dict[str, Any], model_name: str) -> List[Dict[str, Any]]:
    """
    Build the summary rows written to CSV.
    One row is created per combination of path(qa_path) and setting(end_to_end / oracle_passed).
    """
    summary_rows: List[Dict[str, Any]] = []

    for qa_path, qa_dict in performance_metrics.items():
        end_to_end_stage1 = qa_dict['detection_qa']['list']
        oracle_stage1 = qa_dict['detection_qa']['list']
        depth_list = qa_dict['depth']['list']

        e2e_s2 = []
        e2e_s3 = []
        e2e_s4 = []
        orc_s2 = []
        orc_s3 = []
        orc_s4 = []

        if 'contour_evaluation_qa' in qa_dict:
            e2e_s2 = qa_dict['contour_evaluation_qa']['list (end-to-end)']
            orc_s2 = qa_dict['contour_evaluation_qa']['list (oracle-passed)']
        if 'contour_revision_qa' in qa_dict:
            e2e_s3 = qa_dict['contour_revision_qa']['list (end-to-end)']
            orc_s3 = qa_dict['contour_revision_qa']['list (oracle-passed)']
        if 'attribute_extraction_qa' in qa_dict:
            e2e_s4 = qa_dict['attribute_extraction_qa']['list (end-to-end)']
            orc_s4 = qa_dict['attribute_extraction_qa']['list (oracle-passed)']

        summary_rows.append({
            'path': qa_path,
            'model': model_name,
            'setting': 'end_to_end',
            'acc(stage1)': compute_average(end_to_end_stage1),
            'acc(stage2)': compute_average(e2e_s2),
            'acc(stage3)': compute_average(e2e_s3),
            'acc(stage4)': compute_average(e2e_s4),
            'depth': compute_average(depth_list),
        })

        summary_rows.append({
            'path': qa_path,
            'model': model_name,
            'setting': 'oracle_passed',
            'acc(stage1)': compute_average(oracle_stage1),
            'acc(stage2)': compute_average(orc_s2),
            'acc(stage3)': compute_average(orc_s3),
            'acc(stage4)': compute_average(orc_s4),
            'depth': compute_average(depth_list),
        })

    return summary_rows


def save_summary_csv(model_eval_path: str, summary_rows: List[Dict[str, Any]]) -> str:
    """Save the summary results as CSV and return the saved path."""
    csv_path = os.path.join(model_eval_path, "model_performance_summary.csv")
    fieldnames = [
        'path', 'model', 'setting',
        'acc(stage1)', 'acc(stage2)', 'acc(stage3)', 'acc(stage4)',
        'depth',
    ]
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return csv_path


def build_contour_revision_detail_rows(performance_metrics: Dict[str, Any], model_name: str) -> List[Dict[str, Any]]:
    """Detailed analysis rows for contour_revision (qa_path=revision_required, oracle_passed view)."""
    rr = performance_metrics.get('revision_required', {})
    cr_detail = rr.get('contour_revision_qa_detail', {})
    if not cr_detail:
        return []
    return [{
        'path': 'revision_required',
        'model': model_name,
        'acc(expansion)': compute_average(cr_detail.get('expansion', [])),
        'acc(contraction)': compute_average(cr_detail.get('contraction', [])),
        'acc(revision_result)': compute_average(cr_detail.get('revision_result', [])),
    }]


def build_attribute_extraction_detail_rows(performance_metrics: Dict[str, Any], model_name: str) -> List[Dict[str, Any]]:
    """Detailed analysis rows for attribute_extraction (sum of revision_required + revision_free, oracle_passed view)."""
    ae_merged: Dict[str, List[Any]] = {
        'distribution': [],
        'location': [],
        'severity/measurement': [],
        'comparison': [],
    }
    for qa_path in ['revision_required', 'revision_free']:
        ae_detail = performance_metrics.get(qa_path, {}).get('attribute_extraction_qa_detail', {})
        for key in ae_merged:
            ae_merged[key].extend(ae_detail.get(key, []))
    return [{
        'path': 'revision_required+revision_free',
        'model': model_name,
        'acc(distribution)': compute_average(ae_merged['distribution']),
        'acc(location)': compute_average(ae_merged['location']),
        'acc(severity/measurement)': compute_average(ae_merged['severity/measurement']),
        'acc(comparison)': compute_average(ae_merged['comparison']),
    }]


def save_contour_revision_detail_csv(model_eval_path: str, rows: List[Dict[str, Any]]) -> Optional[str]:
    """Save the detailed contour_revision results as CSV."""
    if not rows:
        return None
    csv_path = os.path.join(model_eval_path, "model_performance_detail_contour_revision.csv")
    fieldnames = ['path', 'model', 'acc(expansion)', 'acc(contraction)', 'acc(revision_result)']
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def save_attribute_extraction_detail_csv(model_eval_path: str, rows: List[Dict[str, Any]]) -> Optional[str]:
    """Save the detailed attribute_extraction results as CSV."""
    if not rows:
        return None
    csv_path = os.path.join(model_eval_path, "model_performance_detail_attribute_extraction.csv")
    fieldnames = ['path', 'model', 'acc(distribution)', 'acc(location)', 'acc(severity/measurement)', 'acc(comparison)']
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


# ---------------------------------------------------------------------------
# Single-model analysis helper
# ---------------------------------------------------------------------------

def analyze_single_model(
    output_path: str,
    provider: str,
    model_name: str,
    oracle_setting: str,
    limit: Optional[int] = None,
) -> Tuple[Optional[Dict], List[Dict], List[Dict], List[Dict]]:
    """Run analysis for one model. Returns (performance_metrics, summary_rows, cr_detail_rows, ae_detail_rows)."""
    model_eval_path = os.path.join(output_path, f"{provider}_{model_name}", f"oracle_{oracle_setting}")
    results_path = os.path.join(model_eval_path, "all_results.json")
    if not os.path.exists(results_path):
        print(f"  [skip] No all_results.json found: {results_path}")
        return None, [], [], []

    all_results = json.load(open(results_path, 'r'))
    all_items = list(all_results.items())
    if limit is not None:
        all_items = all_items[:limit]

    performance_metrics = init_performance_metrics()
    for key_id, result in all_items:
        accumulate_result_for_case(performance_metrics, result, key_id)
    finalize_stage_averages(performance_metrics)

    summary_rows = build_summary_rows(performance_metrics, model_name)
    cr_detail_rows = build_contour_revision_detail_rows(performance_metrics, model_name)
    ae_detail_rows = build_attribute_extraction_detail_rows(performance_metrics, model_name)

    # per-model CSVs
    save_summary_csv(model_eval_path, summary_rows)
    save_contour_revision_detail_csv(model_eval_path, cr_detail_rows)
    save_attribute_extraction_detail_csv(model_eval_path, ae_detail_rows)

    return performance_metrics, summary_rows, cr_detail_rows, ae_detail_rows


# ---------------------------------------------------------------------------
# Multi-model discovery
# ---------------------------------------------------------------------------

def discover_models(output_path: str, oracle_setting: str) -> List[Tuple[str, str]]:
    """Scan output_path for {provider}_{model}/oracle_{oracle_setting}/all_results.json.

    Returns list of (provider, model_name) tuples.
    """
    found = []
    if not os.path.isdir(output_path):
        return found
    for entry in sorted(os.listdir(output_path)):
        entry_path = os.path.join(output_path, entry, f"oracle_{oracle_setting}", "all_results.json")
        if os.path.exists(entry_path):
            parts = entry.split('_', 1)
            if len(parts) == 2:
                found.append((parts[0], parts[1]))
    return found


# ---------------------------------------------------------------------------
# Combined CSV save
# ---------------------------------------------------------------------------

def save_all_models_csv(save_dir: str, summary_rows: List[Dict], cr_rows: List[Dict], ae_rows: List[Dict]) -> None:
    """Save combined CSVs for all models to save_dir."""
    os.makedirs(save_dir, exist_ok=True)

    if summary_rows:
        csv_path = os.path.join(save_dir, "all_models_summary.csv")
        fieldnames = ['path', 'model', 'setting', 'acc(stage1)', 'acc(stage2)', 'acc(stage3)', 'acc(stage4)', 'depth']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved: {csv_path}")

    if cr_rows:
        csv_path = os.path.join(save_dir, "all_models_contour_revision_detail.csv")
        fieldnames = ['path', 'model', 'acc(expansion)', 'acc(contraction)', 'acc(revision_result)']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cr_rows)
        print(f"Saved: {csv_path}")

    if ae_rows:
        csv_path = os.path.join(save_dir, "all_models_attribute_extraction_detail.csv")
        fieldnames = ['path', 'model', 'acc(distribution)', 'acc(location)', 'acc(severity/measurement)', 'acc(comparison)']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ae_rows)
        print(f"Saved: {csv_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(
    results_dir: str,
    provider: str = "opensource",
    model_name: str = "medgemma",
    limit: Optional[int] = None,
    oracle_setting: str = "explicit",
    all_models: bool = False,
):
    output_path = results_dir

    if all_models:
        # ---- Batch mode: discover and analyze every model ----
        discovered = discover_models(output_path, oracle_setting)
        if not discovered:
            print(f"No models found under {output_path} with oracle_{oracle_setting}")
            return

        print(f"Found {len(discovered)} model(s): {[f'{p}_{m}' for p, m in discovered]}\n")

        all_summary: List[Dict] = []
        all_cr_detail: List[Dict] = []
        all_ae_detail: List[Dict] = []

        for prov, mname in discovered:
            print(f"  Analyzing {prov}/{mname} ...")
            _, summary_rows, cr_rows, ae_rows = analyze_single_model(
                output_path, prov, mname, oracle_setting, limit
            )
            all_summary.extend(summary_rows)
            all_cr_detail.extend(cr_rows)
            all_ae_detail.extend(ae_rows)

        save_dir = os.path.join(current_dir, f"all_models_oracle_{oracle_setting}")
        save_all_models_csv(save_dir, all_summary, all_cr_detail, all_ae_detail)

        # Visualizations
        plot_stage_accuracy_table(all_summary,
                                  os.path.join(save_dir, "plot_stage_accuracy.png"),
                                  oracle_setting)
        plot_contour_revision_detail(all_cr_detail,
                                     os.path.join(save_dir, "plot_contour_revision_detail.png"))
        plot_attribute_extraction_detail(all_ae_detail,
                                         os.path.join(save_dir, "plot_attribute_extraction_detail.png"))
        plot_depth_heatmap(all_summary,
                           os.path.join(save_dir, "plot_depth_heatmap.png"))

    else:
        # ---- Single model mode ----
        model_eval_path = os.path.join(output_path, f"{provider}_{model_name}", f"oracle_{oracle_setting}")
        performance_metrics, summary_rows, cr_detail_rows, ae_detail_rows = analyze_single_model(
            output_path, provider, model_name, oracle_setting, limit
        )
        if performance_metrics is None:
            return

        print(performance_metrics)
        print(f"Saved model performance summary to: {model_eval_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Aggregate Stage 03 evaluation results across models.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results-dir', type=str, default=DEFAULT_RESULTS_DIR,
        help='Stage 03 output directory containing <provider>_<model>/oracle_<setting>/all_results.json',
    )
    parser.add_argument('--provider', type=str, default='opensource',
                        choices=['opensource', 'openai', 'azure', 'gemini', 'claude', 'dummy'],
                        help='LLM provider (single-model mode only)')
    parser.add_argument('--model', type=str, default='qwen3.5-moe',
                        help='Model name (single-model mode only)')
    parser.add_argument('--limit', type=int,
                        help='Limit number of QA entries (for testing)')
    parser.add_argument('--all_models', action='store_true', default=True,
                        help='Automatically discover and analyze all models in the output folder')
    parser.add_argument(
        '--oracle_setting',
        type=str,
        default='implicit',
        choices=['explicit', 'implicit', 'none'],
        help="Oracle mode: 'explicit', 'implicit', or 'none'.",
    )

    args = parser.parse_args()
    main(
        results_dir=args.results_dir,
        provider=args.provider,
        model_name=args.model,
        limit=args.limit,
        oracle_setting=args.oracle_setting,
        all_models=args.all_models,
    )