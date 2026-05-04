"""Visualization functions for model performance analysis."""

from typing import Dict, List, Any

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


PATH_DISPLAY = {
    'revision_required': 'Rev.Required',
    'revision_free':     'Rev.Free',
    'lesion_free':       'LesionFree',
}


def _pct(v: float) -> str:
    return f"{round(v * 100, 1):.1f}"


def plot_stage_accuracy_table(summary_rows: List[Dict], save_path: str, oracle_setting: str) -> None:
    """Render stage accuracy as a formatted table image (per qa_path × model × setting)."""
    if not HAS_MATPLOTLIB:
        print("[warn] matplotlib not available; skipping visualization.")
        return

    models = sorted({r['model'] for r in summary_rows})
    qa_paths = ['revision_required', 'revision_free', 'lesion_free']
    settings = ['end_to_end', 'oracle_passed']
    stage_cols = ['acc(stage1)', 'acc(stage2)', 'acc(stage3)', 'acc(stage4)', 'depth']
    stage_labels = ['Stage1\n(Detection)', 'Stage2\n(ContourEval)', 'Stage3\n(Revision)', 'Stage4\n(Attribute)', 'Depth']

    # Build lookup: data[model][path][setting] = row
    data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in summary_rows:
        data.setdefault(r['model'], {}).setdefault(r['path'], {})[r['setting']] = r

    n_paths = len(qa_paths)
    fig, axes = plt.subplots(n_paths, 1, figsize=(max(10, len(models) * 2.5), 3.5 * n_paths))
    if n_paths == 1:
        axes = [axes]

    fig.suptitle(f'Stage Accuracy by Model  [oracle: {oracle_setting}]', fontsize=14, fontweight='bold')

    for ax, qa_path in zip(axes, qa_paths):
        ax.axis('off')

        # Build header row
        col_headers = ['Model', 'Setting'] + stage_labels
        rows_data = []

        for model in models:
            model_data = data.get(model, {}).get(qa_path, {})
            for setting in settings:
                row = model_data.get(setting, {})
                row_vals = [model, setting]
                for col in stage_cols:
                    v = row.get(col, 0)
                    row_vals.append(_pct(v) if col != 'depth' else f"{v:.2f}")
                rows_data.append(row_vals)

        if not rows_data:
            ax.set_title(PATH_DISPLAY.get(qa_path, qa_path), fontsize=12, fontweight='bold', pad=4)
            continue

        n_rows = len(rows_data)
        n_cols = len(col_headers)

        # Alternate row colors
        cell_colors = []
        for i, row_vals in enumerate(rows_data):
            setting = row_vals[1]
            if setting == 'end_to_end':
                base = '#D6E4F7'
            else:
                base = '#FAD9D9'
            bg = base if i % 2 == 0 else '#FFFFFF'
            cell_colors.append([bg] * n_cols)

        table = ax.table(
            cellText=rows_data,
            colLabels=col_headers,
            cellLoc='center',
            loc='center',
            cellColours=cell_colors,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)

        # Style header
        for col_idx in range(n_cols):
            cell = table[0, col_idx]
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')

        ax.set_title(PATH_DISPLAY.get(qa_path, qa_path), fontsize=12, fontweight='bold', pad=4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_contour_revision_detail(cr_rows: List[Dict], save_path: str) -> None:
    """Bar chart for contour revision sub-tasks (expansion / contraction / revision_result)."""
    if not HAS_MATPLOTLIB or not cr_rows:
        return

    models = [r['model'] for r in cr_rows]
    sub_keys = ['acc(expansion)', 'acc(contraction)', 'acc(revision_result)']
    sub_labels = ['Expansion', 'Contraction', 'Rev.Result']
    colors = ['#4C72B0', '#DD8452', '#55A868']

    x = np.arange(len(models))
    bar_w = 0.22
    offsets = [-bar_w, 0, bar_w]

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.2), 4.5))
    ax.set_title('Contour Revision Detail (oracle-passed)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')
    ax.yaxis.grid(True, linewidth=0.4, linestyle=':')
    ax.set_axisbelow(True)

    for key, label, color, offset in zip(sub_keys, sub_labels, colors, offsets):
        vals = [float(_pct(r.get(key, 0))) for r in cr_rows]
        bars = ax.bar(x + offset, vals, width=bar_w * 0.9, color=color, alpha=0.85, label=label)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{v:.0f}', ha='center', va='bottom', fontsize=8)

    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_attribute_extraction_detail(ae_rows: List[Dict], save_path: str) -> None:
    """Bar chart for attribute extraction sub-tasks."""
    if not HAS_MATPLOTLIB or not ae_rows:
        return

    models = [r['model'] for r in ae_rows]
    sub_keys = ['acc(distribution)', 'acc(location)', 'acc(severity/measurement)', 'acc(comparison)']
    sub_labels = ['Distribution', 'Location', 'Severity/Meas.', 'Comparison']
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    x = np.arange(len(models))
    bar_w = 0.18
    n = len(sub_keys)
    offsets = np.linspace(-(n - 1) * bar_w / 2, (n - 1) * bar_w / 2, n)

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.3), 4.5))
    ax.set_title('Attribute Extraction Detail (oracle-passed)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right')
    ax.yaxis.grid(True, linewidth=0.4, linestyle=':')
    ax.set_axisbelow(True)

    for key, label, color, offset in zip(sub_keys, sub_labels, colors, offsets):
        vals = [float(_pct(r.get(key, 0))) for r in ae_rows]
        bars = ax.bar(x + offset, vals, width=bar_w * 0.9, color=color, alpha=0.85, label=label)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{v:.0f}', ha='center', va='bottom', fontsize=7)

    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_depth_heatmap(summary_rows: List[Dict], save_path: str) -> None:
    """Heatmap: avg depth per model × qa_path (end-to-end setting)."""
    if not HAS_MATPLOTLIB:
        return

    models = sorted({r['model'] for r in summary_rows})
    qa_paths = ['revision_required', 'revision_free', 'lesion_free']

    depth_matrix = np.zeros((len(qa_paths), len(models)))
    for r in summary_rows:
        if r['setting'] != 'end_to_end':
            continue
        if r['model'] in models and r['path'] in qa_paths:
            ri = qa_paths.index(r['path'])
            ci = models.index(r['model'])
            depth_matrix[ri, ci] = r.get('depth', 0)

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 0.9), 3.5))
    im = ax.imshow(depth_matrix, aspect='auto', cmap='YlGn', vmin=0)
    ax.set_title('Avg. Depth (end-to-end) per Model × QA-Path', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=9)
    ax.set_yticks(range(len(qa_paths)))
    ax.set_yticklabels([PATH_DISPLAY.get(p, p) for p in qa_paths])

    for ri in range(len(qa_paths)):
        for ci in range(len(models)):
            v = depth_matrix[ri, ci]
            ax.text(ci, ri, f'{v:.2f}', ha='center', va='center',
                    fontsize=8, color='black' if v < 2.5 else 'white')

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
