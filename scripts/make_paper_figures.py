from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


TASK_ORDER = [
    "modular_addition",
    "dyck_1",
    "dyck_2",
    "induction",
    "associative_recall",
]
TASK_LABELS = {
    "modular_addition": "Modular Addition",
    "dyck_1": "Dyck-1",
    "dyck_2": "Dyck-2",
    "induction": "Induction",
    "associative_recall": "Associative Recall",
}
TASK_CHANCE = {
    "modular_addition": 0.10,
    "dyck_1": 0.50,
    "dyck_2": 0.50,
    "induction": 0.25,
    "associative_recall": 0.25,
}
ARCH_ORDER = ["transformer", "lstm", "gru"]
ARCH_LABELS = {
    "transformer": "Transformer",
    "lstm": "LSTM",
    "gru": "GRU",
}
LAYER_ORDER = ["embedding", "layer_1", "layer_2"]
LAYER_LABELS = {
    "embedding": "Embedding",
    "layer_1": "Layer 1",
    "layer_2": "Layer 2",
}
PAIR_ORDER = [
    ("transformer", "lstm"),
    ("transformer", "gru"),
    ("lstm", "gru"),
]
PANEL_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 16,
            "legend.fontsize": 9,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _read_summary(results_dir: Path) -> dict[str, Any]:
    summary_path = results_dir / "summary.json"
    with summary_path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def _read_metrics(results_dir: Path, metrics_table_path: str) -> list[dict[str, str]]:
    metrics_path = results_dir / metrics_table_path
    with metrics_path.open("r", encoding="ascii", newline="") as handle:
        return list(csv.DictReader(handle))


def _ordered_tasks(summary_by_task: dict[str, Any]) -> list[str]:
    configured = [task for task in TASK_ORDER if task in summary_by_task]
    extras = sorted(task for task in summary_by_task if task not in configured)
    return configured + extras


def _short_layer_labels() -> list[str]:
    return ["Emb.", "L1", "L2"]


def _panel_label(index: int) -> str:
    if index < len(PANEL_LABELS):
        return PANEL_LABELS[index]
    return f"P{index + 1}"


def _add_panel_badge(ax: plt.Axes, index: int) -> None:
    ax.text(
        0.01,
        0.99,
        _panel_label(index),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2.0},
    )


def _save_figure(fig: plt.Figure, output_stem: Path) -> dict[str, str]:
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": png_path.name, "pdf": pdf_path.name}


def _pair_key(arch_a: str, arch_b: str) -> str:
    return f"{arch_a}_vs_{arch_b}"


def _draw_heatmap(
    ax: plt.Axes,
    matrix: list[list[float]] | np.ndarray,
    show_x_labels: bool,
    show_y_labels: bool,
    y_axis_label: str | None = None,
) -> Any:
    array = np.asarray(matrix, dtype=float)
    im = ax.imshow(array, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    ticks = np.arange(len(_short_layer_labels()))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(_short_layer_labels() if show_x_labels else [])
    ax.set_yticklabels(_short_layer_labels() if show_y_labels else [])
    if show_x_labels:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.tick_params(length=0, pad=2)
    if y_axis_label:
        ax.set_ylabel(y_axis_label, rotation=90, labelpad=18)
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            value = array[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.58 else "black",
                fontsize=8,
                fontweight="medium",
            )
    return im


def build_cross_architecture_figure(
    summary_by_task: dict[str, Any],
    results_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    tasks = _ordered_tasks(summary_by_task)
    fig, axes = plt.subplots(
        len(tasks),
        len(PAIR_ORDER),
        figsize=(10.5, 3.1 * len(tasks)),
        constrained_layout=True,
    )
    axes_array = np.atleast_2d(np.array(axes, dtype=object))
    panel_index = 0
    shared_im = None
    for row, task_name in enumerate(tasks):
        task_summary = summary_by_task[task_name]
        for col, (arch_a, arch_b) in enumerate(PAIR_ORDER):
            ax = axes_array[row, col]
            pair_name = _pair_key(arch_a, arch_b)
            pair_summary = task_summary["cross_architecture"][pair_name]
            shared_im = _draw_heatmap(
                ax,
                pair_summary["heatmap_mean"],
                show_x_labels=True,
                show_y_labels=(col == 0),
                y_axis_label=TASK_LABELS.get(task_name, task_name) if col == 0 else None,
            )
            if row == 0:
                ax.set_title(f"{ARCH_LABELS[arch_a]} vs {ARCH_LABELS[arch_b]}", pad=10, fontsize=11)
            _add_panel_badge(ax, panel_index)
            panel_index += 1
    if shared_im is not None:
        fig.colorbar(shared_im, ax=axes_array.ravel().tolist(), shrink=0.82, pad=0.02, label="CKA")
    fig.suptitle("Cross-Architecture CKA Heatmaps", y=1.01)
    return _save_figure(fig, output_dir / "figure_cross_architecture_cka")


def build_within_architecture_figure(
    summary_by_task: dict[str, Any],
    results_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    tasks = _ordered_tasks(summary_by_task)
    fig, axes = plt.subplots(
        len(tasks),
        len(ARCH_ORDER),
        figsize=(10.5, 3.1 * len(tasks)),
        constrained_layout=True,
    )
    axes_array = np.atleast_2d(np.array(axes, dtype=object))
    panel_index = 0
    shared_im = None
    for row, task_name in enumerate(tasks):
        task_summary = summary_by_task[task_name]
        for col, arch_name in enumerate(ARCH_ORDER):
            ax = axes_array[row, col]
            within_summary = task_summary["within_architecture"][arch_name]
            heatmap = within_summary.get("heatmap_mean")
            if heatmap:
                shared_im = _draw_heatmap(
                    ax,
                    heatmap,
                    show_x_labels=True,
                    show_y_labels=(col == 0),
                    y_axis_label=TASK_LABELS.get(task_name, task_name) if col == 0 else None,
                )
            else:
                ax.imshow(np.ones((32, 32), dtype=np.float32), cmap="gray", vmin=0.0, vmax=1.0)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(
                    0.5,
                    0.5,
                    "No within-arch\nbaseline\n(single seed)",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#555555",
                )
            if row == 0:
                ax.set_title(ARCH_LABELS[arch_name], pad=8)
            _add_panel_badge(ax, panel_index)
            panel_index += 1
    if shared_im is not None:
        fig.colorbar(shared_im, ax=axes_array.ravel().tolist(), shrink=0.82, pad=0.02, label="CKA")
    fig.suptitle("Within-Architecture Seed Baseline CKA", y=1.01)
    return _save_figure(fig, output_dir / "figure_within_architecture_baselines")


def build_accuracy_figure(
    summary_by_task: dict[str, Any],
    metrics_rows: list[dict[str, str]],
    output_dir: Path,
) -> dict[str, str]:
    tasks = _ordered_tasks(summary_by_task)
    x = np.arange(len(tasks))
    width = 0.24
    colors = {
        "transformer": "#4C78A8",
        "lstm": "#F58518",
        "gru": "#54A24B",
    }
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    for index, arch_name in enumerate(ARCH_ORDER):
        offsets = x + (index - 1) * width
        means = [
            summary_by_task[task_name]["within_architecture"][arch_name]["val_accuracy"]["mean"]
            for task_name in tasks
        ]
        stds = [
            summary_by_task[task_name]["within_architecture"][arch_name]["val_accuracy"]["std"]
            for task_name in tasks
        ]
        ax.bar(
            offsets,
            means,
            width=width,
            label=ARCH_LABELS[arch_name],
            color=colors[arch_name],
            alpha=0.88,
            yerr=stds,
            capsize=4,
            linewidth=0,
        )
        for task_pos, task_name in zip(offsets, tasks):
            seed_points = [
                float(row["val_accuracy"])
                for row in metrics_rows
                if row["task"] == task_name and row["architecture"] == arch_name
            ]
            if seed_points:
                jitter = np.linspace(-0.03, 0.03, num=len(seed_points))
                ax.scatter(
                    np.full(len(seed_points), task_pos) + jitter,
                    seed_points,
                    color="black",
                    s=18,
                    zorder=3,
                    alpha=0.7,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(task, task) for task in tasks], rotation=0, ha="center")
    for idx, task_name in enumerate(tasks):
        chance = TASK_CHANCE.get(task_name)
        if chance is None:
            continue
        ax.hlines(
            y=chance,
            xmin=idx - 0.42,
            xmax=idx + 0.42,
            colors="#666666",
            linestyles="--",
            linewidth=1.0,
            alpha=0.75,
            zorder=2,
        )
        ax.text(
            idx,
            chance + 0.015,
            f"chance {chance:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy Across Tasks and Architectures")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    return _save_figure(fig, output_dir / "figure_accuracy_summary")


def build_probe_figure(
    summary_by_task: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    tasks = _ordered_tasks(summary_by_task)
    columns = 3
    rows = int(np.ceil(len(tasks) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 3.5 * rows),
        constrained_layout=True,
    )
    axes_array = np.atleast_2d(np.array(axes, dtype=object))
    colors = {
        "transformer": "#4C78A8",
        "lstm": "#F58518",
        "gru": "#54A24B",
    }
    for panel_index, ax in enumerate(axes_array.flat):
        if panel_index >= len(tasks):
            ax.axis("off")
            continue
        task_name = tasks[panel_index]
        task_summary = summary_by_task[task_name]
        x = np.arange(len(LAYER_ORDER))
        for arch_name in ARCH_ORDER:
            probe_summary = task_summary["probing"][arch_name]
            means = [
                probe_summary[layer_name]["val_accuracy"]["mean"]
                for layer_name in LAYER_ORDER
            ]
            stds = [
                probe_summary[layer_name]["val_accuracy"]["std"]
                for layer_name in LAYER_ORDER
            ]
            ax.plot(
                x,
                means,
                marker="o",
                linewidth=2,
                color=colors[arch_name],
                label=ARCH_LABELS[arch_name],
            )
            ax.fill_between(
                x,
                np.asarray(means) - np.asarray(stds),
                np.asarray(means) + np.asarray(stds),
                color=colors[arch_name],
                alpha=0.16,
            )
        chance = 1.0 / task_summary["probe_num_classes"]
        ax.axhline(
            chance,
            linestyle="--",
            linewidth=1.0,
            color="#666666",
            alpha=0.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([LAYER_LABELS[layer_name] for layer_name in LAYER_ORDER])
        ax.set_ylim(0.0, 1.05)
        ax.set_title(TASK_LABELS.get(task_name, task_name))
        ax.set_ylabel("Probe Accuracy")
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.text(
            0.98,
            0.03,
            f"Target: {task_summary['probe_target_name'].replace('_', ' ')}\nChance: {chance:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "#c7c7c7",
                "boxstyle": "round,pad=0.25",
            },
        )
        _add_panel_badge(ax, panel_index)
    handles, labels = axes_array.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Layer-wise Linear Probe Accuracy Across Tasks and Architectures", y=1.08)
    return _save_figure(fig, output_dir / "figure_probe_summary")


def generate_paper_figures(results_dir: Path, output_dir: Path) -> dict[str, Any]:
    _configure_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = _read_summary(results_dir)
    summary_by_task = summary_payload["summary_by_task"]
    metrics_rows = _read_metrics(results_dir, summary_payload["metrics_table_path"])
    manifest = {
        "cross_architecture": build_cross_architecture_figure(
            summary_by_task, results_dir, output_dir
        ),
        "within_architecture": build_within_architecture_figure(
            summary_by_task, results_dir, output_dir
        ),
        "accuracy_summary": build_accuracy_figure(
            summary_by_task, metrics_rows, output_dir
        ),
        "probe_summary": build_probe_figure(
            summary_by_task, output_dir
        ),
    }
    manifest_path = output_dir / "figure_manifest.json"
    with manifest_path.open("w", encoding="ascii") as handle:
        json.dump(manifest, handle, indent=2)
    manifest["manifest_path"] = manifest_path.name
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready figures from MechInterpret experiment outputs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing summary.json, metrics_table.csv, and heatmap PNGs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_figures"),
        help="Directory where publication-ready figures will be written.",
    )
    args = parser.parse_args()
    manifest = generate_paper_figures(args.results_dir, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
