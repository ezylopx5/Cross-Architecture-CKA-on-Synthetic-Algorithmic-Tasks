from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

from mechcmp.analysis import average_heatmaps, cka_heatmap_from_activations, save_heatmap
from mechcmp.config import ExperimentConfig, ModelConfig, TaskConfig
from mechcmp.models import build_model
from mechcmp.tasks import build_task_datasets
from mechcmp.training import collect_layer_activations, make_loader, set_seed, train_model


def _make_seeded_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def default_tasks() -> list[TaskConfig]:
    return [
        TaskConfig(
            name="modular_addition",
            train_size=2048,
            val_size=512,
            seq_len=6,
            vocab_size=13,
            num_classes=10,
        ),
        TaskConfig(
            name="dyck_1",
            train_size=2048,
            val_size=512,
            seq_len=14,
            vocab_size=3,
            num_classes=2,
        ),
        TaskConfig(
            name="dyck_2",
            train_size=2048,
            val_size=512,
            seq_len=14,
            vocab_size=5,
            num_classes=2,
        ),
        TaskConfig(
            name="induction",
            train_size=2048,
            val_size=512,
            seq_len=6,
            vocab_size=9,
            num_classes=4,
        ),
        TaskConfig(
            name="associative_recall",
            train_size=2048,
            val_size=512,
            seq_len=12,
            vocab_size=11,
            num_classes=4,
            num_pairs=3,
        ),
    ]


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("values must contain at least one element")
    return {
        "mean": round(mean(values), 4),
        "std": round(pstdev(values), 4),
    }


def default_architectures() -> list[str]:
    return ["transformer", "lstm", "gru"]


def _train_model_for_seed(
    arch: str,
    task: TaskConfig,
    model_config: ModelConfig,
    exp_config: ExperimentConfig,
    train_ds: object,
    val_loader: object,
    model_seed: int,
):
    set_seed(model_seed)
    model = build_model(
        arch,
        vocab_size=task.vocab_size,
        num_classes=task.num_classes,
        config=model_config,
        max_seq_len=task.seq_len,
    )
    train_loader = make_loader(
        train_ds,
        exp_config.batch_size,
        shuffle=True,
        generator=_make_seeded_generator(model_seed),
    )
    summary = train_model(model, train_loader, val_loader, exp_config)
    return model, summary


def _pair_name(arch_a: str, arch_b: str) -> str:
    return f"{arch_a}_vs_{arch_b}"


def _save_metrics_table(
    metrics_rows: list[dict[str, float | int | str]],
    output_path: Path,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "seed",
        "architecture",
        "train_loss",
        "val_loss",
        "val_accuracy",
    ]
    with output_path.open("w", encoding="ascii", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)
    return output_path.name


def _save_mean_heatmap(
    heatmaps: list[torch.Tensor | object],
    y_labels: list[str],
    x_labels: list[str],
    output_path: Path,
    title: str,
) -> str:
    mean_heatmap = average_heatmaps(heatmaps)
    save_heatmap(mean_heatmap, y_labels, x_labels, output_path, title)
    return output_path.name


def run_experiment(
    exp_config: ExperimentConfig,
    model_config: ModelConfig,
) -> dict[str, Any]:
    if not exp_config.seeds:
        raise ValueError("ExperimentConfig.seeds must contain at least one seed")
    set_seed(exp_config.seed)
    output_dir = Path(exp_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    architectures = default_architectures()

    metrics_rows: list[dict[str, float | int | str]] = []
    summary: dict[str, dict[str, object]] = {}
    for task in exp_config.tasks:
        train_ds, val_ds = build_task_datasets(task, exp_config.seed)
        val_loader = make_loader(val_ds, exp_config.batch_size, shuffle=False)
        per_seed: dict[str, dict[str, float]] = {}
        activations_by_arch: dict[str, list[dict[str, object]]] = {
            arch: [] for arch in architectures
        }
        accuracies_by_arch: dict[str, list[float]] = {arch: [] for arch in architectures}
        cross_heatmaps_by_pair: dict[str, list[object]] = {}
        cross_labels_by_pair: dict[str, tuple[list[str], list[str]]] = {}

        for model_seed in exp_config.seeds:
            seed_activations: dict[str, dict[str, object]] = {}
            seed_metrics: dict[str, float] = {}
            for arch in architectures:
                model, model_summary = _train_model_for_seed(
                    arch,
                    task,
                    model_config,
                    exp_config,
                    train_ds,
                    val_loader,
                    model_seed,
                )
                activations = collect_layer_activations(model, val_loader, exp_config.device)
                activations_by_arch[arch].append(activations)
                seed_activations[arch] = activations
                model.to("cpu")
                del model
                accuracies_by_arch[arch].append(model_summary.val_accuracy)
                seed_metrics[f"{arch}_val_accuracy"] = round(model_summary.val_accuracy, 4)
                metrics_rows.append(
                    {
                        "task": task.name,
                        "seed": model_seed,
                        "architecture": arch,
                        "train_loss": round(model_summary.train_loss, 6),
                        "val_loss": round(model_summary.val_loss, 6),
                        "val_accuracy": round(model_summary.val_accuracy, 6),
                    }
                )

            for arch_a, arch_b in combinations(architectures, 2):
                y_labels, x_labels, cross_heatmap = cka_heatmap_from_activations(
                    seed_activations[arch_a],
                    seed_activations[arch_b],
                )
                pair_name = _pair_name(arch_a, arch_b)
                cross_heatmaps_by_pair.setdefault(pair_name, []).append(cross_heatmap)
                cross_labels_by_pair[pair_name] = (y_labels, x_labels)
            per_seed[str(model_seed)] = seed_metrics

        cross_architecture: dict[str, dict[str, object]] = {}
        for arch_a, arch_b in combinations(architectures, 2):
            pair_name = _pair_name(arch_a, arch_b)
            y_labels, x_labels = cross_labels_by_pair[pair_name]
            cross_heatmap_filename = _save_mean_heatmap(
                cross_heatmaps_by_pair[pair_name],
                y_labels,
                x_labels,
                output_dir / f"{task.name}_{pair_name}_cross_seed_mean_cka.png",
                f"{task.name}: {arch_a} vs {arch_b} (mean across seeds)",
            )
            cross_architecture[pair_name] = {
                f"{arch_a}_val_accuracy": _summarize(accuracies_by_arch[arch_a]),
                f"{arch_b}_val_accuracy": _summarize(accuracies_by_arch[arch_b]),
                "heatmap_path": cross_heatmap_filename,
            }

        within_architecture: dict[str, dict[str, object]] = {}
        for arch_name in architectures:
            pairwise_heatmaps = []
            y_labels: list[str] | None = None
            x_labels: list[str] | None = None
            for acts_a, acts_b in combinations(activations_by_arch[arch_name], 2):
                y_labels, x_labels, heatmap = cka_heatmap_from_activations(
                    acts_a, acts_b
                )
                pairwise_heatmaps.append(heatmap)
            if pairwise_heatmaps and y_labels is not None and x_labels is not None:
                heatmap_filename = _save_mean_heatmap(
                    pairwise_heatmaps,
                    y_labels,
                    x_labels,
                    output_dir / f"{task.name}_{arch_name}_within_seed_baseline_cka.png",
                    f"{task.name}: {arch_name} within-architecture baseline",
                )
            else:
                heatmap_filename = None
            within_architecture[arch_name] = {
                "num_seed_pairs": len(pairwise_heatmaps),
                "heatmap_path": heatmap_filename,
                "val_accuracy": _summarize(accuracies_by_arch[arch_name]),
            }

        summary[task.name] = {
            "dataset_seed": exp_config.seed,
            "model_seeds": exp_config.seeds,
            "per_seed": per_seed,
            "architectures": architectures,
            "cross_architecture": cross_architecture,
            "within_architecture": within_architecture,
        }

    metrics_table_path = _save_metrics_table(
        metrics_rows,
        output_dir / "metrics_table.csv",
    )
    with (output_dir / "summary.json").open("w", encoding="ascii") as f:
        json.dump(
            {
                "metrics_table_path": metrics_table_path,
                "summary_by_task": summary,
            },
            f,
            indent=2,
        )
    return {
        "metrics_table_path": metrics_table_path,
        "summary_by_task": summary,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_config = ExperimentConfig(
        seed=42,
        seeds=[42, 43],
        device=device,
        batch_size=64,
        lr=1e-3,
        epochs=10,
        output_dir="results",
        tasks=default_tasks(),
    )
    model_config = ModelConfig(name="matched_small", d_model=128, num_layers=2)
    run_experiment(exp_config, model_config)


if __name__ == "__main__":
    main()
