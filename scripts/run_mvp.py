from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch

from mechcmp.analysis import average_heatmaps, cka_heatmap_from_activations, save_heatmap
from mechcmp.config import ExperimentConfig, ModelConfig, TaskConfig
from mechcmp.models import build_model
from mechcmp.probing import run_layerwise_probing
from mechcmp.tasks import build_task_datasets
from mechcmp.training import collect_layer_activations, make_loader, set_seed, train_model
from scripts.activation_patching import fit_linear_projection

try:
    from mechcmp import mlx_backend

    MLX_AVAILABLE = True
except Exception:
    mlx_backend = None
    MLX_AVAILABLE = False


def _make_seeded_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def default_tasks() -> list[TaskConfig]:
    return [
        TaskConfig(
            name="modular_addition",
            train_size=10000,
            val_size=512,
            seq_len=6,
            vocab_size=13,
            num_classes=10,
        ),
        TaskConfig(
            name="dyck_1",
            train_size=10000,
            val_size=512,
            seq_len=14,
            vocab_size=3,
            num_classes=2,
        ),
        TaskConfig(
            name="dyck_2",
            train_size=10000,
            val_size=512,
            seq_len=14,
            vocab_size=5,
            num_classes=2,
        ),
        TaskConfig(
            name="induction",
            train_size=10000,
            val_size=512,
            seq_len=6,
            vocab_size=9,
            num_classes=4,
        ),
        TaskConfig(
            name="associative_recall",
            train_size=10000,
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
    ci_low, ci_high = _bootstrap_mean_ci(values, seed=0)
    return {
        "mean": round(mean(values), 4),
        "std": round(pstdev(values), 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
    }


def default_architectures() -> list[str]:
    return ["transformer", "lstm", "gru"]


def default_model_seeds() -> list[int]:
    return [42, 43, 44, 45, 46, 47, 48, 49]


def _resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "mlx" if MLX_AVAILABLE else "torch"
    if backend == "mlx" and not MLX_AVAILABLE:
        raise ValueError("backend=mlx requested but mlx is not available")
    if backend not in {"torch", "mlx"}:
        raise ValueError(f"Unsupported backend: {backend}")
    return backend


def _bootstrap_mean_ci(
    values: list[float],
    seed: int,
    num_bootstrap_samples: int = 1000,
) -> tuple[float, float]:
    if not values:
        raise ValueError("values must contain at least one element")
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        only = float(array[0])
        return only, only
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(
        0, array.size, size=(num_bootstrap_samples, array.size)
    )
    bootstrap_means = array[bootstrap_indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(ci_low), float(ci_high)


def _round_matrix(matrix: np.ndarray, digits: int = 4) -> list[list[float]]:
    return np.round(matrix.astype(np.float64), digits).tolist()


def _summarize_heatmaps(
    heatmaps: list[np.ndarray],
    seed: int,
    num_bootstrap_samples: int = 500,
) -> dict[str, object]:
    if not heatmaps:
        raise ValueError("heatmaps must contain at least one matrix")
    stack = np.stack(heatmaps, axis=0).astype(np.float64)
    mean_heatmap = stack.mean(axis=0)
    std_heatmap = stack.std(axis=0)
    if stack.shape[0] == 1:
        ci_low_heatmap = mean_heatmap.copy()
        ci_high_heatmap = mean_heatmap.copy()
    else:
        rng = np.random.default_rng(seed)
        bootstrap_indices = rng.integers(
            0, stack.shape[0], size=(num_bootstrap_samples, stack.shape[0])
        )
        bootstrap_means = stack[bootstrap_indices].mean(axis=1)
        ci_low_heatmap, ci_high_heatmap = np.quantile(
            bootstrap_means, [0.025, 0.975], axis=0
        )
    scalar_scores = [float(np.mean(heatmap)) for heatmap in heatmaps]
    scalar_ci_low, scalar_ci_high = _bootstrap_mean_ci(scalar_scores, seed=seed)
    return {
        "mean": _round_matrix(mean_heatmap),
        "std": _round_matrix(std_heatmap),
        "ci_low": _round_matrix(ci_low_heatmap),
        "ci_high": _round_matrix(ci_high_heatmap),
        "scalar_mean": round(float(mean(scalar_scores)), 4),
        "scalar_std": round(float(pstdev(scalar_scores)), 4),
        "scalar_ci_low": round(scalar_ci_low, 4),
        "scalar_ci_high": round(scalar_ci_high, 4),
    }


def _train_model_for_seed(
    arch: str,
    task: TaskConfig,
    model_config: ModelConfig,
    exp_config: ExperimentConfig,
    train_ds: object,
    val_loader: object,
    model_seed: int,
    lr: Optional[float] = None,
    warmup_steps: int = 0,
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
    summary = train_model(
        model,
        train_loader,
        val_loader,
        exp_config,
        lr=lr,
        warmup_steps=warmup_steps,
    )
    return model, summary


def _train_model_for_seed_mlx(
    arch: str,
    task: TaskConfig,
    model_config: ModelConfig,
    exp_config: ExperimentConfig,
    train_tokens: np.ndarray,
    train_labels: np.ndarray,
    val_tokens: np.ndarray,
    val_labels: np.ndarray,
    model_seed: int,
    lr: Optional[float] = None,
    warmup_steps: int = 0,
):
    if mlx_backend is None:
        raise RuntimeError("MLX backend requested but mlx_backend is unavailable")
    mlx_backend.set_seed(model_seed)
    model = mlx_backend.build_model(
        arch,
        vocab_size=task.vocab_size,
        num_classes=task.num_classes,
        config=model_config,
        max_seq_len=task.seq_len,
    )
    summary = mlx_backend.train_model(
        model,
        train_tokens=train_tokens,
        train_labels=train_labels,
        val_tokens=val_tokens,
        val_labels=val_labels,
        batch_size=exp_config.batch_size,
        config=exp_config,
        seed=model_seed,
        lr=lr,
        warmup_steps=warmup_steps,
    )
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


def _save_probe_table(
    probe_rows: list[dict[str, float | int | str]],
    output_path: Path,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "seed",
        "architecture",
        "layer",
        "target_name",
        "num_classes",
        "train_accuracy",
        "val_accuracy",
    ]
    with output_path.open("w", encoding="ascii", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(probe_rows)
    return output_path.name


def _save_convergence_table(
    convergence_rows: list[dict[str, float | str]],
    output_path: Path,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "architecture_pair",
        "cross_cka_mean",
        "within_baseline_mean",
        "convergence_index",
    ]
    with output_path.open("w", encoding="ascii", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(convergence_rows)
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
    backend: str = "torch",
) -> dict[str, Any]:
    if not exp_config.seeds:
        raise ValueError("ExperimentConfig.seeds must contain at least one seed")
    backend = _resolve_backend(backend)
    set_seed(exp_config.seed)
    if backend == "mlx" and mlx_backend is not None:
        mlx_backend.set_seed(exp_config.seed)
    output_dir = Path(exp_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    architectures = default_architectures()

    metrics_rows: list[dict[str, float | int | str]] = []
    probe_rows: list[dict[str, float | int | str]] = []
    convergence_rows: list[dict[str, float | str]] = []
    summary: dict[str, dict[str, object]] = {}
    for task in exp_config.tasks:
        train_ds, val_ds = build_task_datasets(task, exp_config.seed)
        if backend == "mlx":
            train_tokens, train_labels = mlx_backend.dataset_to_arrays(train_ds)
            val_tokens, val_labels = mlx_backend.dataset_to_arrays(val_ds)
        else:
            train_probe_loader = make_loader(
                train_ds, exp_config.batch_size, shuffle=False
            )
            val_loader = make_loader(val_ds, exp_config.batch_size, shuffle=False)
        per_seed: dict[str, dict[str, float]] = {}
        activations_by_arch: dict[str, list[dict[str, object]]] = {
            arch: [] for arch in architectures
        }
        accuracies_by_arch: dict[str, list[float]] = {arch: [] for arch in architectures}
        probe_results_by_arch: dict[str, dict[str, dict[str, list[float]]]] = {
            arch: {} for arch in architectures
        }
        probe_target_name = ""
        probe_num_classes = 0
        cross_heatmaps_by_pair: dict[str, list[object]] = {}
        cross_labels_by_pair: dict[str, tuple[list[str], list[str]]] = {}
        within_scores_by_arch: dict[str, list[float]] = {arch: [] for arch in architectures}

        for model_seed in exp_config.seeds:
            seed_activations: dict[str, dict[str, object]] = {}
            seed_metrics: dict[str, float] = {}
            for arch in architectures:
                # Grokking protocol for Transformer on modular addition
                if arch == "transformer" and task.name == "modular_addition":
                    lr = 1e-3
                    warmup_steps = 50
                    epochs = 500
                    weight_decay = 1.0
                elif arch == "transformer":
                    lr = 3e-4
                    warmup_steps = 100
                    epochs = exp_config.epochs
                    weight_decay = exp_config.weight_decay
                else:
                    lr = None
                    warmup_steps = 0
                    epochs = exp_config.epochs
                    weight_decay = exp_config.weight_decay

                # Temporarily override exp_config epochs for this model
                original_epochs = exp_config.epochs
                exp_config.epochs = epochs
                original_wd = exp_config.weight_decay
                exp_config.weight_decay = weight_decay
                
                try:
                    if backend == "mlx":
                        model, model_summary = _train_model_for_seed_mlx(
                            arch,
                            task,
                            model_config,
                            exp_config,
                            train_tokens,
                            train_labels,
                            val_tokens,
                            val_labels,
                            model_seed,
                            lr=lr,
                            warmup_steps=warmup_steps,
                        )
                        train_activations = mlx_backend.collect_layer_activations(
                            model,
                            train_tokens,
                            batch_size=exp_config.batch_size,
                            seed=model_seed,
                        )
                        val_activations = mlx_backend.collect_layer_activations(
                            model,
                            val_tokens,
                            batch_size=exp_config.batch_size,
                            seed=model_seed,
                        )
                    else:
                        model, model_summary = _train_model_for_seed(
                            arch,
                            task,
                            model_config,
                            exp_config,
                            train_ds,
                            val_loader,
                            model_seed,
                            lr=lr,
                            warmup_steps=warmup_steps,
                        )
                        train_activations = collect_layer_activations(
                            model, train_probe_loader, exp_config.device
                        )
                        val_activations = collect_layer_activations(
                            model, val_loader, exp_config.device
                        )
                finally:
                    exp_config.epochs = original_epochs
                    exp_config.weight_decay = original_wd
                target_name, target_num_classes, probe_metrics = run_layerwise_probing(
                    task_name=task.name,
                    train_dataset=train_ds,
                    val_dataset=val_ds,
                    train_activations=train_activations,
                    val_activations=val_activations,
                    seed=model_seed,
                )
                probe_target_name = target_name
                probe_num_classes = target_num_classes
                for layer_name, layer_metrics in probe_metrics.items():
                    probe_results_by_arch[arch].setdefault(
                        layer_name,
                        {"train_accuracy": [], "val_accuracy": []},
                    )
                    probe_results_by_arch[arch][layer_name]["train_accuracy"].append(
                        layer_metrics.train_accuracy
                    )
                    probe_results_by_arch[arch][layer_name]["val_accuracy"].append(
                        layer_metrics.val_accuracy
                    )
                    probe_rows.append(
                        {
                            "task": task.name,
                            "seed": model_seed,
                            "architecture": arch,
                            "layer": layer_name,
                            "target_name": target_name,
                            "num_classes": target_num_classes,
                            "train_accuracy": round(layer_metrics.train_accuracy, 6),
                            "val_accuracy": round(layer_metrics.val_accuracy, 6),
                        }
                    )
                activations_by_arch[arch].append(val_activations)
                seed_activations[arch] = val_activations
                if backend == "torch":
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
            heatmap_stats = _summarize_heatmaps(
                cross_heatmaps_by_pair[pair_name],
                seed=exp_config.seed + len(task.name) + len(pair_name),
            )
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
                "heatmap_mean": heatmap_stats["mean"],
                "heatmap_std": heatmap_stats["std"],
                "heatmap_ci_low": heatmap_stats["ci_low"],
                "heatmap_ci_high": heatmap_stats["ci_high"],
                "mean_cka": heatmap_stats["scalar_mean"],
                "mean_cka_std": heatmap_stats["scalar_std"],
                "mean_cka_ci_low": heatmap_stats["scalar_ci_low"],
                "mean_cka_ci_high": heatmap_stats["scalar_ci_high"],
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
                within_scores_by_arch[arch_name].append(float(np.mean(heatmap)))
            if pairwise_heatmaps and y_labels is not None and x_labels is not None:
                heatmap_stats = _summarize_heatmaps(
                    pairwise_heatmaps,
                    seed=exp_config.seed + len(task.name) + len(arch_name),
                )
                heatmap_filename = _save_mean_heatmap(
                    pairwise_heatmaps,
                    y_labels,
                    x_labels,
                    output_dir / f"{task.name}_{arch_name}_within_seed_baseline_cka.png",
                    f"{task.name}: {arch_name} within-architecture baseline",
                )
                mean_cka = heatmap_stats["scalar_mean"]
                mean_cka_std = heatmap_stats["scalar_std"]
                mean_cka_ci_low = heatmap_stats["scalar_ci_low"]
                mean_cka_ci_high = heatmap_stats["scalar_ci_high"]
                heatmap_mean = heatmap_stats["mean"]
                heatmap_std = heatmap_stats["std"]
                heatmap_ci_low = heatmap_stats["ci_low"]
                heatmap_ci_high = heatmap_stats["ci_high"]
            else:
                heatmap_filename = None
                mean_cka = None
                mean_cka_std = None
                mean_cka_ci_low = None
                mean_cka_ci_high = None
                heatmap_mean = None
                heatmap_std = None
                heatmap_ci_low = None
                heatmap_ci_high = None
            within_architecture[arch_name] = {
                "num_seed_pairs": len(pairwise_heatmaps),
                "heatmap_path": heatmap_filename,
                "val_accuracy": _summarize(accuracies_by_arch[arch_name]),
                "mean_cka": mean_cka,
                "mean_cka_std": mean_cka_std,
                "mean_cka_ci_low": mean_cka_ci_low,
                "mean_cka_ci_high": mean_cka_ci_high,
                "heatmap_mean": heatmap_mean,
                "heatmap_std": heatmap_std,
                "heatmap_ci_low": heatmap_ci_low,
                "heatmap_ci_high": heatmap_ci_high,
            }

        convergence_index: dict[str, dict[str, float | None]] = {}
        for arch_a, arch_b in combinations(architectures, 2):
            pair_name = _pair_name(arch_a, arch_b)
            cross_mean_cka = cross_architecture[pair_name]["mean_cka"]
            within_a = within_architecture[arch_a]["mean_cka"]
            within_b = within_architecture[arch_b]["mean_cka"]
            if within_a is None or within_b is None:
                within_baseline_geom_mean = None
                pair_convergence_index = None
            else:
                within_baseline_geom_mean = round(math.sqrt(within_a * within_b), 4)
                pair_convergence_index = (
                    round(cross_mean_cka / within_baseline_geom_mean, 4)
                    if within_baseline_geom_mean > 0
                    else 0.0
                )
                convergence_rows.append(
                    {
                        "task": task.name,
                        "architecture_pair": pair_name,
                        "cross_cka_mean": round(cross_mean_cka, 4),
                        "within_baseline_mean": within_baseline_geom_mean,
                        "convergence_index": pair_convergence_index,
                    }
                )
            convergence_index[pair_name] = {
                "cross_cka_mean": round(cross_mean_cka, 4),
                "within_baseline_mean": within_baseline_geom_mean,
                "convergence_index": pair_convergence_index,
            }

        probing: dict[str, dict[str, object]] = {}
        for arch_name in architectures:
            probing[arch_name] = {}
            for layer_name, layer_metrics in probe_results_by_arch[arch_name].items():
                probing[arch_name][layer_name] = {
                    "target_name": probe_target_name,
                    "num_classes": probe_num_classes,
                    "train_accuracy": _summarize(layer_metrics["train_accuracy"]),
                    "val_accuracy": _summarize(layer_metrics["val_accuracy"]),
                }

        # 4.6 Activation Patching (LSTM -> GRU on Dyck-2)
        patching_results: dict[str, dict[str, float]] = {}
        if task.name == "dyck_2":
            # Pick first seed for patching experiment
            model_seed = exp_config.seeds[0]
            set_seed(model_seed)
            lstm_model = build_model("lstm", task.vocab_size, task.num_classes, model_config, task.seq_len).to(exp_config.device)
            gru_model = build_model("gru", task.vocab_size, task.num_classes, model_config, task.seq_len).to(exp_config.device)
            trans_model = build_model("transformer", task.vocab_size, task.num_classes, model_config, task.seq_len).to(exp_config.device)
            
            train_loader = make_loader(train_ds, exp_config.batch_size, shuffle=True)
            val_loader = make_loader(val_ds, exp_config.batch_size, shuffle=False)
            
            # Ensure models are trained
            train_model(lstm_model, train_loader, val_loader, exp_config)
            train_model(gru_model, train_loader, val_loader, exp_config)
            train_model(trans_model, train_loader, val_loader, exp_config)
            
            fit_size = len(val_ds) // 2
            fit_loader = make_loader(torch.utils.data.Subset(val_ds, range(fit_size)), batch_size=exp_config.batch_size, shuffle=False)
            test_loader = make_loader(torch.utils.data.Subset(val_ds, range(fit_size, len(val_ds))), batch_size=exp_config.batch_size, shuffle=False)
            
            def get_layer_acts(model, loader):
                model.eval()
                acts = []
                with torch.no_grad():
                    for x, _ in loader:
                        res = model(x.to(exp_config.device), return_activations=True)
                        acts.append(res.activations['layer_1'].mean(dim=1).cpu().numpy())
                return np.concatenate(acts, axis=0)
            
            lstm_fit_acts = get_layer_acts(lstm_model, fit_loader)
            gru_fit_acts = get_layer_acts(gru_model, fit_loader)
            trans_fit_acts = get_layer_acts(trans_model, fit_loader)
            
            proj_gru = fit_linear_projection(gru_fit_acts, lstm_fit_acts).to(exp_config.device)
            proj_trans = fit_linear_projection(trans_fit_acts, lstm_fit_acts).to(exp_config.device)
            
            def eval_patching(source_model, projection=None, is_random=False):
                correct = 0
                total = 0
                lstm_model.eval()
                if source_model: source_model.eval()
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(exp_config.device), y.to(exp_config.device)
                        if is_random:
                            B, S = x.shape
                            projected = torch.randn(B, S, model_config.d_model).to(exp_config.device)
                        elif source_model is not None:
                            res_source = source_model(x, return_activations=True)
                            source_l1 = res_source.activations['layer_1']
                            B, S, D = source_l1.shape
                            projected = projection(source_l1.reshape(B*S, D)).reshape(B, S, -1)
                        else:
                            # Original LSTM
                            logits = lstm_model(x)
                            correct += (logits.argmax(dim=-1) == y).sum().item()
                            total += y.size(0)
                            continue
                        
                        logits = lstm_model(x, replacement_activations={'layer_1': projected})
                        correct += (logits.argmax(dim=-1) == y).sum().item()
                        total += y.size(0)
                return correct / total

            patching_results = {
                "unpatched": eval_patching(None, is_random=False),
                "gru_source": eval_patching(gru_model, proj_gru),
                "trans_source": eval_patching(trans_model, proj_trans),
                "random_source": eval_patching(None, is_random=True)
            }

        summary[task.name] = {
            "dataset_seed": exp_config.seed,
            "model_seeds": exp_config.seeds,
            "per_seed": per_seed,
            "architectures": architectures,
            "cross_architecture": cross_architecture,
            "within_architecture": within_architecture,
            "probe_target_name": probe_target_name,
            "probe_num_classes": probe_num_classes,
            "probing": probing,
            "convergence_index": convergence_index,
            "patching_results": patching_results,
        }

    metrics_table_path = _save_metrics_table(
        metrics_rows,
        output_dir / "metrics_table.csv",
    )
    probe_table_path = _save_probe_table(
        probe_rows,
        output_dir / "probe_metrics.csv",
    )
    convergence_table_path = _save_convergence_table(
        convergence_rows,
        output_dir / "convergence_table.csv",
    )
    with (output_dir / "summary.json").open("w", encoding="ascii") as f:
        json.dump(
            {
                "metrics_table_path": metrics_table_path,
                "probe_table_path": probe_table_path,
                "convergence_table_path": convergence_table_path,
                "summary_by_task": summary,
            },
            f,
            indent=2,
        )
    return {
        "metrics_table_path": metrics_table_path,
        "probe_table_path": probe_table_path,
        "convergence_table_path": convergence_table_path,
        "summary_by_task": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the MechInterpret cross-architecture comparison benchmark."
    )
    parser.add_argument("--seed", type=int, default=42, help="Dataset seed.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=default_model_seeds(),
        help="Model seeds to train and aggregate over.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device. Defaults to cuda when available, else cpu.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="AdamW weight decay."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where experiment outputs will be written.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        help="Backend to use: torch, mlx, or auto.",
    )
    parser.add_argument("--d-model", type=int, default=128, help="Hidden size.")
    parser.add_argument(
        "--num-layers", type=int, default=2, help="Number of model layers."
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads for the Transformer.",
    )
    args = parser.parse_args()
    if args.device is not None:
        device = args.device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_config = ExperimentConfig(
        seed=args.seed,
        seeds=args.seeds,
        device=device,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        tasks=default_tasks(),
    )
    model_config = ModelConfig(
        name="matched_small",
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    summary = run_experiment(exp_config, model_config, backend=args.backend)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
