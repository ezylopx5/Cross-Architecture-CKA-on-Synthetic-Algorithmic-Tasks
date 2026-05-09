from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .training import collect_layer_activations


def _center_gram(gram: np.ndarray) -> np.ndarray:
    n = gram.shape[0]
    unit = np.ones((n, n), dtype=gram.dtype) / n
    return gram - unit @ gram - gram @ unit + unit @ gram @ unit


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must both be 2D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    gram_x = _center_gram(x @ x.T)
    gram_y = _center_gram(y @ y.T)
    numerator = np.sum(gram_x * gram_y)
    denominator = np.sqrt(np.sum(gram_x * gram_x) * np.sum(gram_y * gram_y))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _get_model_device(model: torch.nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


def cka_heatmap_from_activations(
    acts_a: dict[str, np.ndarray],
    acts_b: dict[str, np.ndarray],
) -> tuple[list[str], list[str], np.ndarray]:
    layers_a = list(acts_a.keys())
    layers_b = list(acts_b.keys())
    heatmap = np.zeros((len(layers_a), len(layers_b)), dtype=np.float32)
    for row, layer_a in enumerate(layers_a):
        for col, layer_b in enumerate(layers_b):
            heatmap[row, col] = linear_cka(acts_a[layer_a], acts_b[layer_b])
    return layers_a, layers_b, heatmap


def cka_heatmap_from_activations(
    activations_a: dict[str, np.ndarray],
    activations_b: dict[str, np.ndarray],
) -> tuple[list[str], list[str], np.ndarray]:
    if not activations_a or not activations_b:
        raise ValueError("Activation dictionaries must be non-empty")
    layers_a = list(activations_a.keys())
    layers_b = list(activations_b.keys())
    heatmap = np.zeros((len(layers_a), len(layers_b)), dtype=np.float32)
    for row, layer_a in enumerate(layers_a):
        for col, layer_b in enumerate(layers_b):
            heatmap[row, col] = linear_cka(
                activations_a[layer_a],
                activations_b[layer_b],
            )
    return layers_a, layers_b, heatmap


def compute_cross_architecture_cka(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> tuple[list[str], list[str], np.ndarray]:
    original_device_a = _get_model_device(model_a)
    original_device_b = _get_model_device(model_b)
    try:
        model_a = model_a.to(device)
        model_b = model_b.to(device)
        acts_a = collect_layer_activations(model_a, dataloader, device)
        acts_b = collect_layer_activations(model_b, dataloader, device)
    finally:
        model_a.to(original_device_a)
        model_b.to(original_device_b)
    return cka_heatmap_from_activations(acts_a, acts_b)


def average_heatmaps(
    heatmaps: Iterable[np.ndarray],
) -> np.ndarray:
    heatmaps = list(heatmaps)
    if not heatmaps:
        raise ValueError("heatmaps must contain at least one matrix")
    reference_shape = heatmaps[0].shape
    if any(heatmap.shape != reference_shape for heatmap in heatmaps):
        raise ValueError("all heatmaps must have the same shape")
    return np.mean(np.stack(heatmaps, axis=0), axis=0)


def save_heatmap(
    heatmap: np.ndarray,
    y_labels: list[str],
    x_labels: list[str],
    output_path: str | Path,
    title: str,
) -> None:
    if heatmap.shape != (len(y_labels), len(x_labels)):
        raise ValueError("heatmap shape must match the provided axis labels")
    
    # Clean labels: layer_1 -> Layer 1, embedding -> Embedding
    def clean_label(label: str) -> str:
        return label.replace("_", " ").title()

    y_labels = [clean_label(l) for l in y_labels]
    x_labels = [clean_label(l) for l in x_labels]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heatmap, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)
    for row in range(heatmap.shape[0]):
        for col in range(heatmap.shape[1]):
            ax.text(
                col,
                row,
                f"{heatmap[row, col]:.2f}",
                ha="center",
                va="center",
                color="white" if heatmap[row, col] < 0.6 else "black",
                fontsize=11,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="CKA")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
