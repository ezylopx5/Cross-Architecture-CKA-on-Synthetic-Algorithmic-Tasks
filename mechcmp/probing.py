from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .tasks import SequenceClassificationDataset


@dataclass(frozen=True)
class ProbeTarget:
    name: str
    labels: np.ndarray
    num_classes: int


@dataclass(frozen=True)
class ProbeMetrics:
    train_accuracy: float
    val_accuracy: float


def _labels_from_dataset(dataset: SequenceClassificationDataset) -> np.ndarray:
    return np.asarray([example.label for example in dataset.examples], dtype=np.int64)


def _tokens_from_dataset(dataset: SequenceClassificationDataset) -> np.ndarray:
    return np.asarray([example.tokens for example in dataset.examples], dtype=np.int64)


def _infer_probe_num_classes(
    task_name: str, dataset: SequenceClassificationDataset
) -> int:
    tokens = _tokens_from_dataset(dataset)
    if tokens.shape[0] == 0:
        raise ValueError("Dataset must contain at least one example for probing")
    if task_name == "modular_addition":
        return int(tokens[0, 1])
    if task_name == "induction":
        return int(tokens[0, -2] // 2)
    if task_name == "associative_recall":
        return int((tokens[0, -3] - 1) // 2)
    raise ValueError(f"No fixed probe class-count rule for task: {task_name}")


def _max_dyck_depth(tokens: np.ndarray, num_bracket_types: int) -> np.ndarray:
    max_depths = np.zeros(tokens.shape[0], dtype=np.int64)
    pad_token = num_bracket_types * 2
    for example_index, sequence in enumerate(tokens):
        depth = 0
        max_depth = 0
        stack: list[int] = []
        for token in sequence.tolist():
            if token == pad_token:
                continue
            if token % 2 == 0:
                stack.append(token // 2)
                depth += 1
                max_depth = max(max_depth, depth)
            elif stack and stack[-1] == (token - 1) // 2:
                stack.pop()
                depth = max(depth - 1, 0)
            else:
                depth = 0
                stack.clear()
        max_depths[example_index] = max_depth
    return max_depths


def build_probe_target(
    task_name: str, dataset: SequenceClassificationDataset
) -> ProbeTarget:
    labels = _labels_from_dataset(dataset)
    if task_name == "modular_addition":
        return ProbeTarget(
            name="answer_class",
            labels=labels,
            num_classes=_infer_probe_num_classes(task_name, dataset),
        )
    if task_name == "dyck_1":
        depth_labels = _max_dyck_depth(_tokens_from_dataset(dataset), num_bracket_types=1)
        return ProbeTarget(
            name="max_depth",
            labels=depth_labels,
            num_classes=int(depth_labels.max()) + 1,
        )
    if task_name == "dyck_2":
        depth_labels = _max_dyck_depth(_tokens_from_dataset(dataset), num_bracket_types=2)
        return ProbeTarget(
            name="max_depth",
            labels=depth_labels,
            num_classes=int(depth_labels.max()) + 1,
        )
    if task_name == "induction":
        return ProbeTarget(
            name="target_value",
            labels=labels,
            num_classes=_infer_probe_num_classes(task_name, dataset),
        )
    if task_name == "associative_recall":
        return ProbeTarget(
            name="target_value",
            labels=labels,
            num_classes=_infer_probe_num_classes(task_name, dataset),
        )
    raise ValueError(f"Unsupported task for probing: {task_name}")


def _standardize(
    train_x: np.ndarray, val_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (val_x - mean) / std


def fit_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    num_classes: int,
    seed: int,
) -> ProbeMetrics:
    if train_x.ndim != 2 or val_x.ndim != 2:
        raise ValueError("Probe features must be 2D")
    if train_x.shape[1] != val_x.shape[1]:
        raise ValueError("Train and validation feature dimensions must match")
    if num_classes < 2:
        raise ValueError("Probe target must have at least 2 classes")
    train_x, val_x = _standardize(train_x.astype(np.float32), val_x.astype(np.float32))
    torch.manual_seed(seed)
    model = nn.Linear(train_x.shape[1], num_classes)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=100,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )
    criterion = nn.CrossEntropyLoss()
    train_features = torch.from_numpy(train_x)
    train_labels = torch.from_numpy(train_y.astype(np.int64))
    val_features = torch.from_numpy(val_x)
    val_labels = torch.from_numpy(val_y.astype(np.int64))

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_features)
        loss = criterion(logits, train_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        train_predictions = model(train_features).argmax(dim=-1)
        val_predictions = model(val_features).argmax(dim=-1)
    return ProbeMetrics(
        train_accuracy=float((train_predictions == train_labels).float().mean().item()),
        val_accuracy=float((val_predictions == val_labels).float().mean().item()),
    )


def run_layerwise_probing(
    task_name: str,
    train_dataset: SequenceClassificationDataset,
    val_dataset: SequenceClassificationDataset,
    train_activations: dict[str, np.ndarray],
    val_activations: dict[str, np.ndarray],
    seed: int,
) -> tuple[str, int, dict[str, ProbeMetrics]]:
    train_target = build_probe_target(task_name, train_dataset)
    val_target = build_probe_target(task_name, val_dataset)
    if train_target.name != val_target.name:
        raise ValueError("Probe targets must match across train and validation datasets")
    num_classes = max(train_target.num_classes, val_target.num_classes)
    probe_metrics: dict[str, ProbeMetrics] = {}
    for layer_name, train_x in train_activations.items():
        if layer_name not in val_activations:
            raise ValueError(f"Missing validation activations for layer {layer_name}")
        probe_metrics[layer_name] = fit_linear_probe(
            train_x=train_x,
            train_y=train_target.labels,
            val_x=val_activations[layer_name],
            val_y=val_target.labels,
            num_classes=num_classes,
            seed=seed,
        )
    return train_target.name, num_classes, probe_metrics
