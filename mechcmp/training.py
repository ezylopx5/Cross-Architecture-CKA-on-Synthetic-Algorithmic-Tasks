from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .models import ForwardResult


@dataclass
class TrainingSummary:
    train_loss: float
    val_loss: float
    val_accuracy: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def evaluate(
    model: nn.Module, dataloader: DataLoader, device: str
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        for tokens, labels in dataloader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            logits = model(tokens)
            loss = criterion(logits, labels)
            total_loss += loss.item() * tokens.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_items += tokens.size(0)
    if total_items == 0:
        raise ValueError("dataloader produced no items during evaluation")
    return total_loss / total_items, total_correct / total_items


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ExperimentConfig,
) -> TrainingSummary:
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    train_loss = 0.0
    for _epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_items = 0
        for tokens, labels in train_loader:
            tokens = tokens.to(config.device)
            labels = labels.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * tokens.size(0)
            total_items += tokens.size(0)
        if total_items == 0:
            raise ValueError("train_loader produced no items during training")
        train_loss = total_loss / total_items
    val_loss, val_accuracy = evaluate(model, val_loader, config.device)
    return TrainingSummary(
        train_loss=train_loss,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
    )


def collect_layer_activations(
    model: nn.Module, dataloader: DataLoader, device: str
) -> dict[str, np.ndarray]:
    was_training = model.training
    model.eval()
    collected: dict[str, list[np.ndarray]] = {}
    try:
        with torch.no_grad():
            for tokens, _labels in dataloader:
                tokens = tokens.to(device)
                output = model(tokens, return_activations=True)
                assert isinstance(output, ForwardResult)
                for name, acts in output.activations.items():
                    pooled = acts.mean(dim=1).detach().cpu().numpy()
                    collected.setdefault(name, []).append(pooled)
    finally:
        model.train(was_training)
    if not collected:
        raise ValueError("dataloader produced no activations")
    return {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}
