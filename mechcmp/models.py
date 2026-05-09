from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .config import ModelConfig


@dataclass
class ForwardResult:
    logits: torch.Tensor
    activations: dict[str, torch.Tensor]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        if max_len < 1:
            raise ValueError("max_len must be positive")
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"sequence length {x.size(1)} exceeds positional encoding max_len {self.pe.size(1)}"
            )
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        config: ModelConfig,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.position = PositionalEncoding(config.d_model, max_len=max_seq_len)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.num_heads,
                    dim_feedforward=config.d_model * 4,
                    dropout=config.dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, num_classes)

    def forward(
        self, tokens: torch.Tensor, return_activations: bool = False
    ) -> ForwardResult | torch.Tensor:
        hidden = self.position(self.embedding(tokens))
        activations: dict[str, torch.Tensor] = {"embedding": hidden}
        for idx, layer in enumerate(self.layers):
            hidden = layer(hidden)
            activations[f"layer_{idx + 1}"] = hidden
        pooled = self.norm(hidden.mean(dim=1))
        logits = self.classifier(pooled)
        if return_activations:
            return ForwardResult(logits=logits, activations=activations)
        return logits


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=config.d_model,
                    hidden_size=config.d_model,
                    batch_first=True,
                    num_layers=1,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, num_classes)

    def forward(
        self, tokens: torch.Tensor, return_activations: bool = False
    ) -> ForwardResult | torch.Tensor:
        hidden = self.embedding(tokens)
        activations: dict[str, torch.Tensor] = {"embedding": hidden}
        for idx, layer in enumerate(self.layers):
            hidden, _ = layer(hidden)
            activations[f"layer_{idx + 1}"] = hidden
        pooled = self.norm(hidden.mean(dim=1))
        logits = self.classifier(pooled)
        if return_activations:
            return ForwardResult(logits=logits, activations=activations)
        return logits


class GRUClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [
                nn.GRU(
                    input_size=config.d_model,
                    hidden_size=config.d_model,
                    batch_first=True,
                    num_layers=1,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, num_classes)

    def forward(
        self, tokens: torch.Tensor, return_activations: bool = False
    ) -> ForwardResult | torch.Tensor:
        hidden = self.embedding(tokens)
        activations: dict[str, torch.Tensor] = {"embedding": hidden}
        for idx, layer in enumerate(self.layers):
            hidden, _ = layer(hidden)
            activations[f"layer_{idx + 1}"] = hidden
        pooled = self.norm(hidden.mean(dim=1))
        logits = self.classifier(pooled)
        if return_activations:
            return ForwardResult(logits=logits, activations=activations)
        return logits


def build_model(
    arch: str,
    vocab_size: int,
    num_classes: int,
    config: ModelConfig,
    max_seq_len: int,
) -> nn.Module:
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if arch == "transformer":
        return TransformerClassifier(vocab_size, num_classes, config, max_seq_len)
    if arch == "lstm":
        return LSTMClassifier(vocab_size, num_classes, config)
    if arch == "gru":
        return GRUClassifier(vocab_size, num_classes, config)
    raise ValueError(f"Unsupported architecture: {arch}")
