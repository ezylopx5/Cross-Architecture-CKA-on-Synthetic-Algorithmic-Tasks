from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaskConfig:
    name: str
    train_size: int
    val_size: int
    seq_len: int
    vocab_size: int
    num_classes: int
    num_pairs: Optional[int] = None


@dataclass
class ModelConfig:
    name: str
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1


@dataclass
class ExperimentConfig:
    seed: int = 42
    seeds: list[int] = field(default_factory=lambda: [42])
    device: str = "cpu"
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 12
    weight_decay: float = 1e-4
    output_dir: str = "results"
    tasks: list[TaskConfig] = field(default_factory=list)
