from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except Exception as exc:  # pragma: no cover - import guard for non-MLX environments
    raise ImportError("MLX backend requested but mlx is not available") from exc

from .config import ModelConfig
from .training import TrainingSummary


@dataclass
class MLXForwardResult:
    logits: mx.array
    activations: dict[str, mx.array]


def set_seed(seed: int) -> None:
    mx.random.seed(seed)


def dataset_to_arrays(dataset: object) -> tuple[np.ndarray, np.ndarray]:
    examples = getattr(dataset, "examples", None)
    if examples is None:
        raise ValueError("dataset must expose an examples attribute")
    tokens = np.asarray([example.tokens for example in examples], dtype=np.int64)
    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    return tokens, labels


def _iter_batches(
    tokens: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    indices = np.arange(len(tokens))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield tokens[batch_idx], labels[batch_idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        if max_len < 1:
            raise ValueError("max_len must be positive")
        position = np.arange(max_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model)
        )
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.pe = mx.array(pe[None, :, :])

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError(
                f"sequence length {x.shape[1]} exceeds positional encoding max_len {self.pe.shape[1]}"
            )
        return x + self.pe[:, : x.shape[1]]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        qkv = mx.reshape(qkv, (batch, seq_len, 3, self.num_heads, self.head_dim))
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale
        attn_weights = mx.softmax(attn_scores, axis=-1)
        context = mx.matmul(attn_weights, v)
        context = mx.transpose(context, (0, 2, 1, 3))
        context = mx.reshape(context, (batch, seq_len, d_model))
        return self.out_proj(context)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, d_model * 4)
        self.ffn2 = nn.Linear(d_model * 4, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array) -> mx.array:
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        ff = mx.maximum(self.ffn1(x), 0)
        ff = self.ffn2(ff)
        return self.norm2(x + ff)


class MLXTransformerClassifier(nn.Module):
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
        self.position = PositionalEncoding(config.d_model, max_seq_len)
        self.layers = [
            TransformerEncoderLayer(config.d_model, config.num_heads)
            for _ in range(config.num_layers)
        ]
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, num_classes)

    def __call__(
        self,
        tokens: mx.array,
        return_activations: bool = False,
        replacement_activations: dict[str, mx.array] | None = None,
    ) -> MLXForwardResult | mx.array:
        hidden = self.position(self.embedding(tokens))
        activations: dict[str, mx.array] = {"embedding": hidden}
        if replacement_activations and "embedding" in replacement_activations:
            hidden = replacement_activations["embedding"]

        for idx, layer in enumerate(self.layers):
            hidden = layer(hidden)
            layer_name = f"layer_{idx + 1}"
            activations[layer_name] = hidden
            if replacement_activations and layer_name in replacement_activations:
                hidden = replacement_activations[layer_name]

        pooled = self.norm(mx.mean(hidden, axis=1))
        logits = self.classifier(pooled)
        if return_activations:
            return MLXForwardResult(logits=logits, activations=activations)
        return logits


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size + hidden_size, hidden_size * 4)
        self.hidden_size = hidden_size

    def __call__(self, x: mx.array, state: tuple[mx.array, mx.array]) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        h, c = state
        combined = mx.concatenate([x, h], axis=-1)
        gates = self.linear(combined)
        i, f, o, g = mx.split(gates, 4, axis=-1)
        i = mx.sigmoid(i)
        f = mx.sigmoid(f)
        o = mx.sigmoid(o)
        g = mx.tanh(g)
        c_new = f * c + i * g
        h_new = o * mx.tanh(c_new)
        return h_new, (h_new, c_new)


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size + hidden_size, hidden_size * 3)
        self.hidden_size = hidden_size

    def __call__(self, x: mx.array, h: mx.array) -> mx.array:
        combined = mx.concatenate([x, h], axis=-1)
        gates = self.linear(combined)
        z, r, h_tilde = mx.split(gates, 3, axis=-1)
        z = mx.sigmoid(z)
        r = mx.sigmoid(r)
        candidate = mx.tanh(h_tilde + r * h)
        return (1.0 - z) * h + z * candidate


class MLXLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, config: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.layers = [
            LSTMCell(config.d_model, config.d_model) for _ in range(config.num_layers)
        ]
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, num_classes)

    def __call__(
        self,
        tokens: mx.array,
        return_activations: bool = False,
        replacement_activations: dict[str, mx.array] | None = None,
    ) -> MLXForwardResult | mx.array:
        hidden = self.embedding(tokens)
        activations: dict[str, mx.array] = {"embedding": hidden}
        if replacement_activations and "embedding" in replacement_activations:
            hidden = replacement_activations["embedding"]

        for idx, cell in enumerate(self.layers):
            hidden = _run_lstm_layer(hidden, cell)
            layer_name = f"layer_{idx + 1}"
            activations[layer_name] = hidden
            if replacement_activations and layer_name in replacement_activations:
                hidden = replacement_activations[layer_name]

        pooled = self.norm(mx.mean(hidden, axis=1))
        logits = self.classifier(pooled)
        if return_activations:
            return MLXForwardResult(logits=logits, activations=activations)
        return logits


class MLXGRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, config: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.layers = [
            GRUCell(config.d_model, config.d_model) for _ in range(config.num_layers)
        ]
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, num_classes)

    def __call__(
        self,
        tokens: mx.array,
        return_activations: bool = False,
        replacement_activations: dict[str, mx.array] | None = None,
    ) -> MLXForwardResult | mx.array:
        hidden = self.embedding(tokens)
        activations: dict[str, mx.array] = {"embedding": hidden}
        if replacement_activations and "embedding" in replacement_activations:
            hidden = replacement_activations["embedding"]

        for idx, cell in enumerate(self.layers):
            hidden = _run_gru_layer(hidden, cell)
            layer_name = f"layer_{idx + 1}"
            activations[layer_name] = hidden
            if replacement_activations and layer_name in replacement_activations:
                hidden = replacement_activations[layer_name]

        pooled = self.norm(mx.mean(hidden, axis=1))
        logits = self.classifier(pooled)
        if return_activations:
            return MLXForwardResult(logits=logits, activations=activations)
        return logits


def _run_lstm_layer(inputs: mx.array, cell: LSTMCell) -> mx.array:
    batch, seq_len, _ = inputs.shape
    h = mx.zeros((batch, cell.hidden_size))
    c = mx.zeros((batch, cell.hidden_size))
    outputs = []
    for t in range(seq_len):
        h, (h, c) = cell(inputs[:, t], (h, c))
        outputs.append(h)
    return mx.stack(outputs, axis=1)


def _run_gru_layer(inputs: mx.array, cell: GRUCell) -> mx.array:
    batch, seq_len, _ = inputs.shape
    h = mx.zeros((batch, cell.hidden_size))
    outputs = []
    for t in range(seq_len):
        h = cell(inputs[:, t], h)
        outputs.append(h)
    return mx.stack(outputs, axis=1)


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
        return MLXTransformerClassifier(vocab_size, num_classes, config, max_seq_len)
    if arch == "lstm":
        return MLXLSTMClassifier(vocab_size, num_classes, config)
    if arch == "gru":
        return MLXGRUClassifier(vocab_size, num_classes, config)
    raise ValueError(f"Unsupported architecture: {arch}")


def _cross_entropy(logits: mx.array, labels: mx.array) -> mx.array:
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    labels = labels.astype(mx.int32)
    gathered = mx.take_along_axis(log_probs, labels[:, None], axis=-1)
    return mx.mean(-mx.squeeze(gathered, axis=-1))


def _accuracy(logits: mx.array, labels: mx.array) -> tuple[int, int]:
    preds = mx.argmax(logits, axis=-1)
    correct = mx.sum(preds == labels)
    mx.eval(correct)
    return int(correct.item()), int(labels.shape[0])


def evaluate(
    model: nn.Module,
    tokens: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    for batch_tokens, batch_labels in _iter_batches(tokens, labels, batch_size, False, seed):
        batch_tokens_mx = mx.array(batch_tokens)
        batch_labels_mx = mx.array(batch_labels)
        logits = model(batch_tokens_mx)
        loss = _cross_entropy(logits, batch_labels_mx)
        correct, count = _accuracy(logits, batch_labels_mx)
        mx.eval(loss)
        total_loss += float(loss.item()) * count
        total_correct += correct
        total_items += count
    if total_items == 0:
        raise ValueError("dataloader produced no items during evaluation")
    return total_loss / total_items, total_correct / total_items


def train_model(
    model: nn.Module,
    train_tokens: np.ndarray,
    train_labels: np.ndarray,
    val_tokens: np.ndarray,
    val_labels: np.ndarray,
    batch_size: int,
    config: object,
    seed: int,
    lr: Optional[float] = None,
    warmup_steps: int = 0,
) -> TrainingSummary:
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    
    actual_lr = lr if lr is not None else config.lr
    
    # Simple linear warmup schedule for MLX
    def lr_schedule(step: int) -> mx.array:
        if step < warmup_steps:
            return mx.array(actual_lr * (float(step) / float(max(1, warmup_steps))))
        return mx.array(actual_lr)

    optimizer = optim.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
    )

    def loss_fn(model_: nn.Module, batch_tokens: mx.array, batch_labels: mx.array) -> mx.array:
        logits = model_(batch_tokens)
        return _cross_entropy(logits, batch_labels)

    global_step = 0
    train_loss = 0.0
    for _epoch in range(config.epochs):
        total_loss = 0.0
        total_items = 0
        for batch_tokens, batch_labels in _iter_batches(
            train_tokens, train_labels, batch_size, True, seed
        ):
            batch_tokens_mx = mx.array(batch_tokens)
            batch_labels_mx = mx.array(batch_labels)
            loss, grads = mx.value_and_grad(loss_fn)(
                model, batch_tokens_mx, batch_labels_mx
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            total_loss += float(loss.item()) * len(batch_tokens)
            total_items += len(batch_tokens)
            global_step += 1
        if total_items == 0:
            raise ValueError("train_loader produced no items during training")
        train_loss = total_loss / total_items

    val_loss, val_accuracy = evaluate(
        model,
        val_tokens,
        val_labels,
        batch_size=batch_size,
        seed=seed,
    )
    return TrainingSummary(
        train_loss=train_loss,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
    )


def collect_layer_activations(
    model: nn.Module,
    tokens: np.ndarray,
    batch_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {}
    dummy_labels = np.zeros((len(tokens),), dtype=np.int64)
    for batch_tokens, _batch_labels in _iter_batches(
        tokens, dummy_labels, batch_size, False, seed
    ):
        batch_tokens_mx = mx.array(batch_tokens)
        output = model(batch_tokens_mx, return_activations=True)
        if not isinstance(output, MLXForwardResult):
            raise ValueError("Expected MLXForwardResult when return_activations=True")
        for name, acts in output.activations.items():
            pooled = mx.mean(acts, axis=1)
            mx.eval(pooled)
            collected.setdefault(name, []).append(np.asarray(pooled))
    if not collected:
        raise ValueError("dataloader produced no activations")
    return {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}
