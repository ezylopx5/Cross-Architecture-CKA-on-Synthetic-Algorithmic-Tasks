from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from mechcmp.analysis import (
    average_heatmaps,
    compute_cross_architecture_cka,
    cka_heatmap_from_activations,
    linear_cka,
    save_heatmap,
)
from mechcmp.config import ExperimentConfig, ModelConfig, TaskConfig
from mechcmp.models import build_model
from mechcmp.tasks import build_task_datasets
from mechcmp.training import (
    collect_layer_activations,
    evaluate,
    make_loader,
    set_seed,
    train_model,
)


def _small_task() -> TaskConfig:
    return TaskConfig(
        name="modular_addition",
        train_size=32,
        val_size=16,
        seq_len=6,
        vocab_size=13,
        num_classes=10,
    )


def test_linear_cka_is_one_for_identical_inputs() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0], [0.5, 0.25]], dtype=np.float32)

    score = linear_cka(x, x.copy())

    assert score == pytest.approx(1.0, abs=1e-6)


def test_linear_cka_rejects_mismatched_sample_counts() -> None:
    x = np.ones((4, 3), dtype=np.float32)
    y = np.ones((5, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="same number of samples"):
        linear_cka(x, y)


def test_set_seed_seeds_all_cuda_devices_when_available() -> None:
    with (
        patch("mechcmp.training.torch.cuda.is_available", return_value=True),
        patch("mechcmp.training.torch.cuda.manual_seed_all") as manual_seed_all,
    ):
        set_seed(123)

    assert any(call.args == (123,) for call in manual_seed_all.call_args_list)


def test_training_and_activation_collection_run_end_to_end() -> None:
    set_seed(0)
    task = _small_task()
    train_ds, val_ds = build_task_datasets(task, seed=0)
    train_loader = make_loader(train_ds, batch_size=8, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=8, shuffle=False)
    model = build_model(
        arch="lstm",
        vocab_size=task.vocab_size,
        num_classes=task.num_classes,
        config=ModelConfig(name="small", d_model=16, num_layers=2, num_heads=4),
        max_seq_len=task.seq_len,
    )
    config = ExperimentConfig(device="cpu", batch_size=8, epochs=1)

    summary = train_model(model, train_loader, val_loader, config)
    val_loss, val_accuracy = evaluate(model, val_loader, "cpu")
    activations = collect_layer_activations(model, val_loader, "cpu")

    assert np.isfinite(summary.train_loss)
    assert np.isfinite(val_loss)
    assert 0.0 <= summary.val_accuracy <= 1.0
    assert 0.0 <= val_accuracy <= 1.0
    assert activations["embedding"].shape == (len(val_ds), 16)
    assert activations["layer_2"].shape == (len(val_ds), 16)


def test_collect_layer_activations_restores_training_mode() -> None:
    task = _small_task()
    _train_ds, val_ds = build_task_datasets(task, seed=0)
    val_loader = make_loader(val_ds, batch_size=8, shuffle=False)
    model = build_model(
        arch="lstm",
        vocab_size=task.vocab_size,
        num_classes=task.num_classes,
        config=ModelConfig(name="small", d_model=16, num_layers=2, num_heads=4),
        max_seq_len=task.seq_len,
    )
    model.train()

    collect_layer_activations(model, val_loader, "cpu")

    assert model.training is True


def test_compute_cross_architecture_cka_restores_original_devices() -> None:
    class StubModel:
        def __init__(self, name: str) -> None:
            self.name = name
            self.devices: list[str] = []

        def to(self, device: str) -> "StubModel":
            self.devices.append(device)
            return self

    model_a = StubModel("a")
    model_b = StubModel("b")
    fake_activations = {
        "embedding": np.ones((4, 3), dtype=np.float32),
        "layer_1": np.eye(4, 3, dtype=np.float32),
    }

    with (
        patch(
            "mechcmp.analysis._get_model_device",
            side_effect=["orig_a", "orig_b"],
        ),
        patch(
            "mechcmp.analysis.collect_layer_activations",
            side_effect=[fake_activations, fake_activations],
        ),
    ):
        y_labels, x_labels, heatmap = compute_cross_architecture_cka(
            model_a, model_b, dataloader=None, device="cpu"
        )

    assert model_a.devices == ["cpu", "orig_a"]
    assert model_b.devices == ["cpu", "orig_b"]
    assert y_labels == ["embedding", "layer_1"]
    assert x_labels == ["embedding", "layer_1"]
    assert heatmap.shape == (2, 2)


def test_cka_heatmap_from_activations_matches_label_order() -> None:
    activations = {
        "embedding": np.ones((4, 3), dtype=np.float32),
        "layer_1": np.eye(4, 3, dtype=np.float32),
    }

    y_labels, x_labels, heatmap = cka_heatmap_from_activations(activations, activations)

    assert y_labels == ["embedding", "layer_1"]
    assert x_labels == ["embedding", "layer_1"]
    assert heatmap.shape == (2, 2)


def test_cross_architecture_cka_and_heatmap_export(tmp_path: Path) -> None:
    set_seed(0)
    task = _small_task()
    _train_ds, val_ds = build_task_datasets(task, seed=1)
    val_loader = make_loader(val_ds, batch_size=8, shuffle=False)
    config = ModelConfig(name="small", d_model=16, num_layers=2, num_heads=4)
    transformer = build_model("transformer", task.vocab_size, task.num_classes, config, task.seq_len)
    lstm = build_model("lstm", task.vocab_size, task.num_classes, config, task.seq_len)

    y_labels, x_labels, heatmap = compute_cross_architecture_cka(
        transformer, lstm, val_loader, "cpu"
    )
    output_path = tmp_path / "heatmap.png"
    save_heatmap(heatmap, y_labels, x_labels, output_path, "test heatmap")

    assert heatmap.shape == (3, 3)
    assert np.isfinite(heatmap).all()
    assert output_path.exists()


def test_average_heatmaps_computes_elementwise_mean() -> None:
    heatmap_a = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    heatmap_b = np.array([[0.0, 1.0], [0.5, 0.0]], dtype=np.float32)

    mean_heatmap = average_heatmaps([heatmap_a, heatmap_b])

    assert np.allclose(mean_heatmap, np.array([[0.5, 0.5], [0.5, 0.25]], dtype=np.float32))


def test_make_loader_can_reproduce_shuffles_with_seeded_generators() -> None:
    task = _small_task()
    train_ds, _val_ds = build_task_datasets(task, seed=0)
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)

    loader_a = make_loader(train_ds, batch_size=8, shuffle=True, generator=generator_a)
    loader_b = make_loader(train_ds, batch_size=8, shuffle=True, generator=generator_b)

    batches_a = [batch_tokens.tolist() for batch_tokens, _labels in loader_a]
    batches_b = [batch_tokens.tolist() for batch_tokens, _labels in loader_b]

    assert batches_a == batches_b
