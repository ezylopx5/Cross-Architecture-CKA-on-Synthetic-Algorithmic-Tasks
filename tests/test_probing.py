import numpy as np

from mechcmp.probing import build_probe_target, fit_linear_probe, run_layerwise_probing
from mechcmp.tasks import build_task_datasets
from mechcmp.config import TaskConfig


def test_build_probe_target_uses_answer_class_for_modular_addition() -> None:
    config = TaskConfig(
        name="modular_addition",
        train_size=8,
        val_size=4,
        seq_len=6,
        vocab_size=13,
        num_classes=10,
    )
    train_ds, _ = build_task_datasets(config, seed=0)

    target = build_probe_target("modular_addition", train_ds)

    assert target.name == "answer_class"
    assert target.num_classes == 10
    assert target.labels.shape == (8,)


def test_build_probe_target_uses_depth_for_dyck() -> None:
    config = TaskConfig(
        name="dyck_2",
        train_size=8,
        val_size=4,
        seq_len=10,
        vocab_size=5,
        num_classes=2,
    )
    train_ds, _ = build_task_datasets(config, seed=0)

    target = build_probe_target("dyck_2", train_ds)

    assert target.name == "max_depth"
    assert target.num_classes >= 2
    assert target.labels.shape == (8,)


def test_build_probe_target_uses_query_token_for_associative_recall() -> None:
    config = TaskConfig(
        name="associative_recall",
        train_size=8,
        val_size=4,
        seq_len=12,
        vocab_size=11,
        num_classes=4,
        num_pairs=3,
    )
    train_ds, _ = build_task_datasets(config, seed=0)

    target = build_probe_target("associative_recall", train_ds)

    assert target.name == "target_value"
    assert target.num_classes == (config.vocab_size - 3) // 2


def test_fit_linear_probe_solves_separable_problem() -> None:
    train_x = np.array(
        [[1.0, 0.0], [1.2, 0.1], [-1.0, 0.0], [-1.1, -0.1]],
        dtype=np.float32,
    )
    train_y = np.array([1, 1, 0, 0], dtype=np.int64)
    val_x = np.array([[0.9, 0.2], [-0.8, -0.2]], dtype=np.float32)
    val_y = np.array([1, 0], dtype=np.int64)

    metrics = fit_linear_probe(train_x, train_y, val_x, val_y, num_classes=2, seed=0)

    assert metrics.train_accuracy >= 0.99
    assert metrics.val_accuracy >= 0.99


def test_run_layerwise_probing_returns_metrics_for_each_layer() -> None:
    config = TaskConfig(
        name="modular_addition",
        train_size=8,
        val_size=4,
        seq_len=6,
        vocab_size=13,
        num_classes=10,
    )
    train_ds, val_ds = build_task_datasets(config, seed=0)
    train_labels = np.array([example.label for example in train_ds.examples], dtype=np.int64)
    val_labels = train_labels[: len(val_ds)]
    for example, label in zip(val_ds.examples, val_labels.tolist(), strict=True):
        example.label = int(label)
    train_activations = {
        "embedding": np.eye(10, dtype=np.float32)[train_labels],
        "layer_1": np.eye(10, dtype=np.float32)[train_labels],
        "layer_2": np.eye(10, dtype=np.float32)[train_labels],
    }
    val_activations = {
        "embedding": np.eye(10, dtype=np.float32)[val_labels],
        "layer_1": np.eye(10, dtype=np.float32)[val_labels],
        "layer_2": np.eye(10, dtype=np.float32)[val_labels],
    }

    target_name, num_classes, metrics = run_layerwise_probing(
        task_name="modular_addition",
        train_dataset=train_ds,
        val_dataset=val_ds,
        train_activations=train_activations,
        val_activations=val_activations,
        seed=0,
    )

    assert target_name == "answer_class"
    assert num_classes == 10
    assert list(metrics.keys()) == ["embedding", "layer_1", "layer_2"]
    assert all(layer_metrics.val_accuracy >= 0.99 for layer_metrics in metrics.values())
