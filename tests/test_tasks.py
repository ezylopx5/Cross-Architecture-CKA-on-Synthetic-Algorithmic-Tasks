import pytest

from mechcmp.config import TaskConfig
from mechcmp.tasks import (
    build_associative_recall_examples,
    build_dyck_examples,
    build_dyck1_examples,
    build_induction_examples,
    build_modular_addition_examples,
    build_task_datasets,
)


def test_modular_addition_examples_are_deterministic_and_well_formed() -> None:
    examples_a = build_modular_addition_examples(count=8, modulus=10, seed=7)
    examples_b = build_modular_addition_examples(count=8, modulus=10, seed=7)

    assert examples_a == examples_b
    assert all(len(example.tokens) == 6 for example in examples_a)
    assert all(0 <= example.label < 10 for example in examples_a)


def test_dyck_examples_return_binary_labels_and_padded_sequences() -> None:
    examples = build_dyck1_examples(count=16, seq_len=10, seed=5)

    assert len(examples) == 16
    assert all(len(example.tokens) == 12 for example in examples)
    assert {example.label for example in examples}.issubset({0, 1})


def test_dyck2_examples_use_two_bracket_types_plus_pad() -> None:
    examples = build_dyck_examples(count=10, seq_len=8, num_bracket_types=2, seed=5)

    assert len(examples) == 10
    assert all(len(example.tokens) == 10 for example in examples)
    assert all(max(example.tokens) <= 4 for example in examples)


def test_induction_examples_validate_sampling_constraints() -> None:
    with pytest.raises(ValueError, match="num_pairs exceeds"):
        build_induction_examples(count=4, vocab_size=4, num_pairs=3, seed=0)


def test_associative_recall_examples_have_expected_length_and_label_range() -> None:
    examples = build_associative_recall_examples(count=6, vocab_size=8, num_pairs=2, seed=0)

    assert len(examples) == 6
    assert all(len(example.tokens) == 9 for example in examples)
    assert all(0 <= example.label < 4 for example in examples)


def test_build_task_datasets_rejects_unknown_task() -> None:
    config = TaskConfig(
        name="unknown",
        train_size=8,
        val_size=4,
        seq_len=6,
        vocab_size=10,
        num_classes=2,
    )

    with pytest.raises(ValueError, match="Unsupported task"):
        build_task_datasets(config, seed=0)


def test_build_task_datasets_returns_expected_sizes() -> None:
    config = TaskConfig(
        name="induction",
        train_size=12,
        val_size=6,
        seq_len=6,
        vocab_size=9,
        num_classes=4,
    )

    train_ds, val_ds = build_task_datasets(config, seed=0)

    assert len(train_ds) == 12
    assert len(val_ds) == 6


def test_build_task_datasets_rejects_too_small_modular_vocab() -> None:
    config = TaskConfig(
        name="modular_addition",
        train_size=8,
        val_size=4,
        seq_len=6,
        vocab_size=12,
        num_classes=10,
    )

    with pytest.raises(ValueError, match="requires vocab_size >="):
        build_task_datasets(config, seed=0)


def test_build_task_datasets_rejects_misaligned_induction_seq_len() -> None:
    config = TaskConfig(
        name="induction",
        train_size=8,
        val_size=4,
        seq_len=5,
        vocab_size=9,
        num_classes=4,
    )

    with pytest.raises(ValueError, match="requires seq_len"):
        build_task_datasets(config, seed=0)


def test_build_task_datasets_rejects_too_few_induction_classes() -> None:
    config = TaskConfig(
        name="induction",
        train_size=8,
        val_size=4,
        seq_len=6,
        vocab_size=9,
        num_classes=3,
    )

    with pytest.raises(ValueError, match="num_classes >="):
        build_task_datasets(config, seed=0)


def test_build_task_datasets_accepts_explicit_induction_num_pairs() -> None:
    config = TaskConfig(
        name="induction",
        train_size=8,
        val_size=4,
        seq_len=8,
        vocab_size=11,
        num_classes=5,
        num_pairs=3,
    )

    train_ds, val_ds = build_task_datasets(config, seed=0)

    assert len(train_ds[0][0]) == 8
    assert len(val_ds[0][0]) == 8


def test_build_task_datasets_supports_dyck2() -> None:
    config = TaskConfig(
        name="dyck_2",
        train_size=8,
        val_size=4,
        seq_len=10,
        vocab_size=5,
        num_classes=2,
    )

    train_ds, val_ds = build_task_datasets(config, seed=0)

    assert len(train_ds[0][0]) == 10
    assert len(val_ds[0][0]) == 10


def test_build_task_datasets_supports_associative_recall() -> None:
    config = TaskConfig(
        name="associative_recall",
        train_size=8,
        val_size=4,
        seq_len=12,
        vocab_size=11,
        num_classes=4,
        num_pairs=3,
    )

    train_ds, val_ds = build_task_datasets(config, seed=0)

    assert len(train_ds[0][0]) == 12
    assert len(val_ds[0][0]) == 12
