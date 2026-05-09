import pytest
import torch

from mechcmp.config import ModelConfig
from mechcmp.models import PositionalEncoding, build_model


def test_positional_encoding_supports_odd_dimensions() -> None:
    module = PositionalEncoding(d_model=5, max_len=8)
    x = torch.zeros(2, 8, 5)

    output = module(x)

    assert output.shape == (2, 8, 5)


def test_positional_encoding_rejects_sequence_lengths_above_max_len() -> None:
    module = PositionalEncoding(d_model=4, max_len=8)
    x = torch.zeros(2, 9, 4)

    with pytest.raises(ValueError, match="exceeds positional encoding max_len"):
        module(x)


@pytest.mark.parametrize("arch", ["transformer", "lstm", "gru"])
def test_models_return_logits_and_layer_activations(arch: str) -> None:
    config = ModelConfig(name="test", d_model=12, num_layers=2, num_heads=3)
    model = build_model(
        arch=arch,
        vocab_size=16,
        num_classes=4,
        config=config,
        max_seq_len=6,
    )
    tokens = torch.randint(0, 16, (3, 6))

    output = model(tokens, return_activations=True)

    assert output.logits.shape == (3, 4)
    assert list(output.activations.keys()) == ["embedding", "layer_1", "layer_2"]
    assert all(acts.shape == (3, 6, 12) for acts in output.activations.values())


def test_transformer_requires_head_compatibility() -> None:
    config = ModelConfig(name="bad", d_model=10, num_layers=2, num_heads=3)

    with pytest.raises(ValueError, match="divisible by num_heads"):
        build_model(
            arch="transformer",
            vocab_size=16,
            num_classes=4,
            config=config,
            max_seq_len=6,
        )
