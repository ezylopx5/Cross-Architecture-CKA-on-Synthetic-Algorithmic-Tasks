from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .config import TaskConfig


@dataclass
class SequenceExample:
    tokens: list[int]
    label: int


class SequenceClassificationDataset(Dataset):
    def __init__(self, examples: list[SequenceExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[idx]
        x = torch.tensor(example.tokens, dtype=torch.long)
        y = torch.tensor(example.label, dtype=torch.long)
        return x, y


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def build_modular_addition_examples(
    count: int, modulus: int = 10, seed: int = 0
) -> list[SequenceExample]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if modulus < 2:
        raise ValueError("modulus must be at least 2")
    rng = _rng(seed)
    plus_token = modulus
    eq_token = modulus + 1
    pad_token = modulus + 2
    examples: list[SequenceExample] = []
    for _ in range(count):
        a = rng.randrange(modulus)
        b = rng.randrange(modulus)
        tokens = [a, plus_token, b, eq_token, pad_token, pad_token]
        label = (a + b) % modulus
        examples.append(SequenceExample(tokens=tokens, label=label))
    return examples


def build_dyck1_examples(
    count: int, seq_len: int = 12, max_depth: int = 4, seed: int = 0
) -> list[SequenceExample]:
    return build_dyck_examples(
        count=count,
        seq_len=seq_len,
        num_bracket_types=1,
        max_depth=max_depth,
        seed=seed,
    )


def build_dyck_examples(
    count: int,
    seq_len: int,
    num_bracket_types: int,
    max_depth: int = 4,
    seed: int = 0,
) -> list[SequenceExample]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    if num_bracket_types < 1:
        raise ValueError("num_bracket_types must be positive")
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    rng = _rng(seed)
    pad = num_bracket_types * 2
    examples: list[SequenceExample] = []
    for _ in range(count):
        stack: list[int] = []
        tokens: list[int] = []
        valid = True
        for _step in range(seq_len):
            can_open = len(stack) < max_depth
            can_close = len(stack) > 0
            if not can_close:
                token = _open_token(rng.randrange(num_bracket_types))
            elif not can_open:
                token = _close_token(stack[-1])
            else:
                if rng.random() < 0.5:
                    token = _open_token(rng.randrange(num_bracket_types))
                else:
                    token = _close_token(stack[-1])
            tokens.append(token)
            if token % 2 == 0:
                stack.append(token // 2)
            elif stack and stack[-1] == (token - 1) // 2:
                stack.pop()
            else:
                valid = False
                stack.clear()
        if stack:
            valid = False
        if rng.random() < 0.3:
            flip_index = rng.randrange(seq_len)
            tokens[flip_index] = rng.randrange(num_bracket_types * 2)
            valid = _is_balanced_multitype(tokens, num_bracket_types)
        examples.append(SequenceExample(tokens=tokens + [pad, pad], label=int(valid)))
    return examples


def _open_token(bracket_type: int) -> int:
    return bracket_type * 2


def _close_token(bracket_type: int) -> int:
    return (bracket_type * 2) + 1


def _is_balanced_multitype(tokens: list[int], num_bracket_types: int) -> bool:
    stack: list[int] = []
    for token in tokens:
        if token < 0 or token >= num_bracket_types * 2:
            return False
        if token % 2 == 0:
            stack.append(token // 2)
        elif not stack or stack[-1] != (token - 1) // 2:
            return False
        else:
            stack.pop()
    return not stack


def build_induction_examples(
    count: int, vocab_size: int = 16, num_pairs: int = 2, seed: int = 0
) -> list[SequenceExample]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if vocab_size < 4 or vocab_size % 2 != 0:
        raise ValueError("vocab_size must be an even number >= 4")
    if num_pairs < 1:
        raise ValueError("num_pairs must be positive")
    rng = _rng(seed)
    query_token = vocab_size
    examples: list[SequenceExample] = []
    keys = list(range(vocab_size // 2))
    values = list(range(vocab_size // 2, vocab_size))
    if num_pairs > len(keys):
        raise ValueError("num_pairs exceeds the number of available keys")
    for _ in range(count):
        chosen_keys = rng.sample(keys, k=num_pairs)
        chosen_values = rng.sample(values, k=num_pairs)
        mapping = dict(zip(chosen_keys, chosen_values))
        tokens: list[int] = []
        for key in chosen_keys:
            tokens.extend([key, mapping[key]])
        query = rng.choice(chosen_keys)
        tokens.extend([query_token, query])
        label = mapping[query] - (vocab_size // 2)
        examples.append(SequenceExample(tokens=tokens, label=label))
    return examples


def build_associative_recall_examples(
    count: int, vocab_size: int = 16, num_pairs: int = 3, seed: int = 0
) -> list[SequenceExample]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if vocab_size < 4 or vocab_size % 2 != 0:
        raise ValueError("vocab_size must be an even number >= 4")
    if num_pairs < 1:
        raise ValueError("num_pairs must be positive")
    rng = _rng(seed)
    pair_sep = vocab_size
    query_token = vocab_size + 1
    pad_token = vocab_size + 2
    examples: list[SequenceExample] = []
    keys = list(range(vocab_size // 2))
    values = list(range(vocab_size // 2, vocab_size))
    if num_pairs > len(keys):
        raise ValueError("num_pairs exceeds the number of available keys")
    for _ in range(count):
        chosen_keys = rng.sample(keys, k=num_pairs)
        chosen_values = rng.sample(values, k=num_pairs)
        mapping = dict(zip(chosen_keys, chosen_values))
        tokens: list[int] = []
        for key in chosen_keys:
            tokens.extend([key, mapping[key], pair_sep])
        query = rng.choice(chosen_keys)
        tokens.extend([query_token, query, pad_token])
        label = mapping[query] - (vocab_size // 2)
        examples.append(SequenceExample(tokens=tokens, label=label))
    return examples


def build_task_datasets(
    config: TaskConfig, seed: int
) -> tuple[SequenceClassificationDataset, SequenceClassificationDataset]:
    if config.train_size < 1 or config.val_size < 1:
        raise ValueError("train_size and val_size must be positive")
    if config.seq_len < 1:
        raise ValueError("seq_len must be positive")
    if config.vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    if config.num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if config.name == "modular_addition":
        required_vocab_size = config.num_classes + 3
        required_seq_len = 6
        if config.vocab_size < required_vocab_size:
            raise ValueError(
                f"modular_addition requires vocab_size >= {required_vocab_size}"
            )
        if config.seq_len != required_seq_len:
            raise ValueError(
                f"modular_addition requires seq_len == {required_seq_len}"
            )
    elif config.name == "dyck_1":
        if config.vocab_size < 3:
            raise ValueError("dyck_1 requires vocab_size >= 3")
        if config.seq_len < 3:
            raise ValueError("dyck_1 requires seq_len >= 3")
    elif config.name == "dyck_2":
        if config.vocab_size < 5:
            raise ValueError("dyck_2 requires vocab_size >= 5")
        if config.seq_len < 3:
            raise ValueError("dyck_2 requires seq_len >= 3")
    elif config.name == "induction":
        if config.vocab_size < 5:
            raise ValueError("induction requires vocab_size >= 5")
        if (config.vocab_size - 1) % 2 != 0:
            raise ValueError(
                "induction requires config.vocab_size - 1 to be an even number"
            )
        available_labels = (config.vocab_size - 1) // 2
        if config.num_classes < available_labels:
            raise ValueError(
                "induction requires num_classes >= (vocab_size - 1) // 2"
            )
        num_pairs = config.num_pairs
        if num_pairs is None:
            if config.seq_len < 4 or (config.seq_len - 2) % 2 != 0:
                raise ValueError(
                    "induction requires seq_len of the form 2 * num_pairs + 2"
                )
            num_pairs = (config.seq_len - 2) // 2
        elif config.seq_len != (2 * num_pairs) + 2:
            raise ValueError("induction requires seq_len == 2 * num_pairs + 2")
        if num_pairs > (config.vocab_size - 1) // 2:
            raise ValueError(
                "induction num_pairs exceeds the number of available key tokens"
            )
    elif config.name == "associative_recall":
        if config.vocab_size < 7:
            raise ValueError("associative_recall requires vocab_size >= 7")
        if (config.vocab_size - 3) % 2 != 0:
            raise ValueError(
                "associative_recall requires config.vocab_size - 3 to be an even number"
            )
        num_pairs = config.num_pairs
        if num_pairs is None:
            if config.seq_len < 6 or (config.seq_len - 3) % 3 != 0:
                raise ValueError(
                    "associative_recall requires seq_len of the form 3 * num_pairs + 3"
                )
            num_pairs = (config.seq_len - 3) // 3
        elif config.seq_len != (3 * num_pairs) + 3:
            raise ValueError(
                "associative_recall requires seq_len == 3 * num_pairs + 3"
            )
        available_keys = (config.vocab_size - 3) // 2
        if config.num_classes != available_keys:
            raise ValueError(
                "associative_recall requires num_classes == (vocab_size - 3) // 2"
            )
        if num_pairs > available_keys:
            raise ValueError(
                "associative_recall num_pairs exceeds the number of available key tokens"
            )
    builders = {
        "modular_addition": lambda size, s: build_modular_addition_examples(
            size, modulus=config.num_classes, seed=s
        ),
        "dyck_1": lambda size, s: build_dyck1_examples(
            size, seq_len=config.seq_len - 2, seed=s
        ),
        "dyck_2": lambda size, s: build_dyck_examples(
            size,
            seq_len=config.seq_len - 2,
            num_bracket_types=2,
            seed=s,
        ),
        "induction": lambda size, s: build_induction_examples(
            size,
            vocab_size=config.vocab_size - 1,
            num_pairs=(config.seq_len - 2) // 2 if config.num_pairs is None else config.num_pairs,
            seed=s,
        ),
        "associative_recall": lambda size, s: build_associative_recall_examples(
            size,
            vocab_size=config.vocab_size - 3,
            num_pairs=(config.seq_len - 3) // 3 if config.num_pairs is None else config.num_pairs,
            seed=s,
        ),
    }
    if config.name not in builders:
        raise ValueError(f"Unsupported task: {config.name}")
    train_examples = builders[config.name](config.train_size, seed)
    val_examples = builders[config.name](config.val_size, seed + 1)
    return (
        SequenceClassificationDataset(train_examples),
        SequenceClassificationDataset(val_examples),
    )
