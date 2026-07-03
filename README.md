# Cross-Architecture CKA on Synthetic Algorithmic Tasks

> **Do architecturally distinct neural networks converge on similar internal representations when trained on the same algorithmic task?**

This repository provides a complete experimental framework for **cross-architecture mechanistic comparison** using [Centered Kernel Alignment (CKA)](https://arxiv.org/abs/1905.00414), linear probing, and activation patching. It accompanies our ICML 2026 Mechanistic Interpretability Workshop paper.

We train matched Transformer, LSTM, and GRU classifiers on five controlled synthetic tasks, then systematically compare their learned representations layer-by-layer — revealing when architectures converge on similar solutions and when their inductive biases drive them apart.

---

## Table of Contents

- [Key Features](#key-features)
- [Synthetic Tasks](#synthetic-tasks)
- [Architectures](#architectures)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running Experiments](#running-experiments)
- [Generating Paper Figures](#generating-paper-figures)
- [Apple Silicon (MLX) Support](#apple-silicon-mlx-support)
- [Running Tests](#running-tests)
- [Outputs](#outputs)
- [Citation](#citation)
- [License](#license)

---

## Key Features

- **Three architectures**: Transformer (multi-head self-attention), LSTM, and GRU — matched in depth (`num_layers=2`) and hidden dimension (`d_model=128`) for fair comparison
- **Five synthetic tasks**: Modular addition, Dyck-1, Dyck-2, induction, and associative recall — each chosen because the ground-truth algorithm is known
- **Linear CKA heatmaps**: Full layer × layer cross-architecture representational similarity, aggregated across multiple seeds with bootstrap confidence intervals
- **Within-architecture baselines**: Seed-to-seed CKA to calibrate expected representation variability
- **Convergence index**: A normalized metric (cross-CKA / geometric mean of within-CKA) quantifying representational convergence
- **Linear probing**: Layer-wise probe accuracy to assess where task-relevant information becomes linearly decodable
- **Activation patching**: Interchange interventions with learned linear projections to test causal interchangeability of representations
- **Dual backend**: Full PyTorch implementation + native Apple MLX backend for Apple Silicon acceleration
- **Publication-ready figures**: Automated composite figure generation (PNG + PDF) with panel labels, CKA heatmaps, accuracy charts, and probing curves

---

## Synthetic Tasks

| Task | # Classes | Description | What It Tests |
|------|-----------|-------------|---------------|
| **Modular Addition** | 10 | `a + b ≡ ? (mod 10)` from token sequences `[a, +, b, =, pad, pad]` | Algebraic / modular arithmetic |
| **Dyck-1** | 2 | Classify whether a sequence of single-type brackets is balanced | Counter / stack computation |
| **Dyck-2** | 2 | Classify whether a sequence with two bracket types is balanced | Hierarchical nesting structure |
| **Induction** | 4 | Given key-value pairs and a query key, predict the associated value | In-context pattern matching |
| **Associative Recall** | 4 | Retrieve the value bound to a queried key from a stored mapping | Binding / associative memory |

All tasks use controlled synthetic data generators with configurable sizes, ensuring reproducible experiments with known ground-truth algorithms.

---

## Architectures

All models share a common interface (`forward(tokens, return_activations=True)` → `ForwardResult` containing logits and a dict of per-layer activations):

| Architecture | Key Design | Layers | Pooling |
|---|---|---|---|
| **TransformerClassifier** | Multi-head self-attention + GELU FFN, sinusoidal positional encoding | `embedding` → `layer_1` → `layer_2` | Mean pool → LayerNorm → Linear head |
| **LSTMClassifier** | Stacked single-layer LSTMs | `embedding` → `layer_1` → `layer_2` | Mean pool → LayerNorm → Linear head |
| **GRUClassifier** | Stacked single-layer GRUs | `embedding` → `layer_1` → `layer_2` | Mean pool → LayerNorm → Linear head |

All models also support **`replacement_activations`** — a dict that, when provided, substitutes a layer's output during the forward pass. This enables the activation patching experiments.

---

## Methodology

```
┌──────────────────┐
│  Generate Task   │  (modular addition, Dyck-1/2, induction, associative recall)
│      Data        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Train Models    │  Transformer, LSTM, GRU × N seeds (default: 8)
│  (per seed)      │  AdamW + optional linear warmup; grokking protocol for Transformer on mod-add
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    Extract       │  Mean-pooled activations at embedding, layer_1, layer_2
│   Activations    │
└────────┬─────────┘
         │
    ┌────┼────────────────┬──────────────────┐
    ▼    ▼                ▼                  ▼
┌────────┐  ┌────────────┐  ┌──────────┐  ┌─────────────┐
│ Cross- │  │  Within-   │  │  Linear  │  │ Activation  │
│ Arch   │  │  Arch      │  │ Probing  │  │  Patching   │
│  CKA   │  │ Baseline   │  │  Suite   │  │ (Dyck-2)    │
│Heatmaps│  │    CKA     │  │          │  │             │
└────────┘  └────────────┘  └──────────┘  └─────────────┘
    │            │                │              │
    └────────────┴────────────────┴──────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  Aggregate & Export    │
            │  summary.json         │
            │  metrics_table.csv    │
            │  probe_metrics.csv    │
            │  convergence_table.csv│
            │  CKA heatmap PNGs     │
            └────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  Publication Figures   │
            │  (PNG + PDF)           │
            └────────────────────────┘
```

### Linear CKA

We compute CKA using the HSIC (Hilbert-Schmidt Independence Criterion) formulation: center the Gram matrices of mean-pooled layer activations, then compute the normalized Frobenius inner product. Produces a scalar similarity in `[0, 1]` for each layer pair.

### Convergence Index

For each architecture pair, the **convergence index** normalizes cross-architecture CKA by the geometric mean of the within-architecture seed baselines:

$$\text{CI}(A, B) = \frac{\text{CKA}_{\text{cross}}(A, B)}{\sqrt{\text{CKA}_{\text{within}}(A) \cdot \text{CKA}_{\text{within}}(B)}}$$

Values near 1.0 indicate that two architectures are as similar to each other as they are to themselves across random seeds (i.e., strong convergence).

### Linear Probing

For each layer, an LBFGS-optimized linear classifier is trained on frozen, standardized activations to predict task-specific targets (e.g., answer class, max nesting depth, target value). This measures how *linearly decodable* the task-relevant information is at each processing stage.

### Activation Patching

On Dyck-2, we fit linear projections from GRU/Transformer layer-1 activations to LSTM layer-1 activations (on a held-out fit set), then replace the LSTM's internal activations with the projected source activations and measure downstream accuracy. This tests whether the representations are *causally interchangeable*.

---

## Project Structure

```text
mechcmp/                        # Core Python package
├── __init__.py                 # Public API exports
├── config.py                   # TaskConfig, ModelConfig, ExperimentConfig dataclasses
├── tasks.py                    # Synthetic task data generators + dataset registry
├── models.py                   # TransformerClassifier, LSTMClassifier, GRUClassifier (PyTorch)
├── training.py                 # Train loop, evaluation, activation extraction
├── analysis.py                 # Linear CKA, CKA heatmap computation, heatmap visualization
├── probing.py                  # Linear probing pipeline (LBFGS probes per layer)
└── mlx_backend.py              # Full MLX reimplementation for Apple Silicon

scripts/
├── run_mvp.py                  # Main experiment runner (CLI entrypoint)
├── make_paper_figures.py       # Publication-quality composite figure generator
└── activation_patching.py      # Standalone activation patching / interchange intervention

tests/
├── test_models.py              # Model shape + activation key tests
├── test_tasks.py               # Task generator correctness + registry tests
├── test_training_and_analysis.py  # End-to-end train → CKA pipeline tests
├── test_probing.py             # Linear probing pipeline tests
└── test_run_mvp.py             # Full integration test for run_mvp

results/                        # Experiment outputs (auto-generated)
paper_figures/                  # Publication-ready composite figures (auto-generated)
```

---

## Getting Started

### Prerequisites

- Python ≥ 3.10
- PyTorch ≥ 2.2
- NumPy ≥ 1.26
- Matplotlib ≥ 3.8
- *(Optional)* MLX ≥ 0.10 — for Apple Silicon acceleration

### Installation

```bash
git clone https://github.com/your-username/Cross-Architecture-CKA-on-Synthetic-Algorithmic-Tasks.git
cd Cross-Architecture-CKA-on-Synthetic-Algorithmic-Tasks

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running Experiments

### Default (all tasks, 8 seeds, auto backend detection)

```bash
PYTHONPATH=. python scripts/run_mvp.py
```

### With custom parameters

```bash
PYTHONPATH=. python scripts/run_mvp.py \
    --backend torch \
    --seeds 42 43 44 45 46 \
    --epochs 100 \
    --d-model 128 \
    --num-layers 2 \
    --num-heads 4 \
    --batch-size 64 \
    --lr 1e-3 \
    --output-dir results
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `auto` | `torch`, `mlx`, or `auto` (prefers MLX on Apple Silicon) |
| `--seeds` | `42 43 44 45 46 47 48 49` | Model seeds for multi-seed aggregation |
| `--seed` | `42` | Dataset generation seed |
| `--epochs` | `100` | Training epochs (overridden per-architecture when needed) |
| `--d-model` | `128` | Hidden dimension for all architectures |
| `--num-layers` | `2` | Number of layers |
| `--num-heads` | `4` | Attention heads (Transformer only) |
| `--batch-size` | `64` | Training / evaluation batch size |
| `--lr` | `1e-3` | Learning rate |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--device` | auto | `cpu`, `cuda`, or `mps` |
| `--output-dir` | `results` | Output directory |

> **Note**: The experiment runner automatically applies a **grokking protocol** for the Transformer on modular addition (500 epochs, weight decay = 1.0, warmup = 50 steps), following the empirical observation that Transformers require extended training to learn modular arithmetic.

---

## Generating Paper Figures

After running the experiment:

```bash
PYTHONPATH=. python scripts/make_paper_figures.py \
    --results-dir results \
    --output-dir paper_figures
```

This generates four composite figures in both PNG (300 DPI) and PDF formats:

| Figure | Description |
|--------|-------------|
| `figure_cross_architecture_cka` | Grid of CKA heatmaps: 5 tasks × 3 architecture pairs, with panel labels |
| `figure_within_architecture_baselines` | Grid of within-architecture seed baseline CKA: 5 tasks × 3 architectures |
| `figure_accuracy_summary` | Grouped bar chart of validation accuracy with per-seed scatter and chance baselines |
| `figure_probe_summary` | Layer-wise linear probe accuracy curves with ±1 std shading per architecture |

A `figure_manifest.json` is also generated listing all output files.

---

## Apple Silicon (MLX) Support

The codebase includes a full native [MLX](https://github.com/ml-explore/mlx) backend (`mechcmp/mlx_backend.py`) with custom implementations of:

- Multi-head self-attention, Transformer encoder layers
- LSTM cells and GRU cells (unrolled over time steps)
- Training with `mx.value_and_grad` and AdamW
- Activation extraction and CKA computation

To run on Apple Silicon with MLX:

```bash
PYTHONPATH=. python scripts/run_mvp.py --backend mlx --output-dir mechinter_results
PYTHONPATH=. python scripts/make_paper_figures.py --results-dir mechinter_results --output-dir mechinter_paper_figures
```

---

## Running Tests

```bash
pip install pytest>=8.0
pytest
```

The test suite covers:

- **Model tests**: Output shapes, activation key correctness for all three architectures
- **Task tests**: Data generator correctness, value ranges, label distributions, registry completeness
- **Training & analysis tests**: End-to-end train → extract → CKA pipeline, CKA value bounds, self-CKA ≈ 1.0, multi-seed aggregation
- **Probing tests**: Linear probe training, accuracy computation, full layer-wise probing suite
- **Integration tests**: Minimal end-to-end experiment run verifying all output files

---

## Outputs

A completed experiment run produces the following in the output directory:

```text
results/
├── summary.json                                      # Full experiment results (all seeds, tasks, metrics)
├── metrics_table.csv                                  # Per-seed train/val loss and accuracy
├── probe_metrics.csv                                  # Per-seed, per-layer probe accuracy
├── convergence_table.csv                              # Cross-CKA, within-CKA baseline, convergence index
├── {task}_{archA}_vs_{archB}_cross_seed_mean_cka.png  # Cross-architecture CKA heatmap
└── {task}_{arch}_within_seed_baseline_cka.png         # Within-architecture baseline CKA heatmap
```

---



- License

This project is released for academic and research purposes.
