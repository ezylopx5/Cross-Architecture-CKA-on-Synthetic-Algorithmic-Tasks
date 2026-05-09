# Mechanistic Comparison Protocol MVP

This repository now contains a minimal implementation of the highest-impact
part of the research plan in `goal.md`: cross-architecture CKA heatmaps for
matched Transformer and LSTM models on controlled synthetic tasks, including
multi-seed and within-architecture baselines.

## What Is Implemented

- Matched 2-layer Transformer and LSTM classifiers
- Synthetic tasks for modular addition, Dyck-1, Dyck-2, and induction
- Training and validation loops
- Layer activation extraction
- Cross-architecture linear CKA computation
- Multi-seed experiment runs with aggregated metrics
- Within-architecture seed-baseline CKA heatmaps
- Metrics table export for paper-style reporting

## Project Layout

```text
mechcmp/
  config.py
  tasks.py
  models.py
  training.py
  analysis.py
scripts/
  run_mvp.py
results/
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python scripts/run_mvp.py
```

Outputs are written into `results/`:

- `summary.json`
- `metrics_table.csv`
- `*_transformer_vs_lstm_cross_seed_mean_cka.png`
- `*_{transformer,lstm}_within_seed_baseline_cka.png`

To generate publication-ready composite figures from those outputs:

```bash
PYTHONPATH=. python scripts/make_paper_figures.py --results-dir results --output-dir paper_figures
```

This writes:

- `paper_figures/figure_cross_architecture_cka.{png,pdf}`
- `paper_figures/figure_within_architecture_baselines.{png,pdf}`
- `paper_figures/figure_accuracy_summary.{png,pdf}`
- `paper_figures/figure_manifest.json`

## Next Steps

- Add GRU and Mamba baselines
- Add probing, SVCCA, and disagreement analysis
- Extend task set to associative recall
