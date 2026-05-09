from pathlib import Path

import torch

from mechcmp.config import ExperimentConfig, ModelConfig
from scripts.make_paper_figures import generate_paper_figures
from scripts.run_mvp import default_model_seeds, default_tasks, run_experiment


def test_local_end_to_end_pipeline() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "mechinter_results"
    figures_dir = project_root / "mechinter_paper_figures"
    device = (
        "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    exp_config = ExperimentConfig(
        seed=42,
        seeds=default_model_seeds(),
        device=device,
        batch_size=64,
        lr=1e-3,
        epochs=10,
        weight_decay=1e-4,
        output_dir=str(results_dir),
        tasks=default_tasks(),
    )
    model_config = ModelConfig(name="matched_small", d_model=128, num_layers=2, num_heads=4)

    summary = run_experiment(exp_config, model_config)
    manifest = generate_paper_figures(results_dir=results_dir, output_dir=figures_dir)

    assert "summary_by_task" in summary
    assert (results_dir / "summary.json").exists()
    assert (results_dir / "probe_metrics.csv").exists()
    assert (results_dir / "convergence_table.csv").exists()
    assert (figures_dir / manifest["cross_architecture"]["png"]).exists()
    assert (figures_dir / manifest["within_architecture"]["png"]).exists()
    assert (figures_dir / manifest["accuracy_summary"]["png"]).exists()
    assert (figures_dir / manifest["probe_summary"]["png"]).exists()
