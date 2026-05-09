import json
from pathlib import Path

from mechcmp.config import ExperimentConfig, ModelConfig, TaskConfig
from scripts.run_mvp import run_experiment


def test_run_experiment_writes_summary_and_heatmap(tmp_path: Path) -> None:
    exp_config = ExperimentConfig(
        seed=99,
        seeds=[0, 1],
        device="cpu",
        batch_size=8,
        lr=1e-3,
        epochs=1,
        output_dir=str(tmp_path),
        tasks=[
            TaskConfig(
                name="modular_addition",
                train_size=32,
                val_size=16,
                seq_len=6,
                vocab_size=13,
                num_classes=10,
            )
        ],
    )
    model_config = ModelConfig(name="tiny", d_model=16, num_layers=1, num_heads=4)

    summary = run_experiment(exp_config, model_config)

    summary_path = tmp_path / "summary.json"
    cross_heatmap_path = tmp_path / "modular_addition_transformer_vs_lstm_cross_seed_mean_cka.png"
    transformer_gru_cross_heatmap_path = tmp_path / "modular_addition_transformer_vs_gru_cross_seed_mean_cka.png"
    lstm_gru_cross_heatmap_path = tmp_path / "modular_addition_lstm_vs_gru_cross_seed_mean_cka.png"
    transformer_baseline_path = tmp_path / "modular_addition_transformer_within_seed_baseline_cka.png"
    lstm_baseline_path = tmp_path / "modular_addition_lstm_within_seed_baseline_cka.png"
    gru_baseline_path = tmp_path / "modular_addition_gru_within_seed_baseline_cka.png"
    metrics_table_path = tmp_path / "metrics_table.csv"

    assert "summary_by_task" in summary
    assert summary_path.exists()
    assert cross_heatmap_path.exists()
    assert transformer_gru_cross_heatmap_path.exists()
    assert lstm_gru_cross_heatmap_path.exists()
    assert transformer_baseline_path.exists()
    assert lstm_baseline_path.exists()
    assert gru_baseline_path.exists()
    assert metrics_table_path.exists()
    task_summary = summary["summary_by_task"]["modular_addition"]
    assert task_summary["cross_architecture"]["transformer_vs_lstm"]["heatmap_path"] == "modular_addition_transformer_vs_lstm_cross_seed_mean_cka.png"
    assert task_summary["cross_architecture"]["transformer_vs_gru"]["heatmap_path"] == "modular_addition_transformer_vs_gru_cross_seed_mean_cka.png"
    assert task_summary["cross_architecture"]["lstm_vs_gru"]["heatmap_path"] == "modular_addition_lstm_vs_gru_cross_seed_mean_cka.png"
    assert task_summary["within_architecture"]["transformer"]["heatmap_path"] == "modular_addition_transformer_within_seed_baseline_cka.png"
    assert task_summary["within_architecture"]["lstm"]["heatmap_path"] == "modular_addition_lstm_within_seed_baseline_cka.png"
    assert task_summary["within_architecture"]["gru"]["heatmap_path"] == "modular_addition_gru_within_seed_baseline_cka.png"
    assert json.loads(summary_path.read_text(encoding="ascii")) == summary


def test_run_experiment_uses_null_heatmap_path_for_single_seed_within_architecture(
    tmp_path: Path,
) -> None:
    exp_config = ExperimentConfig(
        seed=99,
        seeds=[0],
        device="cpu",
        batch_size=8,
        lr=1e-3,
        epochs=1,
        output_dir=str(tmp_path),
        tasks=[
            TaskConfig(
                name="modular_addition",
                train_size=32,
                val_size=16,
                seq_len=6,
                vocab_size=13,
                num_classes=10,
            )
        ],
    )
    model_config = ModelConfig(name="tiny", d_model=16, num_layers=1, num_heads=4)

    summary = run_experiment(exp_config, model_config)

    task_summary = summary["summary_by_task"]["modular_addition"]
    assert task_summary["within_architecture"]["transformer"]["num_seed_pairs"] == 0
    assert task_summary["within_architecture"]["transformer"]["heatmap_path"] is None
    assert task_summary["within_architecture"]["lstm"]["heatmap_path"] is None
    assert task_summary["within_architecture"]["gru"]["heatmap_path"] is None
