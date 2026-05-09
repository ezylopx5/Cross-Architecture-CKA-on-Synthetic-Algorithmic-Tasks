import torch
import numpy as np
from mechcmp.models import build_model
from mechcmp.config import ModelConfig
from mechcmp.training import evaluate, make_loader
from mechcmp.tasks import build_task_datasets, TaskConfig
from torch import nn

def fit_linear_projection(source_acts, target_acts):
    # source_acts: (N, D), target_acts: (N, D)
    # Solve source * W = target
    source_acts_torch = torch.from_numpy(source_acts).float()
    target_acts_torch = torch.from_numpy(target_acts).float()
    
    # Simple linear regression
    # target = source * W + b
    reg = nn.Linear(source_acts_torch.shape[1], target_acts_torch.shape[1])
    optimizer = torch.optim.LBFGS(reg.parameters(), lr=1, max_iter=100)
    criterion = nn.MSELoss()
    
    def closure():
        optimizer.zero_grad()
        loss = criterion(reg(source_acts_torch), target_acts_torch)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    return reg

def run_patching_experiment():
    print("Starting activation patching experiment: LSTM <-> GRU on Dyck-2")
    
    task_config = TaskConfig(
        name="dyck_2",
        train_size=2048,
        val_size=512,
        seq_len=14,
        vocab_size=5,
        num_classes=2
    )
    model_config = ModelConfig(name="test", d_model=128, num_layers=2)
    
    train_ds, val_ds = build_task_datasets(task_config, seed=42)
    val_loader = make_loader(val_ds, batch_size=64, shuffle=False)
    
    # 1. Build and 'train' (or just use initialized for demo if needed, but let's assume they are trained)
    # For a real experiment we'd load trained weights. 
    # Here we'll just show the mechanism works.
    lstm = build_model("lstm", task_config.vocab_size, task_config.num_classes, model_config, task_config.seq_len)
    gru = build_model("gru", task_config.vocab_size, task_config.num_classes, model_config, task_config.seq_len)
    
    # 2. Extract activations
    def collect_acts(model, loader):
        model.eval()
        acts = []
        labels = []
        with torch.no_grad():
            for x, y in loader:
                res = model(x, return_activations=True)
                # Mean pool over sequence length
                acts.append(res.activations['layer_1'].mean(dim=1).numpy())
                labels.append(y.numpy())
        return np.concatenate(acts, axis=0), np.concatenate(labels, axis=0)

    print("Collecting activations...")
    # Split validation set for fitting and testing patching
    # Use first 256 for fitting, last 256 for testing
    fit_loader = make_loader(torch.utils.data.Subset(val_ds, range(256)), batch_size=64, shuffle=False)
    test_loader = make_loader(torch.utils.data.Subset(val_ds, range(256, 512)), batch_size=64, shuffle=False)
    
    lstm_fit_acts, _ = collect_acts(lstm, fit_loader)
    gru_fit_acts, _ = collect_acts(gru, fit_loader)
    
    # 3. Fit projection GRU -> LSTM
    print("Fitting linear projection GRU -> LSTM...")
    projection = fit_linear_projection(gru_fit_acts, lstm_fit_acts)
    
    # 4. Patching test
    print("Running patching test...")
    lstm.eval()
    gru.eval()
    
    correct_original = 0
    correct_patched = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            # Original LSTM accuracy
            logits_orig = lstm(x)
            correct_original += (logits_orig.argmax(dim=-1) == y).sum().item()
            
            # Patched LSTM: replace layer_1 with projected GRU layer_1
            res_gru = gru(x, return_activations=True)
            gru_layer1 = res_gru.activations['layer_1']
            
            # Project each token's activation
            B, S, D = gru_layer1.shape
            gru_layer1_flat = gru_layer1.reshape(B*S, D)
            projected_flat = projection(gru_layer1_flat)
            projected = projected_flat.reshape(B, S, -1)
            
            logits_patched = lstm(x, replacement_activations={'layer_1': projected})
            correct_patched += (logits_patched.argmax(dim=-1) == y).sum().item()
            
            total += y.size(0)
            
    print(f"Original LSTM Accuracy: {correct_original/total:.4f}")
    print(f"Patched LSTM Accuracy: {correct_patched/total:.4f}")

if __name__ == "__main__":
    run_patching_experiment()
