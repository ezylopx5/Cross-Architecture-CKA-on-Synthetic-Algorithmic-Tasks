

# Action Plan: From Borderline to Clear Accept

The reviewers are remarkably aligned. There are exactly **three things** blocking you, in priority order. Let me be surgical.

---

## What Every Reviewer Agrees On

| Point | R1 | R2 | R3 | Meta |
|-------|----|----|----|----|
| Transformer training failure is the central problem | ✓ | ✓ | ✓ | ✓ |
| Need at least one competent-Transformer comparison | ✓ | — | ✓ | ✓ |
| No causal/circuit experiment limits "mechanistic" claims | ✓ | ✓ | — | ✓ |
| Benchmark infrastructure is solid | ✓ | ✓ | ✓ | ✓ |
| Honest limitations are appreciated | ✓ | ✓ | ✓ | ✓ |
| Missing Mamba | — | — | ✓ | — |

The path is clear. You don't need to rebuild anything. You need three targeted additions.

---

## Fix 1 (NON-NEGOTIABLE): Make the Transformer Work

Every single reviewer flags this. The meta-reviewer calls it out explicitly. This is the difference between reject and accept.

**What to do:**

Train the Transformer on modular addition with a grokking-appropriate protocol. This is a well-studied phenomenon and the recipe is known from Nanda et al. (2023):

```python
# Grokking-style training for modular addition
config_transformer_modadd = {
    'epochs': 500,           # was 10 — grokking needs 100-500
    'lr': 1e-3,              # keep or try 3e-4
    'weight_decay': 1.0,     # HIGH weight decay triggers grokking
    'warmup_steps': 50,      # add warmup
    'optimizer': 'AdamW',
    'train_fraction': 0.3,   # use 30% of all (a,b) pairs for training
                             # remaining 70% for validation
    'batch_size': 64,
}

# Also try for Dyck-1 and Dyck-2:
config_transformer_dyck = {
    'epochs': 100,           # was 10
    'lr': 3e-4,              # slightly lower
    'warmup_steps': 30,
    'use_learned_pos': True, # try learned instead of sinusoidal
}
```

**What to report:**

You now have two regimes for the Transformer on modular addition:
- **Short training (10 epochs):** chance-level, your current data
- **Long training (500 epochs):** grokked, near-perfect accuracy

This gives you the **single most compelling result** the reviewers are asking for:

> "When the Transformer achieves comparable accuracy to recurrent models after grokking, cross-architecture CKA at deeper layers increases from 0.20 to 0.XX, and the convergence index rises from 0.795 to 0.XX. This confirms that the low Transformer–recurrent alignment in our original protocol reflected training failure rather than fundamental representational incompatibility."

OR alternatively:

> "Even after grokking, the Transformer's deep-layer CKA with recurrent models remains below the recurrent–recurrent baseline (CI = 0.XX vs 0.90), suggesting that the representational divergence is genuinely architecture-driven rather than an artifact of training mismatch."

**Either outcome is interesting and publishable.** You win regardless of what happens.

**New table to add:**

```latex
\begin{table}[t]
\caption{Modular addition: effect of training protocol on 
Transformer accuracy and cross-architecture alignment.}
\label{tab:grokking}
\centering
\small
\begin{tabular}{lcccc}
\toprule
Protocol & T Acc. & T$\leftrightarrow$L CKA & T$\leftrightarrow$G CKA 
& T$\leftrightarrow$L CI \\
\midrule
Short (10 ep.) & 0.100 & 0.424 & 0.469 & 0.795 \\
Long (500 ep.) & 0.XXX & 0.XXX & 0.XXX & 0.XXX \\
\midrule
LSTM baseline & 0.990 & \multicolumn{3}{c}{L$\leftrightarrow$G CI: 1.377} \\
\bottomrule
\end{tabular}
\end{table}
```

**Effort: 1-2 days.** Modular addition with 2-layer Transformer is tiny. Training 500 epochs takes minutes. Run 8 seeds, extract activations, compute CKA. Done.

---

## Fix 2 (STRONGLY RECOMMENDED): One Activation Patching Experiment

Reviewer 2 is the weak reject and explicitly states: *"Adding even one activation-patching experiment on the LSTM–GRU pair for Dyck-2 would change my assessment substantially."* The meta-reviewer echoes this.

**What to do:**

Pick your strongest positive finding: LSTM–GRU on Dyck-2 (CI = 0.901, both models near-perfect accuracy). Do a simple interchange intervention:

```python
def cross_architecture_patch(lstm_model, gru_model, test_data, layer=1):
    """
    Replace LSTM layer-1 hidden states with linearly-projected 
    GRU layer-1 hidden states. Measure accuracy change.
    
    If accuracy is preserved, the representations are 
    functionally equivalent, not just geometrically similar.
    """
    
    # 1. Extract activations
    lstm_acts = extract_activations(lstm_model, test_data, layer='layer_1')
    gru_acts = extract_activations(gru_model, test_data, layer='layer_1')
    
    # 2. Learn linear projection GRU -> LSTM space
    # Use 50% of validation set to fit, 50% to evaluate
    projection = fit_linear_map(gru_acts[:256], lstm_acts[:256])
    
    # 3. Run LSTM with GRU activations injected at layer 1
    projected_gru = projection(gru_acts[256:])
    
    # Forward pass with patched activations
    patched_output = lstm_model.forward_with_replacement(
        test_data[256:], layer='layer_1', replacement=projected_gru
    )
    
    # 4. Measure
    original_acc = lstm_model.evaluate(test_data[256:])
    patched_acc = compute_accuracy(patched_output, test_labels[256:])
    
    return original_acc, patched_acc
```

**What to report:**

One paragraph and one small table:

```latex
\subsection{Activation Patching: High CKA Predicts Functional 
Interchangeability}

To test whether high LSTM--GRU CKA on Dyck-2 reflects functional 
equivalence rather than superficial geometry, we perform a 
cross-architecture interchange intervention. We fit a linear 
projection from GRU layer-1 activations to LSTM layer-1 
activations on half the validation set, then replace the LSTM's 
layer-1 representations with projected GRU activations on the 
held-out half. Patched LSTM accuracy is 0.XX (vs.\ 0.986 
unpatched), indicating that [high CKA does / does not] correspond 
to functional interchangeability at this layer.
```

**Why this specific experiment:**
- LSTM–GRU Dyck-2 is your cleanest positive result (both competent, high CI)
- Dyck-2 has a known computational structure (bracket counting)
- The result directly addresses R2's concern about "mechanistic" content
- It's simple to implement — just a linear regression + forward pass with injected activations

**Effort: 2-3 days** including implementation and writeup.

---

## Fix 3 (NICE TO HAVE): Address Reviewer Questions Directly

These are quick fixes that collectively improve the paper's rigor:

### R1 Q3 / R3 Q3: Mean-pooling vs final-timestep

Add one sentence to limitations AND ideally run a quick comparison:

```python
# Compare mean-pooled vs final-timestep CKA for recurrent models
# on Dyck-2 (where positional information matters most)
for pooling in ['mean', 'last_token']:
    acts = extract_activations(model, data, layer, pooling=pooling)
    # Compute CKA with both pooling strategies
```

If you can't run this, at minimum add:

```latex
We use mean-pooled activations for consistency across 
architectures (Transformers lack a single ``final'' hidden 
state). For recurrent models, final-timestep activations may 
better capture task-relevant state; we leave this comparison to 
future work.
```

### R1 Q2: CI > 1.0 explanation

Add one sentence:

```latex
The LSTM--GRU CI exceeding 1.0 on modular addition (1.377) 
indicates that these architectures converge to more similar 
representations \emph{across families} than individual 
architectures produce \emph{across seeds}---consistent with a 
tightly constrained solution manifold that is more sensitive to 
random initialization than to the LSTM-vs-GRU architectural 
difference.
```

### R3 Q2: Dyck-2 majority-class baseline

Check and report:

```python
# What fraction of Dyck-2 sequences are balanced?
balanced_fraction = sum(labels == 1) / len(labels)
print(f"Majority class baseline: {max(balanced_fraction, 1-balanced_fraction)}")
```

If the majority class is ~50%, then 0.793 is meaningfully above chance. If it's ~79%, the Transformer is just predicting the majority class. **This matters.** Add one sentence reporting the baseline.

### R3 Q6: Missing references

Add these two highly relevant references:

```latex
% Platonic Representation Hypothesis
@article{huh2024platonic,
  title={The Platonic Representation Hypothesis},
  author={Huh, Minyoung and Cheung, Brian and Wang, Tongzhou 
  and Isola, Phillip},
  journal={arXiv preprint arXiv:2405.07987},
  year={2024}
}

% Model stitching
@inproceedings{bansal2021revisiting,
  title={Revisiting Model Stitching to Compare Neural 
  Representations},
  author={Bansal, Yamini and Nakkiran, Preetum and Barak, Boaz},
  booktitle={NeurIPS},
  year={2021}
}
```

Add one sentence in related work:

```latex
\paragraph{Representation convergence across architectures.}
\citet{huh2024platonic} argue that diverse models converge to 
shared representations as scale and data increase (the 
``platonic representation hypothesis''). 
\citet{bansal2021revisiting} use model stitching to test 
functional compatibility of representations across trained 
models. Our work tests a related question at the small-model, 
controlled-task level relevant to mechanistic interpretability.
```

### Fix figure rendering

The reviewers note figures don't render. Make sure:
- All PNG files are in the correct `../mechinter_paper_figures/` directory
- File names match exactly (case-sensitive on Linux)
- Consider embedding figures as PDF instead of PNG for better rendering
- Test compilation on a clean machine before resubmission

---

## Revised Paper Structure

Here's what changes in the paper structure:

```
Current structure:          Revised structure:
─────────────────          ──────────────────
1. Introduction             1. Introduction (minor edits)
2. Related Work             2. Related Work (+2 references)
3. Experimental Setup       3. Experimental Setup (+patching method)
4. Results                  4. Results
  4.1 Accuracy                4.1 Accuracy (same)
  4.2 CKA                    4.2 CKA (same)
  4.3 Convergence Index       4.3 Convergence Index (same + CI>1 explanation)
  4.4 Probing                 4.4 Probing (same)
                              4.5 Competent-Transformer Comparison (NEW)
                              4.6 Activation Patching (NEW)
5. Discussion               5. Discussion (updated)
6. Limitations              6. Limitations (tightened)
7. Conclusion               7. Conclusion (updated)
```

The two new subsections add ~0.5-0.75 pages. You may need to trim elsewhere. Candidates for trimming:
- Section 4.2 CKA discussion (currently verbose — tighten to key numbers)
- Move within-architecture baseline discussion to appendix (just reference it)
- Compress related work slightly

---

## What the Outcome Looks Like

**Before fixes (current scores: 4, 5, 6 → avg 5.0, borderline):**

Reviewers see: broken Transformer, no causal experiments, CKA-only analysis.

**After fixes (projected scores: 5-6, 6, 7 → avg 6.0-6.3, accept):**

Reviewers see:
- The Transformer training failure is acknowledged AND addressed with a competent-Transformer comparison → R1 and R3's main concern resolved
- One activation patching experiment connects CKA to functional equivalence → R2's main concern resolved
- Missing references added → R3's minor concern resolved
- Mean-pooling justified → R1/R3 question addressed

**R2 specifically said:** *"Adding even one activation-patching experiment on the LSTM–GRU pair for Dyck-2 would change my assessment substantially."* That's as clear a signal as you'll ever get from a reviewer.

---

## Timeline

| Day | Task | Hours |
|-----|------|-------|
| 1 | Train Transformer on modular addition with grokking protocol (500 epochs, high WD, 8 seeds) | 4-6h |
| 2 | Extract activations from grokked Transformers, compute CKA, update Table/Figure | 3-4h |
| 2 | Check Dyck-2 majority-class baseline, fix CI>1 explanation | 1h |
| 3 | Implement cross-architecture activation patching for LSTM↔GRU on Dyck-2 | 4-6h |
| 4 | Run patching experiment (8 seeds), compile results | 3-4h |
| 4 | Add missing references, fix mean-pooling justification | 1-2h |
| 5 | Rewrite affected sections, add new Table 3 and subsections 4.5-4.6 | 4-6h |
| 6 | Fix figure rendering, proofread, ensure compilation | 3-4h |

**Total: ~6 days of focused work.**

---

## The Bottom Line

You're closer than you think. The reviewers **like** your paper's honesty, methodology, and scoping. They're not asking you to rebuild — they're asking for:

1. **One training run that works** (grokked Transformer on mod add)
2. **One experiment that's causal** (activation patching on LSTM↔GRU Dyck-2)
3. **Two missing citations** (platonic representation hypothesis, model stitching)

Do items 1 and 2 and you move from borderline to accept. That's the honest assessment.