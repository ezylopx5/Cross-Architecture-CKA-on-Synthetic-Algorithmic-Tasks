

# Converting This Survey Into a NeurIPS-Caliber Mechanistic Interpretability Research Paper

## The Core Problem

You have an excellent **literature map** but zero original evidence. Here's exactly what to build.

---

## OPTION A: The Empirical Research Paper (Recommended — Highest Acceptance Probability)

### Title Concept
*"Convergent Algorithms, Divergent Circuits: A Mechanistic Comparison of How Transformers and RNNs Solve Identical Algorithmic Tasks"*

---

### The Central Hypothesis To Test

**Hypothesis:** When a Transformer and an RNN achieve identical task accuracy, they converge in *what* features they extract (spectral/statistical) but diverge in *how* they organize those features geometrically — and this divergence point is predictable from task complexity and can be precisely localized to specific layers/timesteps.

Your survey already claims this. Now you need to **prove it with original experiments**.

---

### Experimental Framework: The "Mechanistic Comparison Protocol" (MCP)

This becomes your **novel methodological contribution** — a standardized protocol for cross-architecture mechanistic comparison that doesn't exist yet.

```
┌─────────────────────────────────────────────────────┐
│         MECHANISTIC COMPARISON PROTOCOL (MCP)        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Step 1: TRAIN matched models on identical tasks     │
│  Step 2: EXTRACT representations at every layer      │
│  Step 3: ALIGN representations cross-architecture    │
│  Step 4: LOCATE divergence points                    │
│  Step 5: EXPLAIN divergence mechanistically          │
│  Step 6: INTERVENE to verify causal circuits         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

### Exact Experiments To Run

#### Experiment 1: Spectral vs. Geometric Convergence (Tests RQ1)

**What you do:**

```python
# Models to train (ALL from scratch, matched parameters)
models = {
    'transformer_2L': TransformerLM(layers=2, d_model=256, heads=4),
    'transformer_4L': TransformerLM(layers=4, d_model=256, heads=4),
    'lstm_2L': LSTM(layers=2, hidden=256),
    'lstm_4L': LSTM(layers=4, hidden=256),
    'gru_2L': GRU(layers=2, hidden=256),
    'mamba_2L': Mamba(layers=2, d_model=256),
    'gated_deltanet': GatedDeltaNet(layers=2, d_model=256),
}

# Tasks (graduated complexity)
tasks = {
    'modular_addition': ModularArithmeticDataset(mod=10),
    'dyck_1': DyckLanguageDataset(num_pairs=1, max_depth=5),
    'dyck_2': DyckLanguageDataset(num_pairs=2, max_depth=8),
    'induction': InductionTaskDataset(vocab=50, seq_len=128),
    'associative_recall': AssociativeRecallDataset(pairs=16),
}

# For EACH (model, task) pair:
# 1. Train to convergence (record learning curves)
# 2. Extract embeddings at every layer for test set
# 3. Compute Fourier spectrum of number/token embeddings
# 4. Compute linear probing accuracy at each layer
# 5. Store all activations for alignment analysis
```

**What you measure:**
- **Spectral convergence score:** Cross-model correlation of Fourier power spectra at each layer
- **Geometric convergence score:** Linear probing accuracy for target features at each layer
- **The Gap:** Quantify exactly where spectral convergence holds but geometric convergence fails

**Novel finding you'd likely get:**
> "All 7 architectures achieve spectral convergence (mean cross-model Fourier correlation > 0.92) by layer 1, but geometric convergence (linear probing accuracy > 80%) is achieved only by Transformers and Mamba, not by LSTMs/GRUs. The divergence point occurs at layer 2 for modular addition and layer 3 for Dyck-2."

---

#### Experiment 2: Cross-Architecture Representational Alignment (Tests RQ2)

**What you do:**

```python
# For every pair of models that achieve >95% accuracy on the same task:

def compute_alignment_trajectory(model_A, model_B, test_data):
    """Compute layer-by-layer alignment between two architectures."""
    
    results = {}
    
    for layer_a in model_A.layers:
        for layer_b in model_B.layers:
            acts_a = extract_activations(model_A, test_data, layer_a)
            acts_b = extract_activations(model_B, test_data, layer_b)
            
            results[(layer_a, layer_b)] = {
                'CKA': compute_CKA(acts_a, acts_b),
                'SVCCA': compute_SVCCA(acts_a, acts_b),
                'RSA': compute_RSA(acts_a, acts_b),
                'probing_transfer': train_probe_on_A_test_on_B(acts_a, acts_b),
            }
    
    return results  # This gives you a full alignment HEATMAP

# Also compute WITHIN-architecture alignment across random seeds
# to establish a baseline for "how similar can two solutions be"
```

**What you produce:**
- **Cross-architecture alignment heatmaps** (Figure 1 of your paper — this is your money figure)
- **Divergence layer identification**: The exact layer/timestep where CKA drops below within-architecture baseline
- **Task-dependent divergence**: Show that divergence happens earlier for harder tasks

**Novel finding you'd likely get:**
> "Transformer-LSTM CKA alignment is 0.87 at embedding layer, drops to 0.31 by layer 3 on Dyck-2, while Transformer-Mamba alignment remains 0.72 through layer 3. The divergence point correlates with task hierarchical depth (r=0.83, p<0.01)."

---

#### Experiment 3: Circuit-Level Comparison via Interchange Intervention (Tests RQ1 + RQ2)

This is the **most novel and NeurIPS-worthy** experiment. Nobody has done systematic cross-architecture interchange interventions.

**What you do:**

```python
def cross_architecture_interchange(model_A, model_B, input_x):
    """
    Replace activations at layer L of model_A with 
    linearly-projected activations from layer L' of model_B.
    Measure output change.
    
    If replacing Transformer layer 2 activations with 
    LSTM hidden states (after projection) preserves accuracy,
    those layers compute functionally equivalent representations.
    """
    
    # Step 1: Get activations from both models
    acts_A_L = model_A.get_activations(input_x, layer=L)
    acts_B_L = model_B.get_activations(input_x, layer=L_prime)
    
    # Step 2: Learn a linear projection (on held-out data)
    projection = learn_linear_map(acts_B_L, acts_A_L)
    
    # Step 3: Replace A's activations with projected B's activations
    projected_B = projection(acts_B_L)
    output_patched = model_A.forward_with_replacement(
        input_x, layer=L, replacement=projected_B
    )
    
    # Step 4: Measure accuracy change
    accuracy_drop = accuracy(output_original) - accuracy(output_patched)
    
    return accuracy_drop  # Small drop = functionally equivalent
```

**What you produce:**
- **Functional equivalence matrix**: For each (layer_A, layer_B) pair, how much accuracy survives cross-architecture patching
- **Circuit transferability score**: Novel metric quantifying how much of one architecture's computation is "translatable" to another

**Novel finding you'd likely get:**
> "On the induction task, replacing Transformer layer-1 attention output with projected LSTM hidden state h_t preserves 94% accuracy, confirming functional equivalence of shallow representations. However, replacing Transformer layer-2 (induction head) with LSTM h_t drops accuracy to 61%, confirming architecture-specific circuit implementation despite identical task solutions."

---

#### Experiment 4: Controlled Divergence Analysis — When Models Disagree (Tests RQ2 directly)

**What you do:**

```python
# Find inputs where models DISAGREE
disagreement_set = []
agreement_set = []

for x in test_data:
    pred_transformer = transformer(x)
    pred_lstm = lstm(x)
    
    if pred_transformer != pred_lstm:
        disagreement_set.append(x)
    else:
        agreement_set.append(x)

# For disagreement cases:
# 1. Trace activations through BOTH models layer-by-layer
# 2. Find the EARLIEST layer where representations diverge
#    (using CKA on agreement vs disagreement subsets)
# 3. Apply activation patching to find the CAUSAL circuit
#    responsible for the disagreement

def find_divergence_point(model_A, model_B, agree_set, disagree_set):
    """Find the earliest layer where representations diverge
    MORE for disagreement inputs than agreement inputs."""
    
    for layer in range(num_layers):
        cka_agree = CKA(
            model_A.activations(agree_set, layer),
            model_B.activations(agree_set, layer)
        )
        cka_disagree = CKA(
            model_A.activations(disagree_set, layer),
            model_B.activations(disagree_set, layer)
        )
        
        divergence_gap = cka_agree - cka_disagree
        
        if divergence_gap > threshold:
            return layer  # This is the divergence point
```

**What you produce:**
- **Divergence point localization**: Exact layer where disagreement inputs first show lower cross-architecture alignment than agreement inputs
- **Error mode taxonomy**: Categorize disagreement cases by which architecture is correct and what representational failure caused the error

**Novel finding you'd likely get:**
> "On Dyck-2 with depth 6, Transformer-LSTM disagreements originate at layer 2 of the Transformer (attention head 3 fails to attend to the correct bracket) versus timestep t-4 of the LSTM (hidden state norm drops below the decrement threshold). The Transformer error is a discrete routing failure; the LSTM error is a continuous state decay — confirming architecture-specific failure modes."

---

#### Experiment 5: Sparse Autoencoder Cross-Architecture Feature Comparison (Tests RQ1)

**What you do:**

```python
# Train SAEs on BOTH architectures' internal representations
sae_transformer = SparseAutoencoder(
    input_dim=256, latent_dim=2048, sparsity=0.05
)
sae_lstm = SparseAutoencoder(
    input_dim=256, latent_dim=2048, sparsity=0.05
)

# Train on layer-2 activations of each model
sae_transformer.fit(transformer_layer2_activations)
sae_lstm.fit(lstm_layer2_activations)

# Compare learned features:
# 1. Feature activation correlation across architectures
# 2. Feature interpretability (manual inspection + automated)
# 3. Feature overlap: what fraction of SAE features are 
#    "universal" (activate on same inputs in both architectures)?

def compute_feature_universality(sae_A, sae_B, shared_inputs):
    """What fraction of learned features are shared?"""
    
    features_A = sae_A.encode(model_A.activations(shared_inputs))
    features_B = sae_B.encode(model_B.activations(shared_inputs))
    
    # Compute correlation matrix between all feature pairs
    correlation_matrix = np.corrcoef(features_A.T, features_B.T)
    
    # For each feature in A, find best-matching feature in B
    max_correlations = correlation_matrix[:n_features_A, n_features_A:].max(axis=1)
    
    universal_fraction = (max_correlations > 0.7).mean()
    architecture_specific_fraction = 1 - universal_fraction
    
    return universal_fraction, architecture_specific_fraction
```

**What you produce:**
- **Feature universality ratio**: e.g., "43% of SAE features are universal (r > 0.7 cross-architecture), 57% are architecture-specific"
- **Universal feature characterization**: What do the shared features represent? (likely: token identity, position, basic syntax)
- **Architecture-specific feature characterization**: What do the unique features represent? (likely: attention patterns for Transformers, state trajectory features for LSTMs)

---

### What Your Results Section Looks Like

```
Section 4: Results

4.1 Spectral Convergence is Universal, Geometric Convergence is Not
    - Figure 1: Fourier spectra across 7 architectures (nearly identical)
    - Figure 2: Linear probing accuracy across layers (Transformer/Mamba succeed, LSTM/GRU fail)
    - Table 1: Spectral correlation scores (all > 0.90) vs geometric convergence scores

4.2 The Divergence Point is Task-Dependent and Predictable  
    - Figure 3: Cross-architecture CKA heatmaps for 5 tasks
    - Figure 4: Divergence layer vs. task hierarchical depth (linear fit, R²=0.83)
    - Finding: Divergence occurs 1-2 layers earlier for tasks requiring global structure

4.3 Functional Equivalence Breaks at Circuit Boundaries
    - Figure 5: Cross-architecture interchange accuracy preservation matrix
    - Finding: Shallow circuits are functionally interchangeable; deep circuits are not
    - Table 2: Circuit transferability scores by task and layer

4.4 Disagreement Analysis Reveals Architecture-Specific Failure Modes
    - Figure 6: Divergence point localization for agreement vs disagreement inputs
    - Finding: Transformer failures are discrete (attention routing); RNN failures are continuous (state decay)
    - Table 3: Error mode taxonomy with frequencies

4.5 SAE Feature Universality
    - Figure 7: Feature universality ratio across layers (decreasing with depth)
    - Finding: 43% universal features at layer 1, dropping to 12% by final layer
```

---

## OPTION B: The Theoretical Paper

If you prefer a theory-heavy paper, here are **novel theorems you could prove**.

### Novel Theorem 1: Divergence Point Lower Bound

```
Theorem (Minimum Divergence Depth):
Let T be a Transformer with L layers and R be an RNN with hidden 
dimension d processing sequences of length n.

For any task requiring hierarchical depth k, the minimum layer 
at which CKA(T_layer, R_timestep) drops below ε is:

    L_diverge ≥ ⌈log₂(k)⌉  for Transformers (spatial depth)
    t_diverge ≥ k           for RNNs (temporal steps)

Proof sketch: 
- Transformers in TC⁰ can parallelize hierarchy resolution,
  requiring O(log k) layers to resolve depth-k structure
- RNNs must sequentially process each hierarchical level,
  requiring O(k) steps
- Before L_diverge, both architectures are performing 
  sub-hierarchical (local) computations that are functionally 
  equivalent, hence high CKA
- After L_diverge, the spatial vs temporal resolution strategies
  produce geometrically incompatible representations
```

### Novel Theorem 2: Feature Universality Decay

```
Theorem (Universality Decay Rate):
Let U(l) be the fraction of linearly-matchable features between 
a Transformer and an RNN at depth l. Under mild assumptions on 
the data distribution and task structure:

    U(l) ≤ U(0) · exp(-γ · I(task; architecture | l))

where I(task; architecture | l) is the conditional mutual 
information between task-relevant features and architecture 
type at layer l, and γ depends on the representational capacity 
ratio.

Implication: Feature universality decays exponentially with 
the information that architectural inductive bias contributes 
at each layer.
```

### Novel Theorem 3: Cross-Architecture Patching Bound

```
Theorem (Interchange Intervention Accuracy Bound):
Let acc_patch(l) be the accuracy after replacing Transformer 
layer l activations with linearly-projected RNN activations.

    acc_patch(l) ≥ acc_original - ||Σ_T^l - P·Σ_R^l·P^T||_F · C_task

where Σ_T^l, Σ_R^l are the activation covariance matrices,
P is the optimal linear projection, and C_task is a 
task-dependent Lipschitz constant.

Implication: Patching accuracy degrades proportionally to 
the spectral mismatch between architectures' covariance 
structures, providing a computable prediction of when 
cross-architecture circuits are functionally equivalent.
```

---

## OPTION C: The Novel Framework Paper

### Framework: "Mechanistic Rosetta Stone" (MRS)

A unified framework for translating mechanistic findings across architectures.

```
┌─────────────────────────────────────────────────────────────┐
│              MECHANISTIC ROSETTA STONE (MRS)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LAYER 1: Universal Feature Identification                   │
│  ├── Train SAEs on both architectures                        │
│  ├── Compute cross-architecture feature matching             │
│  └── Output: Universal Feature Set F_u, Specific Sets F_t, F_r│
│                                                             │
│  LAYER 2: Circuit Correspondence Mapping                     │
│  ├── For each circuit in architecture A                       │
│  ├── Find functional equivalent in architecture B            │
│  ├── Via interchange intervention with learned projection     │
│  └── Output: Circuit Translation Dictionary                  │
│                                                             │
│  LAYER 3: Divergence Localization                            │
│  ├── Compute layer-wise alignment trajectory                 │
│  ├── Identify critical divergence points                     │
│  ├── Classify divergence type (geometric vs functional)      │
│  └── Output: Divergence Map with causal explanations         │
│                                                             │
│  LAYER 4: Failure Mode Transfer                              │
│  ├── Identify failure modes in architecture A                │
│  ├── Predict corresponding failure modes in architecture B   │
│  ├── Via circuit translation dictionary                      │
│  └── Output: Cross-architecture vulnerability predictions    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why this is novel:** Nobody has proposed a systematic protocol for mechanistic comparison. Individual papers do one-off comparisons; MRS would be a reusable, standardized toolkit.

**What makes it publishable:**
1. The framework itself (methodology contribution)
2. A reference implementation (code contribution)
3. Validation on 5+ tasks showing the framework produces consistent, reproducible findings
4. Demonstration that the framework predicts cross-architecture transfer success/failure

---

## Minimum Viable Paper (4-6 weeks of work)

If you want the fastest path to a submittable paper:

```
Week 1-2: 
├── Implement matched Transformer (2-layer) and LSTM (2-layer)
├── Train both on 3 tasks: modular addition, Dyck-1, induction
├── Extract all activations
└── Compute Fourier spectra + linear probing at each layer

Week 3-4:
├── Compute full CKA/SVCCA alignment heatmaps
├── Run cross-architecture interchange interventions
├── Identify divergence points
└── Run disagreement analysis

Week 5:
├── Train SAEs on both architectures
├── Compute feature universality ratios
└── Compile results

Week 6:
├── Write paper with proper NeurIPS formatting
├── Create figures (alignment heatmaps, divergence curves, 
│   feature universality decay, interchange accuracy matrices)
└── Frame contribution clearly
```

---

## What Your Paper Structure Should Look Like

```
1. Introduction (1.5 pages)
   - The question: same answer, same algorithm?
   - Why it matters (safety, distillation, architecture design)
   - Our contribution: MCP framework + 5 experiments + 3 findings

2. Related Work (1 page)
   - YOUR CURRENT SURVEY CONTENT, compressed to 1 page
   - This is where your literature knowledge becomes valuable

3. Mechanistic Comparison Protocol (1.5 pages)
   - The MCP framework (your methodological contribution)
   - Formal definitions of spectral/geometric convergence
   - Cross-architecture interchange intervention method

4. Experimental Setup (1 page)
   - Models (exact architectures, sizes, hyperparameters)
   - Tasks (exact specifications, dataset sizes)
   - Metrics (CKA, SVCCA, probing accuracy, interchange accuracy)
   - Compute budget and reproducibility details

5. Results (3 pages)
   - 5.1: Spectral convergence confirmed, geometric divergence quantified
   - 5.2: Divergence point localization and task-dependence
   - 5.3: Circuit-level functional equivalence analysis
   - 5.4: Disagreement analysis and failure mode taxonomy
   - 5.5: SAE feature universality decay

6. Analysis and Discussion (1 page)
   - What's universal: statistical features, shallow circuits
   - What's specific: geometric organization, deep circuits, failure modes
   - Implications for distillation, safety, architecture design

7. Conclusion (0.5 pages)

Appendix:
   - Extended results tables
   - Additional alignment heatmaps
   - Hyperparameter sensitivity analysis
   - Code and data availability
```

---

## The Key Insight For You

Your survey gives you a **massive advantage**: you already know exactly what findings to expect and how to interpret them. Most researchers spend months figuring out what to look for. You already have the roadmap.

**What's missing is the evidence.** Run the experiments, report the numbers, and let your literature knowledge guide the interpretation. That's the difference between a survey and a research paper.

The **single highest-impact thing** you could produce is **Figure 3: Cross-architecture CKA heatmaps across 5 tasks**. That one figure, if done properly with matched architectures and controlled tasks, would be a novel empirical contribution that doesn't exist in the literature. Everything else builds on top of that foundation.