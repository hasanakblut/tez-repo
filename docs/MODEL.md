# GA-Dueling DQN — Model Architecture Documentation

> **Implementation file:** `src/model.py`
>
> **Paper reference:** Xia, L.; Wang, L.; Xie, Z.; Gao, X. *"GA-Dueling DQN Jamming Decision-Making Method for Intra-Pulse Frequency Agile Radar"*, Sensors 2024, 24, 1325.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [NoisyLinear Layer](#2-noisylinear-layer)
3. [GRU Layer — Temporal Encoding](#3-gru-layer--temporal-encoding)
4. [Multi-Head Self-Attention](#4-multi-head-self-attention)
5. [Fully-Connected Feature Extractor](#5-fully-connected-feature-extractor)
6. [Dueling Heads](#6-dueling-heads)
7. [Q-Value Aggregation](#7-q-value-aggregation)
8. [Dimension Flow Summary](#8-dimension-flow-summary)
9. [Hidden State Management](#9-hidden-state-management)
10. [Noise Reset Protocol](#10-noise-reset-protocol)

---

## 1. Architecture Overview

The GA-Dueling DQN (GRU-Attention-based Dueling Deep Q Network) is designed to solve the intelligent jamming frequency selection problem for intra-pulse frequency agile radars. The architecture is constructed to address a fundamental challenge stated in Section 3.3 of the paper:

> *"Even though Dueling DQN can effectively alleviate the overestimation problem and improve the stability of learning [...], it has difficulty providing sufficient information from single observations due to the random generation of radar frequencies."*

The network combines four key ideas to overcome this:

| Component | Purpose | Paper Reference |
|---|---|---|
| GRU | Capture long-term temporal dependencies from sequential radar observations | Section 3.3, Figure 7 |
| Multi-Head Self-Attention | Learn correlations between temporal features at different positions | Section 3.3, Figure 7 |
| NoisyLinear + LayerNorm | Encourage exploration via parametric noise; stabilize gradients | Section 3.3, Figure 8, Ref [28] |
| Dueling Decomposition | Separate state value V(s) from action advantage A(s,a) for more precise Q estimation | Section 3.3, Equation 18, Figure 6 |

### Full Architecture Diagram (Paper Figures 6–8)

```
Input: one-hot(s_t) ∈ R^240        ← sequence of L past radar observations
  │
  ▼
┌──────────────────────────────────┐
│  GRU Layer                       │  input_size=240, hidden_size=128
│  (240 → 128)                     │  captures temporal dependencies
│                                  │  Paper: "the model can capture long-term
│                                  │  dependency relationships in the sequence"
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  Multi-Head Self-Attention       │  embed_dim=128, num_heads=8
│  (128 → 128)                     │  learns inter-feature correlations
│                                  │  Paper: "learn correlations between
│                                  │  different input features"
└──────────────┬───────────────────┘
               ▼  (last time step output)
┌──────────────────────────────────┐
│  FC1: NoisyLinear(128 → 64)      │
│  + LayerNorm(64)                 │  prevents gradient explosion/vanishing
│  + ReLU                          │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  FC2: NoisyLinear(64 → 64)       │
│  + LayerNorm(64)                 │  further feature refinement
│  + ReLU                          │
└──────────────┬───────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
┌─────────────┐ ┌──────────────┐
│ V(s) Stream │ │ A(s,a) Stream│
│ NoisyLinear │ │ NoisyLinear  │
│  64 → 1     │ │  64 → 240    │
└──────┬──────┘ └──────┬───────┘
       │               │
       └───────┬───────┘
               ▼
    Q(s,a) = V(s) + (A(s,a) − mean(A))
```

---

## 2. NoisyLinear Layer

**Implementation:** `NoisyLinear` class in `src/model.py`
**Paper reference:** Section 3.3, referencing Fortunato et al. (2017), Ref [28]

### 2.1 Motivation

From the paper:

> *"The Noisy Net technique is employed in the linear layers, FC_i, by way of introducing Gaussian noise to the weights and biases of the neural network. [...] This augmentation promotes exploration and enhances learning."*

Standard ε-greedy exploration chooses random actions uniformly, which is inefficient in a 240-dimensional action space. NoisyLinear embeds **learned** exploration directly into the network weights, allowing the agent to explore in a state-dependent manner. The noise magnitude is a trainable parameter — the network learns *how much* to explore for each weight.

### 2.2 Factorized Gaussian Noise

We use the **factorized** variant for computational efficiency. For a layer with `p` input features and `q` output features:

**Parameters:**
- `μ_w ∈ R^(q×p)` — deterministic weight mean
- `σ_w ∈ R^(q×p)` — learnable noise scale for weights
- `μ_b ∈ R^q` — deterministic bias mean
- `σ_b ∈ R^q` — learnable noise scale for biases

**Noise generation (factorized):**
```
ε_i = f(randn(p))        // input noise vector
ε_j = f(randn(q))        // output noise vector
ε_w = ε_j ⊗ ε_i          // outer product → (q × p) noise matrix
ε_b = ε_j                // bias noise

where f(x) = sgn(x) · √|x|
```

**Effective weights during training:**
```
w = μ_w + σ_w ⊙ ε_w
b = μ_b + σ_b ⊙ ε_b
```

**During evaluation:** noise is disabled; only `μ_w` and `μ_b` are used.

### 2.3 Parameter Initialization

| Parameter | Initialization | Rationale |
|---|---|---|
| `μ_w` | Uniform(−1/√p, 1/√p) | Standard for linear layers |
| `σ_w` | Constant(σ_init / √p), σ_init=0.5 | Fortunato et al. recommendation |
| `μ_b` | Uniform(−1/√p, 1/√p) | Matches weight initialization |
| `σ_b` | Constant(σ_init / √p) | Consistent noise scale |

### 2.4 Usage in the Architecture

NoisyLinear replaces standard `nn.Linear` in the following layers:
- FC1 (128 → 64)
- FC2 (64 → 64)
- Value stream FC3 (64 → 1)
- Advantage stream FC3 (64 → 240)

The GRU and MultiheadAttention layers use standard (non-noisy) parameters, as their internal recurrence and attention computations already introduce sufficient representational diversity.

---

## 3. GRU Layer — Temporal Encoding

**Implementation:** `nn.GRU(input_size=240, hidden_size=128, batch_first=True)`
**Paper reference:** Section 3.3, Figure 7, Ref [26] (Cho et al., 2014)

### 3.1 Motivation

The core challenge of jamming a frequency agile radar is that a single observation is insufficient:

> *"[Dueling DQN] has difficulty providing sufficient information from single observations due to the random generation of radar frequencies. The GRU layer is a variant of a recurrent neural network (RNN) that can remember and update past state information through a self-feedback mechanism. By using the GRU layer, the model can capture long-term dependency relationships in the sequence by utilizing historical observations, thereby better understanding the pattern of changes in radar frequencies."*

### 3.2 Why GRU over LSTM

GRU (Gated Recurrent Unit) has two gates (reset gate, update gate) compared to LSTM's three gates (input, forget, output). This makes it:
- **Computationally lighter** — fewer parameters, faster training per step
- **Sufficient for this task** — the 240-state frequency pattern does not require the additional expressiveness of LSTM's cell state

### 3.3 GRU Gate Equations

At each time step `t`, given input `x_t` and previous hidden state `h_{t-1}`:

```
z_t = σ(W_z · [h_{t-1}, x_t])          // Update gate
r_t = σ(W_r · [h_{t-1}, x_t])          // Reset gate
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])  // Candidate hidden state
h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t // Final hidden state
```

- **Update gate z_t**: Controls how much of the previous hidden state to retain. When the radar exhibits a stable frequency pattern, z_t stays low to preserve learned history.
- **Reset gate r_t**: Determines how much past information to forget. When the radar changes subband, r_t activates to allow the hidden state to adapt.

### 3.4 Dimensions

| Tensor | Shape | Description |
|---|---|---|
| Input `x` | (batch, seq_len, 240) | Sequence of one-hot encoded radar states |
| Hidden `h` | (1, batch, 128) | Single-layer GRU hidden state |
| Output | (batch, seq_len, 128) | Hidden representation at every time step |

### 3.5 Hidden State Across the Episode

The GRU hidden state is:
- **Initialized to zeros** at the start of each episode (10,000-pulse train)
- **Carried forward** between consecutive pulses within the same episode
- **Detached from the computation graph** when stored in replay buffer to prevent backpropagation through the entire episode history

---

## 4. Multi-Head Self-Attention

**Implementation:** `nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)`
**Paper reference:** Section 3.3, Figure 7, Ref [27] (Vaswani et al., 2017)

### 4.1 Motivation

From the paper:

> *"An eight-headed multi-head self-attention module is employed. This module leverages the attention mechanism to learn correlations between different input features, resulting in improved extraction of relevant features and representations from input observations."*

While GRU captures sequential dependencies, self-attention allows the model to directly relate any two positions in the sequence — even distant ones — with O(1) path length. This is critical when radar frequency patterns have long-range correlations (e.g., a subband pattern that repeats every N pulses).

### 4.2 Mechanism

Self-attention is applied to the **GRU output sequence**. Let `H ∈ R^(seq_len × 128)` be the GRU output for one sample:

```
Q = H · W_Q     // Queries
K = H · W_K     // Keys
V = H · W_V     // Values

Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V

where d_k = 128 / 8 = 16  (per-head dimension)
```

Each of the 8 heads independently computes attention with `d_k = 16`, then the 8 outputs are concatenated and linearly projected back to 128 dimensions.

### 4.3 Why Self-Attention After GRU

The GRU already captures temporal dynamics, but self-attention adds two capabilities:
1. **Parallel pairwise comparison**: The attention matrix explicitly scores how relevant each past observation is to every other, enabling the model to detect periodic patterns in subband selection.
2. **Permutation-aware weighting**: Unlike the GRU's strictly sequential processing, attention can assign high weight to a past observation that occurred many steps ago if it is informative for the current decision.

### 4.4 Output Selection

After attention, we extract only the **last time step** representation:

```python
features = attn_out[:, -1, :]   # (batch, 128)
```

This vector encodes the full temporal context (via GRU + attention) and serves as input to the fully-connected layers.

---

## 5. Fully-Connected Feature Extractor

**Implementation:** Two sequential blocks (FC1, FC2) in `src/model.py`
**Paper reference:** Section 3.3, Figure 8, Table 2

### 5.1 Block Structure

Each FC block follows the same pattern:

```
NoisyLinear → LayerNorm → ReLU
```

From the paper:

> *"The addition of layer normalization after the linear layer helps alleviate the issues of gradient vanishing and exploding, improves the convergence of the network, and enhances its generalization capability."*

### 5.2 FC1: Feature Compression

| Component | Specification |
|---|---|
| NoisyLinear | 128 → 64 |
| LayerNorm | normalized_shape=64 |
| Activation | ReLU |

Compresses the 128-dimensional attention output to 64 dimensions, forcing the network to learn a compact representation of the radar's temporal behavior.

### 5.3 FC2: Feature Refinement

| Component | Specification |
|---|---|
| NoisyLinear | 64 → 64 |
| LayerNorm | normalized_shape=64 |
| Activation | ReLU |

Maintains dimensionality while adding another layer of nonlinear transformation, increasing the network's capacity to model complex frequency-action relationships.

### 5.4 LayerNorm Placement

LayerNorm is applied **before** the activation (post-linear, pre-ReLU). This is the "pre-activation" normalization pattern, which normalizes across the feature dimension:

```
LayerNorm(x)_i = γ_i · (x_i − μ) / √(σ² + ε) + β_i
```

where μ and σ² are computed across the 64 features for each sample independently (not across the batch). This makes it batch-size invariant — important for the PER sampling where effective batch composition varies.

---

## 6. Dueling Heads

**Implementation:** `value_stream` and `advantage_stream` in `src/model.py`
**Paper reference:** Section 3.3, Equations 16–18, Figure 6 (right), Figure 8, Ref [24] (Wang et al., 2016)

### 6.1 Motivation

From the paper:

> *"The Dueling DQN decomposes the action value function, Q(s, a), into a state value function, V(s), and an advantage function, A(s, a). V(s) predicts the expected return of a state, while A(s, a) calculates the relative advantage of each action compared to the expected return."*

In the radar jamming context:
- **V(s)**: Estimates the baseline value of the current radar state regardless of which jamming frequency is chosen. For example, if the radar is in a subband with a pattern the jammer has already learned, V(s) is high.
- **A(s,a)**: Estimates how much *better or worse* each specific jamming frequency is compared to average. In a 240-action space where many actions yield zero reward, A(s,a) helps differentiate the few good actions from the many bad ones.

### 6.2 Stream Architecture (Paper Figure 8)

**State Value Stream:**
```
FC2 output (64) → NoisyLinear → scalar V(s)
```

**Advantage Stream:**
```
FC2 output (64) → NoisyLinear → vector A(s,a) ∈ R^240
```

Both streams receive the same 64-dimensional feature vector from FC2 but have independent weights. This shared-feature + separate-head design is shown in Paper Figure 8.

### 6.3 Dimensions

| Stream | Input | Output | Description |
|---|---|---|---|
| Value V(s) | (batch, 64) | (batch, 1) | Scalar state value |
| Advantage A(s,a) | (batch, 64) | (batch, 240) | Per-action advantage |

---

## 7. Q-Value Aggregation

**Implementation:** Final computation in `forward()` method
**Paper reference:** Equation 18

### 7.1 Aggregation Formula

The paper's Equation 18 defines:

```
Q(s, a; w) = V(s; w^V) + A(s, a; w^A) − max_{a∈A} A(s, a; w^A)
```

In practice, the **mean-centering** variant is used for better stability (as recommended by Wang et al. [24]):

```
Q(s, a; w) = V(s; w^V) + ( A(s, a; w^A) − (1/|A|) · Σ_{a'} A(s, a'; w^A) )
```

### 7.2 Why Mean Instead of Max

The max-based formula from Equation 18 ensures that the optimal action has `A*(s,a*) = 0`, making V(s) exactly equal to Q(s,a*). However, in practice, the mean-based variant:
- **Improves optimization stability**: Gradients flow more uniformly across all actions.
- **Reduces variance**: The mean is a smoother operation than max, leading to less noisy updates.
- **Preserves identifiability**: The advantage function is centered (mean = 0), ensuring V(s) approximates the expected Q-value across actions.

### 7.3 Implementation

```python
q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
```

Broadcasting ensures the scalar `value` (batch, 1) is added to each of the 240 action values.

---

## 8. Dimension Flow Summary

Complete tensor shape trace through the network for a single forward pass:

```
Layer                      Input Shape              Output Shape
─────────────────────────  ───────────────────────  ──────────────────────
One-hot encoding           (batch,)                 (batch, seq_len, 240)
GRU                        (batch, seq_len, 240)    (batch, seq_len, 128)
  └─ hidden state          (1, batch, 128)          (1, batch, 128)
Multi-Head Attention       (batch, seq_len, 128)    (batch, seq_len, 128)
Last step extraction       (batch, seq_len, 128)    (batch, 128)
FC1: NoisyLinear           (batch, 128)             (batch, 64)
FC1: LayerNorm + ReLU      (batch, 64)              (batch, 64)
FC2: NoisyLinear           (batch, 64)              (batch, 64)
FC2: LayerNorm + ReLU      (batch, 64)              (batch, 64)
Value Stream               (batch, 64)              (batch, 1)
Advantage Stream           (batch, 64)              (batch, 240)
Q-Value Aggregation        V:(batch,1) A:(batch,240)(batch, 240)
```

### Parameter Count Estimate

| Component | Parameters (approx.) |
|---|---|
| GRU(240, 128) | ~142,080 |
| MultiheadAttention(128, 8) | ~66,048 |
| NoisyLinear FC1 (128→64) | ~16,576 |
| LayerNorm(64) | 128 |
| NoisyLinear FC2 (64→64) | ~8,320 |
| LayerNorm(64) | 128 |
| NoisyLinear Value (64→1) | ~194 |
| NoisyLinear Advantage (64→240) | ~31,200 |
| **Total** | **~264,674** |

---

## 9. Hidden State Management

The GRU hidden state `h ∈ R^(1, batch, 128)` is a critical component that accumulates temporal context across pulses.

### 9.1 Episode Lifecycle

```
Episode start:
    h_0 = zeros(1, 1, 128)            ← init_hidden()

Each pulse t:
    q_values, h_{t+1} = model(x_t, h_t)   ← hidden state passed forward

Episode end:
    h is discarded
```

### 9.2 Training Considerations

During training with PER, transitions are sampled non-sequentially. Two strategies exist:

1. **Stored hidden states** (used here): Store the GRU hidden state as part of each transition in the replay buffer. When sampling, use the stored hidden state to reconstruct the temporal context.

2. **Burn-in sequences**: Store a short prefix of observations before the transition and re-run the GRU to reconstruct the hidden state. This is more memory-efficient but slower.

### 9.3 Utility Method

```python
model.init_hidden(batch_size=1, device=torch.device("cuda"))
```

Returns a zero tensor of shape `(1, batch_size, 128)`.

---

## 10. Noise Reset Protocol

### 10.1 When to Reset

Noise is re-sampled **once per training step** (not per forward pass). This ensures:
- Each gradient update uses a consistent set of noisy weights.
- Exploration varies across training steps but remains stable within a single update.

### 10.2 Method

```python
model.reset_noise()
```

This calls `reset_noise()` on all four NoisyLinear layers (FC1, FC2, Value, Advantage), generating new factorized noise vectors ε_i and ε_j for each.

### 10.3 Training vs. Evaluation

| Mode | Noise Behavior |
|---|---|
| `model.train()` | Noisy weights: `w = μ + σ ⊙ ε` |
| `model.eval()` | Clean weights: `w = μ` (deterministic action selection) |

During evaluation (e.g., computing hit rate on the last episode), the model should be set to `eval()` mode to use the learned deterministic policy without exploration noise.

---

## Appendix: Paper Figure Cross-Reference

| This Document Section | Paper Figure / Equation |
|---|---|
| Architecture Overview | Figure 6 (right), Figure 7 |
| NoisyLinear | Figure 8 (FC_i structure) |
| GRU | Figure 7 (NN Module) |
| Multi-Head Attention | Figure 7 (NN Module) |
| FC1 & FC2 | Figure 8 (FC_i detail), Table 2 |
| Dueling Heads | Figure 6 (left: original, right: GA variant), Figure 8 |
| Q-Value Aggregation | Equation 18 |
| Training Algorithm | Algorithm 1 |
