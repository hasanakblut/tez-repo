# Data Generation, Random Generators, and Model Connection

> Single reference for how data is created, what random generation options exist,
> and how the generator connects to the environment and model.

---

## 1. Overview

There is **no offline dataset**. All data is generated **on the fly** during training:

- **Radar** produces the next frequency index \(s_{t+1}\) via `FrequencyGenerator.next(prev_state)` in `env_utils.py`.
- **Jammer** selects action \(a_t\) based on observation history and receives reward \(r_t = \text{JSR} \times \text{Num}\).
- **Transition** \((s_{\text{history}}, a_t, r_t, s'_{\text{history}}, \text{done})\) is stored in Prioritized Experience Replay.
- **Model** consumes `state_history` (list of int indices) as input: indices → Embedding → GRU–Attention → Q-values.

---

## 2. Data Flow

```
┌─────────────────────────┐     ┌─────────────┐     ┌──────────────────────┐
│ FrequencyGenerator      │────▶│ RadarEnv    │────▶│ Agent / Model        │
│ next(prev_state) → int  │     │ _history[]  │     │ history (indices)    │
└─────────────────────────┘     └─────────────┘     │   → Embedding        │
         │                              │           │   → GRU → Q(s,a)     │
         │                              │           └──────────────────────┘
         │   reset() + step()           │   get_history()
         │   call generator             │   → list of ints
         └──────────────────────────────┴────────────────────────────────────
```

- **Reset:** `generator.reset(seed)` → `generator.next(None)` → \(s_0\) → `_history = [s_0]`.
- **Step:** `generator.next(self._current_state)` → \(s_{t+1}\); compare \(a_t\) vs \(s_{t+1}\); append \(s_{t+1}\) to `_history`.
- **Agent:** `env.get_history()` → list of int indices → `(batch, seq_len)` tensor → model embeds → GRU → Q-values.
- **Learning:** Transitions (histories as list of ints) are sampled from PER; model receives `(batch, seq_len)` indices, embeds, and computes Q.

---

## 3. Random Generation Options

### 3.1 Uniform (Baseline)

| Property | Detail |
|----------|--------|
| **Logic** | `rng.integers(0, state_dim)` — i.i.d. uniform over [0, 239]. |
| **History** | **Not useful.** Each \(s_t\) is independent; GRU cannot exploit patterns. |
| **Use case** | Paper baseline, sanity check. |

### 3.2 Periodic

| Property | Detail |
|----------|--------|
| **Logic** | Cycle through `config["radar"]["periodic_sequence"]`. |
| **History** | **Useful.** GRU can memorize the cycle; hit rate can reach 100%. |
| **Use case** | Sanity check: verify GRU exploits temporal structure. |

### 3.3 LCG

| Property | Detail |
|----------|--------|
| **Logic** | \(x_{n+1} = (a \cdot x_n + c) \bmod m\); map to index in [0, 239]. |
| **History** | **Useful.** Agent can learn recurrence if (a, c, m) are stable. |
| **Use case** | Test model's ability to learn deterministic recurrence. |

### 3.4 Markov

| Property | Detail |
|----------|--------|
| **Logic** | Transition matrix P(i→j); next state sampled from P. Band-limited: from state i, nonzero only for j in [max(0, i-delta), min(239, i+delta)]. |
| **History** | **Partially useful.** Correlations help; randomness limits perfect prediction. |
| **Use case** | Thesis: decision-making under uncertainty. |

### 3.5 Deferred: LFSR, Biased

- **LFSR / Gold Sequences:** Complex bitwise patterns; military-standard frequency hopping.
- **Biased:** Non-uniform distribution over 240; stress tests or curriculum.

---

## 4. Generator Interface

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `FrequencyGenerator(config, state_dim=240, rng=None)` | Set up mode and parameters from config. |
| `reset` | `reset(seed=None)` | Reseed RNG, reset LCG state or Markov position. |
| `next` | `next(prev_state: int \| None) -> int` | Produce next frequency index. |

---

## 5. Config Keys

| Key | Section | Purpose |
|-----|---------|---------|
| `generator_mode` | radar | `"uniform"` \| `"periodic"` \| `"lcg"` \| `"markov"` |
| `reset_generator_on_episode` | radar | If True, call `generator.reset(seed)` on `env.reset()`. |
| `periodic_sequence` | radar | List of indices for periodic mode. |
| `lcg_params` | radar | `{a, c, m}` for LCG. |
| `markov_params` | radar | `{sparsity, delta_limit}` for band-limited P. |
| `markov_transition_path` | radar | Optional path to .npy file for custom P. |
| `embed_dim` | network | Embedding dimension (32, 64, 128); GRU input size. |

---

## 6. Model Input

| Stage | Shape | Description |
|-------|-------|-------------|
| Agent `_encode_history` | `(seq_len,)` | Long tensor of indices; pad with 0 for short histories. |
| Agent `_encode_batch` | `(batch, seq_len)` | Long tensor of indices. |
| Model embedding | `(batch, seq_len, embed_dim)` | Embedding layer output. |
| GRU input | `(batch, seq_len, embed_dim)` | Same as embedding; `input_size=embed_dim`. |

---

## 7. Verification Checklist

When tracing data from generator to model:

1. **Env:** `FrequencyGenerator.next(prev_state)` returns int in [0, 239].
2. **Env:** `get_history()` returns `list` of ints.
3. **Agent:** Stores `state_history` (list of ints) in PER.
4. **Agent learn:** Batch of histories → `(batch, seq_len)` longs → model forward.
5. **Model:** `embedding(x)` → GRU → Q-values.
6. **Logging:** `generator_mode` printed at training start.
