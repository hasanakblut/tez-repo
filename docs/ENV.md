# RadarEnv — Environment Architecture Documentation

> **Implementation file:** `src/env.py`
>
> **Paper reference:** Xia, L.; Wang, L.; Xie, Z.; Gao, X. *"GA-Dueling DQN Jamming Decision-Making Method for Intra-Pulse Frequency Agile Radar"*, Sensors 2024, 24, 1325.

---

## Table of Contents

1. [Environment Overview](#1-environment-overview)
2. [Intra-Pulse Frequency Agile Radar Model](#2-intra-pulse-frequency-agile-radar-model)
3. [Frequency Space & Combinatorial Mapping](#3-frequency-space--combinatorial-mapping)
4. [MDP Formulation](#4-mdp-formulation)
5. [Reward Function — JSR Derivation](#5-reward-function--jsr-derivation)
6. [Sub-Pulse Matching Logic](#6-sub-pulse-matching-logic)
7. [Gymnasium Interface Contract](#7-gymnasium-interface-contract)
8. [Observation History Buffer](#8-observation-history-buffer)
9. [Configuration Parameters](#9-configuration-parameters)
10. [Utility Functions](#10-utility-functions)

---

## 1. Environment Overview

The `RadarEnv` simulates the adversarial interaction between an **intra-pulse frequency agile radar** and a **cognitive jammer**. The jammer (RL agent) must learn the radar's frequency-hopping patterns to maximize jamming effectiveness.

From the paper (Section 2.2):

> *"The mission of the jammer is to protect the target by reducing the probability of detection by the radar. Therefore, the Jamming-to-Signal ratio (JSR) is chosen as the objective function for the jammer."*

### Key Design Principles

| Principle | Implementation |
|---|---|
| Faithful physics | JSR computation follows Equation 15 with Table 1 constants |
| Correct combinatorics | 240-state space from 10 subbands × 24 permutations |
| Gymnasium compliance | Standard `reset()` / `step()` API with proper termination signals |
| Agent-friendly | History buffer, decode/encode helpers, rich info dict |

---

## 2. Intra-Pulse Frequency Agile Radar Model

### 2.1 Pulse Structure (Paper Section 2.1, Figure 1)

The radar transmits a train of pulses with a fixed Pulse Repetition Period (T). Each pulse is divided into K sub-pulses, where each sub-pulse carries a different carrier frequency:

```
Pulse n:
┌──────────┬──────────┬──────────┬──────────┐
│ Sub-pulse│ Sub-pulse│ Sub-pulse│ Sub-pulse│
│    k=0   │    k=1   │    k=2   │    k=3   │
│  f_0^(n) │  f_1^(n) │  f_2^(n) │  f_3^(n) │
│   Tc     │   Tc     │   Tc     │   Tc     │
└──────────┴──────────┴──────────┴──────────┘
           ◄── Pulse width Tp = K × Tc ──►
```

Paper Equation 1 defines the transmitted signal:

```
s_T^(n)(t) = Σ_{k=0}^{K-1} rect(t - k·Tc) · u(t - k·Tc) · exp(j2π·f_k·t)
```

### 2.2 Frequency Band Organization (Paper Section 2.1, Figure 2)

The radar operates within a frequency band F = [f_L, f_H]:

```
F = [10 GHz ──────────────────────────── 11 GHz]
     │  SB0  │  SB1  │  SB2  │ ... │  SB9  │
     │100 MHz│100 MHz│100 MHz│     │100 MHz│

Within each subband (100 MHz):
     │ Slot 0│ Slot 1│ Slot 2│ Slot 3│
     │ 25 MHz│ 25 MHz│ 25 MHz│ 25 MHz│
```

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| Lower frequency | f_L | 10 GHz | Section 4 |
| Upper frequency | f_H | 11 GHz | Section 4 |
| Total bandwidth | B | 1 GHz | Section 4 |
| Number of subbands | M | 10 | Section 4 |
| Subband width | Δf | 100 MHz | Section 4 |
| Sub-pulses per pulse | K | 4 | Section 4 |
| Sub-pulse width | Tc | 40 ns | Section 4 |
| Sub-pulse bandwidth | Δf/K | 25 MHz | Section 4 |

### 2.3 Frequency Agility

The radar has **two levels** of frequency agility:

1. **Inter-pulse agility:** Between consecutive pulses, the radar randomly selects a different subband (Section 2.1: *"the radar can randomly select different subbands for each pulse"*).

2. **Intra-pulse agility:** Within a single pulse, the K=4 sub-pulses use different frequency slots within the selected subband, in a random permutation order.

This dual-level agility creates the 240-dimensional state space that makes jamming extremely challenging.

---

## 3. Frequency Space & Combinatorial Mapping

### 3.1 State/Action Space Construction

The total number of distinct radar configurations is:

```
|S| = M × P(K, K) = 10 × 4! = 10 × 24 = 240
```

Paper Section 4: *"the state space of the frequency agile radar is 24 × 10 = 240. The action space of the jammer is the same as the state space."*

### 3.2 Index Encoding Scheme

Each state/action is mapped to a flat integer index:

```
index = subband_id × 24 + permutation_id

where:
    subband_id    ∈ {0, 1, ..., 9}
    permutation_id ∈ {0, 1, ..., 23}
```

**Example decoding:**

| Index | Subband | Perm ID | Sub-pulse order | Frequencies (GHz) |
|---|---|---|---|---|
| 0 | 0 | 0 | (0,1,2,3) | (10.000, 10.025, 10.050, 10.075) |
| 1 | 0 | 1 | (0,1,3,2) | (10.000, 10.025, 10.075, 10.050) |
| 23 | 0 | 23 | (3,2,1,0) | (10.075, 10.050, 10.025, 10.000) |
| 24 | 1 | 0 | (0,1,2,3) | (10.100, 10.125, 10.150, 10.175) |
| 239 | 9 | 23 | (3,2,1,0) | (10.975, 10.950, 10.925, 10.900) |

### 3.3 Permutation Table

All 24 permutations of `(0, 1, 2, 3)` are pre-computed at module load time and stored in `PERMUTATION_TABLE`. The ordering follows Python's `itertools.permutations` — lexicographic order:

```
Index 0:  (0, 1, 2, 3)
Index 1:  (0, 1, 3, 2)
Index 2:  (0, 2, 1, 3)
...
Index 23: (3, 2, 1, 0)
```

### 3.4 Utility Functions

| Function | Signature | Description |
|---|---|---|
| `index_to_subpulses` | `(index) → (subband_id, perm, frequencies)` | Decode flat index to physical representation |
| `subpulses_to_index` | `(subband_id, perm) → index` | Encode back to flat index |
| `env.decode_index` | `(index) → (subband_id, perm, frequencies)` | Instance method wrapper |
| `env.encode_index` | `(subband_id, perm) → index` | Instance method wrapper |

---

## 4. MDP Formulation

### 4.1 Interaction Model (Paper Section 3.2, Figure 4)

The radar-jammer interaction is modeled as a Markov Decision Process:

```
                    ┌─────────────┐
    s_t ──────────► │   JAMMER    │ ──── a_t (prediction for t+1)
                    │   (Agent)   │
                    └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
    s_{t+1} ◄────── │    RADAR    │
    r_t     ◄────── │   (Env)     │ ──── compares a_t with s_{t+1}
                    └─────────────┘
```

From the paper (Section 3.2):

> *"At time t, the jammer is able to capture radar pulse signal s_t at time t, and takes the optimal jamming action, a_t, according to policy function π. After the jammer executes the action, the environment state transitions from s_t to s_{t+1}. The reward, r_t, is the Jamming-to-Signal ratio received by the radar."*

### 4.2 MDP Elements

| Element | Symbol | Definition | Implementation |
|---|---|---|---|
| State | s_t | Radar frequency at time t | `observation_space = Discrete(240)` |
| Action | a_t | Jammer's predicted frequency for t+1 | `action_space = Discrete(240)` |
| Transition | P(s'|s,a) | Radar selects next frequency randomly | `_generate_next_state()` |
| Reward | r_t | JSR_base × Num | `jsr_base * count_subpulse_matches(...)` |
| Discount | γ | 0.9 | Configured in agent, not in environment |

### 4.3 Step Function Flow

```
step(action=a_t):
    ┌──────────────────────────────────────────────────┐
    │ 1. s_{t+1} = _generate_next_state()              │
    │    (radar picks new random frequency)             │
    │                                                    │
    │ 2. Num = count_subpulse_matches(a_t, s_{t+1})    │
    │    (how many sub-pulses did the jammer get right?) │
    │                                                    │
    │ 3. reward = JSR_base × Num                        │
    │                                                    │
    │ 4. pulse_count += 1                               │
    │    terminated = (pulse_count >= 10,000)            │
    │                                                    │
    │ 5. current_state = s_{t+1}                        │
    │    append to history buffer                        │
    │                                                    │
    │ 6. return (s_{t+1}, reward, terminated, info)     │
    └──────────────────────────────────────────────────┘
```

### 4.4 Episode Structure

One episode represents one complete **pulse train**:

- **Length:** 10,000 pulses (Paper Section 4)
- **Termination:** When `pulse_count >= max_pulses`
- **No truncation:** Episodes always run to completion

---

## 5. Reward Function — JSR Derivation

### 5.1 Power Calculations (Paper Equations 3–4)

The transmit powers of the radar and jammer depend on pulse amplitude and width:

```
P_R = (A_R² × τ_R) / (2T)     [Equation 3]
P_J = (A_J² × τ_J) / (2T)     [Equation 4]
```

Since both the radar and jammer operate with the same pulse structure (K=4 sub-pulses, each of width Tc), we have τ_R = τ_J. The pulse repetition period T is also shared. Therefore, τ and T cancel when computing the JSR ratio.

### 5.2 Jamming-to-Signal Ratio (Paper Equation 5)

The received JSR at the radar is:

```
JSR = (P_J × h_j × L) / (P_R × h_t² × σ)     [Equation 5]
```

After τ and T cancellation:

```
JSR_base = (A_J² × h_j × L) / (A_R² × h_t² × σ)
```

### 5.3 Per-Step Reward (Paper Equation 15)

```
r_t = JSR_base × Num
```

where Num ∈ {0, 1, 2, 3, 4} is the number of sub-pulses where the jammer's frequency matches the radar's frequency at the corresponding position.

### 5.4 Numerical Computation with Paper Constants

Using the values from Paper Table 1:

```
JSR_base = (5² × 0.1 × 0.05) / (1² × 0.1² × 0.1)
         = (25 × 0.1 × 0.05) / (1 × 0.01 × 0.1)
         = 0.125 / 0.001
         = 125.0
```

| Num (matched sub-pulses) | Reward |
|---|---|
| 0 | 0.0 |
| 1 | 125.0 |
| 2 | 250.0 |
| 3 | 375.0 |
| 4 (perfect jam) | 500.0 |

### 5.5 Objective

From the paper (Section 3.2):

> *"The goal of the cognitive jammer is to find the jamming frequency selection policy, π*(a|s), that has the maximum average Jamming-to-Signal ratio."*

The agent maximizes the cumulative reward across the 10,000-pulse episode.

---

## 6. Sub-Pulse Matching Logic

### 6.1 Position-Wise Comparison

Matching is performed **position by position** across the K=4 sub-pulses. This means the order of sub-pulse frequencies matters — not just which frequencies are used.

```
Radar  s_{t+1} → subband=3, perm=(2, 0, 3, 1)
Jammer a_t     → subband=3, perm=(2, 1, 3, 0)

Position:   k=0    k=1    k=2    k=3
Radar:       2      0      3      1
Jammer:      2      1      3      0
Match?       ✓      ✗      ✓      ✗

Num = 2
```

### 6.2 Cross-Subband Actions

If the jammer selects a different subband than the radar, all sub-pulse frequencies are in entirely different 100 MHz bands. No position can match:

```
Radar  → subband=3 → frequencies in [10.300, 10.400) GHz
Jammer → subband=7 → frequencies in [10.700, 10.800) GHz

Num = 0 (guaranteed)
```

### 6.3 Same-Subband, Same-Permutation

The only way to achieve Num=4 (perfect jam) is to match both the subband AND the exact permutation:

```
Radar  → subband=3, perm=(2, 0, 3, 1) → index = 3×24 + perm_id
Jammer → subband=3, perm=(2, 0, 3, 1) → same index

Num = 4 (perfect jam, all sub-pulses matched)
```

### 6.4 Probability Analysis

For a **random** jammer (uniform action selection):
- P(correct subband) = 1/10
- P(correct permutation | correct subband) = 1/24
- P(Num = 4) = 1/240 ≈ 0.42%
- Expected Num per step = 4/240 ≈ 0.0167

The GA-Dueling DQN achieves **97.14%** hit rate (Num=4), a ~230× improvement over random.

---

## 7. Gymnasium Interface Contract

### 7.1 `reset(seed, options) → (observation, info)`

| Return | Type | Description |
|---|---|---|
| observation | `int` | Initial radar state s_0 ∈ {0, ..., 239} |
| info | `dict` | `{"pulse": 0, "num_matches": 0, "hit_rate": 0.0, ...}` |

### 7.2 `step(action) → (observation, reward, terminated, truncated, info)`

| Return | Type | Description |
|---|---|---|
| observation | `int` | Next radar state s_{t+1} ∈ {0, ..., 239} |
| reward | `float` | JSR_base × Num |
| terminated | `bool` | `True` when pulse_count ≥ max_pulses |
| truncated | `bool` | Always `False` |
| info | `dict` | See info dict specification below |

### 7.3 Info Dict Specification

| Key | Type | Description |
|---|---|---|
| `pulse` | `int` | Current pulse count (1-indexed after step) |
| `num_matches` | `int` | Num value for this step (0–4) |
| `hit_rate` | `float` | Cumulative fraction of perfect jams (Num=4) so far |
| `total_matches` | `int` | Cumulative sum of all Num values |
| `total_full_hits` | `int` | Count of steps where Num=4 |
| `jsr_base` | `float` | The constant JSR_base value |

---

## 8. Observation History Buffer

### 8.1 Purpose

The GA-Dueling DQN model uses a GRU layer that requires a **sequence** of past observations. The environment maintains a rolling history buffer of the last `history_len` observations.

### 8.2 Configuration

| Parameter | Default | Description |
|---|---|---|
| `history_len` | 10 | Number of past states to retain |

### 8.3 Usage

```python
env = RadarEnv(config={"history_len": 10})
obs, info = env.reset()

for t in range(10_000):
    history = env.get_history()  # list of up to 10 recent state indices
    # Agent converts history to one-hot sequence for GRU input
    action = agent.select_action(history)
    obs, reward, done, truncated, info = env.step(action)
```

### 8.4 Behavior

- On `reset()`: history is initialized with `[s_0]`.
- On each `step()`: the new state s_{t+1} is appended. If the buffer exceeds `history_len`, the oldest entry is removed (FIFO).

---

## 9. Configuration Parameters

All parameters can be overridden via the `config` dict passed to `RadarEnv.__init__()`:

```python
config = {
    # Physical (Paper Table 1)
    "A_R": 1.0,
    "A_J": 5.0,
    "h_t": 0.1,
    "h_j": 0.1,
    "sigma": 0.1,
    "L": 0.05,

    # Radar structure (Paper Section 4)
    "M": 10,
    "K": 4,
    "f_low": 10e9,
    "f_high": 11e9,

    # Episode
    "max_pulses": 10_000,

    # History buffer
    "history_len": 10,
}
```

These values are also centralized in `configs/default.yaml` for reproducibility.

---

## 10. Utility Functions

### 10.1 Module-Level Functions

These are importable directly from `src.env`:

```python
from src.env import (
    index_to_subpulses,
    subpulses_to_index,
    count_subpulse_matches,
    PERMUTATION_TABLE,
)
```

| Function | Purpose |
|---|---|
| `build_permutation_table(k)` | Generate all k! permutations; called once at import |
| `index_to_subpulses(index, ...)` | Flat index → (subband, perm, frequencies) |
| `subpulses_to_index(subband, perm, ...)` | (subband, perm) → flat index |
| `count_subpulse_matches(action, radar, ...)` | Compute Num (0–4) for reward calculation |

### 10.2 Instance Methods

| Method | Purpose |
|---|---|
| `env.decode_index(index)` | Wrapper for `index_to_subpulses` with env parameters |
| `env.encode_index(subband, perm)` | Wrapper for `subpulses_to_index` |
| `env.get_history()` | Return recent observation sequence for GRU input |

---

## Appendix: Paper Figure Cross-Reference

| This Document Section | Paper Figure / Equation / Section |
|---|---|
| Pulse Structure | Section 2.1, Figure 1, Equation 1 |
| Frequency Band Organization | Section 2.1, Figure 2, Equation 2 |
| Countermeasure Scenario | Section 2.2, Figure 3 |
| MDP Formulation | Section 3.2, Figure 4 |
| Power Calculations | Section 2.2, Equations 3–4 |
| JSR Formula | Section 2.2, Equation 5 |
| Reward Function | Section 3.2, Equation 15 |
| State/Action Space | Section 3.2, Section 4 |
| Algorithm Flow | Algorithm 1 |
