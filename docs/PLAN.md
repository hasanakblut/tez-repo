# GA-Dueling DQN Jamming Decision-Making System — Project Plan & Status

> **Paper:** Xia, L.; Wang, L.; Xie, Z.; Gao, X. *"GA-Dueling DQN Jamming Decision-Making Method for Intra-Pulse Frequency Agile Radar"*, Sensors 2024, 24, 1325.
>
> **Objective:** Faithfully reproduce the paper's simulation results, then extend with thesis-original contributions (tracked separately under Future Work).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Scenario & Domain Model](#2-scenario--domain-model)
3. [Architecture & Module Map](#3-architecture--module-map)
4. [Implementation Phases](#4-implementation-phases)
5. [Detailed Task Backlog](#5-detailed-task-backlog)
6. [Paper Reference Parameters](#6-paper-reference-parameters)
7. [Validation Criteria](#7-validation-criteria)
8. [Future Work (Thesis Novelty)](#8-future-work-thesis-novelty)
9. [Status Log](#9-status-log)

---

## 1. Project Overview

### 1.1 Problem Statement

Intra-Pulse Frequency Agile Radars divide each pulse into K sub-pulses, each on a different carrier frequency within a subband. This creates a large frequency agility space (240 states) that classical jamming methods cannot efficiently learn. A cognitive jammer must predict the radar's next frequency selection to maximize the Jamming-to-Signal Ratio (JSR).

### 1.2 Approach

The paper models the radar-jammer interaction as a Markov Decision Process (MDP) and solves it with GA-Dueling DQN — a Dueling Deep Q-Network augmented with GRU temporal processing and Multi-Head Self-Attention.

### 1.3 Scope Decisions

| Decision | Detail |
|---|---|
| Core scope | Full reproduction of the paper (Sections 2–4) |
| Thesis extensions | Deferred to Future Work; code hooks prepared but not active during baseline training |
| Language / Framework | Python 3.10+, PyTorch, Gymnasium |
| Deliverables | `env.py`, `model.py`, `agent.py`, `train.py`, `visualize.py` |

---

## 2. Scenario & Domain Model

### 2.1 Intra-Pulse Frequency Agile Radar

- Carrier frequency band: F = [10 GHz, 11 GHz], bandwidth B = 1 GHz.
- M = 10 subbands, each 100 MHz wide (Δf = 100 MHz).
- Each pulse contains K = 4 sub-pulses with pairwise-distinct frequencies within the subband.
- Sub-pulse width: Tc = 40 ns; sub-pulse bandwidth = 25 MHz.
- Number of unique sub-pulse permutations per subband: P(4,4) = 24.
- **Total state/action space: 10 × 24 = 240.**

### 2.2 Cognitive Jammer

- Shares the same frequency band and PRI as the radar (obtained via ELINT).
- At time step t, observes radar frequency f_t^(r) and selects jamming frequency f_t^(j).
- Goal: maximize cumulative JSR across the pulse train.

### 2.3 Reward Function

```
r_t = JSR_t × Num

where:
  JSR_t = (A_J² × τ_J × h_j × L) / (A_R² × τ_R × h_t² × σ)
  Num   = number of sub-pulses where jammer frequency matches radar frequency (0–4)
```

### 2.4 MDP Formulation

| Element | Definition |
|---|---|
| State s_t | Radar pulse frequency at time t: f_t^(r) ∈ {0, 1, ..., 239} |
| Action a_t | Jammer pulse frequency for t+1: f_t^(j) ∈ {0, 1, ..., 239} |
| Transition | Radar selects next frequency (random or pattern-based) |
| Reward r_t | JSR × Num (see above) |
| Discount γ | 0.9 |

---

## 3. Architecture & Module Map

### 3.1 File Structure

```
_tez/
├── docs/
│   ├── PLAN.md          # This file — project plan & status tracker
│   ├── MODEL.md         # Detailed model architecture documentation
│   └── ENV.md           # Detailed environment documentation
├── prompts/
│   ├── 01_system_overview_and_deliverables.txt
│   ├── 02_model_architecture_and_agent.txt
│   └── 03_environment_and_radar_simulation.txt
├── src/
│   ├── __init__.py
│   ├── env.py           # RadarEnv (Gymnasium custom environment)        ✅
│   ├── model.py         # GA-Dueling DQN network + NoisyLinear           ✅
│   ├── agent.py         # Agent logic (PER, Double DQN, ε-greedy)        ✅
│   ├── train.py         # Training loop & episode management            ✅
│   └── visualize.py     # Plotting (reward curves, hit rate, freq comparison)
├── configs/
│   └── default.yaml     # Hyperparameters & physical constants           ✅
├── results/             # Saved models, logs, plots
└── requirements.txt     # Dependencies with versions
```

### 3.2 GA-Dueling DQN Network (model.py)

```
Input: one-hot(s_t) ∈ R^240  (or sequence of L past states)
  │
  ▼
┌─────────────────────────────┐
│  GRU Layer (240 → 128)      │  ← captures temporal dependencies
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  Multi-Head Self-Attention   │  ← 8 heads, embed_dim=128
│  (128 → 128)                │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  NoisyLinear FC1 (128 → 64) │
│  + LayerNorm                │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  NoisyLinear FC2 (64 → 64)  │
│  + LayerNorm                │
└──────────┬──────────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌──────────┐
│ V(s)    │ │ A(s,a)   │
│ FC3→1   │ │ FC3→240  │
│ (Noisy) │ │ (Noisy)  │
└────┬────┘ └────┬─────┘
     │           │
     └─────┬─────┘
           ▼
  Q(s,a) = V(s) + (A(s,a) - mean(A))
```

### 3.3 Agent Architecture (agent.py)

- **Policy Network:** GA-Dueling DQN (updated every step).
- **Target Network:** Cloned from policy network, soft/hard updated periodically.
- **Action Selection:** ε-greedy with exponential decay (0.995 → 0.005).
- **Experience Replay:** Prioritized (PER) with TD-error priorities, SumTree structure.
- **Double DQN:** Policy net selects action → Target net evaluates Q-value for TD target.

### 3.4 Training Flow (train.py)

```
for episode in 1..100:
    reset environment → initial radar frequency
    hidden_state = zeros(...)

    for pulse in 1..10_000:
        observe state s_t
        select action a_t (ε-greedy from policy net)
        step environment → s_{t+1}, r_t
        store (s_t, a_t, r_t, s_{t+1}) in PER buffer

        if buffer_size >= batch_size:
            sample prioritized minibatch
            compute Double DQN TD targets
            update policy network
            update priorities in PER

    update target network
    decay epsilon
    log episode reward & hit rate
```

---

## 4. Implementation Phases

### Phase 1: Environment (`env.py`)
Build the Gymnasium-compatible radar simulation with correct physics.

### Phase 2: Model (`model.py`)
Implement NoisyLinear, GA-Dueling DQN network with GRU + Attention + Dueling heads.

### Phase 3: Agent (`agent.py`)
Implement PER (SumTree), Double DQN logic, ε-greedy, and training step.

### Phase 4: Training Loop (`train.py`)
Wire everything together: episode loop, logging, model saving, target net sync.

### Phase 5: Visualization (`visualize.py`)
Reproduce Figure 9 (reward curves) and Table 3 (hit rate) from the paper.

### Phase 6: Validation & Tuning
Run experiments, compare with paper results, tune hyperparameters if needed.

---

## 5. Detailed Task Backlog

### Phase 1 — Environment

| ID | Task | Status | Notes |
|---|---|---|---|
| E-01 | Define state/action space (Discrete 240) | ✅ DONE | 10 subbands × 24 permutations |
| E-02 | Implement radar frequency selection logic | ✅ DONE | Random subband + random permutation per pulse |
| E-03 | Implement sub-pulse matching logic (Num calculation) | ✅ DONE | Compare 4 sub-pulses pairwise |
| E-04 | Implement JSR reward calculation | ✅ DONE | Use paper's physical constants |
| E-05 | Implement `reset()` and `step()` methods | ✅ DONE | Gymnasium API compliance |
| E-06 | Add observation history buffer (sequence of L past states) | ✅ DONE | For GRU input |
| E-07 | Unit tests for environment | ⬜ TODO | Verify reward range, space dims |

### Phase 2 — Model

| ID | Task | Status | Notes |
|---|---|---|---|
| M-01 | Implement `NoisyLinear` layer (Gaussian noise on weights/biases) | ✅ DONE | Custom nn.Module, factorized Gaussian |
| M-02 | Implement GRU layer (240 → 128) | ✅ DONE | Handle hidden state across steps |
| M-03 | Implement Multi-Head Self-Attention (128, 8 heads) | ✅ DONE | nn.MultiheadAttention, batch_first=True |
| M-04 | Implement FC1 (NoisyLinear 128→64 + LayerNorm) | ✅ DONE | NoisyLinear + LayerNorm + ReLU |
| M-05 | Implement FC2 (NoisyLinear 64→64 + LayerNorm) | ✅ DONE | NoisyLinear + LayerNorm + ReLU |
| M-06 | Implement Dueling heads (V: 64→1, A: 64→240) | ✅ DONE | Both NoisyLinear |
| M-07 | Implement Q-value aggregation (mean-centering) | ✅ DONE | Q = V + (A - mean(A)) |
| M-08 | Implement `reset_noise()` method for all NoisyLinear layers | ✅ DONE | Called each training step |
| M-09 | Implement `forward()` with proper GRU hidden state handling | ✅ DONE | Sequence input support |
| M-10 | Unit tests for model (forward pass shape validation) | ⬜ TODO | |

### Phase 3 — Agent

| ID | Task | Status | Notes |
|---|---|---|---|
| A-01 | Implement SumTree data structure for PER | ✅ DONE | Iterative traversal, O(log n) |
| A-02 | Implement PrioritizedReplayBuffer class | ✅ DONE | Store, sample, update priorities, β annealing |
| A-03 | Implement ε-greedy action selection | ✅ DONE | Exponential decay 0.995 → 0.005 |
| A-04 | Implement Double DQN TD target calculation | ✅ DONE | Policy net → argmax, Target net → Q-value |
| A-05 | Implement `learn()` / training step method | ✅ DONE | PER sample, Huber loss, IS-weighted backprop |
| A-06 | Implement target network sync (hard or soft update) | ✅ DONE | Hard copy, configurable frequency |
| A-07 | Implement GRU hidden state management across episode | ✅ DONE | Carried online, None for batch training |
| A-08 | Unit tests for agent | ⬜ TODO | |

### Phase 4 — Training Loop

| ID | Task | Status | Notes |
|---|---|---|---|
| T-01 | Implement main training loop (100 episodes × 10,000 pulses) | ✅ DONE | Algorithm 1 full implementation |
| T-02 | Implement episode logging (total reward, hit rate, epsilon) | ✅ DONE | Console + JSONL metrics file |
| T-03 | Implement model checkpoint saving | ✅ DONE | Periodic + final save |
| T-04 | Implement config loading (hyperparameters from YAML) | ✅ DONE | `load_config()` + `build_env_config()` |
| T-05 | Add CLI arguments / entry point | ✅ DONE | `--config` flag, `python -m src.train` |

### Phase 5 — Visualization

| ID | Task | Status | Notes |
|---|---|---|---|
| V-01 | Plot total reward per episode (Figure 9 reproduction) | ⬜ TODO | All 4 methods if comparing |
| V-02 | Compute and display jamming success probability (Table 3) | ⬜ TODO | Target: ~97.14% |
| V-03 | Plot radar vs jammer carrier frequency comparison (Figure 11) | ⬜ TODO | First 100 pulses of last episode |
| V-04 | Plot epsilon decay curve | ⬜ TODO | |

### Phase 6 — Validation

| ID | Task | Status | Notes |
|---|---|---|---|
| X-01 | Full training run with paper parameters | ⬜ TODO | 100 episodes |
| X-02 | Compare reward curve shape with paper's Figure 9 | ⬜ TODO | |
| X-03 | Verify hit rate converges near 97% (Table 3) | ⬜ TODO | |
| X-04 | Hyperparameter sensitivity analysis (if needed) | ⬜ TODO | |

---

## 6. Paper Reference Parameters

### 6.1 Physical / Simulation Parameters

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| Radar pulse amplitude | A_R | 1 V | Table 1 |
| Jammer pulse amplitude | A_J | 5 V | Table 1 |
| Target RCS | σ | 0.1 m² | Table 1 |
| Radar-target channel gain | h_t | 0.1 dB | Table 1 |
| Radar-jammer channel gain | h_j | 0.1 dB | Table 1 |
| Sidelobe loss | L | 0.05 dB | Table 1 |
| Carrier frequency range | F | [10, 11] GHz | Section 4 |
| Bandwidth | B | 1 GHz | Section 4 |
| Number of subbands | M | 10 | Section 4 |
| Subband interval | Δf | 100 MHz | Section 4 |
| Sub-pulses per pulse | K | 4 | Section 4 |
| Sub-pulse width | Tc | 40 ns | Section 4 |
| Sub-pulse bandwidth | — | 25 MHz | Section 4 |
| Pulses per episode | — | 10,000 | Section 4 |
| Total episodes | — | 100 | Section 4 |

### 6.2 Network & Training Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| GRU dimensions | (240, 128) | Table 2 |
| Multi-Head Attention | (128, 8 heads) | Table 2 |
| FC1 | (128, 64) | Table 2 |
| FC2 | (64, 64) | Table 2 |
| FC3 (Value) | (64, 1) | Table 2 |
| FC3 (Advantage) | (64, 240) | Table 2 |
| Discount factor γ | 0.9 | Table 2 |
| Epsilon start | 0.995 | Table 2 |
| Epsilon end | 0.005 | Table 2 |
| Learning rate | 0.009 | Table 2 |
| Batch size | 256 | Table 2 |

---

## 7. Validation Criteria

The implementation is considered successful when:

| Criterion | Target | Paper Reference |
|---|---|---|
| Reward curve shape | Monotonically increasing, converging around episode 60–80 | Figure 9 |
| Final episode total reward | ~2450 (approximate from Figure 9) | Figure 9 |
| Jamming success probability (no prior knowledge) | ≥ 95% (paper: 97.14%) | Table 3 |
| Jamming success probability (with prior knowledge) | ≥ 98% (paper: 99.41%) | Table 3 |
| Convergence speed | Faster than standalone DQN and Dueling DQN | Figure 9 |

---

## 8. Future Work (Thesis Novelty)

These extensions are **out of scope** for the baseline implementation but will be built on top of the validated core system. Code hooks/interfaces will be prepared where feasible.

| ID | Extension | Description | Prep Status |
|---|---|---|---|
| FW-01 | N-Step Returns | Calculate n-step discounted returns instead of 1-step TD. Speeds up reward propagation. | ⬜ Interface planned |
| FW-02 | Curriculum Learning | 3-phase training scheduler: (1) fixed sub-pulses, (2) limited combinations, (3) full 240-state. | ⬜ Interface planned |
| FW-03 | RAdam + Lookahead Optimizer | Replace Adam with RAdam wrapped in Lookahead for more stable convergence. | ⬜ Interface planned |
| FW-04 | Dynamic GRU Window | Variable-length input sequences (L=5 to L=20) to experiment with temporal depth. | ⬜ Interface planned |
| FW-05 | Prior Knowledge Mode | Jammer knows first sub-pulse frequency via ELINT (Section 4.2 of paper). | ⬜ Interface planned |
| FW-06 | Baseline Comparisons | Implement plain DQN, Dueling DQN, Q-Learning for side-by-side comparison. | ⬜ Interface planned |

---

## 9. Status Log

| Timestamp | Update |
|---|---|
| 2026-02-23 21:30 | Project plan created. Prompts formatted. Paper reviewed. Scope defined: baseline reproduction first, thesis novelty as future work. |
| 2026-02-23 22:00 | Phase 2 (Model) completed: `src/model.py` — NoisyLinear + GADuelingDQN. Documentation: `docs/MODEL.md`. |
| 2026-02-23 22:30 | Phase 1 (Environment) completed: `src/env.py` — RadarEnv with full frequency mapping, JSR reward, Gymnasium API. Documentation: `docs/ENV.md`. |
| 2026-02-23 22:45 | Config created: `configs/default.yaml` — all physical, network, and training parameters centralized. |
| 2026-02-23 22:50 | Prompts reorganized into `prompts/` folder with descriptive filenames. |
| 2026-02-23 23:30 | Phase 3 (Agent) completed: `src/agent.py` — SumTree, PER, JammingAgent with Double DQN. Documentation: `docs/AGENT.md`. |
| 2026-02-23 23:30 | Phase 4 (Training) completed: `src/train.py` — Full training loop with YAML config, JSONL logging, checkpointing, CLI entry point. |
| 2026-02-23 23:30 | Prompt 4 formatted and renamed to `prompts/04_agent_logic_and_training_loop.txt`. |

---

*Last updated: 2026-02-23 23:30*
