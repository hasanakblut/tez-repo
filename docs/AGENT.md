# JammingAgent — Agent & Training Documentation

> **Implementation files:** `src/agent.py`, `src/train.py`
>
> **Paper reference:** Xia, L.; Wang, L.; Xie, Z.; Gao, X. *"GA-Dueling DQN Jamming Decision-Making Method for Intra-Pulse Frequency Agile Radar"*, Sensors 2024, 24, 1325.

---

## Table of Contents

1. [Agent Overview](#1-agent-overview)
2. [Dual Network Architecture](#2-dual-network-architecture)
3. [Epsilon-Greedy Exploration](#3-epsilon-greedy-exploration)
4. [Double DQN Update Rule](#4-double-dqn-update-rule)
5. [Prioritized Experience Replay](#5-prioritized-experience-replay)
6. [SumTree Data Structure](#6-sumtree-data-structure)
7. [Training Loop](#7-training-loop)
8. [GRU Hidden State Management](#8-gru-hidden-state-management)
9. [State Encoding Pipeline](#9-state-encoding-pipeline)
10. [Hyperparameter Reference](#10-hyperparameter-reference)

---

## 1. Agent Overview

The `JammingAgent` translates the theoretical framework of Paper Algorithm 1 into an operational RL agent. It orchestrates:

| Component | Role | Paper Reference |
|---|---|---|
| Policy Network | Selects actions & receives gradient updates | Figure 5 (left net) |
| Target Network | Provides stable TD-target Q-values | Figure 5 (right net) |
| ε-greedy | Balances exploration vs. exploitation | Section 3.3 |
| PER Buffer | Stores & prioritizes transitions by TD-error | Algorithm 1, line 11 |
| Double DQN | Mitigates Q-value overestimation | Section 3.3, Ref [25] |

### Agent Lifecycle per Episode

```
episode_start:
    agent.reset_hidden()                    ← zero GRU hidden state

for each pulse t in 1..10,000:
    history = env.get_history()             ← last L observations
    action = agent.select_action(history)   ← ε-greedy over policy net
    obs, reward, done, info = env.step(action)
    agent.store_transition(...)             ← add to PER
    agent.learn()                           ← sample PER, Double DQN update

episode_end:
    agent.decay_epsilon()                   ← exponential ε decay
    agent.update_target_network()           ← hard-copy policy → target
    agent.memory.anneal_beta(...)           ← increase IS weight β
```

---

## 2. Dual Network Architecture

### 2.1 Policy Network vs. Target Network (Paper Figure 5)

From the paper:

> *"This paper adopts a double Q-learning algorithm to update parameters. The first step involves selecting an action that maximizes the output of the DQN, based on state s, using the policy network. The second step calculates the value using the target network."*

Both networks are identical `GADuelingDQN` instances:

```
Policy Network (θ)                 Target Network (θ⁻)
┌────────────────────┐             ┌────────────────────┐
│  GRU → Attention   │             │  GRU → Attention   │
│  FC1 → FC2         │             │  FC1 → FC2         │
│  V(s) + A(s,a)     │             │  V(s) + A(s,a)     │
└────────┬───────────┘             └────────┬───────────┘
         │                                  │
    Updated every step              Periodically synced
    via backpropagation             from policy network
```

### 2.2 Target Network Synchronization

The target network is updated via **hard copy** (not soft/Polyak averaging):

```python
target_net.load_state_dict(policy_net.state_dict())
```

This happens every `target_update_freq` episodes (default: 1, i.e., after each episode). The hard copy ensures the target Q-values are periodically refreshed while remaining stable within an episode.

---

## 3. Epsilon-Greedy Exploration

### 3.1 Strategy (Paper Section 3.3)

From the paper:

> *"At each time step, the jammer selects the frequency using an ε-greedy algorithm based on the current state's Q-values. It selects the action with the largest Q-value, with a probability of 1 − ε, while randomly choosing actions with a probability of ε."*

```
if random() < ε:
    action = random_action(0, 239)        ← exploration
else:
    action = argmax_a Q_policy(s, a)      ← exploitation
```

### 3.2 Exponential Decay Schedule

From the paper:

> *"Exploration rate ε follows an exponential decay schedule from 0.995 to 0.005, maintaining a consistent decay rate."*

The decay factor per episode is computed as:

```
decay = (ε_end / ε_start) ^ (1 / num_episodes)
      = (0.005 / 0.995) ^ (1/100)
      ≈ 0.9474

After each episode:
    ε ← max(ε_end, ε × decay)
```

| Episode | ε (approximate) |
|---|---|
| 1 | 0.995 |
| 10 | 0.582 |
| 25 | 0.253 |
| 50 | 0.064 |
| 75 | 0.016 |
| 100 | 0.005 |

This aggressive early decay ensures the agent moves quickly from random exploration to learned behavior, which is critical in a 240-action space where random hits have only 1/240 ≈ 0.42% probability.

---

## 4. Double DQN Update Rule

### 4.1 Motivation (Paper Section 3.3)

From the paper:

> *"In Equation (9), Q-learning uses its own estimates to update Q-values, which leads to bias propagation. Moreover, the maximization operation often results in overestimating the TD targets [...] To alleviate overestimation, this paper adopts a double Q-learning algorithm."*

Standard DQN uses `max_a Q(s', a; θ)` for both action selection and evaluation, leading to systematic overestimation. Double DQN decouples these:

### 4.2 Two-Step Process

**Step 1 — Action selection** (using policy network θ):
```
a* = argmax_a Q(s_{t+1}, a; θ)
```

**Step 2 — Value evaluation** (using target network θ⁻):
```
y_t = r_t + γ · Q(s_{t+1}, a*; θ⁻)
```

### 4.3 Loss Computation

The loss for a single transition is:

```
δ_t = y_t - Q(s_t, a_t; θ)                    ← TD error
L = SmoothL1Loss(Q(s_t, a_t; θ), y_t)         ← Huber loss (robust to outliers)
```

For PER, the loss is weighted by importance-sampling weights:

```
L_weighted = (1/B) · Σ_i  w_i · SmoothL1(Q(s_i, a_i; θ), y_i)
```

### 4.4 Gradient Update (Paper Equation 14)

```
θ ← θ − α · ∇_θ L_weighted
```

Gradient clipping (`max_norm=10.0`) prevents exploding gradients from large TD-errors in the early unstable training phase.

---

## 5. Prioritized Experience Replay

### 5.1 Motivation (Paper Section 3.3, Algorithm 1 line 11)

From the paper:

> *"During the learning phase, the jammer samples a small batch of experiences from the prioritized experience replay buffer (PER) according to non-uniform weights to update the network."*

Uniform sampling wastes training signal on already well-predicted transitions. PER samples transitions proportional to their TD-error magnitude, focusing learning on surprising or poorly predicted experiences.

### 5.2 Priority Computation

Each transition `i` is assigned a priority:

```
p_i = (|δ_i| + ε_min)^α
```

where:
- `|δ_i|` = absolute TD-error of the transition
- `ε_min` = small constant (1e-6) preventing zero-priority
- `α` = priority exponent (0.6); controls prioritization strength

### 5.3 Sampling Probability

The probability of sampling transition `i`:

```
P(i) = p_i / Σ_k p_k
```

### 5.4 Importance-Sampling Correction

Prioritized sampling introduces bias (non-uniform distribution). This is corrected via importance-sampling weights:

```
w_i = (1 / (N · P(i)))^β / max_j(w_j)
```

where:
- `N` = buffer size
- `β` = annealed from 0.4 → 1.0 over training (full correction at convergence)
- Normalization by `max_j(w_j)` ensures weights ≤ 1 for stability

### 5.5 Transition Format

Each stored transition is a tuple:

```
(state_history, action, reward, next_state_history, done)
```

| Field | Type | Description |
|---|---|---|
| state_history | `List[int]` | Last L observation indices at time of action |
| action | `int` | Chosen frequency index (0–239) |
| reward | `float` | JSR_base × Num |
| next_state_history | `List[int]` | Last L observation indices after env step |
| done | `bool` | Whether episode terminated |

Histories are stored as raw integer lists (memory-efficient) and converted to one-hot tensors only during batch sampling.

---

## 6. SumTree Data Structure

### 6.1 Structure

The SumTree enables O(log n) proportional sampling from the priority distribution. It is a complete binary tree where:
- **Leaf nodes** store individual transition priorities.
- **Internal nodes** store the sum of their children.
- **Root** stores the total priority sum.

```
                    [Total = 42]                          ← root
                   /            \
             [25]                [17]                     ← internal
            /    \              /    \
         [12]    [13]        [9]     [8]                  ← internal
        /   \   /   \       /  \    /  \
       [5] [7][6] [7]     [4] [5] [3] [5]                ← leaves (priorities)
```

### 6.2 Operations

| Operation | Complexity | Description |
|---|---|---|
| `add(priority, data)` | O(log n) | Insert transition at write position |
| `sample(value)` | O(log n) | Find leaf containing cumulative value |
| `update(idx, priority)` | O(log n) | Change a leaf's priority, propagate up |
| `total` | O(1) | Read root node |

### 6.3 Proportional Sampling

To sample a batch of B transitions:

1. Divide `[0, total_priority)` into B equal segments.
2. Sample one uniform random value from each segment.
3. Traverse the tree to find the leaf containing each value.

This stratified sampling ensures coverage across the full priority distribution.

---

## 7. Training Loop

### 7.1 Overall Structure (`src/train.py`)

```
load config from YAML
create RadarEnv
create JammingAgent

for episode = 1 to 100:
    env.reset()
    agent.reset_hidden()

    for pulse = 1 to 10,000:
        history = env.get_history()
        action = agent.select_action(history)
        obs, reward, done, info = env.step(action)
        next_history = env.get_history()
        agent.store_transition(history, action, reward, next_history, done)
        agent.learn()

    agent.decay_epsilon()
    agent.update_target_network()
    agent.memory.anneal_beta(episode, 100)

    log metrics (reward, hit_rate, loss, epsilon)
    save checkpoint if needed

save final model
save metrics to JSONL
```

### 7.2 Metrics Tracked Per Episode

| Metric | Description | Paper Reference |
|---|---|---|
| total_reward | Cumulative JSR × Num over 10,000 pulses | Figure 9 (y-axis) |
| hit_rate | Fraction of pulses with Num=4 (perfect jam) | Table 3 |
| avg_loss | Mean Huber loss across all training steps | — |
| epsilon | Current exploration rate | Section 3.3 |
| beta | Current PER importance-sampling exponent | — |
| time_sec | Wall-clock time for the episode | — |

### 7.3 Output Files

| File | Format | Content |
|---|---|---|
| `results/metrics.jsonl` | JSON Lines | One JSON object per episode with all metrics |
| `results/checkpoint_ep{N}.pt` | PyTorch | Model state, optimizer state, epsilon, step count |
| `results/final_model.pt` | PyTorch | Final trained model |

---

## 8. GRU Hidden State Management

### 8.1 Online Inference

During action selection within an episode, the GRU hidden state is **carried forward** between consecutive `select_action` calls:

```
pulse 1: q, h1 = policy_net(history_window, h0=zeros)
pulse 2: q, h2 = policy_net(history_window, h1)
pulse 3: q, h3 = policy_net(history_window, h2)
...
```

This allows the GRU to accumulate temporal context beyond the observation window length, capturing long-range frequency patterns.

### 8.2 Batch Training

During `learn()`, transitions are sampled non-sequentially from the PER buffer. The GRU hidden state is set to `None` (zero-initialized internally), and the full observation window provides temporal context:

```
q_values, _ = policy_net(state_batch, hidden=None)
```

The observation window (default L=10) gives the GRU sufficient input to extract temporal features. This trade-off (no stored hidden state during training) is standard in recurrent DRL and avoids the memory/staleness issues of storing hidden states in replay.

### 8.3 Episode Boundaries

At the start of each episode, `agent.reset_hidden()` zeros the GRU state, preventing information leakage between independent pulse trains.

---

## 9. State Encoding Pipeline

### 9.1 Raw State → Network Input

The encoding pipeline transforms raw integer state indices into the one-hot tensor format expected by the GA-Dueling DQN network:

```
env.get_history() → [142, 57, 203, 88, 15]     (list of int indices)
        │
        ▼ _encode_history()
    left-pad to seq_len=10 with zero vectors
        │
        ▼
    one-hot encode each index
        │
        ▼
    Tensor shape: (10, 240)                      (seq_len, state_dim)
        │
        ▼ .unsqueeze(0)  or  _encode_batch()
    Tensor shape: (1, 10, 240)  or  (batch, 10, 240)
```

### 9.2 Left-Padding

At the start of an episode, the history is shorter than `seq_len`. Short histories are left-padded with zero vectors:

```
Episode step 1: history = [s0]
Encoded:  [0, 0, 0, 0, 0, 0, 0, 0, 0, s0]     (9 zero vectors + 1 one-hot)

Episode step 5: history = [s0, s1, s2, s3, s4]
Encoded:  [0, 0, 0, 0, 0, s0, s1, s2, s3, s4]  (5 zero vectors + 5 one-hots)

Episode step 10+: history = [s_{t-9}, ..., s_t]
Encoded:  [s_{t-9}, s_{t-8}, ..., s_t]          (10 one-hots, no padding)
```

### 9.3 Memory-Efficient Storage

Transitions in the PER buffer store histories as raw `List[int]` (not tensors). One-hot conversion happens only during the `learn()` call, when a minibatch is sampled. This reduces buffer memory from ~960 MB to ~8 MB for 100K transitions.

---

## 10. Hyperparameter Reference

All values sourced from `configs/default.yaml`:

### Training

| Parameter | Value | Paper Source |
|---|---|---|
| Discount factor γ | 0.9 | Table 2 |
| Learning rate α | 0.009 | Table 2 |
| Batch size | 256 | Table 2 |
| ε start | 0.995 | Table 2 |
| ε end | 0.005 | Table 2 |
| ε decay episodes | 100 | Section 3.3 |
| Target update freq | 1 (per episode) | Section 3.3 |
| Gradient clip norm | 10.0 | — |

### Prioritized Replay

| Parameter | Value | Role |
|---|---|---|
| Buffer capacity | 100,000 | Max stored transitions |
| α (priority exponent) | 0.6 | Prioritization strength |
| β start | 0.4 | Initial IS correction |
| β end | 1.0 | Full IS correction at convergence |
| ε_min (min priority) | 1e-6 | Prevent zero-probability transitions |

---

## Appendix: Paper Cross-Reference

| This Document Section | Paper Figure / Equation / Section |
|---|---|
| Dual Network Architecture | Figure 5 |
| ε-greedy | Section 3.3 |
| Double DQN | Section 3.3, Equation 11, Ref [25] |
| PER | Section 3.3, Algorithm 1 line 11, Ref [14] |
| Training Algorithm | Algorithm 1 |
| Loss Function | Equations 12–14 |
| Target Network Update | Figure 5 |
