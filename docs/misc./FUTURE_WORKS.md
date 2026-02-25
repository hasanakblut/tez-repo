# Future Works — Thesis Extensions & Novelty

> Consolidated list of future-work topics from **docs/future_works_dummy.txt** and **docs/PLAN.md** (Section 8), with detailed explanations, motivation, benefits, and critical implementation notes.  
> Concepts cross-checked with standard RL/optimization literature where noted.

---

## Table of Contents

1. [Autonomous Subband Discovery (Clustering)](#1-autonomous-subband-discovery-clustering)
2. [Curriculum Learning (CL) Framework](#2-curriculum-learning-cl-framework)
3. [N-Step Returns](#3-n-step-returns)
4. [RAdam + Lookahead Optimizer](#4-radam--lookahead-optimizer)
5. [Dynamic GRU Window](#5-dynamic-gru-window)
6. [Prior Knowledge Mode](#6-prior-knowledge-mode)
7. [Baseline Comparisons](#7-baseline-comparisons)

---

## 1. Autonomous Subband Discovery (Clustering)

**Source:** future_works_dummy.txt

### What It Is

Unsupervised learning to **estimate the number of active radar subbands $M$** without prior hardware/ELINT knowledge. The radar’s 70/30 subband persistence creates dense clusters in the frequency (index) domain that can be separated from noise.

### Methodology

- **Algorithms:** Hidden Markov Models (HMM) and Gaussian Mixture Models (GMM) to capture temporal and distributional structure. HMMs model sequential dependence; GMMs (or HMMs with Gaussian emissions) model observations per “state” or cluster. In RL/radar terms, subbands behave like latent states with temporal persistence.
- **Validation:** Bayesian Information Criterion (BIC) or Silhouette Analysis to choose the number of clusters $N$ so that the discovered state space matches the physical setting (e.g. $N \approx M = 10$).

### Motivation & Benefits

- **Structural identification:** Wrong $M$ (e.g. 9 instead of 10) causes mapping errors and can cap jamming success (e.g. wrong action space → irreducible loss in $\mathit{Num}$).
- **Autonomous setup:** The agent can **self-configure** the action space (e.g. $N \times 24$ or a reduced set) before the main RL loop, enabling deployment when $M$ is unknown.

### Consistency Check

HMM/GMM for sequence and state discovery are standard (e.g. Baum–Welch, mixture models). Using BIC or Silhouette for model order ($N$) is consistent with unsupervised model selection. The 70/30 persistence giving “high-density clusters” is consistent with subband persistence: same-subband runs produce index clusters.

### Critical Points

- **Temporal vs. i.i.d.:** If the radar were pure i.i.d. uniform, clustering might not recover subbands. With 70/30 (or similar) persistence, temporal correlation supports HMM/clustering.
- **Index space:** Clustering is on **state indices** (0–239) or derived features; subbands are index blocks (e.g. 0–23, 24–47). Ensure the feature space (e.g. index, or frequency if available) matches what HMM/GMM see.
- **Bootstrap:** Discovered $N$ can drive curriculum Phase 1 (reduced action space) before scaling to 240.

---

## 2. Curriculum Learning (CL) Framework

**Sources:** future_works_dummy.txt, PLAN.md (FW-02)

### What It Is

A **hierarchical training strategy** that gradually increases task difficulty instead of training on the full 240-state problem from the start.

### Methodology (Dummy: 3 phases)

- **Phase 1 (Macro):** Train in a **reduced action space** (e.g. $N$ discovered subbands only, or subband-level decisions). Goal: learn high-level Markovian transition structure (e.g. which subband to jam).
- **Phase 2 (Complexity scaling):** Introduce a **limited set of intra-pulse permutations** so the agent learns sub-pulse agility within a subset of the full 24.
- **Phase 3 (Full):** Switch to the **full 240-state** space, reusing (e.g. via transfer or warm start) weights from earlier phases.

PLAN.md variant: (1) fixed sub-pulses, (2) limited combinations, (3) full 240-state. Same idea: easy → hard.

### Motivation & Benefits

- **Reward sparsity:** In 240 dimensions, random exploration rarely hits the right state; early phases with smaller spaces increase the chance of non-zero reward and more stable gradients.
- **Optimization:** Reduces time in pure random exploration and can avoid bad local minima by first learning a simpler policy (e.g. “which subband”) then refining (exact permutation).

### Consistency Check

Curriculum learning in RL is well established: training on progressively harder tasks or data improves sample efficiency and stability. The description (reduce action space first, then scale up) is consistent with standard CL and with the dummy’s emphasis on mitigating sparsity.

### Critical Points

- **Phase boundaries:** When to switch (e.g. by episode, by success rate) must be defined; too early or too late can hurt.
- **Catastrophic forgetting:** Moving to a harder phase can make the agent “forget” earlier skills; consider replay from earlier phases or gradual blending of tasks.
- **Compatibility with subband discovery:** If $N$ is found autonomously, Phase 1 can use $N$ subbands; otherwise use a fixed reduced setup (e.g. 10 subbands, permutation subset).

---

## 3. N-Step Returns

**Sources:** future_works_dummy.txt, PLAN.md (FW-01)

### What It Is

Instead of 1-step TD targets, use **n-step returns**: sum of $n$ discounted rewards plus a bootstrapped value at the $n$-th next state.

### Formula

$$
G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})
$$

### Motivation & Benefits

- **Bias–variance trade-off:** 1-step TD (e.g. $n=1$) is low variance but biased; Monte Carlo (large $n$) is low bias but high variance. Intermediate $n$ balances the two; standard RL literature supports this.
- **Temporal credit assignment:** Rewards from a successful jamming event at $t+n$ are attributed back to the action at $t$ over $n$ steps, which helps in pattern-based or correlated radar dynamics.
- **Synergy with GRU:** The GRU carries **past** context; n-step returns add a **multi-step future** horizon in the target, which can stabilize learning in non-stationary or Markovian environments.

### Consistency Check

This matches standard treatments of n-step returns (e.g. Sutton & Barto): same formula, same bias–variance interpretation, same credit-assignment benefit. The dummy’s “GRU synergy” is a reasonable application note, not a contradiction.

### Critical Points

- **Off-policy:** With PER and ε-greedy, behavior is off-policy; n-step off-policy corrections (e.g. importance sampling) may be needed if $n$ is large.
- **Delay:** Updates are delayed by $n$ steps; buffer and training loop must store and use multi-step transitions correctly.
- **Hyperparameter:** $n$ (e.g. 3–5) is a tunable choice; too large can increase variance.

---

## 4. RAdam + Lookahead Optimizer

**Source:** PLAN.md (FW-03)

### What It Is

- **RAdam (Rectified Adam):** Adjusts the adaptive learning rate in early training to reduce variance and often reduces the need for manual warmup.
- **Lookahead:** Keeps “fast” and “slow” weights; periodically interpolates fast weights toward slow weights (e.g. “k steps forward, 1 step back”), smoothing the optimization path.

Combined (e.g. “Ranger”): RAdam as the inner optimizer, Lookahead wrapping it.

### Motivation & Benefits

- **Stability:** Fewer blow-ups in early training; less sensitivity to learning-rate and warmup choices.
- **Convergence:** In many settings, RAdam+Lookahead performs on par or better than plain Adam, with modest extra cost.

### Consistency Check

RAdam and Lookahead are established (PyTorch includes RAdam; Lookahead from Zhang et al.). The “stability and lower variance” claim is consistent with the literature.

### Critical Points

- **Reproducibility:** Paper baseline uses Adam; switching to RAdam+Lookahead is a fair thesis extension but should be reported as a change for comparability.
- **LR schedule:** Some recommendations use a flat LR then short decay; worth aligning with the rest of the training setup (e.g. warmup, decay).
- **Implementation:** Use a maintained implementation (e.g. Ranger) or PyTorch’s RAdam plus a Lookahead wrapper.

---

## 5. Dynamic GRU Window

**Source:** PLAN.md (FW-04)

### What It Is

Allow **variable-length input sequences** for the GRU (e.g. history length $L$ in 5–20 or configurable range) instead of a fixed $L=10$.

### Motivation & Benefits

- **Temporal depth:** Shorter $L$ focuses on recent pulses; longer $L$ can capture slower patterns. Ablating $L$ tests how much history the agent needs.
- **Efficiency vs. expressiveness:** Short $L$ is cheaper; long $L$ may help on strongly correlated radar dynamics.

### Consistency Check

Variable sequence length is standard in RNN/GRU applications; the paper uses a fixed architecture, so making $L$ configurable is a natural extension.

### Critical Points

- **Padding/masking:** Variable length requires consistent padding or masking so that the GRU and downstream layers ignore padding.
- **Batch training:** In PER, batches contain sequences of possibly different lengths; either pad to max $L$ in the batch or use a fixed max $L$.
- **Comparison:** When comparing to the paper, fix $L=10$ (or the paper’s value) for baseline fairness.

---

## 6. Prior Knowledge Mode

**Sources:** PLAN.md (FW-05), prompts 07 & 08

### What It Is

The jammer receives **partial information** about the **next** radar pulse (e.g. the first subpulse frequency $f_1$ of $s_{t+1}$) via an ELINT-style “leading edge” measurement, and uses it to narrow the action space.

### Methodology

- **Observation:** In addition to history, the env provides a **hint** $= f_1(s_{t+1})$. So the env must “peek” at $s_{t+1}$ before the agent chooses $a_t$.
- **Search space:** With $f_1$ known, subband and first slot are fixed; only the ordering of the other three subpulses remains → **6 options** instead of 240 ($3!$ remaining permutations).
- **Paper alignment:** Table 3 reports ~97.14% without prior knowledge and ~99.41% with; the 6 vs 240 interpretation matches the “search space reduction” explanation in the prompts.

### Motivation & Benefits

- **Realism:** Models the fact that a fast intercept receiver can detect $f_1$ before the pulse ends (LPI/leading edge).
- **Reproducibility:** Replicating Table 3 requires implementing this mode and logging success rate with and without the hint.
- **Thesis:** Clear before/after comparison (autonomous vs. assisted jammer).

### Consistency Check

The prompts’ timing (hint $= f_1$ of **upcoming** $s_{t+1}$, not current pulse) and the 240→6 reduction ($3!$ remaining permutations) are consistent with the paper’s state space and with the reported accuracy jump.

### Critical Points

- **Env contract:** When `use_prior_knowledge=True`, the env must expose $f_1(s_{t+1})$ in the observation **before** the agent selects $a_t$; when `False`, hint is masked. This is a strict requirement for a fair comparison.
- **Model:** Add an embedding or small branch for the hint and concatenate to the GRU/decision path; keep the GRU’s role (history) independent of the hint so that long-term dependency is still learned from the sequence.
- **Action space:** With hint, the agent can either choose among 6 valid actions or still output 240 with a mask; both are valid if documented.

---

## 7. Baseline Comparisons

**Source:** PLAN.md (FW-06)

### What It Is

Implement **plain DQN**, **Dueling DQN** (without GRU–Attention), and optionally **tabular Q-Learning** (if feasible) so that GA-Dueling DQN can be compared to simpler baselines.

### Motivation & Benefits

- **Ablation:** Quantify the gain from GRU, Attention, and dueling architecture (e.g. via Figure 9–style curves and Table 3–style success rates).
- **Thesis narrative:** “Our method outperforms DQN and Dueling DQN” is supported by like-for-like experiments (same env, same reward, same evaluation protocol).

### Consistency Check

Standard practice in RL papers: report baselines under the same conditions. No conflict with the rest of the future works.

### Critical Points

- **Same setup:** Same env (e.g. same generator_mode, seed policy), same episode length, same evaluation metric (e.g. hit rate, total reward).
- **Compute:** Multiple algorithms mean more runs; fix seeds and report variance (e.g. std over runs).
- **Scope:** Tabular Q-Learning on 240 states may be heavy; often DQN vs. Dueling DQN vs. GA-Dueling DQN is enough.

---

## Summary Table

| Topic | Source | Main benefit | Critical note |
|-------|--------|--------------|----------------|
| Autonomous Subband Discovery | dummy | Self-configure $M$; avoid wrong state space | Needs temporal structure (e.g. 70/30); validate $N$ with BIC/Silhouette |
| Curriculum Learning | dummy, PLAN | Less sparsity; better optimization path | Phase switching and forgetting |
| N-Step Returns | dummy, PLAN | Better credit assignment; bias–variance balance | Off-policy correction if $n$ large |
| RAdam + Lookahead | PLAN | Training stability | Report as change vs. paper’s Adam |
| Dynamic GRU Window | PLAN | Ablate temporal depth | Padding/masking; fix $L$ for baseline |
| Prior Knowledge | PLAN, 07/08 | Reproduce Table 3; realism | Env peek and 6 vs 240 semantics |
| Baseline Comparisons | PLAN | Ablation and thesis narrative | Same env and evaluation |

---

*Document draws on future_works_dummy.txt, PLAN.md §8, prompts 07–08, and standard RL/optimization references. Last structured for project consistency check.*
