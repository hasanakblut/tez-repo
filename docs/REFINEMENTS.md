# Architectural Refinement Analysis

> Comparison of proposed refinements against the paper's baseline, with independent assessment of each point's significance.

---

## 1. Hybrid Exploration & Failback

### What the paper does
The paper uses **both** NoisyLinear layers (Ref [28], Section 3.3) **and** ε-greedy (Section 3.3) simultaneously, but never justifies why both are needed or analyzes their individual contributions.

### What is proposed
- A toggle to isolate three modes: Pure NoisyNet, Pure ε-greedy, Hybrid.
- A failback mechanism: if hit rate stays below 10% for 5,000 pulses, temporarily boost ε.

### Assessment: Medium

The **toggle** is a standard ablation study — useful for the thesis discussion but not a novel contribution in itself. It does expose a real gap in the paper: the authors combine two exploration strategies without demonstrating their interaction effects.

The **failback** addresses a practical concern (local optima in a 240-action space) but is unlikely to activate once training stabilizes. Its value is defensive — it prevents catastrophic failure in edge cases rather than improving the expected case.

**Recommendation:** Implement the toggle for ablation experiments. Implement the failback as a lightweight safeguard, but don't position it as a primary contribution.

---

## 2. State Representation: One-Hot vs. Learned Embedding

### What the paper does
States are fed to the GRU as 240-dimensional one-hot vectors (implicit from the architecture: GRU input_size=240, Table 2). This treats every frequency index as equally distant from every other — index 0 (subband 0, perm 0) is no closer to index 1 (subband 0, perm 1) than to index 239 (subband 9, perm 23).

### What is proposed
Replace one-hot input with `nn.Embedding(240, 64)` — a learnable dense vector that maps each index to a 64-dim representation before the GRU.

### Assessment: High

This is the most architecturally significant refinement. The one-hot encoding has two concrete problems:

1. **No structural prior.** Indices within the same subband (e.g., 0–23) share 100% of their sub-pulse frequencies — they only differ in permutation order. One-hot encoding erases this subband locality, forcing the GRU to rediscover it from scratch.

2. **Dimensional waste.** A 240-dim sparse vector fed into GRU(240→128) means the first GRU layer has ~142K parameters mostly operating on zeros. An embedding(240→64) feeding GRU(64→128) would cut the GRU parameter count roughly in half while encoding richer relationships.

The paper's Section 3.3 states the GRU's purpose is *"to capture long-term dependency relationships in the sequence."* An embedding layer doesn't compete with this — it **enhances** it by giving the GRU a semantically meaningful input space instead of raw sparse vectors.

**Recommendation:** Implement as a configurable option (`use_embedding: true/false` in config). This is a strong thesis contribution — directly testable via convergence speed comparison against the paper's baseline.

---

## 3. Reward Shaping: Subband-Match Partial Reward

### What the paper does
Reward is strictly `r_t = JSR × Num` (Equation 15). When the jammer picks the wrong subband, Num=0 and reward=0. No gradient signal, no learning.

### The underlying problem
With 240 actions and uniform random exploration:
- P(correct subband) = 24/240 = 10%
- P(Num=4 | random action) = 1/240 ≈ 0.42%

During early training (ε ≈ 0.995), the agent is nearly random. Roughly **90% of actions yield exactly zero reward**. The agent must stumble into the correct subband to receive any learning signal at all. This is a textbook sparse reward problem that the paper does not acknowledge.

### What is proposed
- If subband matches but Num=0, award a small partial reward (e.g., `0.1 × JSR`).
- Fade this partial reward out over episodes (curriculum-style) so final evaluation uses the paper's strict formula.

### Assessment: High

This directly attacks a convergence bottleneck. The partial reward provides a gradient pathway from "completely wrong" → "right subband" → "right permutation," turning a single cliff into a learnable staircase.

The curriculum fading is important: without it, the agent might settle for subband-only matching and never optimize for full permutation matching. With fading, the soft reward bootstraps learning but doesn't distort the final policy.

**Recommendation:** Implement with a configurable fade schedule. Compare convergence speed in the first 20 episodes (where sparse reward bites hardest) against the strict baseline. This is a defensible thesis contribution with clear before/after metrics.

---

## 4. Stress Testing: Non-Stationary Radar

### What the paper does
The radar selects frequencies uniformly at random with a **fixed** strategy for the entire episode and across all episodes (Section 4). No strategy changes, no adversarial adaptation.

### What is proposed
Mid-episode (e.g., at pulse 5,000), change the radar's random seed or modulation pattern. Measure how many pulses the agent needs to recover 97% hit rate.

### Assessment: Medium-High

The paper's conclusion (Section 5) claims the method *"provides an effective solution to the optimal jamming frequency selection problem"* — but only tests against the simplest possible radar behavior (i.i.d. uniform random). A truly frequency-agile radar in adversarial conditions might change its hopping strategy when it detects jamming.

This stress test doesn't require any algorithm changes — only an environment modification and a new metric ("adaptation time"). The value is in the **experimental analysis**, not the implementation complexity.

However, since the paper's radar is already i.i.d. uniform random, changing the seed mid-episode **should not** change the statistical distribution at all. To make this test meaningful, the non-stationary mode should introduce genuinely different behavior: e.g., switching from uniform random to a biased subband preference, or introducing temporal correlations.

**Recommendation:** Implement as a separate test script with multiple non-stationarity modes (seed change, distribution shift, pattern introduction). Position as thesis robustness validation rather than an algorithm improvement.

---

## Priority Ranking

| # | Refinement | Importance | Type | Effort |
|---|---|---|---|---|
| 1 | Learned Embedding | High | Architectural improvement | Low |
| 2 | Reward Shaping | High | Training improvement | Low |
| 3 | Stress Testing | Medium-High | Experimental contribution | Medium |
| 4 | Hybrid Exploration | Medium | Ablation study | Low |

---

## Implementation Status

All four items are **not yet implemented** — they are analysis-stage refinements to be built on top of the validated baseline system.
