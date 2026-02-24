# Prior Knowledge / Reconnaissance Guidance — Paper Analysis

Independent analysis of the paper's "prior knowledge" concept and its implementation implications, without being driven by a specific prompt.

---

## 1. What the Paper Says

### Section 4.2: "Intelligent Frequency Selection by Jammer under Reconnaissance Guidance"

> *"Furthermore, the experiment compared the case where the jammer can obtain some prior information about the radar. It is assumed that the cognitive jammer can obtain the carrier frequency of the first subpulse of each radar pulse through interception and parameter estimation. Also, the cognitive jammer knows that although frequency agility is performed between radar subpulses, the change is within the subband range. Jamming decisions are made on this basis."*

### Table 3: Jamming Success Probability

| Algorithm | Without Prior Knowledge | With Prior Knowledge |
|-----------|-------------------------|----------------------|
| GA-Dueling DQN | 97.14% | 99.41% |
| Dueling DQN | 68.66% | 89.27% |
| DQN | 72.98% | 86.56% |
| Q-Learning | 45.32% | 75.01% |

### Figure 10 caption

> *"Comparison of jammer gains with and without sub-pulse scouting a priori information case."*

### Section 5 (Conclusion)

> *"Furthermore, for the case with subpulse reconnaissance prior knowledge, the algorithm proposed in this paper holds a distinct advantage in enhancing jamming effects. [...] The improved method proposed in this paper can achieve an over 97% jamming pulse hit rate without radar prior knowledge. With radar prior knowledge, the jamming hit rate of the proposed method is 99.41%."*

---

## 2. What "Prior Knowledge" Means in the Paper

The prior knowledge consists of two parts:

1. **f₁ (first subpulse frequency):** The jammer can obtain the carrier frequency of the **first** subpulse of each radar pulse via interception and parameter estimation (ELINT-like capability).

2. **Structural knowledge:** All four subpulses remain within the same subband (100 MHz band) as the first subpulse.

### Effect on the Decision Space

- Without prior knowledge: the jammer must choose among all **240** states (10 subbands × 24 permutations).
- With prior knowledge: f₁ identifies the subband, so the jammer only chooses among **24** permutations within that subband.

The paper states that *"the acquisition of prior knowledge greatly reduces the jamming decision space."* That reduction (240 → 24) is the core mechanism: from full space to subband-constrained space.

---

## 3. How f₁ Maps to Subband

In the paper's model (Section 2.1, Section 4):

- 10 subbands, each 100 MHz wide.
- Each pulse has K=4 subpulses with pairwise-distinct frequencies within one subband.
- Subband width = Δf = 100 MHz; subpulse bandwidth = 25 MHz.

Given an absolute carrier frequency f₁ (e.g. 10.225 GHz), we can compute:

- `subband_id = floor((f₁ - f_L) / Δf)` → integer in [0, 9]
- All four subpulses are in that subband, so only the permutation (24 options) remains unknown.

So **f₁ as “first subpulse frequency”** is operationally equivalent to **subband index** for decision-making: it pins the radar to one subband, and the jammer’s problem becomes selecting the correct permutation (0..23) within that subband.

---

## 4. Paper vs. Prompt: Representation Choices

The paper does **not** specify:

- How f₁ is represented (raw Hz, subband index, etc.).
- How the model incorporates prior knowledge.
- Any change to the GA-Dueling DQN architecture.

The prompt proposes:

- Passing f₁ (or a masked placeholder) as an extra observation.
- `nn.Embedding(10, 8)` for f₁ (10 classes → subband index).
- Concatenating this 8-dim embedding with the 128-dim GRU-Attention output before the dueling heads.

That is a concrete design, not a paper requirement. The paper only defines the **information** (first subpulse frequency + subband constraint) and the **result** (reduced decision space and higher hit rate). The implementation is left open.

---

## 5. Logical Interpretation

### Option A: Action-Space Reduction (Logical)

- If the jammer knows the subband from f₁, it can **restrict its action space** to the 24 actions that belong to that subband.
- No architecture change needed: same GA-Dueling DQN, but at decision time only 24 Q-values (those in the known subband) are considered.
- This directly matches the paper’s “greatly reduces the jamming decision space” and explains the hit-rate gains.

### Option B: State Augmentation (Prompt’s Approach)

- f₁ (or subband index) is added as extra input to the network.
- The network learns to combine sequence history with subband information.
- Architecturally richer but not explicitly described in the paper.

### Option C: Hybrid

- Subband index (from f₁) is both used to **restrict** the action space to 24 actions and optionally fed as **auxiliary input** to the model.
- Captures both “smaller decision set” and “richer state.”

---

## 6. Critical Detail: MDP and Observation Timing

The paper’s MDP (Section 3.2):

- At time t, the jammer observes s_t (radar pulse frequency) and chooses a_t.
- The reward depends on a_t vs s_{t+1}.

So the jammer decides a_t **before** s_{t+1} is revealed. The question is: when is f₁ available?

- **“First subpulse of each radar pulse”** suggests f₁ is from the **current** pulse the jammer just observed (s_t).
- Then f₁ tells us the **subband of s_t**.
- For s_{t+1}, the radar may switch subband. So f₁ of the current pulse does **not** directly reveal the subband of the next pulse.

### Timing Consistency

If the paper assumes f₁ is from the **upcoming** pulse (the one we are trying to jam), then the jammer would know the subband of s_{t+1} before choosing a_t. That would justify narrowing the action space to 24 for that step.

The text says *“the carrier frequency of the first subpulse of **each** radar pulse”* and *“Jamming decisions are made **on this basis**”*. The most natural reading is that the jammer obtains f₁ for the pulse it is about to jam, i.e. for s_{t+1}. So:

- **f₁ = first subpulse frequency of s_{t+1}** (the next radar pulse)
- The jammer uses this to infer subband(s_{t+1}) and restricts its action space to the 24 actions in that subband.

This timing resolves the “when is f₁ observed?” question and aligns with the paper’s claim about reducing the decision space.

---

## 7. Implementation Recommendations

1. **Clarify timing:** Treat f₁ as the first subpulse frequency of the pulse we are jamming (s_{t+1}). If the simulation generates s_{t+1} in `step(action)`, f₁ can be derived from s_{t+1} and returned in `info` or as part of the observation for the next step.

2. **Representation:** Use subband index 0..9 derived from f₁. An `nn.Embedding(10, 8)` or similar is reasonable for a learned representation.

3. **Action-space restriction:** At least implement action masking: when prior knowledge is on, only allow actions whose subband matches the one inferred from f₁. This matches the paper’s core mechanism.

4. **Architecture:** The prompt’s idea (concatenate f₁ embedding with GRU output) is a sensible extension. The paper does not prescribe it, so treat it as an optional enhancement on top of action-space restriction.

5. **GRU independence:** Keep f₁ out of the GRU input so that the recurrent state remains a function of the sequence of full states only; f₁ can be merged later (e.g. before dueling heads) as auxiliary information.

---

## 8. Summary

| Aspect | Paper | Implementation Choice |
|--------|-------|------------------------|
| Prior knowledge | First subpulse frequency + subband constraint | Represent as subband index 0..9 |
| Effect | Reduces decision space 240→24 | Action masking or optional network augmentation |
| Timing | “Each radar pulse” | f₁ = first subpulse of the pulse being jammed (s_{t+1}) |
| Architecture | Not specified | Optional: embed subband, concatenate with GRU output before dueling heads |
