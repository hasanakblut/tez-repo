# Run Readiness: Data, Config, Randomness & Open Issues

Single reference for how the project is wired, how data is generated, what the config does, and what remains to fix or decide before/during training.

---

## 1. Project structure and config flow

### 1.1 Import and entry

- **Entry:** `python -m src.train` (must be run from **repo root** `_tez/`).
- **train.py** loads `configs/default.yaml` and builds:
  - **Env:** `build_env_config(cfg)` → flat dict from `physics`, `radar`, `episode`, `environment`.
  - **Agent:** `JammingAgent(config=cfg)` receives the **full** nested `cfg`.
- **agent.py** uses `cfg["network"]`, `cfg["training"]`, `cfg["replay"]`, `cfg["environment"]` and instantiates `GADuelingDQN` with network kwargs (including `sigma_init` if passed from config).

### 1.2 Config → Env mapping

`build_env_config` merges in order: `physics` → `radar` → `episode` → `environment`. Resulting keys used by `RadarEnv`:

| Key | Source | Used in env |
|-----|--------|-------------|
| A_R, A_J, h_t, h_j, sigma, L | physics | JSR (Table 1) |
| M, K, f_low, f_high, generator_mode, reset_generator_on_episode, periodic_sequence, lcg_params, markov_params | radar | state_dim=240, FrequencyGenerator |
| max_pulses, num_episodes | episode | max_pulses (num_episodes only in train) |
| history_len | environment | observation history length |

All of these exist in `default.yaml`; no missing-key issue for env.

### 1.3 Config → Agent / Model mapping

- **Agent** reads: `network.*`, `training.*`, `replay.*`, `environment.history_len`.
- **GADuelingDQN** is built with: `state_dim`, `action_dim`, `embed_dim`, `hidden_dim`, `num_heads`, `fc_dim`, and optionally `sigma_init` from `config["network"]`.

---

## 2. How data is created (no offline dataset)

There is no separate dataset. Data is generated **on the fly** by the environment:

1. **Radar:** `FrequencyGenerator` (in `env_utils.py`) produces the next index via `next(prev_state)`. Modes: `uniform`, `periodic`, `lcg`, `markov`. See [docs/DATA.md](docs/DATA.md).
2. **Jammer:** Agent chooses action in `[0, 239]` from the policy (or random).
3. **Transition:** `(state_history, action, reward, next_state_history, done)` is pushed into the **Prioritized Replay Buffer**; reward = `JSR_base × Num` (Num = sub-pulse matches 0–4).

So “data” = the stream of transitions from env–agent interaction. The only randomness in data creation is the **radar’s** choice of the next state and the **agent’s** exploration (ε-greedy / NoisyNet).

**Implication:** The agent is not learning a hidden “pattern” in the radar; it is learning which actions yield high reward under a fixed (i.i.d. uniform) radar distribution, with history providing optional context. This matches the paper’s baseline setup (Section 4: “fully randomly switched carrier frequency”).

---

## 3. Randomness: is the radar “pattern” or random?

### 3.1 Current implementation

- **Radar:** `_generate_next_state()` uses Gymnasium’s `self.np_random.integers(0, self.state_dim)` → **i.i.d. uniform** over `{0, 1, …, 239}`. No memory, no correlation, no pattern.
- **Paper (Section 2.1):** “the radar can **randomly select** different subbands for each pulse” and “the modulation code d_n [is] a **random integer**”. The paper does not specify a distribution; uniform over 240 states is a reasonable interpretation.
- **Conclusion:** There is **no** “hopping pattern” or deterministic rule in the code — it is **pure i.i.d. random**. Learning is about good action selection under this fixed distribution, not exploiting a temporal or correlated pattern.

### 3.2 Future: adding a pattern (e.g. for stress tests)

For thesis or robustness (e.g. REFINEMENTS.md stress test), you could add a **radar mode** in config (e.g. `radar.mode: "uniform" | "biased" | "markov"`) and in `_generate_next_state()` implement different logic (biased subband, transition matrix, etc.). Current code is explicitly pattern-free.

---

## 4. Config quick reference

| Section | Keys | Purpose |
|---------|------|---------|
| **physics** | A_R, A_J, h_t, h_j, sigma, L | Table 1 constants for JSR |
| **radar** | M, K, f_low, f_high, generator_mode, reset_generator_on_episode, periodic_sequence, lcg_params, markov_params | FrequencyGenerator config |
| **episode** | max_pulses=10_000, num_episodes=100 | Episode length and count |
| **environment** | history_len=10 | GRU input sequence length |
| **network** | state_dim=240, action_dim=240, embed_dim=64, hidden_dim=128, num_heads=8, fc_dim=64, sigma_init=0.5 | Model dimensions (embedding + GRU) |
| **training** | gamma=0.9, lr=0.009, batch_size=256, epsilon 0.995→0.005 over 100 episodes, target_update_freq=1 | Learning and exploration |
| **replay** | buffer_size=100_000, alpha=0.6, beta 0.4→1.0, min_priority=1e-6 | PER |
| **logging** | log_interval=1, save_interval=10, results_dir="results" | Output paths and frequency |

---

## 5. Issues: resolved vs open

### 5.1 Resolved (or documented as such in prior checks)

| Issue | Status | Note |
|-------|--------|------|
| **sigma_init not passed to model** | Resolved | Agent can pass `sigma_init=net_cfg.get("sigma_init", 0.5)` to `GADuelingDQN` (verify in current agent.py). |
| **requirements.txt missing** | Resolved | Added at repo root (gymnasium, numpy, torch, PyYAML). |

### 5.2 Open: run environment and reproducibility

| # | Issue | Severity | Recommendation |
|---|-------|----------|----------------|
| 1 | **Run from repo root** | High | If you run from another directory, `configs/default.yaml` and `from src.xxx` can fail. Run: `cd _tez` then `python -m src.train` (or pass `--config` with an absolute path). Alternatively, resolve config/results paths from `Path(__file__).resolve().parent.parent` so they are independent of cwd. |
| 2 | **No seed passed to env.reset()** | Medium | For reproducible debug or paper reproduction, call `env.reset(seed=...)` (e.g. per-episode or global seed). |
| 3 | **Config/results paths depend on cwd** | Medium | Make paths robust by deriving project root from `__file__` and using root-relative paths for `configs/` and `results/`. |

### 5.3 Open: logic and design

| # | Issue | Severity | Recommendation |
|---|-------|----------|----------------|
| 4 | **GRU hidden state only updated on exploitation** | Medium | In `select_action`, when `random.random() < self.epsilon` the branch returns a random action without running the policy net, so `self.hidden` is not updated. With high ε (e.g. 0.995), hidden state effectively “freezes”. Either: (a) document as intentional (“carry hidden only when exploiting”), or (b) run a forward pass (e.g. with no_grad) on the random branch too so hidden state advances every step. |
| 5 | **K vs PERMUTATION_TABLE fixed** | Low | **Design decision:** K is kept fixed at 4. `PERMUTATION_TABLE` is built at module load with 24 permutations; state_dim = M×24. If config sets K≠4, behaviour is undefined. May be revisited later if needed. |

### 5.4 Resolved (implemented)

| # | Issue | Resolution |
|---|-------|------------|
| 1 & 3 | Paths / cwd | Config and results paths are resolved from `PROJECT_ROOT = Path(__file__).resolve().parent.parent` in `train.py`. |
| 2 | Seed | `env.reset(seed=...)` is used when `config.get("seed")` is set; per-episode seed = `seed + episode` for reproducibility. |
| 4 | GRU hidden on random | `select_action` now always runs a forward pass to update `self.hidden`, then chooses random vs argmax; recurrent chain stays consistent. |
| 6 | sigma_init | Read from `config["network"].get("sigma_init", 0.5)` and passed into `GADuelingDQN` and all `NoisyLinear` layers. |

### 5.5 Minor

| # | Issue | Note |
|---|-------|------|
| 6 | **YAML numeric types** | `f_low: 10.0e9`, `f_high: 11.0e9` are read as floats; env’s `cfg.get("f_low", 10e9)` is compatible. No change needed. |

---

## 6. Minimal run checklist

1. Run from any cwd: `python -m src.train` (config path is relative to project root).
2. Optional: set `seed: 42` (or any int) in config for reproducible env randomness per episode.
3. Env uses merged config; agent uses full config; both read the same `default.yaml`.
4. Data = online transitions; radar state from `FrequencyGenerator` (mode in config).
5. `generator_mode` is logged at training start.

**Data flow verification (docs/DATA.md):** Generator → env.get_history() → agent (indices) → model embedding → GRU → Q-values.
