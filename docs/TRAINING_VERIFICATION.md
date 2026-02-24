# Training Verification Checklist

Quick reference to verify env–agent–train wiring and to run a dry-run before full training.

---

## 1. Config flow

| Source | Used by | Keys |
|--------|---------|------|
| `physics` | RadarEnv | A_R, A_J, h_t, h_j, sigma, L (JSR) |
| `radar` | RadarEnv, FrequencyGenerator | M, K, generator_mode, start_index, markov_transition_path, ... |
| `episode` | RadarEnv, train.py | max_pulses, num_episodes |
| `environment` | RadarEnv, Agent | history_len |
| `network` | JammingAgent, GADuelingDQN | state_dim, action_dim, embed_dim, ... |
| `training` | JammingAgent | gamma, learning_rate, epsilon_*, ... |
| `replay` | JammingAgent | buffer_size, alpha, beta_*, ... |

- **Full training:** `configs/default.yaml` → `python -m src.train`
- **Dry-run / test:** `configs/test_training.yaml` → `python -m scripts.training_test` or `python -m src.train --config configs/test_training.yaml`

---

## 2. Step order (single step)

1. `state_history = env.get_history()`
2. `action = agent.select_action(state_history)`
3. `next_obs, reward, terminated, truncated, info = env.step(action)`
4. `next_state_history = env.get_history()`
5. `agent.store_transition(state_history, action, reward, next_state_history, terminated)`
6. `agent.learn()` (in full training)

---

## 3. Invariants to check

- After reset: `len(env.get_history())` in `[1, history_len]`; all indices in `[0, 239]`.
- After each step: `action` in `[0, 239]`, `reward` in `[0, JSR_base * 4]`, `next_obs` in `[0, 239]`.
- Config: `generator_mode` (and if Markov, `markov_transition_path`) set in the env config used by the script.

---

## 4. Dry-run

1. Run: `python -m scripts.training_test --config configs/test_training.yaml --plot`
2. Script uses the same `build_env_config` and `JammingAgent(config=cfg)` as `train.py`.
3. One episode (or `--max-steps`) is run; each step is recorded in memory (state_history, action, reward, next_state_history, info).
4. With `--plot`, figures are saved under `results/training_test_plots/` (reward vs t, state index vs t, action distribution).
5. Without `--no-verify`, assertions check history length, action and reward bounds.

---

## 5. Training log and outputs

- Full training appends to `results/training.log`.
- Each run: header with timestamp and config path.
- Each episode: `Episode N seed=... first_100_indices=[...]` then a summary line `reward=... hit_rate=... time=...`.
- Skim the `.log` file to confirm first 100 indices and per-episode metrics.
- **Run directory:** each training creates `results/runs/run_<datetime>_<config>_ep<N>/` (e.g. `runs/run_20250224_153000_default_ep100` or with seed `run_20250224_153000_default_s42_ep100`). All outputs for that run live there.
- **Log:** `run_*/training.log` — header with timestamp, config path, num_episodes, max_pulses, seed, run_dir; then per-episode lines.
- **Metrics:** `run_*/metrics.jsonl` — one JSON per episode (total_reward, hit_rate, avg_loss, epsilon, env_steps, last_obs_index, last_action, …).
- **Checkpoints:** `run_*/checkpoint_ep10.pt`, … `run_*/final_model.pt`.
- **Training plots:** `run_*/training_plots/` — during training updated every `plot_interval`; files: `reward_curve.png`, `hit_rate_curve.png`, `epsilon_curve.png`, `loss_curve.png`.
- **Terminal:** tqdm progress bar with ETA; every `log_interval` episodes an ML-style line (state, action, return, hit_rate, TD_loss, ε, env_steps, train_steps, wall).
