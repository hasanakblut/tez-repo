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
| `training` | JammingAgent | gamma, learning_rate, epsilon_*, epsilon_decay_mode, ... |
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
- **Epsilon decay:** Default is per-episode; optional per-step via `training.epsilon_decay_mode` and `epsilon_decay_steps`. See `docs/EPSILON_DECAY.md`.
- **Checkpoints:** `run_*/checkpoint_ep10.pt`, … `run_*/final_model.pt`.
- **Training plots:** `run_*/training_plots/` — during training updated every `plot_interval`; files: `reward_curve.png`, `hit_rate_curve.png`, `epsilon_curve.png`, `loss_curve.png`.
- **Terminal:** tqdm progress bar with ETA; every `log_interval` episodes an ML-style line (state, action, return, hit_rate, TD_loss, ε, env_steps, train_steps, wall).

---

## 7. Logged metrics: mantık ve sağlıklı eğitim eğilimleri

Eğitim sırasında (pulse bar ve episode özetinde) yazdırılan **ret**, **hit**, **sub**, **num**, **eff**, **H** değerlerinin hesabı ve “episode’lar arası nasıl bir artış görürsen eğitim sağlıklı ilerliyor” kısa özeti aşağıda.

### 7.1 Metriklerin tanımı ve hesap mantığı

| Kısaltma | Tam ad / kaynak | Hesaplama mantığı |
|----------|------------------|-------------------|
| **ret** | Episode total reward | Episode boyunca toplanan ödül: `ret = Σ (JSR_base × Num)` her pulse için. Num = jammer aksiyonu ile radarın bir sonraki durumu arasındaki pozisyon bazlı sub-pulse eşleşme sayısı (0–4). Kaynak: `env.step` → `reward` birikimi (`train.py` içinde `episode_reward`). |
| **hit** | Hit rate (4/4 tam vuruş oranı) | O ana kadar geçen pulse’lar içinde **Num=4** (tam jam) olanların oranı. `hit_rate = total_full_hits / pulse_count`. Ortamda her step’te jammer hem subband hem permutasyonu doğru tahmin ederse Num=4. Kaynak: `env._build_info()` → `info['hit_rate']`. |
| **sub** | Subband rate | O ana kadar pulse’lar içinde **doğru subband** tahmin edilenlerin oranı. `subband_rate = total_subband_hits / pulse_count`. Subband doğru olsa bile permütasyon yanlış olabilir (Num 1–3). Kaynak: `env._build_info()` → `info['subband_rate']`. |
| **num** | Avg match (Num ortalaması) | Pulse başına ortalama eşleşme sayısı: `avg_match = total_matches / pulse_count`. 0–4 arası; 4’e ne kadar yakınsa o kadar iyi. Kaynak: `env._build_info()` → `info['avg_match']`. |
| **eff** | Efficiency (örnek verimliliği) | Pulse başına ortalama ödül: `eff = episode_reward / env_steps`. Yani `ret / pulse_sayısı`. Aynı episode içinde pulse arttıkça ret artar ama eff genelde daha stabil bir “ne kadar ödül per step” göstergesidir. Kaynak: `train.py` içinde `episode_reward / env_steps`. |
| **H** | Policy entropy (bit) | O ana kadar o episode’ta alınan aksiyonlar için softmax(Q) üzerinden hesaplanan entropi ortalaması: `H = -Σ p_i log2(p_i)` (bit). Yüksek H = politika belirsiz/keşifçi, düşük H = politika kararlı/exploit. Kaynak: `agent.select_action_with_info` → her step’te entropy; `train.py`’de `ep_entropy_sum / env_steps`. |

Özet: **ret**, **hit**, **sub**, **num** ortamdan (env) ve biriken ödül/istatistikten; **eff** ret ve step sayısından; **H** ajanın Q dağılımından türetilir.

### 7.2 Episode’lar arası hangi artış = sağlıklı eğitim?

- **ret (total reward):** Episode’lar ilerledikçe **genel trend yukarı** olmalı (özellikle 20–40. episode’tan sonra). Makale Figure 9 ile uyumlu şekilde yaklaşık 60–80. episode civarında plato yapıp ~2450 civarına yaklaşması beklenir. Zaman zaman düşüşler (özellikle ε hâlâ yüksekken) normaldir.
- **hit (hit rate):** **Artış** beklenir; son episode’larda makale Tablo 3’e göre %95+ (no prior) veya %98+ (prior) hedeflenir. Düşük kalıyorsa öğrenme yavaş veya exploration/exploitation dengesi gözden geçirilmeli.
- **sub (subband rate):** Hit’ten önce yükselir (önce doğru subband, sonra doğru permütasyon öğrenilir). **Episode’lar arası artış** sağlıklı; sub > hit her zaman (çünkü subband doğru olsa bile Num<4 olabilir).
- **num (avg match):** 0’dan 4’e doğru **ortalama artış** beklenir. Son episode’larda 3.5–4.0 civarı iyi işaret.
- **eff (efficiency):** ret ile uyumlu; **episode’lar arası genel artış** ve sonlara doğru yüksek değerler (pulse başına yüksek ödül) sağlıklı eğitim göstergesidir.
- **H (entropy):** Başta yüksek (ε≈0.995, keşif), episode’lar ve ε decay ile **genel düşüş** beklenir. Son episode’larda düşük H = politikanın kararlı/exploit moda geçtiğini gösterir. Sürekli çok yüksek kalıyorsa politika öğrenemiyor olabilir.

**Kısa özet:** Episode’lar arası **ret, hit, sub, num, eff** değerlerinde genel **artış** ve **H**’de genel **azalış** görüyorsan eğitim sağlıklı ilerliyor kabul edilir. Ret/hit plato yapıp makale değerlerine yaklaşması (Figure 9, Table 3) başarı kriteridir.

---

## 8. GPU / device options and benchmarking

- **Default:** Single GPU only, `cuda:0` (config: `device_id: 0`, `use_multi_gpu: false`).
- **CLI override:** `python -m src.train --device 0 | 1 | multi`
  - `--device 0` → single GPU cuda:0
  - `--device 1` → single GPU cuda:1
  - `--device multi` → DataParallel over all GPUs (batch split)
- **Config:** `device_id`, `use_multi_gpu` in `configs/default.yaml` (overridden by `--device` when given).

**Benchmarking task (performance / CPU–GPU activity):**  
Run separate training processes with `--device 0` and `--device 1` (e.g. two terminals), same config and seed where applicable. Monitor CPU activity (and optionally GPU via `nvidia-smi` or `watch -n 1 nvidia-smi`). Compare wall-clock time and resource usage; document or plot CPU activity. Task is tracked in **TASKLOG.md** Phase 6 (X-05).
