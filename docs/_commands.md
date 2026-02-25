# Komut Özeti — Önemli Python Script'leri

Proje kökünden (`codes/`) çalıştırın. Tüm komutlar: `python -m ...` ile verilmiştir.

---

## 1. Training (eğitim çalıştırma)

| Amaç | Komut |
|------|--------|
| **Tam eğitim** (default config) | `python -m src.train` |
| **Özel config ile eğitim** | `python -m src.train --config configs/default_embedding.yaml` |
| **GPU seçimi** | `python -m src.train --config configs/default.yaml --device 0` veya `--device 1` veya `--device multi` |
| **Kısa dry-run / test** | `python -m scripts.training_test --config configs/test_training.yaml --plot` |
| **4 epsilon senaryosunu sırayla çalıştır** | `python -m scripts.run_epsilon_scenarios` (opsiyonel: `--device 0`, `--config-dir ...`) |

- Config’ler: `configs/default.yaml`, `configs/test_training.yaml`, `configs/epsilon_scenario*.yaml`, `configs/default_embedding.yaml`.
- Çıktılar: `results/runs/run_<datetime>_<config>_ep<N>/` (log, checkpoint, metrics, plot’lar).
- **TensorBoard:** Eğitim sırasında metrikler `run_*/tensorboard/` altına yazılır. İnteraktif takip için ayrı terminalde:
  - Tek run: `tensorboard --logdir=results/runs/run_<id>/tensorboard`
  - Tüm run’lar: `tensorboard --logdir=results/runs`
  - Tarayıcıda http://localhost:6006 açılır. `step/*` = pulse bazlı, `episode/*` = episode sonu metrikler. Terminalde anlık loglama `logging.verbose_terminal: false` ile kapatılır (varsayılan TensorBoard kullanırken).

---

## 2. Markov matrisi (oluşturma ve kontrol)

| Amaç | Komut |
|------|--------|
| **Matris oluştur ve kaydet** (eğitimden önce gerekli) | `python -m scripts.init_markov_matrix` |
| Aynı iş, özel seed/mode | `python -m scripts.init_markov_matrix --seed 42 --mode markov` |
| **Matrisi yazdır + .npy/.csv kaydet** | `python -m scripts.print_markov_matrix --config configs/default.yaml --mode markov_subband` |
| **Matrisi test et + görselleştir** | `python -m scripts.test_markov_matrix --config configs/default.yaml --mode markov` (plot atlamak: `--no-plot`) |

- `init_markov_matrix`: Varsayılan çıktı `results/markov_matrices/markov_P_markov_seed42.npy` — config’teki `markov_transition_path` ile aynı olmalı.
- `--mode`: `markov` veya `markov_subband`.

---

## 3. Pulse train (üretim ve görselleştirme)

| Amaç | Komut |
|------|--------|
| **Pulse train üret ve plot** | `python -m scripts.plot_pulse_train --seed 42 --start-index 0 --num-pulses 500` |
| Farklı mod / hazır P matrisi | `python -m scripts.plot_pulse_train --mode markov --markov-npy results/markov_matrices/markov_P_markov_seed42.npy --num-pulses 1000` |
| Seriyi dosyaya kaydet | `python -m scripts.plot_pulse_train --num-pulses 500 --save-train` |
| **Seed/start_index tekrarlanabilirlik testi** | `python -m scripts.test_pulse_train_randomness_in_markov --config configs/default.yaml --markov-npy results/markov_matrices/markov_P_markov_seed42.npy` |

- Plot çıktısı: `results/pulse_train_plots/` (varsayılan).
- `--mode`: `uniform`, `periodic`, `lcg`, `markov`, `markov_subband`.

---

## 4. Diğer faydalı script’ler

| Amaç | Komut |
|------|--------|
| **GPU kullanımını doğrula** | `python -m scripts.check_gpu_usage` veya `--config configs/test_training.yaml` |

- Env CPU’da; model/agent GPU’da. Eğitim sırasında GPU izlemek için ayrı terminalde: `watch -n 1 nvidia-smi`.

---

## Hızlı başlangıç sırası

1. **Markov matrisi** (config’te `generator_mode: markov` kullanıyorsanız):  
   `python -m scripts.init_markov_matrix`
2. **Kısa test**:  
   `python -m scripts.training_test --config configs/test_training.yaml --plot`
3. **Tam eğitim**:  
   `python -m src.train`  
   veya özel config:  
   `python -m src.train --config configs/default_embedding.yaml`

Detaylı eğitim çıktıları ve metrik açıklamaları: `docs/TRAINING_VERIFICATION.md`.
