# Training Run Metrics — Referans

Eğitim sırasında (terminal, log ve `metrics.jsonl`) yazdırılan metriklerin kısa referansı. Run kurulumu ve doğrulama için `TRAINING_VERIFICATION.md` dosyasına bakın.

---

## 1. Metriklerin tanımı ve hesap mantığı

| Kısaltma | Tam ad / kaynak | Hesaplama mantığı |
|----------|------------------|-------------------|
| **ret** | Episode total reward | Episode boyunca toplanan ödül: `ret = Σ (JSR_base × Num)` her pulse için. Num = jammer aksiyonu ile radarın bir sonraki durumu arasındaki pozisyon bazlı sub-pulse eşleşme sayısı (0–4). Kaynak: `env.step` → `reward` birikimi (`train.py` içinde `episode_reward`). |
| **hit** | Hit rate (4/4 tam vuruş oranı) | O ana kadar geçen pulse'lar içinde **Num=4** (tam jam) olanların oranı. `hit_rate = total_full_hits / pulse_count`. Ortamda her step'te jammer hem subband hem permutasyonu doğru tahmin ederse Num=4. Kaynak: `env._build_info()` → `info['hit_rate']`. |
| **sub** | Subband rate | O ana kadar pulse'lar içinde **doğru subband** tahmin edilenlerin oranı. `subband_rate = total_subband_hits / pulse_count`. Subband doğru olsa bile permütasyon yanlış olabilir (Num 1–3). Kaynak: `env._build_info()` → `info['subband_rate']`. |
| **num** | Avg match (Num ortalaması) | Pulse başına ortalama eşleşme sayısı: `avg_match = total_matches / pulse_count`. 0–4 arası; 4'e ne kadar yakınsa o kadar iyi. Kaynak: `env._build_info()` → `info['avg_match']`. |
| **eff** | Efficiency (örnek verimliliği) | Pulse başına ortalama ödül: `eff = episode_reward / env_steps`. Yani `ret / pulse_sayısı`. Aynı episode içinde pulse arttıkça ret artar ama eff genelde daha stabil bir "ne kadar ödül per step" göstergesidir. Kaynak: `train.py` içinde `episode_reward / env_steps`. |
| **H** | Policy entropy (bit) | O ana kadar o episode'ta alınan aksiyonlar için softmax(Q) üzerinden hesaplanan entropi ortalaması: `H = -Σ p_i log2(p_i)` (bit). Yüksek H = politika belirsiz/keşifçi, düşük H = politika kararlı/exploit. Kaynak: `agent.select_action_with_info` → her step'te entropy; `train.py`'de `ep_entropy_sum / env_steps`. |

Özet: **ret**, **hit**, **sub**, **num** ortamdan (env) ve biriken ödül/istatistikten; **eff** ret ve step sayısından; **H** ajanın Q dağılımından türetilir.

---

## 2. Metriklerin nerede göründüğü

- **Terminal (tqdm):** Pulse bar'da `ret`, `hit`, `sub`, `num`, `eff`; periyodik log satırlarında ε ve **H** eklenir.
- **Episode özet satırı:** `reward=... hit_rate=... time=...` (ve isteğe bağlı ε, loss vb.).
- **`run_*/metrics.jsonl`:** Her episode için bir JSON (total_reward, hit_rate, epsilon, avg_loss, env_steps, train_steps, …).
- **`run_*/training_plots/`:** `reward_curve.png`, `hit_rate_curve.png`, `epsilon_curve.png`, `loss_curve.png` — her `plot_interval` episode'ta güncellenir.

---

## 3. Episode'lar arası hangi artış = sağlıklı eğitim?

- **ret (total reward):** Episode'lar ilerledikçe **genel trend yukarı** olmalı (özellikle 20–40. episode'tan sonra). Makale Figure 9 ile uyumlu şekilde yaklaşık 60–80. episode civarında plato yapıp ~2450 civarına yaklaşması beklenir. Zaman zaman düşüşler (özellikle ε hâlâ yüksekken) normaldir.
- **hit (hit rate):** **Artış** beklenir; son episode'larda makale Tablo 3'e göre %95+ (no prior) veya %98+ (prior) hedeflenir. Düşük kalıyorsa öğrenme yavaş veya exploration/exploitation dengesi gözden geçirilmeli.
- **sub (subband rate):** Hit'ten önce yükselir (önce doğru subband, sonra doğru permütasyon öğrenilir). **Episode'lar arası artış** sağlıklı; sub > hit her zaman (çünkü subband doğru olsa bile Num<4 olabilir).
- **num (avg match):** 0'dan 4'e doğru **ortalama artış** beklenir. Son episode'larda 3.5–4.0 civarı iyi işaret.
- **eff (efficiency):** ret ile uyumlu; **episode'lar arası genel artış** ve sonlara doğru yüksek değerler (pulse başına yüksek ödül) sağlıklı eğitim göstergesidir.
- **H (entropy):** Başta yüksek (ε≈0.995, keşif), episode'lar ve ε decay ile **genel düşüş** beklenir. Son episode'larda düşük H = politikanın kararlı/exploit moda geçtiğini gösterir. Sürekli çok yüksek kalıyorsa politika öğrenemiyor olabilir.

**Kısa özet:** Episode'lar arası **ret, hit, sub, num, eff** değerlerinde genel **artış** ve **H**'de genel **azalış** görüyorsan eğitim sağlıklı ilerliyor kabul edilir. Ret/hit plato yapıp makale değerlerine yaklaşması (Figure 9, Table 3) başarı kriteridir.

---

## 4. Policy entropy (H) — yorumlama ve uyarılar

**Bit aralıkları:** Maksimum entropi = log₂(N) bit (N = aksiyon sayısı; 240 aksiyonda ≈7.9 bit — tam rastgele). Minimum 0 bit = ajan tek bir aksiyona kilitlenmiş. **İdeal seyir:** Eğitim başında ~4–4.5 bit, ilerledikçe 0.5–1.0 bit (veya 0.8–1.5 bit) bandına inmesi beklenir; bu aralık başarılı pattern matching göstergesi sayılır.

**Kritik durum:** Reward artıyor ama H hâlâ yüksek (ör. ~3 bit) → ajan büyük ölçüde şansla doğruyu buluyor, öğrenme tam oturmamış olabilir.

**Best practice:** Entropi her step'te Q üzerinden hesaplanıp episode sonunda ortalaması alınarak loglanmalı (projede zaten `ep_entropy_sum / env_steps` ile yapılıyor). Kümülatif ortalama baştaki “bilgisiz” dönemi vurgulayacağı için episode bazlı takip tercih edilir.

**Policy collapse:** H çok hızlı 0'a iner ama reward artmaz — ajan radar örüntüsünü çözmeden tek bir frekansa takılmış olabilir. Çözüm olarak cyclical epsilon (epsilon'u zaman zaman zıplatarak keşfi artırmak) kullanılabilir.

**Görselleştirme ipucu:** H ile hit rate (success rate) aynı grafikte dual-axis çizilirse; hit artarken H düşüyorsa modelin ezberlemeden öğrendiğinin güçlü bir göstergesi olur.

---

## 5. Run çıktıları (kısa referans)

| Çıktı | Konum | Açıklama |
|--------|--------|----------|
| Run directory | `results/runs/run_<datetime>_<config>_ep<N>/` | Tek bir training run'ının tüm çıktıları. |
| Log | `run_*/training.log` | Header (config, seed, run_dir) ve episode bazlı satırlar. |
| Metrics | `run_*/metrics.jsonl` | Her episode için bir JSON (total_reward, hit_rate, epsilon, avg_loss, env_steps, …). |
| Checkpoints | `run_*/checkpoint_ep*.pt`, `run_*/final_model.pt` | Model snapshot'ları. |
| Curves | `run_*/training_plots/*.png` | Reward, hit rate, epsilon, loss vs episode. |

Epsilon decay, `training.epsilon_decay_mode` ve `epsilon_decay_episodes` / `epsilon_decay_steps` ile kontrol edilir; ayrıntı için `docs/EPSILON_DECAY.md`.
