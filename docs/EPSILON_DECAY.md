# Epsilon Decay: Per Step vs Per Episode

Bu belge, ε (exploration rate) azalmasının **adım bazlı (per step)** mı yoksa **bölüm sonu bazlı (per episode)** mı uygulanacağını netleştirir ve projedeki davranışı açıklar.

---

## Makale ne diyor?

GA-Dueling DQN makalesi (Xia et al., Sensors 2024, 24, 1325), Table 2 ve Section 3.3’te şunları belirtir:

- **ε başlangıç:** 0.995  
- **ε bitiş:** 0.005  
- **Azalma:** Üstel (exponential) decay, “consistent decay rate” ile.

Makale, azalmanın **her adımda** mı yoksa **her episode sonunda** mı uygulanacağını **açıkça belirtmez**. Birçok DQN implementasyonu (ör. OpenAI Baselines, Dopamine) ε’yu **her environment step** sonrası günceller; bazı eğitim scriptleri ise **episode sonunda** bir kez günceller.

---

## Projedeki davranış

| Seçenek | Açıklama | Varsayılan |
|--------|----------|------------|
| **per_episode** | ε, her episode bittiğinde bir kez güncellenir. 100 episode × 10.000 pulse ile toplam 1M adımda 100 kez decay uygulanır. | **Evet** (mevcut davranış) |
| **per_step** | ε, her environment step (pulse) sonrası güncellenir. `epsilon_decay_steps` adım boyunca 0.995→0.005 üstel azalma uygulanır. | Hayır |

Config’te `training.epsilon_decay_mode` ile seçilir:

- `epsilon_decay_mode: "per_episode"` → `epsilon_decay_episodes` kullanılır (örn. 100).  
- `epsilon_decay_mode: "per_step"` → `epsilon_decay_steps` kullanılır (örn. 1_000_000).

Aynı toplam “ilerleme” için (100 episode × 10k pulse = 1M step) iki mod da benzer ε eğrisi üretir; per_step, episode uzunluğu değişse bile aynı step sayısında aynı ε’ya ulaşır.

---

## Referanslar

- **Makale:** Xia et al., *GA-Dueling DQN*, Sensors 2024, 24, 1325 — Table 2, Section 3.3.  
- **Kod:** `src/agent.py` (`decay_epsilon`, `epsilon_decay` hesabı), `src/train.py` (decay çağrısının yeri).  
- **Config:** `configs/default.yaml` → `training.epsilon_decay_mode`, `epsilon_decay_episodes`, `epsilon_decay_steps`.
