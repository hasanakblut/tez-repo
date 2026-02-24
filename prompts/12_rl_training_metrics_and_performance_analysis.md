# RL Training Metrics and Performance Analysis Documentation

Bu döküman, bilişsel karıştırıcı (cognitive jammer) eğitiminin başarısını, kararlılığını ve hızını ölçmek için kullanılan gelişmiş metrikleri ve analiz yöntemlerini tanımlar.

---

## 1. Öğrenme Hızının Nicelleştirilmesi (Quantifying Learning Speed)

Eğitimin ne kadar hızlı ilerlediğini doğrulamak için sadece görsel trendler yerine aşağıdaki istatistiksel yöntemler kullanılmalıdır:

- **Öğrenme Eğrisi Eğimi (Slope of Learning Curve):** Son $N$ bölümdeki (episode) return değerleri üzerinden hesaplanan lineer regresyon katsayısı ($\beta$).
  - Formül: $y = \beta x + \alpha$ (burada $y$ episodik ödül, $x$ bölüm numarasıdır).
  - $\beta > 0$: Sürekli öğrenme gerçekleşmektedir.
  - $\beta \approx 0$: Model doyuma (plateau) ulaşmıştır.

- **Eşiğe Ulaşma Süresi (Steps to Threshold):** Önceden belirlenen bir başarı eşiğine (örneğin %90 isabet oranı) ulaşmak için gereken toplam darbe (pulse) sayısı. Bu metrik, farklı hiperparametrelerin örneklem verimliliğini (sample efficiency) kıyaslamak için temel ölçüttür.

- **Kümülatif Ödül Artışı:** Birim pulse başına kazanılan ek ödül miktarı; ajanın 240 durumlu geniş eylem uzayındaki arama verimliliğini gösterir.

---

## 2. İsabet Oranı Analizi (Match Distribution)

Radar hiyerarşisindeki (subband ve subpulse) başarıyı ayrıştırmak için ikili (binary) başarı yerine parçalı eşleşme ($Num$) analizi uygulanmalıdır.

- **Ağırlıklı Başarı Skoru (Weighted Success Score):**

  $$
  Score = \frac{1}{K} \sum_{k=1}^{K} \mathbb{1}(f_{agent,k} == f_{radar,k})
  $$

  Burada $K=4$ alt-darbe (sub-pulse) sayısıdır.

- **Makro vs. Mikro Başarı Analizi:**
  - **Subband Success Rate:** Ajanın sadece doğru alt-bandı seçme oranı.
  - **Permutation Success Rate:** Ajan doğru alt-bandı bulduğunda, içindeki 4'lü dizilimi doğru tahmin etme oranı.

---

## 3. Kararlılık ve Varyans Metrikleri (Stability Metrics)

Eğitimin güvenilirliğini ölçmek için kullanılan metriklerdir:

- **Bağıl Standart Sapma (Relative Standard Deviation – RSD):** Son $P$ penceresindeki episodik ödüllerin standart sapmasının ortalamaya oranı. Düşük RSD kararlı bir politikayı; yüksek RSD, radarın %30 zıplama olasılığı karşısında bocalayan kararsız bir ajanı simgeler.

- **Maksimum Pişmanlık (Maximum Regret):** İdeal bir ajanın alabileceği maksimum ödül ile mevcut ajanın aldığı ödül arasındaki farkın kümülatif toplamı.

---

## 4. Gelişmiş Bilişsel İçgörüler (Advanced Insights)

Modelin ezber yapıp yapmadığını anlamak için politika entropisi (policy entropy) takip edilmelidir:

- **Politika Entropisi (Policy Entropy):** Modelin aksiyon seçimindeki belirsizliğini ölçer.
  - Eğitim başında yüksek entropi, keşif (exploration) aşamasını gösterir.
  - Eğitim sonunda düşük entropi ve yüksek başarı, modelin belirli paternleri çözdüğünü doğrular.
- **Analiz:** Başarı artarken entropi düşmüyorsa, ajan stratejik bir öğrenme yerine frekans olasılıklarını (bias) ezberliyor olabilir.

---

## 5. Önerilen Loglama ve Terminal Formatı

İzlenebilirliği artırmak için Cursor terminal çıktısında şu yapının kullanılması planlanmaktadır:

```
[Episode 150] | Progress: 50% | Total Pulses: 75,000
---------------------------------------------------------
Performance Metrics:
  Return: 18,400 (MA50: 17,200 | Reg. Slope: +0.45)
  Hit Rate (4/4): 0.62 | Avg Match (Num): 3.15 / 4.0
  Efficiency: 0.24 reward/pulse
  Stability: RSD 12% (Stable Trend ↑)
---------------------------------------------------------
Discovery Analysis:
  Subband Detection Rate: 98% (Phase 1 Logic)
  Intra-Pulse Precision: 45% (Phase 2-3 Logic)
  Policy Entropy: 1.25 (Certainty Increasing)
```
