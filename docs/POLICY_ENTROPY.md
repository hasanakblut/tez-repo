# Policy Entropy — Değerlendirme Kılavuzu

## 1. Metriği Nasıl Değerlendirmeliyiz? (Bit Cinsinden)

Bu değer, ajanın aksiyon seçerken ne kadar "kararsız" olduğunu söyler.

**Maksimum Entropi** ($log_2(N)$): Eğer 24 aksiyonun varsa, $\log_2(24) \approx 4.58$ bit maksimumdur. Bu, ajanın tamamen rastgele (zar atarak) hareket ettiği anlamına gelir.

**Minimum Entropi (0 bit):** Ajan tek bir aksiyona %100 emin şekilde kilitlenmiş demektir.

**İdeal Senaryo:** Eğitimin başında bu değerin 4.0 - 4.5 bandında başlaması, eğitim ilerledikçe (epsilon düştükçe) 0.5 - 1.0 bandına (veya altına) inmesi beklenir.

**Kritik Analiz:** Eğer Reward artıyor ama Entropi hala çok yüksekse (örneğin 3 bit), ajan "şans eseri" doğruyu buluyor ama hala çok fazla rastgele deneme yapıyor demektir. Bu, öğrenmenin tam "oturmadığına" işarettir.

---

## 2. Best Practice: Ne Zaman ve Nasıl Hesaplanmalı?

RL literatüründeki en iyi uygulama (best practice) şudur: **Adım Başına Hesaplama, Episode Başına Ortalama.** Her step içinde o anki $Q$ değerleri üzerinden entropiyi hesapla, bu değerleri bir listede topla, episode bittiğinde bu listenin ortalamasını alıp "Mean Policy Entropy" olarak logla.

**Neden Episode Bazlı?** Tüm training başından beri kümülatif hesaplarsan, baştaki "bilgisiz" dönem verileri, sondaki "uzman" dönemi gölgeler. Episode bazlı takip, ajanın her rauntta radarın yeni örüntüsünü ne kadar sürede çözdüğünü görmeni sağlar.

---

## 3. İleri Düzey (Advanced) Okuma: "Collapse" Takibi

Bu metrikte en çok korkulan şey **"Policy Collapse"**tır.

- **Belirti:** Entropi çok hızlı bir şekilde 0'a düşer ama Reward artmaz.
- **Anlamı:** Ajan daha radarın örüntüsünü çözmeden "hep aynı frekansa basma" gibi kötü bir alışkanlık edinmiş ve yeni şeyler denemeyi bırakmıştır.
- **Çözüm:** İşte burada senin önceki mesajında önerdiğin "Cyclical Epsilon" devreye girer. Entropi 0'a çakıldığında epsilonu zıplatarak entropiyi tekrar yukarı çekebilirsin.

---

## 4. Makale ile İlişkisi

Makaledeki radarın frekans atlama hızı (intra-pulse) çok yüksek olduğu için, entropinin tamamen 0 olması zordur. Çünkü radarın alt darbelerdeki (subpulses) rastgeleliği, ajanda her zaman bir miktar "belirsizlik" bırakacaktır.

**Tahmin:** Senin modelinde entropinin 0.8 - 1.5 bit arasına düşmesi, başarılı bir "pattern matching" (örüntü eşleme) yapıldığını kanıtlar.
