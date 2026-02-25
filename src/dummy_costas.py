import numpy as np
from itertools import permutations

def create_radar_transition_matrix():
    """
    240 durumlu (10 Subband x 24 Pattern) radar geçiş matrisini oluşturur.
    Mantık:
    1. Subband: Aynı bantta kalma yok, yakın bantlara geçiş olasılığı daha yüksek (alpha decay).
    2. Intra-pulse: 12 Costas dizisi arasında gürültülü döngüsel (noisy cyclic) geçiş.
    """
    
    # --- 1. PARAMETRE TANIMLAMALARI ---
    NUM_SUBBANDS = 10
    NUM_PATTERNS = 24
    TOTAL_STATES = NUM_SUBBANDS * NUM_PATTERNS # 240
    
    # N=4 için doğrulanmış Costas indisi listesi (Lexicographical order)
    # Bu indisler 24 permütasyon içindeki ideal radar dizilimleridir
    costas_ring = [1, 3, 4, 11, 22, 20, 19, 17, 15, 12, 8, 6]
    non_costas = [i for i in range(NUM_PATTERNS) if i not in costas_ring]
    
    # Olasılık Katsayıları
    alpha = 1.2          # Subband geçişleri için yakınlık katsayısı
    lambda_p = 1.5       # Intra-pulse Hamming mesafesi etkisi
    p_next = 0.7         # Döngüde bir sonraki Costas'a geçme olasılığı
    p_stay = 0.1         # Aynı pattern'da kalma olasılığı
    p_skip = 0.15        # Halka içindeki diğer Costas'lara sıçrama
    p_out = 0.05         # Rastgele (Costas olmayan) dizine çıkış
    
    all_perms = list(permutations([1, 2, 3, 4]))

    # --- 2. INTER-PULSE (SUBBAND) GEÇİŞ MATRİSİ (10x10) ---
    P_inter = np.zeros((NUM_SUBBANDS, NUM_SUBBANDS))
    for i in range(NUM_SUBBANDS):
        for j in range(NUM_SUBBANDS):
            if i != j:
                # Mesafe arttıkça olasılık exponantial olarak düşer
                P_inter[i, j] = np.exp(-alpha * abs(i - j))
        P_inter[i] /= P_inter[i].sum() # Satır normalizasyonu

    # --- 3. INTRA-PULSE (PATTERN) GEÇİŞ MATRİSİ (24x24) ---
    P_intra = np.zeros((NUM_PATTERNS, NUM_PATTERNS))
    for i in range(NUM_PATTERNS):
        weights = np.zeros(NUM_PATTERNS)
        
        # Hamming mesafesi bazlı temel ağırlıklar (donanım kısıtı simülasyonu)
        for j in range(NUM_PATTERNS):
            dist = sum(1 for k in range(4) if all_perms[i][k] != all_perms[j][k])
            weights[j] = np.exp(-lambda_p * dist)
            
        if i in costas_ring:
            # Mevcut durum Costas ise stratejik Noisy Cyclic logic uygula
            curr_pos = costas_ring.index(i)
            next_val = costas_ring[(curr_pos + 1) % 12]
            
            # Değerleri normalize etmeden önce stratejik ağırlıkları ekle
            weights[next_val] += p_next * 10 
            weights[i] += p_stay * 10
            for skip_idx in costas_ring:
                if skip_idx != i and skip_idx != next_val:
                    weights[skip_idx] += (p_skip / (len(costas_ring)-2)) * 10
            for out_idx in non_costas:
                weights[out_idx] += (p_out / len(non_costas)) * 10
        else:
            # Eğer sistem rastgele bir durumdaysa Costas halkasına dönme eğilimi
            for j in costas_ring:
                weights[j] *= 5 
        
        P_intra[i] = weights / weights.sum()

    # --- 4. GLOBAL 240x240 MATRİSİN BİRLEŞTİRİLMESİ ---
    # Durum İndisleme Formülü: (subband_index * 24) + pattern_index
    P_total = np.zeros((TOTAL_STATES, TOTAL_STATES))
    
    for i_sub in range(NUM_SUBBANDS):
        for i_pat in range(NUM_PATTERNS):
            idx_i = i_sub * NUM_PATTERNS + i_pat
            
            for j_sub in range(NUM_SUBBANDS):
                for j_pat in range(NUM_PATTERNS):
                    idx_j = j_sub * NUM_PATTERNS + j_pat
                    
                    # Birleşik olasılık: P(Sj | Si) = P_inter(sub_j | sub_i) * P_intra(pat_j | pat_i)
                    P_total[idx_i, idx_j] = P_inter[i_sub, j_sub] * P_intra[i_pat, j_pat]
    
    return P_total

# KULLANIM
transition_matrix = create_radar_transition_matrix()

print(f"Toplam Durum Sayısı: {transition_matrix.shape[0]}")
print(f"Matris Normalizasyon Kontrolü (Satır toplamları): {np.allclose(transition_matrix.sum(axis=1), 1.0)}")

# Örnek: Durum 0'dan sonraki en muhtemel 3 durumu bulma
current_state = 0
next_probs = transition_matrix[current_state]
top_indices = np.argsort(next_probs)[-3:][::-1]
print(f"\nDurum {current_state} sonrası en muhtemel geçişler:")
for idx in top_indices:
    print(f"Hedef Durum: {idx}, Olasılık: {next_probs[idx]:.4f}")