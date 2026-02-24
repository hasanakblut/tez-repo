# PyTorch (CUDA) + Gymnasium — Windows / Miniconda

Python 3.11 Miniconda ortamında CUDA destekli PyTorch ve proje bağımlılıklarını kurmak için.

---

## 1. Ortam

```bash
conda create -n tez python=3.11 -y
conda activate tez
```

---

## 2. PyTorch (CUDA)

**PyPI’daki `pip install torch` Windows’ta CUDA’sız (CPU) kurar.** CUDA için resmi wheel index’i kullanın.

Sisteminizdeki **CUDA sürümüne** göre **tek** komut çalıştırın. `nvidia-smi` çıktısında sağ üstte **"CUDA Version"** (sürücünün desteklediği en yüksek sürüm) yazar.

- **CUDA Version 13.0 veya 12.x** (örn. RTX 4070) → **cu128** kullanın (sürücü geriye uyumludur).
- **CUDA 12.1** → cu121.
- **CUDA 11.x** → cu118.

**Önerilen (CUDA 12.8, driver 12.x/13.x) — requirements ile:**
```bash
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu128
```

**Tek komut (torch yalnız):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**Diğer CUDA:** cu118, cu121, cu124 için `requirements-torch.txt` ile aynı komutta sadece `cu128` yerine cu118/cu121/cu124 yazın.

Kontrol:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Kaynak: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally) (OS: Windows, Package: Pip, Compute: CUDA x.x seçin).

---

## 3. Gymnasium ve GPU

**Gymnasium GPU kullanmaz.** Ortam (env) CPU’da, NumPy ile çalışır. GPU’yu kullanan kısım **PyTorch** tarafıdır: policy/target ağları, forward/backward, optimizer. Yani `pip install gymnasium` GPU’yu “etkinleştirmez” veya kullanmaz; GPU kullanımı tamamen PyTorch’un CUDA kurulumuna bağlıdır.

**Test:** `python -m scripts.check_gpu_usage` çalıştırın. Çıktıda `PyTorch CUDA available: True` ve `Policy network parameters on: cuda:0` görürseniz eğitim GPU kullanıyordur. Eğitim sırasında başka bir terminalde `nvidia-smi` ile GPU bellek ve kullanımını izleyebilirsiniz.

Gymnasium **PyTorch’a bağımlı değil**; sadece NumPy kullanır. PyTorch sürümüyle uyum seçimi gerekmez. Proje gereksinimleriyle birlikte kurulur:

```bash
pip install -r requirements.txt
```

(Bu aşamada `torch` zaten 2. adımda kurulmuş olmalı; `requirements.txt` içinde torch satırı yok, CUDA kurulumu ayrı yapılıyor.)

---

## 4. Sıra özeti

1. `conda activate tez`
2. PyTorch CUDA: `pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu128`
3. `pip install -r requirements.txt`
