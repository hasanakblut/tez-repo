"""
Create Custom Markov Transition Matrix (vectorized, ergodic)
==========================================================
Vektörize, ergodik (tek çevrim) geçiş matrisi oluşturur:
- Baskın geçiş: Her satırda tek bir hedefe 0.8–0.9 olasılık (permütasyon çevrimi).
- Gürültü: Kalan olasılık diğer hücrelere rastgele dağıtılır.
- Nested loop yok; NumPy indexing ve permutation kullanılır.

Ayrıca --mode deterministic_cycle ile deterministik çevrim matrisi: bir state'ten
sonrakine tam olasılıkla (1.0) geçiş, her state bir kez gezilip çevrim tamamlanır.

Usage:
    python -m scripts.create_custom_markov_transition_matrix
    python -m scripts.create_custom_markov_transition_matrix --seed 42 --states 240
    python -m scripts.create_custom_markov_transition_matrix --mode deterministic_cycle --seed 42

Creates:
    results/markov_matrices/markov_P_custom_seed<seed>.npy  (veya deterministic_cycle)
    results/markov_matrices/markov_P_<mode>_seed<seed>.png
"""

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_custom_transition_matrix(
    n: int,
    rng: np.random.Generator,
    dominant_low: float = 0.8,
    dominant_high: float = 0.9,
) -> np.ndarray:
    """
    n x n geçiş matrisi: vektörize, ergodik (tek çevrim), baskın geçiş + gürültü.
    Nested loop kullanılmaz.
    """
    # Ergodisite: tek bir n-çevrimi — her state'in tam bir girişi ve bir çıkışı
    perm = np.concatenate([[0], rng.permutation(np.arange(1, n))])
    dominant_next = np.empty(n, dtype=np.intp)
    dominant_next[perm[:-1]] = perm[1:]
    dominant_next[perm[-1]] = perm[0]

    # Baskın geçiş olasılığı: satır başına [dominant_low, dominant_high]
    dominant_prob = rng.uniform(dominant_low, dominant_high, size=n)

    # Baskın hücrelere ata (vektörize)
    P = np.zeros((n, n))
    P[np.arange(n), dominant_next] = dominant_prob

    # Gürültü: kalan olasılığı diğer hücrelere rastgele dağıt (vektörize)
    remainder = 1.0 - dominant_prob
    noise = rng.random((n, n))
    noise[np.arange(n), dominant_next] = 0
    row_sums = noise.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)  # satır tamamen 0 olmasın
    noise = noise / row_sums
    P = P + noise * remainder[:, np.newaxis]

    # Normalizasyon: satır toplamları tam 1.0
    P = P / P.sum(axis=1, keepdims=True)
    return P


def create_deterministic_cycle_transition_matrix(
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    n x n deterministik çevrim geçiş matrisi: bir state'ten bir sonrakine tam olasılıkla (1.0)
    geçiş; tüm state'ler tek bir çevrimde bir kez gezilir, sonra döngü tamamlanır.

    Matris bir permütasyon matrisidir: her satırda tam bir tane 1, geri kalan 0.
    Geçişler tek bir n-çevrimi oluşturur (ergodik). Nested loop yok.
    """
    # Tek n-çevrimi: perm[0]->perm[1]->...->perm[n-1]->perm[0]
    perm = np.concatenate([[0], rng.permutation(np.arange(1, n))])
    next_state = np.empty(n, dtype=np.intp)
    next_state[perm[:-1]] = perm[1:]
    next_state[perm[-1]] = perm[0]

    P = np.zeros((n, n))
    P[np.arange(n), next_state] = 1.0
    return P


def plot_matrix(P: np.ndarray, out_path: Path, title_label: str, n: int = 240) -> None:
    """Matrisi [0,1] olasılık skalasında heatmap olarak çiz; 240x240 ise subband grid."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(P, cmap="viridis", aspect="equal", vmin=0.0, vmax=1.0)

    if n == 240:
        for k in range(1, 10):
            ax.axhline(k * 24 - 0.5, color="white", linewidth=0.5)
            ax.axvline(k * 24 - 0.5, color="white", linewidth=0.5)
        ax.axhline(239.5, color="white", linewidth=0.5)
        ax.axvline(239.5, color="white", linewidth=0.5)

    ax.set_xlabel("Next state (column)")
    ax.set_ylabel("Current state (row)")
    ax.set_title(
        f"Markov Transition Matrix P ({title_label})\n"
        "probability [0, 1]" + ("; subband boundaries in white" if n == 240 else "")
    )
    plt.colorbar(im, ax=ax, label="P")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create custom ergodic Markov transition matrix (vectorized, no nested loops)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility",
    )
    parser.add_argument(
        "--states",
        type=int,
        default=240,
        help="Number of states (matrix size n x n)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/markov_matrices",
        help="Directory to save .npy and .png",
    )
    parser.add_argument(
        "--dominant-low",
        type=float,
        default=0.8,
        help="Lower bound for dominant transition probability",
    )
    parser.add_argument(
        "--dominant-high",
        type=float,
        default=0.9,
        help="Upper bound for dominant transition probability",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="custom",
        choices=["custom", "deterministic_cycle"],
        help="custom: baskın geçiş 0.8-0.9 + gürültü; deterministic_cycle: tam olasılıkla (1.0) tek çevrim",
    )
    args = parser.parse_args()

    n = args.states
    if n < 2:
        raise ValueError("--states must be >= 2")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    mode = args.mode
    seed = args.seed

    if mode == "deterministic_cycle":
        print("Building deterministic cycle transition matrix (1.0 per row, single n-cycle)...")
        P = create_deterministic_cycle_transition_matrix(n, rng)
        stem = f"markov_P_deterministic_cycle_seed{seed}"
    else:
        print("Building custom transition matrix (vectorized, ergodic)...")
        P = create_custom_transition_matrix(
            n,
            rng,
            dominant_low=args.dominant_low,
            dominant_high=args.dominant_high,
        )
        stem = f"markov_P_custom_seed{seed}"

    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Rows must sum to 1.0"
    print(f"  Shape: {P.shape}, row sums OK (min={row_sums.min():.6f}, max={row_sums.max():.6f})")

    npy_path = out_dir / f"{stem}.npy"
    np.save(npy_path, P)
    print(f"Saved: {npy_path}")

    png_path = out_dir / f"{stem}.png"
    try:
        plot_matrix(P, png_path, title_label=f"{mode}, seed={seed}", n=n)
        print(f"Saved plot: {png_path}")
    except ImportError as e:
        print(f"Skipping plot (matplotlib not available): {e}")

    nnz = (P > 1e-10).sum(axis=1)
    print(f"Nonzeros/row: min={nnz.min()}, max={nnz.max()}, mean={nnz.mean():.1f}")
    try:
        rel = npy_path.relative_to(PROJECT_ROOT)
        print(f"Use in config: markov_transition_path: {rel.as_posix()}")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
