"""
Create Markov Transition Matrix (Costas-based 240×240)
======================================================
Builds the Costas/subband transition matrix from src.dummy_costas logic,
saves it as .npy, and plots it as .png. Uses the same seed/out-dir conventions
as init_markov_matrix, print_markov_matrix, and test_markov_matrix.

Can also plot an existing .npy matrix (e.g. markov_P_markov_seed42.npy).

Usage:
    python -m scripts.create_markov_transition_matrix
    python -m scripts.create_markov_transition_matrix --seed 42 --out-dir results/markov_matrices
    python -m scripts.create_markov_transition_matrix --plot-npy results/markov_matrices/markov_P_markov_seed42.npy

Creates (when not using --plot-npy):
    results/markov_matrices/markov_P_costas_seed<seed>.npy
    results/markov_matrices/markov_P_costas_seed<seed>.png

When using --plot-npy: loads the .npy and saves <stem>.png in --out-dir.
"""

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_radar_transition_matrix() -> np.ndarray:
    """
    240 durumlu (10 Subband x 24 Pattern) radar geçiş matrisini oluşturur.
    Mantık:
    1. Subband: Aynı bantta kalma yok, yakın bantlara geçiş olasılığı daha yüksek (alpha decay).
    2. Intra-pulse: 12 Costas dizisi arasında gürültülü döngüsel (noisy cyclic) geçiş.
    """
    from itertools import permutations

    NUM_SUBBANDS = 10
    NUM_PATTERNS = 24
    TOTAL_STATES = NUM_SUBBANDS * NUM_PATTERNS

    costas_ring = [1, 3, 4, 11, 22, 20, 19, 17, 15, 12, 8, 6]
    non_costas = [i for i in range(NUM_PATTERNS) if i not in costas_ring]

    alpha = 1.2
    lambda_p = 1.5
    p_next = 0.7
    p_stay = 0.1
    p_skip = 0.15
    p_out = 0.05

    all_perms = list(permutations([1, 2, 3, 4]))

    # --- P_inter (10x10) ---
    P_inter = np.zeros((NUM_SUBBANDS, NUM_SUBBANDS))
    for i in range(NUM_SUBBANDS):
        for j in range(NUM_SUBBANDS):
            if i != j:
                P_inter[i, j] = np.exp(-alpha * abs(i - j))
        P_inter[i] /= P_inter[i].sum()

    # --- P_intra (24x24) ---
    P_intra = np.zeros((NUM_PATTERNS, NUM_PATTERNS))
    for i in range(NUM_PATTERNS):
        weights = np.zeros(NUM_PATTERNS)
        for j in range(NUM_PATTERNS):
            dist = sum(1 for k in range(4) if all_perms[i][k] != all_perms[j][k])
            weights[j] = np.exp(-lambda_p * dist)

        if i in costas_ring:
            curr_pos = costas_ring.index(i)
            next_val = costas_ring[(curr_pos + 1) % 12]
            weights[next_val] += p_next * 10
            weights[i] += p_stay * 10
            for skip_idx in costas_ring:
                if skip_idx != i and skip_idx != next_val:
                    weights[skip_idx] += (p_skip / (len(costas_ring) - 2)) * 10
            for out_idx in non_costas:
                weights[out_idx] += (p_out / len(non_costas)) * 10
        else:
            for j in costas_ring:
                weights[j] *= 5

        P_intra[i] = weights / weights.sum()

    # --- P_total 240x240 ---
    P_total = np.zeros((TOTAL_STATES, TOTAL_STATES))
    for i_sub in range(NUM_SUBBANDS):
        for i_pat in range(NUM_PATTERNS):
            idx_i = i_sub * NUM_PATTERNS + i_pat
            for j_sub in range(NUM_SUBBANDS):
                for j_pat in range(NUM_PATTERNS):
                    idx_j = j_sub * NUM_PATTERNS + j_pat
                    P_total[idx_i, idx_j] = P_inter[i_sub, j_sub] * P_intra[i_pat, j_pat]

    return P_total


def plot_matrix(P: np.ndarray, out_path: Path, title_label: str) -> None:
    """Plot 240×240 matrix as heatmap with subband grid (probability scale [0, 1])."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(P, cmap="viridis", aspect="equal", vmin=0.0, vmax=1.0)

    for k in range(1, 10):
        ax.axhline(k * 24 - 0.5, color="white", linewidth=0.5)
        ax.axvline(k * 24 - 0.5, color="white", linewidth=0.5)
    ax.axhline(239.5, color="white", linewidth=0.5)
    ax.axvline(239.5, color="white", linewidth=0.5)

    ax.set_xlabel("Next state (column)")
    ax.set_ylabel("Current state (row)")
    ax.set_title(
        f"Markov Transition Matrix P ({title_label})\n"
        "probability [0, 1]; subband boundaries in white"
    )
    plt.colorbar(im, ax=ax, label="P")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Costas-based 240×240 Markov matrix or plot an existing .npy matrix"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for file naming when building costas matrix (matrix is deterministic)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/markov_matrices",
        help="Directory to save .npy and .png",
    )
    parser.add_argument(
        "--plot-npy",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to existing .npy matrix to plot only (e.g. results/markov_matrices/markov_P_markov_seed42.npy)",
    )
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_npy is not None:
        # Load existing .npy and plot
        npy_path = Path(args.plot_npy)
        if not npy_path.is_absolute():
            npy_path = PROJECT_ROOT / npy_path
        if not npy_path.exists():
            raise FileNotFoundError(f"Matrix file not found: {npy_path}")
        print(f"Loading matrix from {npy_path}...")
        P = np.load(str(npy_path)).astype(np.float64)
        if P.shape != (240, 240):
            print(f"Warning: shape {P.shape} is not (240, 240); subband grid may not align.")
        stem = npy_path.stem
        png_path = out_dir / f"{stem}.png"
        try:
            plot_matrix(P, png_path, title_label=stem)
            print(f"Saved plot: {png_path}")
        except ImportError as e:
            print(f"Skipping plot (matplotlib not available): {e}")
        row_sums = P.sum(axis=1)
        nnz = (P > 1e-10).sum(axis=1)
        print(f"Shape: {P.shape}")
        print(f"Row sums: min={row_sums.min():.6f}, max={row_sums.max():.6f}")
        print(f"Nonzeros/row: min={nnz.min()}, max={nnz.max()}, mean={nnz.mean():.1f}")
        return

    # Build Costas matrix, save .npy and plot
    mode = "costas"
    seed = args.seed

    print("Building Costas-based transition matrix (deterministic)...")
    P = create_radar_transition_matrix()

    # Sanity check
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Rows must sum to 1"
    print(f"  Shape: {P.shape}, row sums OK")

    # Save .npy (same convention as init/print/test markov scripts)
    npy_path = out_dir / f"markov_P_{mode}_seed{seed}.npy"
    np.save(npy_path, P)
    print(f"Saved: {npy_path}")

    # Plot and save .png
    png_path = out_dir / f"markov_P_{mode}_seed{seed}.png"
    try:
        plot_matrix(P, png_path, title_label=f"costas, seed={seed}")
        print(f"Saved plot: {png_path}")
    except ImportError as e:
        print(f"Skipping plot (matplotlib not available): {e}")

    # Summary
    nnz = (P > 1e-10).sum(axis=1)
    print(f"Row sums: min={row_sums.min():.6f}, max={row_sums.max():.6f}")
    print(f"Nonzeros/row: min={nnz.min()}, max={nnz.max()}, mean={nnz.mean():.1f}")
    try:
        rel = npy_path.relative_to(PROJECT_ROOT)
        print(f"Use in config: markov_transition_path: {rel.as_posix()}")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
