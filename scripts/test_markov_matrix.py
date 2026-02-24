"""
Test & Plot Markov Transition Matrix
====================================
Builds the transition matrix with a given seed, runs basic tests,
and plots the full 240×240 matrix as a heatmap for visual inspection.

Usage:
    python -m scripts.test_markov_matrix [--seed 42] [--mode markov_subband]
    python -m scripts.test_markov_matrix --seed 42 --mode markov_subband --no-plot
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env_config(cfg: dict) -> dict:
    env_cfg = {}
    env_cfg.update(cfg.get("physics", {}))
    env_cfg.update(cfg.get("radar", {}))
    env_cfg.update(cfg.get("episode", {}))
    env_cfg.update(cfg.get("environment", {}))
    return env_cfg


def run_tests(P: np.ndarray, mode: str) -> None:
    """Run basic sanity tests on the transition matrix."""
    from src.env_utils import NUM_SUBBANDS, PERMS_PER_SUBBAND

    print("\n--- Tests ---")

    # 1. Row stochastic
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Rows must sum to 1"
    print("  [PASS] Row stochastic (all rows sum to 1)")

    # 2. No absorbing states (all rows have multiple non-tiny entries)
    nnz = (P > 1e-6).sum(axis=1)
    assert np.all(nnz >= 2), "Each row should have at least 2 non-tiny entries"
    print("  [PASS] No absorbing states")

    # 3. Subband stay probability (markov_subband only)
    if mode == "markov_subband":
        for k in range(NUM_SUBBANDS):
            rows = list(range(k * PERMS_PER_SUBBAND, (k + 1) * PERMS_PER_SUBBAND))
            cols_same = list(rows)
            block = P[np.ix_(rows, cols_same)]
            stay = block.sum(axis=1)
            assert np.all(stay > 0.5), f"Subband {k} stay should be ~0.7"
        print("  [PASS] Subband stay probability ~70%")


def plot_matrix(P: np.ndarray, out_path: Path, mode: str, seed: int) -> None:
    """Plot full 240×240 matrix as heatmap with subband grid."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))

    # Use log scale for contrast (many near-zero values)
    P_plot = np.where(P > 1e-10, P, 1e-10)
    im = ax.imshow(np.log10(P_plot + 1e-12), cmap="viridis", aspect="equal")

    # Subband grid lines
    for k in range(1, 10):
        ax.axhline(k * 24 - 0.5, color="white", linewidth=0.5)
        ax.axvline(k * 24 - 0.5, color="white", linewidth=0.5)
    ax.axhline(239.5, color="white", linewidth=0.5)
    ax.axvline(239.5, color="white", linewidth=0.5)

    ax.set_xlabel("Next state (column)")
    ax.set_ylabel("Current state (row)")
    ax.set_title(f"Markov Transition Matrix P ({mode}, seed={seed})\n"
                 "log10(probability); subband boundaries in white")
    plt.colorbar(im, ax=ax, label="log10(P)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test and plot Markov transition matrix"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible matrix",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="markov_subband",
        choices=["markov", "markov_subband"],
        help="Generator mode",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/markov_matrices",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (run tests only)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = build_env_config(cfg)
    env_cfg["generator_mode"] = args.mode

    from src.env_utils import FrequencyGenerator, NUM_SUBBANDS, PERMS_PER_SUBBAND

    rng = np.random.default_rng(args.seed)
    gen = FrequencyGenerator(
        config={"radar": env_cfg},
        state_dim=240,
        rng=rng,
    )

    P = gen.get_transition_matrix()
    if P is None:
        print("ERROR: Generator mode does not produce a transition matrix.")
        return

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save
    npy_path = out_dir / f"markov_P_{args.mode}_seed{args.seed}.npy"
    np.save(npy_path, P)
    print(f"Saved: {npy_path}")

    csv_path = out_dir / f"markov_P_{args.mode}_seed{args.seed}.csv"
    np.savetxt(csv_path, P, fmt="%.6e", delimiter=",")
    print(f"Saved: {csv_path}")

    # Run tests
    run_tests(P, args.mode)

    # Plot
    if not args.no_plot:
        try:
            plot_path = out_dir / f"markov_P_{args.mode}_seed{args.seed}.png"
            plot_matrix(P, plot_path, args.mode, args.seed)
        except ImportError as e:
            print(f"Skipping plot (matplotlib not available): {e}")

    # Summary stats
    print("\n--- Summary ---")
    print(f"Shape: {P.shape}")
    print(f"Row sums: min={P.sum(axis=1).min():.6f}, max={P.sum(axis=1).max():.6f}")
    nnz = (P > 1e-10).sum(axis=1)
    print(f"Nonzeros/row: min={nnz.min()}, max={nnz.max()}, mean={nnz.mean():.1f}")


if __name__ == "__main__":
    main()
