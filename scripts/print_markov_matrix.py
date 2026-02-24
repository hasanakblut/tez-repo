"""
Print Markov Transition Matrix
==============================
Build the transition matrix P for a given seed and generator mode,
then save and print it for inspection.

Usage:
    python -m scripts.print_markov_matrix [--seed 42] [--mode markov_subband]
    python -m scripts.print_markov_matrix --seed 42 --mode markov_subband
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and print Markov transition matrix"
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
        help="Generator mode (must build a transition matrix)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/markov_matrices",
        help="Directory to save matrix files",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = build_env_config(cfg)
    env_cfg["generator_mode"] = args.mode

    from src.env_utils import (
        FrequencyGenerator,
        NUM_SUBBANDS,
        PERMS_PER_SUBBAND,
        _subband_of,
    )

    rng = np.random.default_rng(args.seed)
    gen = FrequencyGenerator(
        config={"radar": env_cfg},
        state_dim=240,
        rng=rng,
    )

    P = gen.get_transition_matrix()
    if P is None:
        print("ERROR: Generator mode does not produce a transition matrix.")
        print("Use --mode markov or --mode markov_subband")
        return

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save as .npy
    npy_path = out_dir / f"markov_P_{args.mode}_seed{args.seed}.npy"
    np.save(npy_path, P)
    print(f"Saved: {npy_path}")

    # Save as CSV (for spreadsheet/viewer)
    csv_path = out_dir / f"markov_P_{args.mode}_seed{args.seed}.csv"
    np.savetxt(csv_path, P, fmt="%.6e", delimiter=",")
    print(f"Saved: {csv_path}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print(f"Transition Matrix P ({args.mode}, seed={args.seed})")
    print("=" * 60)
    print(f"Shape: {P.shape}")
    print(f"Row sums (min, max, mean): {P.sum(axis=1).min():.6f}, "
          f"{P.sum(axis=1).max():.6f}, {P.sum(axis=1).mean():.6f}")

    # Per-subband stay probability check (for markov_subband)
    if args.mode == "markov_subband":
        print("\nSubband stay probability (row i in subband k -> sum P[i,j] for j in k):")
        for k in range(NUM_SUBBANDS):
            rows = list(range(k * PERMS_PER_SUBBAND, (k + 1) * PERMS_PER_SUBBAND))
            cols_same = list(rows)
            block = P[np.ix_(rows, cols_same)]
            stay = block.sum(axis=1).mean()
            print(f"  Subband {k}: mean stay = {stay:.4f}")

    # Print a block: first 24 rows (subband 0), first 72 columns (subbands 0-2)
    print("\nBlock P[0:24, 0:72] (rows=subband 0, cols=subbands 0-2):")
    block = P[:24, :72]
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print(block)

    # Sparse structure: nnz per row
    nnz = (P > 1e-10).sum(axis=1)
    print(f"\nNonzeros per row: min={nnz.min()}, max={nnz.max()}, mean={nnz.mean():.1f}")


if __name__ == "__main__":
    main()
