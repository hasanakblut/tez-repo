"""
Test Markov Transition Matrix — En yüksek olasılıklı geçiş
==========================================================
Verilen bir .npy Markov geçiş matrisinde her state index'i için,
en yüksek olasılıkla geçilecek bir sonraki state index'ini yazdırır.

Usage:
    python -m scripts.test_markov_transition_matrix --npy results/markov_matrices/markov_P_costas_seed42.npy
    python -m scripts.test_markov_transition_matrix --npy results/markov_matrices/markov_P_markov_seed42.npy
"""

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="For each state, print the next state with highest transition probability"
    )
    parser.add_argument(
        "--npy",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to .npy transition matrix (e.g. results/markov_matrices/markov_P_costas_seed42.npy)",
    )
    args = parser.parse_args()

    npy_path = Path(args.npy)
    if not npy_path.is_absolute():
        npy_path = PROJECT_ROOT / npy_path
    if not npy_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {npy_path}")

    P = np.load(str(npy_path)).astype(np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {P.shape}")

    n = P.shape[0]
    # Her satır için en yüksek olasılıklı sütun (sonraki state)
    next_states = np.argmax(P, axis=1)
    next_probs = np.max(P, axis=1)

    print(f"Matrix: {npy_path.name}")
    print(f"Shape: {P.shape}")
    print(f"State index -> next state (highest prob)\n")
    print("state  next   P(state->next)")
    print("-" * 32)
    for i in range(n):
        j = next_states[i]
        p = next_probs[i]
        print(f"{i:5d}  {j:5d}  {p:.6f}")
    print("-" * 32)
    print(f"Total states: {n}")


if __name__ == "__main__":
    main()
