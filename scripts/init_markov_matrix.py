"""
Init Markov Transition Matrix (seed 42, default for training)
=============================================================
Builds and saves the Markov transition matrix with seed 42 so training
can use a fixed, reproducible P by default via markov_transition_path.

Usage:
    python -m scripts.init_markov_matrix
    python -m scripts.init_markov_matrix --seed 42 --mode markov

Creates: results/markov_matrices/markov_P_<mode>_seed<seed>.npy
Default path used by training: results/markov_matrices/markov_P_markov_seed42.npy
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default path used in configs/default.yaml
DEFAULT_MARKOV_PATH = "results/markov_matrices/markov_P_markov_seed42.npy"


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


def _build_P_once(
    seed: int,
    mode: str,
    config_path: str,
) -> np.ndarray:
    """Build transition matrix once from seed (in-code, no file load). Returns P."""
    from src.env_utils import FrequencyGenerator

    cfg = load_config(config_path)
    env_cfg = build_env_config(cfg)
    env_cfg["generator_mode"] = mode
    env_cfg.pop("markov_transition_path", None)

    rng = np.random.default_rng(seed)
    gen = FrequencyGenerator(
        config={"radar": env_cfg},
        state_dim=240,
        rng=rng,
    )
    P = gen.get_transition_matrix()
    if P is None:
        raise RuntimeError(f"Mode {mode!r} does not produce a transition matrix. Use markov or markov_subband.")
    return P


def ensure_markov_matrix(
    seed: int = 42,
    mode: str = "markov",
    config_path: str = "configs/default.yaml",
    out_dir: str = "results/markov_matrices",
) -> Path:
    """Build matrix twice from seed 42, save both, load and verify identical. Return canonical .npy path."""
    out_path = PROJECT_ROOT / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    canonical_name = f"markov_P_{mode}_seed{seed}.npy"
    run2_name = f"markov_P_{mode}_seed{seed}_run2.npy"
    canonical_path = out_path / canonical_name
    run2_path = out_path / run2_name

    # Build twice from same seed (in-code, no file load)
    print("Build 1 (seed=%d, mode=%s)..." % (seed, mode))
    P1 = _build_P_once(seed, mode, config_path)
    np.save(canonical_path, P1)
    print("  Saved: %s" % canonical_path)

    print("Build 2 (seed=%d, mode=%s)..." % (seed, mode))
    P2 = _build_P_once(seed, mode, config_path)
    np.save(run2_path, P2)
    print("  Saved: %s" % run2_path)

    # Load both and check identical
    print("Reproducibility check: load both and compare...")
    Q1 = np.load(canonical_path)
    Q2 = np.load(run2_path)
    if not np.allclose(Q1, Q2):
        run2_path.unlink(missing_ok=True)
        raise RuntimeError("Reproducibility FAILED: two builds from seed %d differ." % seed)
    print("  PASS: two builds from seed %d are identical." % seed)
    print("  Deleting %s in 5 seconds..." % run2_path)
    time.sleep(5)
    run2_path.unlink(missing_ok=True)
    return canonical_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create default Markov transition matrix (seed 42) for training"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="markov", choices=["markov", "markov_subband"])
    parser.add_argument("--out-dir", type=str, default="results/markov_matrices")
    args = parser.parse_args()

    npy_path = ensure_markov_matrix(
        seed=args.seed,
        mode=args.mode,
        config_path=args.config,
        out_dir=args.out_dir,
    )
    # Relative path for use in YAML
    try:
        rel = npy_path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = npy_path
    print(f"Saved: {npy_path}")
    print(f"Use in config: markov_transition_path: {rel.as_posix()}")


if __name__ == "__main__":
    main()
