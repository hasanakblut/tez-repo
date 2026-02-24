"""
Test: Pulse Train Reproducibility and Sensitivity (Markov, Pre-built P)
======================================================================
1. Same seed + same start_index → two runs must yield identical series (reproducibility).
2. Different seed, same start_index → two runs must yield different series.
3. Same seed, different start_index → two runs must yield different series.

Usage:
    python -m scripts.test_pulse_train_randomness_in_markov
    python -m scripts.test_pulse_train_randomness_in_markov --num-pulses 300
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_radar_config_for_markov(cfg: dict, start_index: int, markov_npy: str) -> dict:
    radar = dict(cfg.get("radar", {}))
    radar["generator_mode"] = "markov"
    radar["start_index"] = start_index
    radar["markov_transition_path"] = markov_npy
    return radar


def generate_pulse_train(gen, start_index: int, num_pulses: int) -> np.ndarray:
    train = np.zeros(num_pulses, dtype=np.int32)
    train[0] = start_index
    prev = start_index
    for t in range(1, num_pulses):
        prev = gen.next(prev)
        train[t] = prev
    return train


def run_one(cfg: dict, markov_npy: str, seed: int, start_index: int, num_pulses: int):
    from src.env_utils import FrequencyGenerator

    radar_cfg = build_radar_config_for_markov(cfg, start_index, markov_npy)
    rng = np.random.default_rng(seed)
    gen = FrequencyGenerator(config={"radar": radar_cfg}, state_dim=240, rng=rng)
    gen.reset(seed=seed)
    return generate_pulse_train(gen, start_index, num_pulses)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test pulse train reproducibility and sensitivity to seed/start_index (Markov P)"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--markov-npy",
        type=str,
        default="results/markov_matrices/markov_P_markov_subband_seed101.npy",
    )
    parser.add_argument("--seed", type=int, default=42, help="Primary RNG seed")
    parser.add_argument("--seed2", type=int, default=43, help="Alternate seed for 'different seed' test")
    parser.add_argument("--start-index", type=int, default=0, help="Primary start index (0–239)")
    parser.add_argument(
        "--start-index2",
        type=int,
        default=24,
        help="Alternate start index for 'different start' test",
    )
    parser.add_argument("--num-pulses", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config(args.config)
    npy_name = Path(args.markov_npy).name

    print("=" * 60)
    print("Pulse train tests (Markov, pre-built P)")
    print("=" * 60)
    print(f"  P matrix: {npy_name}  |  num_pulses: {args.num_pulses}")
    print()

    all_ok = True

    # --- Test 1: same seed, same start_index → identical ---
    train_a = run_one(cfg, args.markov_npy, args.seed, args.start_index, args.num_pulses)
    train_b = run_one(cfg, args.markov_npy, args.seed, args.start_index, args.num_pulses)
    ok1 = np.array_equal(train_a, train_b)
    all_ok = all_ok and ok1
    print("  [1] Same seed, same start_index → identical series:")
    print(f"      {'PASS' if ok1 else 'FAIL'} (seed={args.seed}, start={args.start_index})")
    if not ok1:
        d = np.where(train_a != train_b)[0]
        print(f"      First difference at t={d[0]}")
    print()

    # --- Test 2: different seed, same start_index → different ---
    train_s1 = run_one(cfg, args.markov_npy, args.seed, args.start_index, args.num_pulses)
    train_s2 = run_one(cfg, args.markov_npy, args.seed2, args.start_index, args.num_pulses)
    ok2 = not np.array_equal(train_s1, train_s2)
    all_ok = all_ok and ok2
    print("  [2] Different seed, same start_index → different series:")
    print(f"      {'PASS' if ok2 else 'FAIL'} (seed {args.seed} vs {args.seed2}, start={args.start_index})")
    if not ok2:
        print("      Series were identical (unexpected).")
    else:
        d = np.where(train_s1 != train_s2)[0]
        print(f"      First difference at t={d[0]}")
    print()

    # --- Test 3: same seed, different start_index → different ---
    train_i1 = run_one(cfg, args.markov_npy, args.seed, args.start_index, args.num_pulses)
    train_i2 = run_one(cfg, args.markov_npy, args.seed, args.start_index2, args.num_pulses)
    ok3 = not np.array_equal(train_i1, train_i2)
    all_ok = all_ok and ok3
    print("  [3] Same seed, different start_index → different series:")
    print(f"      {'PASS' if ok3 else 'FAIL'} (seed={args.seed}, start {args.start_index} vs {args.start_index2})")
    if not ok3:
        print("      Series were identical (unexpected).")
    else:
        d = np.where(train_i1 != train_i2)[0]
        print(f"      First difference at t={d[0]}")
    print()

    print("  First 10 indices (Test 1 run):", train_a[:10].tolist())
    print("=" * 60)
    print("  Overall:", "PASS (all 3 checks)" if all_ok else "FAIL (see above)")
    print("=" * 60)

    exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
