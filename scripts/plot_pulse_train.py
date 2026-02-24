"""
Plot Pulse Train from FrequencyGenerator
========================================
Given a seed and initial index, run the generator for a configurable number
of steps to produce a pulse train (time series of state indices), then plot it.
Supports using pre-built Markov matrices from results/*.npy (e.g. markov_subband).

Usage:
    python -m scripts.plot_pulse_train --seed 42 --start-index 0 --num-pulses 500
    python -m scripts.plot_pulse_train --seed 101 --start-index 24 --mode markov --markov-npy results/markov_P_markov_subband_seed101.npy --num-pulses 1000
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

# Project root (repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_radar_config(cfg: dict, mode: str, start_index: int, markov_npy: str | None) -> dict:
    """Build radar config for FrequencyGenerator; optionally use pre-built P from .npy."""
    radar = dict(cfg.get("radar", {}))
    radar["generator_mode"] = mode
    radar["start_index"] = start_index
    if markov_npy:
        radar["generator_mode"] = "markov"
        radar["markov_transition_path"] = markov_npy
    return radar


def generate_pulse_train(gen, start_index: int, num_pulses: int) -> np.ndarray:
    """Run generator to produce a pulse train of length num_pulses (indices)."""
    train = np.zeros(num_pulses, dtype=np.int32)
    train[0] = start_index
    prev = start_index
    for t in range(1, num_pulses):
        prev = gen.next(prev)
        train[t] = prev
    return train


def plot_pulse_train(
    pulse_train: np.ndarray,
    out_path: Path,
    title: str | None = None,
    color_by_subband: bool = True,
) -> None:
    """Plot pulse train as time series (t vs index). Optionally color by subband."""
    import matplotlib.pyplot as plt

    num_pulses = len(pulse_train)
    PERMS_PER_SUBBAND = 24
    subband = pulse_train // PERMS_PER_SUBBAND  # 0..9

    fig, ax = plt.subplots(figsize=(12, 4))
    t = np.arange(num_pulses)

    if color_by_subband:
        for k in range(10):
            mask = subband == k
            if not np.any(mask):
                continue
            ax.scatter(t[mask], pulse_train[mask], s=4, alpha=0.7, label=f"SB{k}")
        ax.legend(loc="upper right", ncol=2, fontsize=8)
    else:
        ax.plot(t, pulse_train, linewidth=0.5, alpha=0.9)

    ax.set_xlabel("Pulse index (time)")
    ax.set_ylabel("State index (0–239)")
    ax.set_ylim(-5, 244)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pulse train from FrequencyGenerator and plot as time series"
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
        help="RNG seed for reproducible sequence",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Initial radar state index (0–239)",
    )
    parser.add_argument(
        "--num-pulses",
        type=int,
        default=None,
        help="Number of pulses to generate (default: from config episode.max_pulses)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="uniform",
        choices=["uniform", "periodic", "lcg", "markov", "markov_subband"],
        help="Generator mode",
    )
    parser.add_argument(
        "--markov-npy",
        type=str,
        default=None,
        help="Path to pre-built P matrix .npy (e.g. results/markov_matrices/markov_P_markov_subband_seed101.npy). Uses mode=markov.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/pulse_train_plots",
        help="Directory to save plot and optional pulse train data",
    )
    parser.add_argument(
        "--save-train",
        action="store_true",
        help="Save pulse train as .npy and .csv",
    )
    parser.add_argument(
        "--no-subband-color",
        action="store_true",
        help="Plot single line instead of coloring by subband",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    num_pulses = args.num_pulses if args.num_pulses is not None else cfg.get("episode", {}).get("max_pulses", 1000)
    num_pulses = min(num_pulses, 50_000)  # cap for quick tests

    radar_cfg = build_radar_config(cfg, args.mode, args.start_index, args.markov_npy)
    config_for_gen = {"radar": radar_cfg}

    from src.env_utils import FrequencyGenerator

    rng = np.random.default_rng(args.seed)
    gen = FrequencyGenerator(
        config=config_for_gen,
        state_dim=240,
        rng=rng,
    )
    gen.reset(seed=args.seed)

    pulse_train = generate_pulse_train(gen, args.start_index, num_pulses)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output filename base
    npy_label = ""
    if args.markov_npy:
        npy_label = "_" + Path(args.markov_npy).stem
    base = f"pulse_train_{args.mode}{npy_label}_seed{args.seed}_start{args.start_index}_n{num_pulses}"

    title = f"Pulse train: mode={args.mode}, seed={args.seed}, start={args.start_index}, N={num_pulses}"
    if args.markov_npy:
        title += f" (P from {Path(args.markov_npy).name})"

    plot_path = out_dir / f"{base}.png"
    plot_pulse_train(
        pulse_train,
        plot_path,
        title=title,
        color_by_subband=not args.no_subband_color,
    )
    print(f"Plot saved: {plot_path}")

    if args.save_train:
        np.save(out_dir / f"{base}.npy", pulse_train)
        np.savetxt(out_dir / f"{base}.csv", pulse_train, fmt="%d", delimiter="\n")
        print(f"Pulse train saved: {out_dir / f'{base}.npy'}, {base}.csv")


if __name__ == "__main__":
    main()
