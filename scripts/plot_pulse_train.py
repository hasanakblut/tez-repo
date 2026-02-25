"""
Plot Pulse Train from FrequencyGenerator
========================================
Given a seed and initial index, run the generator for a configurable number
of steps to produce a pulse train (time series of state indices), then plot it.
By default uses the env's frequency generator (RadarEnv) so the next frequency
is determined the same way as in training (reset + step). Use --no-use-env to
generate with a standalone FrequencyGenerator instead.
Supports using pre-built Markov matrices from results/*.npy (e.g. markov_subband).
When using markov_P_deterministic_cycle_*.npy, pulse train is drawn with lines
connecting each transition (frequency → next frequency).

Usage:
    python -m scripts.plot_pulse_train --seed 42 --start-index 0 --num-pulses 500
    python -m scripts.plot_pulse_train --seed 42 --mode markov --markov-npy results/markov_matrices/markov_P_deterministic_cycle_seed42.npy --num-pulses 240
    python -m scripts.plot_pulse_train --no-use-env --seed 42  # use standalone generator
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


def build_env_config(cfg: dict) -> dict:
    """Build full env config (physics + radar + episode + environment) for RadarEnv."""
    env_cfg = {}
    env_cfg.update(cfg.get("physics", {}))
    env_cfg.update(cfg.get("radar", {}))
    env_cfg.update(cfg.get("episode", {}))
    env_cfg.update(cfg.get("environment", {}))
    return env_cfg


def generate_pulse_train(gen, start_index: int, num_pulses: int) -> np.ndarray:
    """Run generator to produce a pulse train of length num_pulses (indices)."""
    train = np.zeros(num_pulses, dtype=np.int32)
    train[0] = start_index
    prev = start_index
    for t in range(1, num_pulses):
        prev = gen.next(prev)
        train[t] = prev
    return train


def generate_pulse_train_from_env(env, start_index: int, num_pulses: int, seed: int) -> np.ndarray:
    """Produce pulse train using the env's frequency generator (reset + step).
    Same probabilistic next-frequency logic as in training."""
    obs, _ = env.reset(seed=seed, options={"start_index": start_index})
    train = np.zeros(num_pulses, dtype=np.int32)
    train[0] = int(obs)
    for t in range(1, num_pulses):
        next_obs, _, terminated, _, _ = env.step(0)
        train[t] = int(next_obs)
        if terminated:
            # Env ended (max_pulses); fill rest with last state if needed
            train[t:] = train[t]
            break
    return train


def plot_pulse_train(
    pulse_train: np.ndarray,
    out_path: Path,
    title: str | None = None,
    color_by_subband: bool = True,
    connect_lines: bool = False,
) -> None:
    """Plot pulse train as time series (t vs index). Optionally color by subband.
    When connect_lines=True (e.g. deterministic_cycle), consecutive states are
    connected with line segments so each frequency→next frequency transition is visible."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    num_pulses = len(pulse_train)
    PERMS_PER_SUBBAND = 24
    subband = pulse_train // PERMS_PER_SUBBAND  # 0..9

    fig, ax = plt.subplots(figsize=(12, 4))
    t = np.arange(num_pulses)

    if connect_lines:
        # Çizgilerle bağla: (t[i], state[i]) -> (t[i+1], state[i+1]) her i için
        segments = np.stack([np.column_stack([t[:-1], pulse_train[:-1]]),
                             np.column_stack([t[1:], pulse_train[1:]])], axis=1)
        lc = LineCollection(segments, linewidths=0.8, alpha=0.9)
        if color_by_subband:
            # Her segmenti başlangıç state'inin subband'ına göre renklendir
            lc.set_array(subband[:-1].astype(float))
            lc.set_cmap(plt.cm.tab10)
            lc.set_clim(-0.5, 9.5)
            ax.add_collection(lc)
            fig.colorbar(lc, ax=ax, label="Subband", ticks=range(10))
        else:
            lc.set_color("steelblue")
            ax.add_collection(lc)
        ax.set_xlim(t[0], t[-1])
        ax.autoscale_view(scalex=False, scaley=True)
    elif color_by_subband:
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
    parser.add_argument(
        "--no-use-env",
        action="store_true",
        help="Use standalone FrequencyGenerator instead of env's generator (default: use RadarEnv)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    num_pulses = args.num_pulses if args.num_pulses is not None else cfg.get("episode", {}).get("max_pulses", 1000)
    num_pulses = min(num_pulses, 50_000)  # cap for quick tests

    radar_cfg = build_radar_config(cfg, args.mode, args.start_index, args.markov_npy)

    if args.no_use_env:
        # Standalone FrequencyGenerator (önceki davranış)
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
    else:
        # Env içindeki frequency generator ile (reset + step ile aynı olasılıksal geçiş)
        env_cfg = build_env_config(cfg)
        env_cfg.update(radar_cfg)
        from src.env import RadarEnv
        env = RadarEnv(config=env_cfg)
        pulse_train = generate_pulse_train_from_env(
            env, args.start_index, num_pulses, args.seed
        )

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

    # Deterministic cycle matrisi kullanılıyorsa geçişleri çizgilerle göster
    use_connect_lines = bool(
        args.markov_npy and "deterministic_cycle" in Path(args.markov_npy).stem
    )

    plot_path = out_dir / f"{base}.png"
    plot_pulse_train(
        pulse_train,
        plot_path,
        title=title,
        color_by_subband=not args.no_subband_color,
        connect_lines=use_connect_lines,
    )
    print(f"Plot saved: {plot_path}")

    if args.save_train:
        np.save(out_dir / f"{base}.npy", pulse_train)
        np.savetxt(out_dir / f"{base}.csv", pulse_train, fmt="%d", delimiter="\n")
        print(f"Pulse train saved: {out_dir / f'{base}.npy'}, {base}.csv")


if __name__ == "__main__":
    main()
