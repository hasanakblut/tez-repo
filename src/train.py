"""
Training Loop — GA-Dueling DQN Cognitive Jamming
=================================================
Main entry point for training the JammingAgent against the RadarEnv.

Paper: Xia et al., "GA-Dueling DQN Jamming Decision-Making Method
       for Intra-Pulse Frequency Agile Radar", Sensors 2024.

Usage:
    python -m src.train                          # default config
    python -m src.train --config path/to.yaml    # custom config
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.agent import JammingAgent
from src.env import RadarEnv
from src.env_utils import FrequencyGenerator

# Project root (so config/results work regardless of cwd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_id(cfg: dict, config_path: str) -> str:
    """Run identifier: datetime + config stem + episodes (+ seed if set) for log/plot/checkpoint naming."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    config_stem = Path(config_path).stem
    num_ep = cfg["episode"]["num_episodes"]
    seed = cfg.get("seed")
    parts = [ts, config_stem, f"ep{num_ep}"]
    if seed is not None:
        parts.insert(2, f"s{seed}")
    return "_".join(parts)


def _regression_slope(values: list) -> float:
    """OLS slope β of values vs index. β>0 → improving, β≈0 → plateau."""
    n = len(values)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = np.array(values, dtype=np.float64)
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom < 1e-12:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def _rsd(values: list) -> float:
    """Relative Standard Deviation (%) = 100 * std / |mean|. Low → stable policy."""
    if len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    mean = arr.mean()
    if abs(mean) < 1e-9:
        return 0.0
    return float(100.0 * arr.std() / abs(mean))


def _trend_symbol(slope: float, rsd: float) -> str:
    """↑ improving, ↓ degrading, → stable, ↗ noisy-up, ↘ noisy-down."""
    if abs(slope) < 1e-6:
        return "→"
    if rsd > 30:
        return "↗" if slope > 0 else "↘"
    return "↑" if slope > 0 else "↓"


def _episode_report(all_metrics: list, ep_metrics: dict, max_pulses: int,
                    window: int = 10) -> str:
    """12.md formatında bölüm sonu raporu: Performance + Discovery + Stability."""
    ep = ep_metrics["episode"]
    ret = ep_metrics["total_reward"]
    hit = ep_metrics["hit_rate"]
    avg_match = ep_metrics.get("avg_match", 0.0)
    subband = ep_metrics.get("subband_rate", 0.0)
    perm_rate = ep_metrics.get("perm_rate", 0.0)
    td_loss = ep_metrics["avg_loss"]
    eps = ep_metrics["epsilon"]
    entropy = ep_metrics.get("entropy", 0.0)
    eff = ret / max(ep_metrics["env_steps"], 1)
    n_ep = all_metrics[-1]["episode"] if all_metrics else ep

    recent = all_metrics[-window:]
    rets = [m["total_reward"] for m in recent]
    hits = [m["hit_rate"] for m in recent]
    ma_ret = sum(rets) / len(rets)
    ma_hit = sum(hits) / len(hits)
    slope = _regression_slope(rets)
    rsd = _rsd(rets)
    trend = _trend_symbol(slope, rsd)

    lines = [
        f"─── [Episode {ep}] ──────────────────────────────────────────",
        f"  Performance:",
        f"    Return: {ret:,.0f} (MA{window}: {ma_ret:,.0f} | Slope: {slope:+.2f}) {trend}",
        f"    Hit Rate (4/4): {hit:.4f} | Avg Match (Num): {avg_match:.2f}/4.0",
        f"    Efficiency: {eff:.3f} reward/pulse",
        f"    Stability: RSD {rsd:.1f}%",
        f"  Discovery:",
        f"    Subband Detection: {subband:.2%} | Intra-Pulse Precision: {perm_rate:.2%}",
        f"    Policy Entropy: {entropy:.2f} bits | ε: {eps:.4f}",
        f"    TD Loss: {td_loss:.4f}",
        f"─────────────────────────────────────────────────────────────",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env_config(cfg: dict) -> dict:
    """Flatten nested YAML sections into a single dict for RadarEnv."""
    env_cfg: dict = {}
    env_cfg.update(cfg.get("physics", {}))
    env_cfg.update(cfg.get("radar", {}))
    env_cfg.update(cfg.get("episode", {}))
    env_cfg.update(cfg.get("environment", {}))
    return env_cfg


def _save_training_plots(
    all_metrics: list,
    results_dir: Path,
    config_path: str,
    num_episodes: int,
    max_pulses: int,
    plot_dir: Path | None = None,
) -> Path:
    """Save reward, hit_rate, epsilon (and loss) curves. If plot_dir is given, use it (incremental); else create run_YYYYMMDD_HHMMSS_<config_stem>."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if plot_dir is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        config_stem = Path(config_path).stem
        plot_dir = results_dir / "training_plots" / f"run_{run_id}_{config_stem}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    episodes = [m["episode"] for m in all_metrics]
    rewards = [m["total_reward"] for m in all_metrics]
    hit_rates = [m["hit_rate"] for m in all_metrics]
    epsilons = [m["epsilon"] for m in all_metrics]
    losses = [m["avg_loss"] for m in all_metrics]

    # Reward curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, rewards, color="C0")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(f"Training reward ({num_episodes} ep × {max_pulses:,} pulses)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "reward_curve.png", dpi=150)
    plt.close(fig)

    # Hit rate curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, hit_rates, color="C1")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Hit rate")
    ax.set_title("Jamming success (hit rate) per episode")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "hit_rate_curve.png", dpi=150)
    plt.close(fig)

    # Epsilon decay
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, epsilons, color="C2")
    ax.set_xlabel("Episode")
    ax.set_ylabel("ε")
    ax.set_title("Epsilon (exploration rate) decay")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "epsilon_curve.png", dpi=150)
    plt.close(fig)

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, losses, color="C3", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg loss")
    ax.set_title("Average TD loss per episode")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # Subband detection rate
    if "subband_rate" in all_metrics[0]:
        sub_rates = [m["subband_rate"] for m in all_metrics]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(episodes, sub_rates, color="C4")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Subband detection rate")
        ax.set_title("Subband detection (Phase 1) per episode")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "subband_rate_curve.png", dpi=150)
        plt.close(fig)

    # Avg match (Num / 4)
    if "avg_match" in all_metrics[0]:
        avg_matches = [m["avg_match"] for m in all_metrics]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(episodes, avg_matches, color="C5")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg sub-pulse matches (Num)")
        ax.set_title("Average Num per pulse (max 4)")
        ax.set_ylim(0, 4.1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "avg_match_curve.png", dpi=150)
        plt.close(fig)

    # Policy entropy
    if "entropy" in all_metrics[0]:
        entropies = [m["entropy"] for m in all_metrics]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(episodes, entropies, color="C6")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Policy entropy (bits)")
        ax.set_title("Policy entropy — low = confident, high = uncertain")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "entropy_curve.png", dpi=150)
        plt.close(fig)

    # Efficiency (reward/pulse)
    if "efficiency" in all_metrics[0]:
        effs = [m["efficiency"] for m in all_metrics]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(episodes, effs, color="C7")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward / pulse")
        ax.set_title("Sample efficiency per episode")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "efficiency_curve.png", dpi=150)
        plt.close(fig)

    return plot_dir


def precompute_first_100_indices(env_cfg: dict, seed: int | None, state_dim: int = 240) -> list[int]:
    """Pre-compute the first 100 radar indices for this episode (same config + seed as env).
    Used so logs can show 'first 100 indices for this trial' for later verification."""
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    gen = FrequencyGenerator(config={"radar": env_cfg}, state_dim=state_dim, rng=rng)
    gen.reset(seed=seed)
    indices = []
    prev = gen.next(None)
    indices.append(prev)
    for _ in range(99):
        prev = gen.next(prev)
        indices.append(prev)
    return indices


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config_path: str = "configs/default.yaml") -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Environment ------------------------------------------------------
    env_cfg = build_env_config(cfg)
    env = RadarEnv(config=env_cfg)
    gen_mode = env_cfg.get("generator_mode", "uniform")
    print(f"Radar generator_mode: {gen_mode} (config: radar.generator_mode; 'markov' için markov_transition_path veya markov_subband_params kullanılır)")

    # --- Agent ------------------------------------------------------------
    agent = JammingAgent(config=cfg)
    print(f"Policy net parameters: "
          f"{sum(p.numel() for p in agent.policy_net.parameters()):,}")

    # --- Logging setup: one run dir per training (datetime + config + ep + seed) ---
    results_dir = PROJECT_ROOT / cfg["logging"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    num_episodes = cfg["episode"]["num_episodes"]
    max_pulses = cfg["episode"]["max_pulses"]
    target_update_freq = cfg["training"]["target_update_freq"]
    log_interval = cfg["logging"]["log_interval"]
    save_interval = cfg["logging"]["save_interval"]
    plot_interval = cfg["logging"].get("plot_interval", save_interval)

    run_id = _run_id(cfg, config_path)
    run_dir = results_dir / "runs" / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.log"
    metrics_path = run_dir / "metrics.jsonl"
    plot_dir = run_dir / "training_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list = []
    ts_iso = time.strftime("%Y-%m-%d %H:%M:%S")
    run_header = (
        f"=== RUN {ts_iso} ===\n"
        f"  config: {config_path}\n"
        f"  num_episodes: {num_episodes}  max_pulses: {max_pulses}\n"
        f"  seed: {cfg.get('seed')}\n"
        f"  run_dir: {run_dir}\n"
        f"===\n"
    )
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(run_header)

    # --- Training loop (Paper Algorithm 1) --------------------------------
    print(f"\nStarting training: {num_episodes} episodes × {max_pulses:,} pulses")
    print(f"Run dir: {run_dir}")
    print(f"Log every {log_interval} ep | Save/plot every {save_interval}/{plot_interval} ep\n")

    global_seed = cfg.get("seed")
    # Dış bar: bölüm ilerlemesi + toplam ETA
    pbar_ep = tqdm(
        range(1, num_episodes + 1),
        unit="ep",
        desc="Train",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for episode in pbar_ep:
        ep_start = time.time()
        reset_seed = (global_seed + episode) if global_seed is not None else None

        # Pre-compute first 100 indices for this trial (same sequence as env will produce)
        first_100 = precompute_first_100_indices(env_cfg, reset_seed, state_dim=240)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"Episode {episode} seed={reset_seed} first_100_indices={first_100}\n")
        if episode == 1 or episode % log_interval == 0:
            tqdm.write(f"  [ep {episode}] seed={reset_seed} first_100={first_100[:12]}... (len=100)")

        obs, info = env.reset(seed=reset_seed)
        agent.reset_hidden()

        episode_reward = 0.0
        episode_loss = 0.0
        loss_count = 0
        env_steps = 0
        ep_entropy_sum = 0.0
        last_obs_index: int | None = None
        last_action: int | None = None

        # İç bar: pulse ilerlemesi + biriken metrikler + bölüm ETA
        pbar_pulse = tqdm(
            range(1, max_pulses + 1),
            unit="pulse",
            desc=f"ep {episode}/{num_episodes}",
            leave=False,
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        for pulse in pbar_pulse:
            state_history = env.get_history()

            action, entropy = agent.select_action_with_info(state_history)
            ep_entropy_sum += entropy

            next_obs, reward, terminated, truncated, info = env.step(action)
            env_steps += 1
            last_obs_index = int(next_obs)
            last_action = int(action)

            next_state_history = env.get_history()
            agent.store_transition(
                state_history, action, reward,
                next_state_history, terminated)

            loss = agent.learn()
            if loss is not None:
                episode_loss += loss
                loss_count += 1

            episode_reward += reward

            # Pulse bar postfix: ret, hit_rate, subband, avg_match, eff (her 50 pulse)
            if pulse % 50 == 0 or pulse == 1 or pulse == max_pulses:
                eff = episode_reward / env_steps
                pbar_pulse.set_postfix(
                    ret=f"{episode_reward:.0f}",
                    hit=f"{info['hit_rate']:.3f}",
                    sub=f"{info['subband_rate']:.2f}",
                    num=f"{info['avg_match']:.1f}",
                    eff=f"{eff:.2f}",
                    refresh=False,
                )
                pbar_pulse.refresh()

            # Terminale granular satır (her %5 = her 500 pulse)
            if pulse % 500 == 0 or pulse == 1 or pulse == max_pulses:
                pct = 100.0 * pulse / max_pulses
                eff = episode_reward / env_steps
                avg_ent = ep_entropy_sum / env_steps
                tqdm.write(
                    f"    [{pct:5.1f}%] pulse {pulse:>5}/{max_pulses} │ "
                    f"ret={episode_reward:>8.0f} │ hit={info['hit_rate']:.4f} │ "
                    f"sub={info['subband_rate']:.3f} │ Num={info['avg_match']:.2f}/4 │ "
                    f"eff={eff:.3f} │ H={avg_ent:.2f}b"
                )

            if terminated:
                break

        pbar_pulse.close()

        # --- End of episode bookkeeping -----------------------------------

        # Decay exploration rate
        agent.decay_epsilon()

        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()

        # Anneal PER importance-sampling β
        agent.memory.anneal_beta(episode, num_episodes)

        ep_time = time.time() - ep_start
        avg_loss = episode_loss / max(loss_count, 1)
        hit_rate = info["hit_rate"]
        subband_rate = info["subband_rate"]
        avg_match = info["avg_match"]
        avg_entropy = ep_entropy_sum / max(env_steps, 1)
        perm_rate = (hit_rate / subband_rate) if subband_rate > 1e-9 else 0.0

        # --- Metrics record -----------------------------------------------
        metrics = {
            "episode": episode,
            "total_reward": round(episode_reward, 2),
            "hit_rate": round(hit_rate, 6),
            "avg_loss": round(avg_loss, 6),
            "epsilon": round(agent.epsilon, 6),
            "beta": round(agent.memory.beta, 4),
            "train_steps": agent.train_steps,
            "env_steps": env_steps,
            "time_sec": round(ep_time, 1),
            "last_obs_index": last_obs_index,
            "last_action": last_action,
            "subband_rate": round(subband_rate, 6),
            "avg_match": round(avg_match, 4),
            "perm_rate": round(perm_rate, 6),
            "entropy": round(avg_entropy, 4),
            "efficiency": round(episode_reward / max(env_steps, 1), 4),
        }
        all_metrics.append(metrics)

        # Dış bar postfix
        pbar_ep.set_postfix(
            ret=f"{episode_reward:.0f}",
            hit=f"{hit_rate:.3f}",
            sub=f"{subband_rate:.2f}",
            eps=f"{agent.epsilon:.3f}",
            refresh=True,
        )

        # --- Console: detaylı bölüm sonu raporu (12.md formatı) --------------
        if episode % log_interval == 0:
            report = _episode_report(all_metrics, metrics, max_pulses, window=10)
            tqdm.write(report)

        # --- Checkpoint saving --------------------------------------------
        if episode % save_interval == 0:
            ckpt_path = run_dir / f"checkpoint_ep{episode}.pt"
            agent.save(str(ckpt_path))

        # --- Incremental training curves (same folder, updated PNGs) -------
        if episode % plot_interval == 0 and all_metrics:
            try:
                _save_training_plots(
                    all_metrics=all_metrics,
                    results_dir=results_dir,
                    config_path=config_path,
                    num_episodes=num_episodes,
                    max_pulses=max_pulses,
                    plot_dir=plot_dir,
                )
            except Exception:
                pass  # don't break training

        # --- Log file summary (skimmable) ----------------------------------
        with open(log_path, "a", encoding="utf-8") as log_file:
            report = _episode_report(all_metrics, metrics, max_pulses, window=10)
            log_file.write(report + "\n")

    # --- Final save -------------------------------------------------------
    agent.save(str(run_dir / "final_model.pt"))

    with open(metrics_path, "w", encoding="utf-8") as f:
        for m in all_metrics:
            f.write(json.dumps(m) + "\n")

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"Training complete. Final hit_rate={all_metrics[-1]['hit_rate']:.4f} epsilon={all_metrics[-1]['epsilon']:.6f}\n")

    # --- Final training plots (same run folder) --------------------------
    try:
        _save_training_plots(
            all_metrics=all_metrics,
            results_dir=results_dir,
            config_path=config_path,
            num_episodes=num_episodes,
            max_pulses=max_pulses,
            plot_dir=plot_dir,
        )
        print(f"Plots saved: {plot_dir}")
    except Exception as e:
        print(f"Warning: could not save training plots: {e}")

    print(f"\nTraining complete. Results saved to {run_dir}/")
    print(f"Log: {log_path}")
    print(f"Final hit rate: {all_metrics[-1]['hit_rate']:.4f}")
    print(f"Final epsilon:  {all_metrics[-1]['epsilon']:.6f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train GA-Dueling DQN Jamming Agent")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file (relative to project root)")
    args = parser.parse_args()
    train(config_path=args.config)


if __name__ == "__main__":
    main()
