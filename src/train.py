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
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]

from src.agent import JammingAgent
from src.env import RadarEnv

# Project root (so config/results work regardless of cwd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _seed_everything(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible training.
    Episode reset_seed is controlled by config same_episode_seed (default True:
    same pulse train every episode; False: seed+episode per episode)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log_line(log_file: None | object, line: str, log_path: Path | None = None) -> None:
    """Write one line to the run log and flush (realtime, safe on Ctrl+C).
    Call with open file handle: _log_line(log_file, line). If log_path is given and
    log_file is None, opens in append mode (fallback for one-off writes)."""
    if log_file is not None:
        log_file.write(line + "\n")
        log_file.flush()
    elif log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


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
                    window: int = 10, use_emoji: bool = True) -> str:
    """Bölüm sonu raporu. use_emoji=True → terminal ve training.log (emojili); False → düz metin."""
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
    recent = all_metrics[-window:]
    rets = [m["total_reward"] for m in recent]
    ma_ret = sum(rets) / len(rets)
    slope = _regression_slope(rets)
    rsd = _rsd(rets)
    trend = _trend_symbol(slope, rsd)

    if use_emoji:
        lines = [
            f"━━━ 📦 Episode {ep} ━━━",
            f"  📈 Return: {ret:,.0f}  (MA{window}: {ma_ret:,.0f}  {trend})  🎯 Hit: {hit:.4f}  # Num: {avg_match:.2f}/4",
            f"  ⚡ Eff: {eff:.3f}  📶 Subband: {subband:.2%}  🔀 Perm: {perm_rate:.2%}  H: {entropy:.2f}b  ε: {eps:.4f}  📉 Loss: {td_loss:.4f}",
            f"  📊 RSD: {rsd:.1f}%",
            f"━━━━━━━━━━━━━━━━━━━━━━",
        ]
    else:
        lines = [
            f"--- [Episode {ep}] ---",
            f"  Return: {ret:,.0f} (MA{window}: {ma_ret:,.0f} | Slope: {slope:+.2f}) {trend}",
            f"  Hit: {hit:.4f} | Num: {avg_match:.2f}/4 | Eff: {eff:.3f}",
            f"  Subband: {subband:.2%} | Perm: {perm_rate:.2%} | H: {entropy:.2f}b | ε: {eps:.4f} | Loss: {td_loss:.4f}",
            f"  RSD: {rsd:.1f}%",
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

    # Episode-end reward bar chart (her episode sonundaki reward)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(episodes, rewards, color="C0", alpha=0.8, edgecolor="C0")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward (episode end)")
    ax.set_title(f"Episode-end reward ({num_episodes} ep × {max_pulses:,} pulses)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "episode_rewards.png", dpi=150)
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config_path: str = "configs/default.yaml", device_override: str | None = None) -> None:
    cfg = load_config(config_path)
    # CLI --device overrides config device_id and use_multi_gpu
    if device_override is not None:
        if device_override == "multi":
            cfg["use_multi_gpu"] = True
            # primary GPU for multi: keep config device_id (default 0)
        else:
            cfg["device_id"] = int(device_override)
            cfg["use_multi_gpu"] = False
    gpu_id = cfg.get("device_id", 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Reproducibility: fix Python, NumPy, PyTorch so same seed → same run
    global_seed = cfg.get("seed")
    if global_seed is not None:
        _seed_everything(global_seed)
        print(f"Seed: {global_seed} (reproducible)")
    else:
        print("Seed: None (non-reproducible)")

    # --- Environment ------------------------------------------------------
    env_cfg = build_env_config(cfg)
    env = RadarEnv(config=env_cfg)
    print("Radar pulse train: placeholder (uniform random); new structure to be added.")

    # --- Agent ------------------------------------------------------------
    agent = JammingAgent(config=cfg)
    nparams = sum(p.numel() for p in agent.policy_net.parameters())
    print(f"Policy net parameters: {nparams:,}")
    if cfg.get("use_multi_gpu", False) and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Multi-GPU: DataParallel over {torch.cuda.device_count()} GPUs (batch split across devices)")

    # --- Logging setup: one run dir per training (datetime + config + ep + seed) ---
    results_dir = PROJECT_ROOT / cfg["logging"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    num_episodes = cfg["episode"]["num_episodes"]
    max_pulses = cfg["episode"]["max_pulses"]
    target_update_freq = cfg["training"]["target_update_freq"]
    epsilon_decay_mode = cfg["training"].get("epsilon_decay_mode", "per_episode")
    early_stop_reward = cfg["training"].get("early_stop_reward")  # float or None; stop when episode total_reward >= this
    log_interval = cfg["logging"]["log_interval"]
    save_interval = cfg["logging"]["save_interval"]
    plot_interval = cfg["logging"].get("plot_interval", save_interval)
    use_tensorboard = cfg["logging"].get("use_tensorboard", True)
    verbose_terminal = cfg["logging"].get("verbose_terminal", False)

    run_id = _run_id(cfg, config_path)
    run_dir = results_dir / "runs" / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.log"
    metrics_path = run_dir / "metrics.jsonl"
    plot_dir = run_dir / "training_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if use_tensorboard and SummaryWriter is not None:
        tb_dir = run_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"📊 TensorBoard: logdir={tb_dir}  (run: tensorboard --logdir={tb_dir})")
    elif use_tensorboard and SummaryWriter is None:
        print("⚠️ TensorBoard is enabled but torch.utils.tensorboard not available; install with: pip install tensorboard")

    all_metrics: list = []
    ts_iso = time.strftime("%Y-%m-%d %H:%M:%S")
    run_header = (
        f"🚀 RUN {ts_iso}\n"
        f"  📄 config: {config_path}\n"
        f"  📋 num_episodes: {num_episodes}  max_pulses: {max_pulses}\n"
        f"  🎲 seed: {cfg.get('seed')}\n"
        f"  📁 run_dir: {run_dir}\n"
        f"  ℹ️  Pulse train: new structure to be integrated.\n"
    )
    run_header += "━━━━━━━━━━━━━━━━━━━━━━\n"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(run_header)
        f.flush()

    # Keep log open for the run (write+flush per line, no open/close per line → faster)
    log_file = open(log_path, "a", encoding="utf-8")
    try:
        # --- Training loop (Paper Algorithm 1) --------------------------------
        print(f"\n🚀 Training: {num_episodes} ep × {max_pulses:,} pulses")
        print(f"📁 {run_dir}")
        print(f"📋 Log every {log_interval} ep │ Save/plot every {save_interval}/{plot_interval} ep")
        if use_tensorboard and writer is not None and not verbose_terminal:
            print("📊 Terminal: sadece episode özeti (detay için TensorBoard)")
        print()

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
            same_episode_seed = cfg.get("same_episode_seed", True)
            if global_seed is not None:
                reset_seed = global_seed if same_episode_seed else (global_seed + episode)
            else:
                reset_seed = None

            # Optional: random start index per episode (Markov matrix unchanged; only first state varies)
            state_dim = 240
            random_start = env_cfg.get("random_start_index", True)
            if random_start:
                if global_seed is not None:
                    start_rng = np.random.default_rng(global_seed + 99999 + episode)
                    start_index = int(start_rng.integers(0, state_dim))
                else:
                    start_index = int(np.random.randint(0, state_dim))
            else:
                start_index = None

            ep_start_line = f"🔄 Episode {episode} │ seed={reset_seed} start_index={start_index}"
            _log_line(log_file, ep_start_line)
            if episode == 1 or episode % log_interval == 0:
                tqdm.write(f"  🔄 Ep {episode} │ seed={reset_seed} start={start_index}")

            reset_options = {"start_index": start_index} if start_index is not None else None
            obs, info = env.reset(seed=reset_seed, options=reset_options)
            agent.reset_hidden()

            # Epsilon from waypoint schedule (smooth [0,1] over run) when configured
            if cfg["training"].get("epsilon_waypoints"):
                progress = (episode - 1) / max(num_episodes - 1, 1)
                agent.update_epsilon_from_schedule(progress)

            episode_reward = 0.0
            episode_loss = 0.0
            loss_count = 0
            env_steps = 0
            ep_entropy_sum = 0.0
            last_obs_index: int | None = None
            last_action: int | None = None
            ep_epsilon_history: list[float] = []
            ep_step_rewards: list[float] = []  # per-pulse reward for 500-pulse window plots
            # Sampled every 500 pulses for per-episode dashboard
            ep_pulse_samples: list[int] = []
            ep_hit_samples: list[float] = []
            ep_avg_match_samples: list[float] = []
            ep_eff_samples: list[float] = []
            ep_entropy_samples: list[float] = []

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
                ep_step_rewards.append(reward)

                # Epsilon decay: per-step (each pulse) or per-episode (once at end)
                if epsilon_decay_mode == "per_step":
                    agent.decay_epsilon()

                # Track epsilon for intra-episode curve
                ep_epsilon_history.append(agent.epsilon)

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

                # Realtime log + terminal: one line per 500 pulses (emojis for terminal)
                if pulse % 500 == 0 or pulse == 1 or pulse == max_pulses:
                    pct = 100.0 * pulse / max_pulses
                    eff = episode_reward / env_steps
                    avg_ent = ep_entropy_sum / env_steps
                    # Sample for per-episode dashboard
                    ep_pulse_samples.append(pulse)
                    ep_hit_samples.append(info["hit_rate"])
                    ep_avg_match_samples.append(info["avg_match"])
                    ep_eff_samples.append(eff)
                    ep_entropy_samples.append(avg_ent)
                    pct_step = int(max_pulses * 0.05)
                    if ep_epsilon_history and pct_step > 0:
                        window = ep_epsilon_history[-pct_step:] if len(ep_epsilon_history) >= pct_step else ep_epsilon_history
                        eps_mean = sum(window) / len(window)
                        eps_var = sum((e - eps_mean) ** 2 for e in window) / len(window)
                        eps_suffix = f" │ ε̄={eps_mean:.4f} σ²={eps_var:.6f} ε={agent.epsilon:.4f}"
                    else:
                        eps_suffix = f" │ ε={agent.epsilon:.4f}"
                    try:
                        max_theoretical = float(env.max_pulses) * float(env.jsr_base) * int(env.K)
                    except (TypeError, AttributeError, ZeroDivisionError):
                        max_theoretical = 0.0
                    if max_theoretical > 0:
                        pct_of_max = 100.0 * episode_reward / max_theoretical
                        pct_str = f" ({pct_of_max:.2f}%)"
                    else:
                        pct_str = ""
                    pulse_line = (
                        f"  📍 {pct:5.1f}% │ pulse {pulse:>5}/{max_pulses} │ "
                        f"📈 {episode_reward:>8.0f}{pct_str} │ 🎯 {info['hit_rate']:.4f} │ "
                        f"📶 {info['subband_rate']:.3f} │ # {info['avg_match']:.2f}/4 │ "
                        f"⚡ {eff:.3f} │ H {avg_ent:.2f}b{eps_suffix}"
                    )
                    _log_line(log_file, pulse_line)
                    if verbose_terminal:
                        tqdm.write(pulse_line)
                    # TensorBoard: step-level metrics (every 500 pulses)
                    if writer is not None:
                        global_step = (episode - 1) * max_pulses + pulse
                        writer.add_scalar("step/reward_cumul", episode_reward, global_step)
                        writer.add_scalar("step/hit_rate", info["hit_rate"], global_step)
                        writer.add_scalar("step/subband_rate", info["subband_rate"], global_step)
                        writer.add_scalar("step/avg_match", info["avg_match"], global_step)
                        writer.add_scalar("step/efficiency", eff, global_step)
                        writer.add_scalar("step/epsilon", agent.epsilon, global_step)
                        writer.add_scalar("step/entropy", avg_ent, global_step)

                if terminated:
                    break

            pbar_pulse.close()

            # --- Reward in 500-pulse windows (per episode) --------------------
            if ep_step_rewards:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    WINDOW = 500
                    window_rewards = [
                        sum(ep_step_rewards[i : i + WINDOW])
                        for i in range(0, len(ep_step_rewards), WINDOW)
                    ]
                    if window_rewards:
                        rwd_plot_dir = plot_dir / "reward_500pulse_windows"
                        rwd_plot_dir.mkdir(parents=True, exist_ok=True)
                        fig, ax = plt.subplots(figsize=(10, 3.5))
                        x = range(1, len(window_rewards) + 1)
                        ax.bar(x, window_rewards, color="C0", alpha=0.8, edgecolor="C0")
                        ax.set_xlabel("500-pulse window (1 = pulses 1–500, 2 = 501–1000, …)")
                        ax.set_ylabel("Total reward in window")
                        ax.set_title(f"Episode {episode} — reward per {WINDOW}-pulse window ({len(ep_step_rewards)} pulses)")
                        ax.grid(True, alpha=0.3, axis="y")
                        fig.tight_layout()
                        fig.savefig(rwd_plot_dir / f"reward_ep{episode:04d}.png", dpi=120)
                        fig.savefig(plot_dir / "reward_500pulse_windows_latest.png", dpi=120)
                        plt.close(fig)
                except Exception:
                    pass

            # --- Per-episode dashboard (reward 500-win + epsilon + hit/avg_match/eff/entropy sampled every 500 pulses) ---
            if ep_step_rewards and ep_epsilon_history and ep_pulse_samples:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    dash_dir = plot_dir / "per_episode_dashboard"
                    dash_dir.mkdir(parents=True, exist_ok=True)
                    W = 500
                    window_rewards = [
                        sum(ep_step_rewards[i : i + W])
                        for i in range(0, len(ep_step_rewards), W)
                    ]
                    n_win = len(window_rewards)
                    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
                    # (1) Reward per 500-pulse window
                    ax = axes[0, 0]
                    ax.bar(range(1, n_win + 1), window_rewards, color="C0", alpha=0.8)
                    ax.set_xlabel("500-pulse window")
                    ax.set_ylabel("Total reward")
                    ax.set_title(f"Ep {episode}: reward per 500-pulse window")
                    ax.grid(True, alpha=0.3, axis="y")
                    # (2) Epsilon over pulses
                    ax = axes[0, 1]
                    ax.plot(range(1, len(ep_epsilon_history) + 1), ep_epsilon_history, color="C2", linewidth=0.6)
                    ax.set_xlabel("Pulse")
                    ax.set_ylabel("ε")
                    ax.set_ylim(-0.02, 1.02)
                    ax.set_title(f"Ep {episode}: ε over pulses")
                    ax.grid(True, alpha=0.3)
                    # (3) Hit rate (sampled every 500 pulses)
                    ax = axes[1, 0]
                    ax.plot(ep_pulse_samples, ep_hit_samples, color="C1", marker="o", markersize=3)
                    ax.set_xlabel("Pulse")
                    ax.set_ylabel("Hit rate")
                    ax.set_ylim(0, 1.02)
                    ax.set_title("Hit rate (sampled every 500 pulses)")
                    ax.grid(True, alpha=0.3)
                    # (4) Avg match (sampled every 500 pulses)
                    ax = axes[1, 1]
                    ax.plot(ep_pulse_samples, ep_avg_match_samples, color="C5", marker="s", markersize=3)
                    ax.set_xlabel("Pulse")
                    ax.set_ylabel("Avg match (Num)")
                    ax.set_ylim(0, 4.1)
                    ax.set_title("Avg match (sampled every 500 pulses)")
                    ax.grid(True, alpha=0.3)
                    # (5) Efficiency (sampled every 500 pulses)
                    ax = axes[2, 0]
                    ax.plot(ep_pulse_samples, ep_eff_samples, color="C7", marker="^", markersize=3)
                    ax.set_xlabel("Pulse")
                    ax.set_ylabel("Efficiency (reward/pulse)")
                    ax.set_title("Efficiency (sampled every 500 pulses)")
                    ax.grid(True, alpha=0.3)
                    # (6) Entropy (sampled every 500 pulses)
                    ax = axes[2, 1]
                    ax.plot(ep_pulse_samples, ep_entropy_samples, color="C6", marker="d", markersize=3)
                    ax.set_xlabel("Pulse")
                    ax.set_ylabel("Policy entropy (bits)")
                    ax.set_title("Entropy (sampled every 500 pulses)")
                    ax.grid(True, alpha=0.3)
                    fig.suptitle(f"Episode {episode} — step-by-step metrics", fontsize=11)
                    fig.tight_layout()
                    fig.savefig(dash_dir / f"dashboard_ep{episode:04d}.png", dpi=120)
                    fig.savefig(plot_dir / "per_episode_dashboard_latest.png", dpi=120)
                    plt.close(fig)
                except Exception:
                    pass

            # --- End of episode bookkeeping -----------------------------------

            # Decay exploration rate (once per episode when mode is per_episode)
            if epsilon_decay_mode == "per_episode":
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

            # TensorBoard: episode-level metrics
            if writer is not None:
                writer.add_scalar("episode/reward", metrics["total_reward"], episode)
                writer.add_scalar("episode/hit_rate", metrics["hit_rate"], episode)
                writer.add_scalar("episode/subband_rate", metrics["subband_rate"], episode)
                writer.add_scalar("episode/avg_match", metrics["avg_match"], episode)
                writer.add_scalar("episode/epsilon", metrics["epsilon"], episode)
                writer.add_scalar("episode/avg_loss", metrics["avg_loss"], episode)
                writer.add_scalar("episode/entropy", metrics["entropy"], episode)
                writer.add_scalar("episode/efficiency", metrics["efficiency"], episode)
                writer.add_scalar("episode/env_steps", metrics["env_steps"], episode)
                writer.add_scalar("episode/time_sec", metrics["time_sec"], episode)

            # Dış bar postfix
            pbar_ep.set_postfix(
                ret=f"{episode_reward:.0f}",
                hit=f"{hit_rate:.3f}",
                sub=f"{subband_rate:.2f}",
                eps=f"{agent.epsilon:.3f}",
                refresh=True,
            )

            # --- Console: detaylı bölüm sonu raporu (emojili) --------------
            if episode % log_interval == 0:
                report_terminal = _episode_report(all_metrics, metrics, max_pulses, window=10, use_emoji=True)
                tqdm.write(report_terminal)

            # --- Checkpoint saving --------------------------------------------
            if episode % save_interval == 0:
                ckpt_path = run_dir / f"checkpoint_ep{episode}.pt"
                agent.save(str(ckpt_path))

            # --- Per-episode epsilon curve (pulse-level, smooth) — every episode ---
            if ep_epsilon_history:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    eps_plot_dir = plot_dir / "epsilon_per_episode"
                    eps_plot_dir.mkdir(parents=True, exist_ok=True)
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(range(1, len(ep_epsilon_history) + 1),
                            ep_epsilon_history, color="C2", linewidth=0.5)
                    ax.set_xlabel("Pulse")
                    ax.set_ylabel("ε")
                    ax.set_ylim(-0.02, 1.02)
                    ax.set_title(f"Episode {episode} — ε over {len(ep_epsilon_history)} pulses")
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(eps_plot_dir / f"epsilon_ep{episode:04d}.png", dpi=120)
                    fig.savefig(plot_dir / "epsilon_over_pulses_latest.png", dpi=120)
                    plt.close(fig)
                except Exception:
                    pass

            # --- Run-level curves: update after every episode (TensorBoard-style incremental) ---
            if all_metrics:
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

            # --- Log file: episode report (emojili, terminal ile aynı) ---
            report_log = _episode_report(all_metrics, metrics, max_pulses, window=10, use_emoji=True)
            for report_line in report_log.split("\n"):
                _log_line(log_file, report_line)

            # --- Early stop: when episode total reward exceeds threshold (e.g. scenario runner) ---
            if early_stop_reward is not None and episode_reward >= early_stop_reward:
                msg = (
                    f"Early stop: episode {episode} total_reward={episode_reward:.0f} >= {early_stop_reward}"
                )
                _log_line(log_file, f"  ⏹ {msg}")
                tqdm.write(f"  ⏹ {msg}")
                break
    finally:
        log_file.close()
        if writer is not None:
            writer.close()

    # --- Final save -------------------------------------------------------
    agent.save(str(run_dir / "final_model.pt"))

    with open(metrics_path, "w", encoding="utf-8") as f:
        for m in all_metrics:
            f.write(json.dumps(m) + "\n")

    _log_line(
        None,
        f"✅ Training complete. 🎯 Final hit_rate={all_metrics[-1]['hit_rate']:.4f}  ε={all_metrics[-1]['epsilon']:.6f}",
        log_path,
    )

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

    print(f"\n✅ Training complete. Results: {run_dir}/")
    print(f"📄 Log: {log_path}")
    print(f"🎯 Final hit rate: {all_metrics[-1]['hit_rate']:.4f}")
    print(f"ε  Final epsilon:  {all_metrics[-1]['epsilon']:.6f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train GA-Dueling DQN Jamming Agent")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file (relative to project root)")
    parser.add_argument(
        "--device", type=str, default=None, choices=["0", "1", "multi"],
        help="GPU: 0 or 1 = single GPU (cuda:0 / cuda:1); multi = DataParallel over all GPUs (overrides config)")
    args = parser.parse_args()
    train(config_path=args.config, device_override=args.device)


if __name__ == "__main__":
    main()
