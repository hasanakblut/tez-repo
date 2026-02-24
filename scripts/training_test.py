"""
Training test (dry-run): run env + agent for a short horizon, record state/action/reward
in memory, run verification checks, and optionally plot.
Uses configs/test_training.yaml by default so a separate config drives the test.
"""

import argparse
from pathlib import Path

import yaml

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
    parser = argparse.ArgumentParser(description="Dry-run training: record state/action/reward, plot, verify")
    parser.add_argument("--config", type=str, default="configs/test_training.yaml")
    parser.add_argument("--max-steps", type=int, default=None, help="Cap steps per episode (default: from config)")
    parser.add_argument("--plot", action="store_true", help="Save matplotlib figures to results/training_test_plots")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification assertions")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = build_env_config(cfg)

    from src.env import RadarEnv
    from src.agent import JammingAgent

    env = RadarEnv(config=env_cfg)
    agent = JammingAgent(config=cfg)
    history_len = cfg.get("environment", {}).get("history_len", 10)
    max_pulses = cfg["episode"]["max_pulses"]
    max_steps = args.max_steps if args.max_steps is not None else max_pulses
    max_steps = min(max_steps, max_pulses)

    jsr_base = env.jsr_base
    max_reward = jsr_base * 4.0

    # --- Single episode, record in memory ---
    records: list[dict] = []
    obs, info = env.reset(seed=42)
    agent.reset_hidden()

    for t in range(1, max_steps + 1):
        state_history = env.get_history()
        action = agent.select_action(state_history)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state_history = env.get_history()
        agent.store_transition(state_history, action, reward, next_state_history, terminated)

        rec = {
            "t": t,
            "state_history": list(state_history),
            "action": action,
            "reward": reward,
            "next_state_history": list(next_state_history),
            "num_matches": info.get("num_matches", 0),
            "hit_rate": info.get("hit_rate", 0.0),
        }
        records.append(rec)

        if not args.no_verify:
            assert 1 <= len(state_history) <= history_len, f"state_history length {len(state_history)}"
            assert 0 <= action < 240, f"action {action}"
            assert 0 <= reward <= max_reward + 1e-6, f"reward {reward}"
            assert 0 <= next_obs < 240, f"next_obs {next_obs}"

        if terminated:
            break

    # --- Optional: direct write to file (later work)
    # out_path = PROJECT_ROOT / cfg["logging"]["results_dir"] / "training_test_transitions.jsonl"
    # with open(out_path, "w", encoding="utf-8") as f:
    #     for r in records:
    #         f.write(json.dumps({k: v for k, v in r.items() if k != "state_history" and k != "next_state_history"}) + "\n")

    print(f"Recorded {len(records)} steps. Reward range: {min(r['reward'] for r in records):.2f} .. {max(r['reward'] for r in records):.2f}")
    print(f"Final hit_rate: {records[-1]['hit_rate']:.4f}")

    if args.plot:
        import matplotlib.pyplot as plt

        out_dir = PROJECT_ROOT / cfg["logging"]["results_dir"] / "training_test_plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        t_axis = [r["t"] for r in records]
        rewards = [r["reward"] for r in records]
        actions = [r["action"] for r in records]
        state_last = [r["state_history"][-1] if r["state_history"] else 0 for r in records]

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(t_axis, rewards, color="C0")
        axes[0].set_ylabel("Reward")
        axes[0].set_title("Reward vs step")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_axis, state_last, color="C1", alpha=0.8)
        axes[1].set_ylabel("State index")
        axes[1].set_title("Radar state (last in history) vs step")
        axes[1].grid(True, alpha=0.3)

        axes[2].hist(actions, bins=min(50, len(set(actions))), color="C2", alpha=0.8, edgecolor="black")
        axes[2].set_xlabel("Action")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Action distribution")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = out_dir / "training_test_plots.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Plots saved: {plot_path}")


if __name__ == "__main__":
    main()
