#!/usr/bin/env python3
"""
Run the 4 epsilon scenarios as separate training runs in sequence.

Scenarios 1–3: waypoint-based epsilon schedules (smooth [0,1]).
Scenario 4: VDBE — TD-error-based adaptive epsilon (Tokic 2010).
When an episode's total reward reaches training.early_stop_reward,
that run stops and the next scenario starts.

Usage (from repo root, e.g. codes/):
    python -m scripts.run_epsilon_scenarios
    python -m scripts.run_epsilon_scenarios --device 0

Runs:
  1. epsilon_scenario1_full_exploration: 0.995 → 0.005
  2. epsilon_scenario2_exploit_explore: 0.25 → 0.75 → 0.5 → 0.25
  3. epsilon_scenario3_low_high_low: 0.005 → 0.995 → 0.005
  4. epsilon_scenario4_vdbe: ε = 1 − exp(−β·EMA|δ|)
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCENARIO_CONFIGS = [
    "configs/epsilon_scenario1_full_exploration.yaml",
    "configs/epsilon_scenario2_exploit_explore.yaml",
    "configs/epsilon_scenario3_low_high_low.yaml",
    "configs/epsilon_scenario4_vdbe.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 4 epsilon-scenario trainings in sequence")
    parser.add_argument("--config-dir", type=str, default=None, help="Directory containing configs (default: project root)")
    parser.add_argument("--device", type=str, default=None, help="Device override: 0, 1, or multi")
    args = parser.parse_args()

    config_dir = Path(args.config_dir) if args.config_dir else PROJECT_ROOT
    runs_dir = PROJECT_ROOT / "results" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    cmd_base = [sys.executable, "-m", "src.train"]
    if args.device is not None:
        cmd_base.extend(["--device", args.device])

    log_paths = []
    for i, rel_config in enumerate(SCENARIO_CONFIGS, start=1):
        config_path = config_dir / rel_config
        if not config_path.exists():
            print(f"Config not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        n = len(SCENARIO_CONFIGS)
        print(f"\n{'='*60}\nScenario {i}/{n}: {rel_config}\n{'='*60}")
        cmd = cmd_base + ["--config", str(config_path)]
        ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if ret.returncode != 0:
            print(f"Training failed with exit code {ret.returncode}", file=sys.stderr)
            sys.exit(ret.returncode)
        # Latest run dir for this config stem is the one just created (by mtime)
        stem = config_path.stem
        matching = sorted(runs_dir.glob(f"run_*{stem}*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if matching:
            log_paths.append((stem, matching[0] / "training.log", matching[0]))

    print("\n" + "="*60)
    print(f"All {len(SCENARIO_CONFIGS)} scenarios finished. Run directories and logs:")
    print("="*60)
    for stem, log_path, run_dir in log_paths:
        print(f"  {stem}:")
        print(f"    run_dir: {run_dir}")
        print(f"    log:     {log_path}")
        if run_dir.exists():
            metrics = run_dir / "metrics.jsonl"
            if metrics.exists():
                print(f"    metrics: {metrics}")
    print()


if __name__ == "__main__":
    main()
