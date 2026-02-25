"""
Check whether training uses GPU and where.
- Gymnasium (env): CPU only, NumPy. Does NOT use GPU.
- PyTorch (model, agent): uses GPU when available (policy_net.to(device), tensors on device).
Run: python -m scripts.check_gpu_usage [--config configs/test_training.yaml]
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path: str) -> dict:
    import yaml
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env_config(cfg: dict) -> dict:
    env_cfg = {}
    for key in ("physics", "radar", "episode", "environment"):
        env_cfg.update(cfg.get(key, {}))
    return env_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test_training.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    env_cfg = build_env_config(cfg)

    import torch
    from src.agent import JammingAgent
    from src.env import RadarEnv

    print("=" * 60)
    print("GPU / device check")
    print("=" * 60)
    print("PyTorch CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem_gb:.1f} GB)")
        print("Current device (PyTorch):", torch.cuda.current_device())
    else:
        print("(Training will use CPU.)")
    print()

    env = RadarEnv(config=env_cfg)
    agent = JammingAgent(config=cfg)
    param_device = next(agent.policy_net.parameters()).device
    print("Policy network parameters on:", param_device)
    print("Agent device:", agent.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        using_dp = isinstance(agent.policy_net, torch.nn.DataParallel)
        print("Multi-GPU (DataParallel):", using_dp, f"— batch split across {torch.cuda.device_count()} GPUs in learn()")
    print()

    obs, _ = env.reset(seed=42)
    agent.reset_hidden()
    history = env.get_history()
    action = agent.select_action(history)
    print("After one step: action =", action)
    print("(Forward pass ran on:", param_device, ")")
    print()
    print("Gymnasium/RadarEnv: runs on CPU (NumPy). No GPU.")
    print("GPU is used for: policy_net forward, loss, backward, optimizer (PyTorch).")
    print("To watch GPU during training: run training, then in another terminal: nvidia-smi")
    print("=" * 60)


if __name__ == "__main__":
    main()
