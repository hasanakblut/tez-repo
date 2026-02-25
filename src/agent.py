"""
JammingAgent — Intelligent Decision-Making Agent
=================================================
Implements the full RL agent for cognitive jamming against intra-pulse
frequency agile radar, including Prioritized Experience Replay, Double
DQN updates, and ε-greedy exploration.

Paper: Xia et al., "GA-Dueling DQN Jamming Decision-Making Method
       for Intra-Pulse Frequency Agile Radar", Sensors 2024.

Components:
    SumTree                  – Binary tree for O(log n) priority sampling
    PrioritizedReplayBuffer  – TD-error prioritized experience storage
    JammingAgent             – Policy/Target net management, action selection, learning
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.model import GADuelingDQN


# ---------------------------------------------------------------------------
# SumTree — Efficient priority-based sampling (Paper Section 3.3)
# ---------------------------------------------------------------------------

class SumTree:
    """Complete binary tree where leaf nodes store priorities and internal
    nodes store the sum of their children.

    Supports O(log n) proportional sampling and O(log n) priority updates.
    Used as the backbone data structure for Prioritized Experience Replay.

    Paper Section 3.3: "the jammer samples a small batch of experiences
    from the prioritized experience replay buffer (PER) according to
    non-uniform weights to update the network."
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_pos = 0
        self.size = 0

    @property
    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data) -> None:
        tree_idx = self.write_pos + self.capacity - 1
        self.data[self.write_pos] = data
        self._update(tree_idx, priority)
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _update(self, tree_idx: int, priority: float) -> None:
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def update(self, tree_idx: int, priority: float) -> None:
        self._update(tree_idx, priority)

    def sample(self, value: float) -> Tuple[int, float, object]:
        """Retrieve the leaf whose cumulative priority range contains *value*.

        Traverses the tree iteratively (no recursion) from root to leaf.
        """
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                break
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right

        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# ---------------------------------------------------------------------------
# Prioritized Experience Replay (Paper Section 3.3, Ref [14])
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Experience replay buffer with TD-error based priority sampling.

    Each transition is stored with a priority proportional to its TD-error,
    so that surprising transitions are replayed more frequently.

    Paper Section 3.3: "Sample a Minibatch of transitions from the replay
    buffer B based on priorities" (Algorithm 1, line 11).

    Args:
        capacity:     Maximum number of transitions.
        alpha:        Priority exponent (0 = uniform, 1 = full prioritization).
        beta_start:   Initial importance-sampling exponent.
        beta_end:     Final importance-sampling exponent.
        min_priority: Small constant added to TD-error to prevent zero priority.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        min_priority: float = 1e-6,
    ):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.min_priority = min_priority
        self.max_priority = 1.0

    def add(self, transition: tuple) -> None:
        """Store a transition with maximum priority (to ensure it gets
        sampled at least once)."""
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> Tuple[list, list, np.ndarray]:
        """Sample a batch proportional to stored priorities.

        Returns:
            transitions: List of sampled transition tuples.
            tree_indices: SumTree indices for later priority updates.
            is_weights:   Importance-sampling weights (normalized).
        """
        transitions = []
        tree_indices = []
        priorities = []

        segment = self.tree.total / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = random.uniform(low, high)
            idx, priority, data = self.tree.sample(value)

            if data is None:
                value = random.uniform(0, self.tree.total)
                idx, priority, data = self.tree.sample(value)

            tree_indices.append(idx)
            priorities.append(priority)
            transitions.append(data)

        probs = np.array(priorities, dtype=np.float64) / self.tree.total
        is_weights = (self.tree.size * probs) ** (-self.beta)
        is_weights /= is_weights.max()

        return transitions, tree_indices, is_weights.astype(np.float32)

    def update_priorities(self, tree_indices: list,
                          td_errors: np.ndarray) -> None:
        """Recompute priorities from new TD-errors after a learning step."""
        for idx, td_err in zip(tree_indices, td_errors):
            priority = (abs(td_err) + self.min_priority) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def anneal_beta(self, episode: int, total_episodes: int) -> None:
        """Linearly anneal β from beta_start to beta_end."""
        frac = min(episode / max(total_episodes, 1), 1.0)
        self.beta = self.beta_start + frac * (self.beta_end - self.beta_start)

    def __len__(self) -> int:
        return self.tree.size


# ---------------------------------------------------------------------------
# JammingAgent (Paper Section 3.3, Algorithm 1, Figures 5–8)
# ---------------------------------------------------------------------------

class JammingAgent:
    """Cognitive jammer agent using GA-Dueling DQN with Double DQN updates.

    Manages:
        - Policy Network (for action selection and gradient updates).
        - Target Network (for stable TD-target computation).
        - Prioritized Experience Replay buffer.
        - ε-greedy exploration with exponential decay.
        - GRU hidden state across episode steps.

    Paper Algorithm 1 describes the full decision-making loop that this
    class implements.
    """

    def __init__(self, config: Dict):
        net_cfg = config["network"]
        train_cfg = config["training"]
        replay_cfg = config["replay"]

        self.state_dim: int = net_cfg["state_dim"]
        self.action_dim: int = net_cfg["action_dim"]
        self.seq_len: int = config["environment"]["history_len"]
        self.gamma: float = train_cfg["gamma"]
        self.batch_size: int = train_cfg["batch_size"]
        gpu_id = config.get("device_id", 0)
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        # --- Networks (Paper Figure 5: policy net + target net) -----------
        net_kwargs = dict(
            state_dim=net_cfg["state_dim"],
            action_dim=net_cfg["action_dim"],
            embed_dim=net_cfg.get("embed_dim", 64),
            hidden_dim=net_cfg["hidden_dim"],
            num_heads=net_cfg["num_heads"],
            fc_dim=net_cfg["fc_dim"],
            sigma_init=net_cfg.get("sigma_init", 0.5),
            use_embedding=net_cfg.get("use_embedding", False),
        )
        self.policy_net = GADuelingDQN(**net_kwargs).to(self.device)
        self.target_net = GADuelingDQN(**net_kwargs).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # --- Multi-GPU: DataParallel only when use_multi_gpu=true and 2+ GPUs ---
        # device_ids[0] must match self.device (where the module already lives)
        if config.get("use_multi_gpu", False) and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            primary = self.device.index if self.device.type == "cuda" else 0
            device_ids = [primary] + [i for i in range(torch.cuda.device_count()) if i != primary]
            self.policy_net = torch.nn.DataParallel(self.policy_net, device_ids=device_ids)
            self.target_net = torch.nn.DataParallel(self.target_net, device_ids=device_ids)

        # --- Optimizer ----------------------------------------------------
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=train_cfg["learning_rate"])

        # --- Epsilon schedule (Paper Section 3.3) -------------------------
        # "Exploration rate ε follows an exponential decay schedule from
        #  0.995 to 0.005, maintaining a consistent decay rate."
        # Optional: epsilon_waypoints = [[progress_0, eps_0], [progress_1, eps_1], ...]
        # with progress in [0, 1] over the run; epsilon is linearly interpolated (smooth).
        waypoints = train_cfg.get("epsilon_waypoints")
        if waypoints and len(waypoints) >= 2:
            self._epsilon_waypoints: list[tuple[float, float]] = sorted(
                [(float(p), float(e)) for p, e in waypoints]
            )
            self._use_epsilon_schedule = True
            self.epsilon = max(0.0, min(1.0, self._epsilon_waypoints[0][1]))
        else:
            self._epsilon_waypoints = []
            self._use_epsilon_schedule = False
            self.epsilon = train_cfg["epsilon_start"]

        self.epsilon_start: float = train_cfg["epsilon_start"]
        self.epsilon_end: float = train_cfg["epsilon_end"]
        decay_mode = train_cfg.get("epsilon_decay_mode", "per_episode")
        if decay_mode == "per_step":
            decay_denom = max(int(train_cfg.get("epsilon_decay_steps", 1_000_000)), 1)
        else:
            decay_denom = max(int(train_cfg.get("epsilon_decay_episodes", 100)), 1)
        self.epsilon_decay: float = (
            (self.epsilon_end / self.epsilon_start) ** (1.0 / decay_denom)
        )

        # --- VDBE: TD-error-based adaptive epsilon (Tokic, KI 2010) -------
        self._vdbe_mode: bool = train_cfg.get("epsilon_mode") == "vdbe"
        if self._vdbe_mode:
            self._vdbe_beta: float = float(train_cfg.get("vdbe_beta", 0.2))
            self._vdbe_alpha: float = float(train_cfg.get("vdbe_ema_alpha", 0.1))
            self._vdbe_eps_min: float = float(train_cfg.get("vdbe_eps_min", 0.01))
            self._vdbe_eps_max: float = float(train_cfg.get("vdbe_eps_max", 0.99))
            self._vdbe_td_scale: float = float(train_cfg.get("vdbe_td_scale", 20.0))
            self._vdbe_ema_td: float = 1.0
            self._use_epsilon_schedule = True

        # --- Prioritized Experience Replay --------------------------------
        self.memory = PrioritizedReplayBuffer(
            capacity=replay_cfg["buffer_size"],
            alpha=replay_cfg["alpha"],
            beta_start=replay_cfg["beta_start"],
            beta_end=replay_cfg["beta_end"],
            min_priority=replay_cfg["min_priority"],
        )

        # --- GRU hidden state for online inference ------------------------
        self.hidden: Optional[torch.Tensor] = None

        # --- Training step counter ----------------------------------------
        self.train_steps: int = 0

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def _encode_history(self, history: List[int]) -> torch.Tensor:
        """Convert a history of state indices to a long tensor for embedding.

        Short histories are left-padded with 0 (valid embedding index).

        Args:
            history: List of state indices (length ≤ seq_len).

        Returns:
            Tensor of shape (seq_len,) dtype long.
        """
        tensor = torch.zeros(self.seq_len, dtype=torch.long)
        trimmed = history[-self.seq_len:]
        start = self.seq_len - len(trimmed)
        for t, idx in enumerate(trimmed):
            tensor[start + t] = idx
        return tensor

    def _encode_batch(self, histories: List[List[int]]) -> torch.Tensor:
        """Encode a batch of histories into a single tensor.

        Returns:
            Tensor of shape (batch, seq_len) dtype long.
        """
        return torch.stack([self._encode_history(h) for h in histories])

    # ------------------------------------------------------------------
    # Action selection (Paper Algorithm 1, lines 5–7)
    # ------------------------------------------------------------------

    def select_action(self, history: List[int]) -> int:
        """Choose an action using ε-greedy over the policy network.

        Paper Section 3.3: "the jammer selects the frequency using an
        ε-greedy algorithm based on the current state's Q-values. It
        selects the action with the largest Q-value, with a probability
        of 1 − ε, while randomly choosing actions with a probability of ε."

        GRU hidden state is updated on every step (including random actions)
        so the recurrent chain stays temporally consistent.
        """
        with torch.no_grad():
            state_seq = self._encode_history(history).unsqueeze(0).to(
                self.device)
            self.policy_net.eval()
            q_values, self.hidden = self.policy_net(state_seq, self.hidden)
            self.policy_net.train()

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return q_values.argmax(dim=-1).item()

    def select_action_with_info(self, history: List[int]) -> Tuple[int, float]:
        """Like select_action but also returns policy entropy (bits).

        Entropy H = -Σ p_i log2(p_i) over softmax(Q).
        High entropy → uncertain/exploring; low entropy → confident/exploiting.
        """
        with torch.no_grad():
            state_seq = self._encode_history(history).unsqueeze(0).to(
                self.device)
            self.policy_net.eval()
            q_values, self.hidden = self.policy_net(state_seq, self.hidden)
            self.policy_net.train()

            probs = F.softmax(q_values.squeeze(0), dim=-1)
            log_probs = torch.log2(probs + 1e-10)
            entropy = -(probs * log_probs).sum().item()

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1), entropy
        return q_values.argmax(dim=-1).item(), entropy

    def reset_hidden(self) -> None:
        """Zero-initialize the GRU hidden state at the start of an episode."""
        net = getattr(self.policy_net, "module", self.policy_net)
        self.hidden = net.init_hidden(batch_size=1, device=self.device)

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state_history: List[int],
        action: int,
        reward: float,
        next_state_history: List[int],
        done: bool,
    ) -> None:
        """Add a transition to the PER buffer.

        Paper Algorithm 1, line 9: "Add transition (s_t, a_t, r_{t+1},
        s_{t+1}) to replay buffer B."
        """
        self.memory.add((
            list(state_history),
            action,
            reward,
            list(next_state_history),
            done,
        ))

    # ------------------------------------------------------------------
    # Learning step (Paper Algorithm 1, lines 11–16)
    # ------------------------------------------------------------------

    def learn(self) -> Optional[float]:
        """Sample a prioritized minibatch, compute Double DQN loss,
        and update the policy network.

        Returns:
            Loss value (float) if an update was performed, else None.
        """
        if len(self.memory) < self.batch_size:
            return None

        # --- Sample from PER (Algorithm 1, line 11) ---
        transitions, tree_indices, is_weights = self.memory.sample(
            self.batch_size)
        is_weights_t = torch.tensor(
            is_weights, dtype=torch.float32, device=self.device)

        # --- Unpack and encode ---
        state_hists = [t[0] for t in transitions]
        actions = torch.tensor(
            [t[1] for t in transitions], dtype=torch.long, device=self.device)
        rewards = torch.tensor(
            [t[2] for t in transitions], dtype=torch.float32, device=self.device)
        next_state_hists = [t[3] for t in transitions]
        dones = torch.tensor(
            [t[4] for t in transitions], dtype=torch.float32, device=self.device)

        state_batch = self._encode_batch(state_hists).to(self.device)
        next_state_batch = self._encode_batch(next_state_hists).to(self.device)

        # --- Re-sample noise for this training step ---
        getattr(self.policy_net, "module", self.policy_net).reset_noise()
        getattr(self.target_net, "module", self.target_net).reset_noise()

        # --- Current Q-values: Q_policy(s, a) ---
        # Algorithm 1, line 14
        q_values, _ = self.policy_net(state_batch)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- Double DQN TD target (Paper Figure 5, Equation 11) ---
        # Step 1: Policy net selects best action for s'
        # Step 2: Target net evaluates Q-value of that action
        with torch.no_grad():
            next_q_policy, _ = self.policy_net(next_state_batch)
            best_next_actions = next_q_policy.argmax(dim=-1, keepdim=True)

            next_q_target, _ = self.target_net(next_state_batch)
            next_q = next_q_target.gather(1, best_next_actions).squeeze(1)

            td_targets = rewards + self.gamma * next_q * (1.0 - dones)

        # --- TD errors for priority update ---
        td_errors = (td_targets - q_sa).detach().cpu().numpy()

        # --- Weighted Huber loss ---
        elementwise_loss = F.smooth_l1_loss(q_sa, td_targets, reduction="none")
        loss = (is_weights_t * elementwise_loss).mean()

        # --- Backpropagation (Algorithm 1, line 15) ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # --- Update PER priorities ---
        self.memory.update_priorities(tree_indices, td_errors)

        # --- VDBE: update epsilon from batch TD error ---
        if self._vdbe_mode:
            mean_abs_td = float(np.abs(td_errors).mean())
            scale = max(1e-6, self._vdbe_td_scale)
            scaled_td = mean_abs_td / scale
            self._vdbe_ema_td = (
                self._vdbe_alpha * scaled_td
                + (1.0 - self._vdbe_alpha) * self._vdbe_ema_td
            )
            raw_eps = 1.0 - float(np.exp(-self._vdbe_beta * self._vdbe_ema_td))
            self.epsilon = max(self._vdbe_eps_min, min(self._vdbe_eps_max, raw_eps))

        self.train_steps += 1
        return loss.item()

    # ------------------------------------------------------------------
    # Network management
    # ------------------------------------------------------------------

    def update_target_network(self) -> None:
        """Hard-copy policy network weights to the target network.

        Paper Figure 5: The target network is periodically synchronized
        with the policy network to stabilize TD targets.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        """Apply one step of exponential ε decay (ignored when using waypoints or VDBE)."""
        if self._use_epsilon_schedule or self._vdbe_mode:
            return
        self.epsilon = max(
            self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_epsilon_from_schedule(self, progress: float) -> None:
        """Set epsilon from waypoint schedule by linear interpolation (smooth in [0,1]).

        progress: 0 = start of run, 1 = end of run (e.g. (episode - 1) / (num_episodes - 1)).
        Only has effect when training.epsilon_waypoints is set in config.
        """
        if not self._use_epsilon_schedule or not self._epsilon_waypoints:
            return
        progress = max(0.0, min(1.0, float(progress)))
        wp = self._epsilon_waypoints
        if progress <= wp[0][0]:
            self.epsilon = wp[0][1]
        elif progress >= wp[-1][0]:
            self.epsilon = wp[-1][1]
        else:
            for i in range(len(wp) - 1):
                if wp[i][0] <= progress <= wp[i + 1][0]:
                    denom = wp[i + 1][0] - wp[i][0]
                    t = (progress - wp[i][0]) / denom if denom > 0 else 1.0
                    self.epsilon = wp[i][1] + t * (wp[i + 1][1] - wp[i][1])
                    break
        self.epsilon = max(0.0, min(1.0, self.epsilon))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        # state_dict() on DataParallel returns inner module's state (no "module." prefix)
        policy_sd = getattr(self.policy_net, "module", self.policy_net).state_dict()
        target_sd = getattr(self.target_net, "module", self.target_net).state_dict()
        torch.save({
            "policy_net": policy_sd,
            "target_net": target_sd,
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        getattr(self.policy_net, "module", self.policy_net).load_state_dict(ckpt["policy_net"])
        getattr(self.target_net, "module", self.target_net).load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.train_steps = ckpt.get("train_steps", 0)
