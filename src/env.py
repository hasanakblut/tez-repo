"""
RadarEnv — Gymnasium Environment for Cognitive Jamming Simulation
=================================================================
Simulates the adversarial interaction between an intra-pulse frequency
agile radar and a cognitive jammer, modeled as a Markov Decision Process.

Paper: Xia et al., "GA-Dueling DQN Jamming Decision-Making Method
       for Intra-Pulse Frequency Agile Radar", Sensors 2024.

MDP Flow (Paper Section 3.2, Figure 4):
    1. Radar transmits pulse at frequency s_t.
    2. Jammer observes s_t and selects action a_t (predicted next frequency).
    3. Radar transmits next pulse at frequency s_{t+1}.
    4. Reward = JSR_base × Num, where Num = matched sub-pulses between a_t and s_{t+1}.

Frequency Space (Paper Section 2.1, Section 4):
    - 10 subbands × 24 permutations of 4 sub-pulses = 240 discrete states/actions.
"""

from itertools import permutations
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env_utils import FrequencyGenerator


# ---------------------------------------------------------------------------
# Frequency mapping utilities
# ---------------------------------------------------------------------------

def build_permutation_table(k: int = 4) -> list:
    """Pre-compute all K! permutations of sub-pulse frequency slots.

    For K=4 sub-pulses with pairwise-distinct frequencies within a subband,
    each permutation represents a unique ordering of the 4 frequency slots.
    Paper Section 2.1: "the frequencies of the four subpulses are pairwise
    distinct, so there are 24 combinations in total."

    Returns:
        List of 24 tuples, each a permutation of (0, 1, 2, 3).
    """
    return list(permutations(range(k)))


PERMUTATION_TABLE = build_permutation_table(k=4)


def index_to_subpulses(
    index: int,
    num_subbands: int = 10,
    num_perms: int = 24,
    f_low: float = 10e9,
    delta_f: float = 100e6,
    k: int = 4,
) -> Tuple[int, tuple, tuple]:
    """Decode a flat action/state index (0–239) into physical frequencies.

    The 240-state space is organized as:
        index = subband_id × 24 + permutation_id

    Paper Section 4: "the state space of the frequency agile radar is
    24 × 10 = 240."

    Args:
        index:         Flat index in [0, num_subbands × num_perms).
        num_subbands:  M = 10 subbands.
        num_perms:     4! = 24 permutations.
        f_low:         Lower bound of frequency band (10 GHz).
        delta_f:       Subband width (100 MHz).
        k:             Number of sub-pulses per pulse.

    Returns:
        subband_id:    Integer in [0, M).
        perm:          Tuple of sub-pulse slot indices, e.g. (2, 0, 3, 1).
        frequencies:   Tuple of 4 absolute carrier frequencies in Hz.
    """
    subband_id = index // num_perms
    perm_id = index % num_perms
    perm = PERMUTATION_TABLE[perm_id]

    base_freq = f_low + subband_id * delta_f
    sub_bw = delta_f / k  # 25 MHz per sub-pulse slot
    frequencies = tuple(base_freq + slot * sub_bw for slot in perm)

    return subband_id, perm, frequencies


def subpulses_to_index(
    subband_id: int,
    perm: tuple,
    num_perms: int = 24,
) -> int:
    """Encode (subband, permutation) back to a flat index.

    Inverse of :func:`index_to_subpulses`.
    """
    perm_id = PERMUTATION_TABLE.index(tuple(perm))
    return subband_id * num_perms + perm_id


# ---------------------------------------------------------------------------
# Sub-pulse matching
# ---------------------------------------------------------------------------

def count_subpulse_matches(action_index: int, radar_index: int,
                           num_perms: int = 24) -> int:
    """Count position-wise sub-pulse frequency matches (Num).

    Paper Section 3.2, Equation 15: "Num is the number of radar subpulses
    that are equal to the jamming subpulses."

    Matching is **position-wise**: sub-pulse k of the jammer must equal
    sub-pulse k of the radar for it to count.  If the jammer and radar
    are in different subbands, Num = 0 immediately because all absolute
    frequencies differ.

    Args:
        action_index: Jammer's chosen frequency index (0–239).
        radar_index:  Radar's actual frequency index (0–239).

    Returns:
        Num ∈ {0, 1, 2, 3, 4}.
    """
    action_subband = action_index // num_perms
    radar_subband = radar_index // num_perms

    if action_subband != radar_subband:
        return 0

    action_perm = PERMUTATION_TABLE[action_index % num_perms]
    radar_perm = PERMUTATION_TABLE[radar_index % num_perms]

    return sum(a == r for a, r in zip(action_perm, radar_perm))


# ---------------------------------------------------------------------------
# RadarEnv
# ---------------------------------------------------------------------------

class RadarEnv(gym.Env):
    """Gymnasium environment for cognitive jamming against intra-pulse
    frequency agile radar.

    Observation:
        Current radar pulse frequency index s_t ∈ {0, ..., 239}.

    Action:
        Jammer's predicted next frequency index a_t ∈ {0, ..., 239}.
        The jammer tries to predict s_{t+1} given the history of past states.

    Reward:
        r_t = JSR_base × Num  (Paper Equation 15)
        where Num = number of position-wise sub-pulse matches between
        the jammer's action a_t and the radar's next state s_{t+1}.

    Episode:
        One episode = one pulse train of ``max_pulses`` pulses (default 10,000).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()

        cfg = config or {}

        # --- Physical parameters (Paper Table 1) -------------------------
        # Loaded from config (physics section via build_env_config); used to compute JSR coefficient below.
        self.A_R: float = cfg.get("A_R", 1.0)       # Radar pulse amplitude (V)
        self.A_J: float = cfg.get("A_J", 5.0)       # Jammer pulse amplitude (V)
        self.h_t: float = cfg.get("h_t", 0.1)       # Radar-target channel gain
        self.h_j: float = cfg.get("h_j", 0.1)       # Radar-jammer channel gain
        self.sigma: float = cfg.get("sigma", 0.1)    # Target RCS (m²)
        self.L: float = cfg.get("L", 0.05)           # Sidelobe loss

        # --- Radar structure (Paper Section 2.1, Section 4) ---------------
        self.M: int = cfg.get("M", 10)               # Number of subbands
        self.K: int = cfg.get("K", 4)                 # Sub-pulses per pulse
        # PyYAML can load e.g. 10.0e9 as str; coerce to float (see YAML scientific-notation parsing)
        self.f_low: float = float(cfg.get("f_low", 10e9))    # Band lower bound (Hz)
        self.f_high: float = float(cfg.get("f_high", 11e9))  # Band upper bound (Hz)
        self.B: float = self.f_high - self.f_low             # Total bandwidth
        self.delta_f: float = self.B / self.M           # Subband width

        self.num_perms: int = len(PERMUTATION_TABLE)   # 24
        self.state_dim: int = self.M * self.num_perms  # 240

        # --- Episode parameters -------------------------------------------
        self.max_pulses: int = cfg.get("max_pulses", 10_000)

        # --- Observation history for GRU input ----------------------------
        self.history_len: int = cfg.get("history_len", 10)

        # --- Pre-compute base JSR (constant across episode) ---------------
        # Paper Equation 5 (JSR at radar) then Eq.15 (reward): r_t = JSR_base × Num.
        # With Table 1: JSR_base = (A_J² × h_j × L) / (A_R² × h_t² × σ); τ_R = τ_J cancels.
        self.jsr_base: float = (
            (self.A_J ** 2 * self.h_j * self.L)
            / (self.A_R ** 2 * self.h_t ** 2 * self.sigma)
        )
        # Theoretical max (for reference): one pulse = jsr_base × K; episode = max_pulses × jsr_base × K.

        # --- Gymnasium spaces ---------------------------------------------
        self.observation_space = spaces.Discrete(self.state_dim)
        self.action_space = spaces.Discrete(self.state_dim)

        # --- Episode state (set in reset) ---------------------------------
        self._current_state: int = 0
        self._pulse_count: int = 0
        self._total_matches: int = 0
        self._total_full_hits: int = 0
        self._total_subband_hits: int = 0
        self._history: list = []

        # --- Frequency generator (prompts 06, 9) --------------------------
        gen_cfg = {"radar": cfg}
        self._generator = FrequencyGenerator(
            config=gen_cfg,
            state_dim=self.state_dim,
            rng=None,
        )
        self._reset_generator_on_episode: bool = cfg.get(
            "reset_generator_on_episode", True
        )

    # ------------------------------------------------------------------
    # Radar frequency generation
    # ------------------------------------------------------------------

    def _generate_next_state(self, prev_state: Optional[int] = None) -> int:
        """Select the radar's next frequency index via FrequencyGenerator.

        Returns:
            Frequency index in [0, state_dim).
        """
        return self._generator.next(
            prev_state if prev_state is not None else self._current_state
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[int, dict]:
        """Initialize a new episode (pulse train).

        Returns the first radar observation s_0 and an info dict.
        """
        super().reset(seed=seed)
        if self._reset_generator_on_episode:
            start_index = options.get("start_index") if options else None
            self._generator.reset(seed=seed, start_index=start_index)

        self._current_state = self._generate_next_state(None)
        self._pulse_count = 0
        self._total_matches = 0
        self._total_full_hits = 0
        self._total_subband_hits = 0
        self._history = [self._current_state]

        info = self._build_info(num_matches=0)
        return self._current_state, info

    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """Execute one MDP step.

        MDP flow (Paper Section 3.2, Figure 4, Algorithm 1 lines 4–10):
            1. s_t = current radar frequency (already observed by jammer).
            2. a_t = ``action`` (jammer's predicted next frequency).
            3. s_{t+1} = new radar frequency (generated now).
            4. Num = position-wise sub-pulse matches between a_t and s_{t+1}.
            5. reward = JSR_base × Num.

        Args:
            action: Jammer frequency index a_t ∈ {0, ..., 239}.

        Returns:
            observation: Next radar state s_{t+1}.
            reward:      JSR_base × Num.
            terminated:  True when pulse count reaches max_pulses.
            truncated:   Always False (no external truncation).
            info:        Logging dict with hit rate, match count, etc.
        """
        next_state = self._generate_next_state(self._current_state)

        num_matches = count_subpulse_matches(
            action, next_state, self.num_perms)
        # Reward = JSR coefficient (from config, Eq.5) × Num: linear scale; episode total = sum of these.
        reward = self.jsr_base * num_matches

        action_subband = action // self.num_perms
        radar_subband = next_state // self.num_perms
        subband_hit = int(action_subband == radar_subband)

        self._pulse_count += 1
        self._total_matches += num_matches
        self._total_subband_hits += subband_hit
        if num_matches == self.K:
            self._total_full_hits += 1

        self._current_state = next_state
        self._history.append(next_state)
        if len(self._history) > self.history_len:
            self._history.pop(0)

        terminated = self._pulse_count >= self.max_pulses
        info = self._build_info(num_matches=num_matches)

        return next_state, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_info(self, num_matches: int) -> dict:
        """Assemble the info dict returned by reset/step."""
        pc = max(self._pulse_count, 1)
        hit_rate = self._total_full_hits / pc
        subband_rate = self._total_subband_hits / pc
        avg_match = self._total_matches / pc
        return {
            "pulse": self._pulse_count,
            "num_matches": num_matches,
            "hit_rate": hit_rate,
            "subband_rate": subband_rate,
            "avg_match": avg_match,
            "total_matches": self._total_matches,
            "total_full_hits": self._total_full_hits,
            "total_subband_hits": self._total_subband_hits,
            "jsr_base": self.jsr_base,
        }

    def get_history(self) -> list:
        """Return the recent observation history for GRU sequence input.

        The agent can call this to obtain the last ``history_len``
        observations as a list of integer indices.
        """
        return list(self._history)

    def decode_index(self, index: int) -> Tuple[int, tuple, tuple]:
        """Public wrapper around :func:`index_to_subpulses`."""
        return index_to_subpulses(
            index,
            num_subbands=self.M,
            num_perms=self.num_perms,
            f_low=self.f_low,
            delta_f=self.delta_f,
            k=self.K,
        )

    def encode_index(self, subband_id: int, perm: tuple) -> int:
        """Public wrapper around :func:`subpulses_to_index`."""
        return subpulses_to_index(subband_id, perm, self.num_perms)

    def get_transition_matrix(self):
        """Return a copy of the generator's Markov P if mode is markov/markov_subband, else None."""
        return self._generator.get_transition_matrix()
