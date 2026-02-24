"""
FrequencyGenerator — Radar frequency index generation for RadarEnv
==================================================================
Supports uniform, periodic, LCG, and Markov (band-limited or subband
persistence) modes for thesis ablation studies.

Paper: Xia et al., "GA-Dueling DQN Jamming Decision-Making Method
       for Intra-Pulse Frequency Agile Radar", Sensors 2024.
Prompts: 10.md, 11.md — Markov subband persistence (70/30), absorbing prevention.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# State space: 240 = 10 subbands × 24 permutations
NUM_SUBBANDS = 10
PERMS_PER_SUBBAND = 24


def _subband_of(index: int) -> int:
    """Return subband index (0–9) for state index 0–239."""
    return index // PERMS_PER_SUBBAND


def _indices_in_subband(k: int) -> range:
    """Return range of state indices belonging to subband k."""
    return range(k * PERMS_PER_SUBBAND, (k + 1) * PERMS_PER_SUBBAND)


def _apply_absorbing_prevention(P: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Add epsilon to all cells and renormalize rows. Ensures irreducibility."""
    P = P + epsilon
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / row_sums
    return P


class FrequencyGenerator:
    """Produces the next radar frequency index given config and optional
    previous state. Used by RadarEnv._generate_next_state().

    Modes:
        uniform:        i.i.d. uniform over [0, state_dim). History not useful.
        periodic:       Cycle through periodic_sequence. History useful.
        lcg:            Linear Congruential Generator. History useful.
        markov:         Band-limited transition matrix P. History partially useful.
        markov_subband: 70/30 subband persistence (11.md). History partially useful.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        state_dim: int = 240,
        rng: Optional[np.random.Generator] = None,
    ):
        self.config = config
        self.state_dim = state_dim
        self._rng = rng if rng is not None else np.random.default_rng()
        self._owns_rng = rng is None

        radar_cfg = config.get("radar", self.config)
        self._mode = str(radar_cfg.get("generator_mode", "uniform")).lower()
        self._start_index: int = int(radar_cfg.get("start_index", 0))

        # --- Mode-specific state ---
        self._periodic_idx: int = 0
        self._lcg_state: int = 0
        self._markov_P: Optional[np.ndarray] = None
        self._markov_cumsum: Optional[np.ndarray] = None

        self._init_mode()

    def _init_mode(self) -> None:
        radar_cfg = self.config.get("radar", self.config)
        if self._mode == "periodic":
            seq = radar_cfg.get("periodic_sequence")
            if seq is None:
                seq = list(range(0, self.state_dim, max(1, self.state_dim // 10)))
            self._periodic_sequence = [int(x) % self.state_dim for x in seq]
            self._periodic_len = len(self._periodic_sequence)
        elif self._mode == "lcg":
            params = radar_cfg.get("lcg_params", {})
            self._lcg_a = int(params.get("a", 1103515245))
            self._lcg_c = int(params.get("c", 12345))
            self._lcg_m = int(params.get("m", 2**31))
        elif self._mode == "markov":
            path = radar_cfg.get("markov_transition_path")
            if path:
                p = Path(path)
                if p.is_absolute() or not p.exists():
                    full_path = p
                else:
                    full_path = Path(__file__).resolve().parent.parent / p
                self._markov_P = np.load(str(full_path)).astype(np.float64)
            else:
                params = radar_cfg.get("markov_params", {})
                delta = int(params.get("delta_limit", 20))
                sparsity = float(params.get("sparsity", 0.1))
                self._markov_P = self._build_band_limited_P(delta, sparsity)
            self._markov_P = _apply_absorbing_prevention(
                self._markov_P, epsilon=1e-6
            )
            self._markov_cumsum = np.cumsum(self._markov_P, axis=1)
        elif self._mode == "markov_subband":
            params = radar_cfg.get("markov_subband_params", {})
            stay_prob = float(params.get("stay_prob", 0.7))
            sparsity = float(params.get("sparsity", 0.1))
            sigma = float(params.get("sigma_subband", 1.5))
            self._markov_P = self._build_subband_persistence_P(
                stay_prob, sparsity, sigma
            )
            self._markov_P = _apply_absorbing_prevention(
                self._markov_P, epsilon=1e-6
            )
            self._markov_cumsum = np.cumsum(self._markov_P, axis=1)

    def _build_band_limited_P(
        self, delta_limit: int, sparsity: float
    ) -> np.ndarray:
        """Build band-limited transition matrix: from i, nonzero only for
        j in [max(0, i-delta), min(state_dim-1, i+delta)].
        sparsity: fraction of band that is nonzero (density).
        """
        n = self.state_dim
        P = np.zeros((n, n))
        for i in range(n):
            lo = max(0, i - delta_limit)
            hi = min(n - 1, i + delta_limit)
            width = hi - lo + 1
            nnz = max(1, int(width * sparsity))
            indices = self._rng.choice(
                np.arange(lo, hi + 1), size=min(nnz, width), replace=False
            )
            probs = self._rng.dirichlet(np.ones(len(indices)))
            for idx, p in zip(indices, probs):
                P[i, idx] = p
            row_sum = P[i].sum()
            if row_sum > 0:
                P[i] /= row_sum
            else:
                P[i, i] = 1.0
        return P

    def _build_subband_persistence_P(
        self, stay_prob: float, sparsity: float, sigma_subband: float
    ) -> np.ndarray:
        """Build 70/30 subband persistence matrix (11.md).

        Stay 70%: same subband. Jump 30%: other subbands, Gaussian over
        subband distance. Sparsity applied within each allocation.
        """
        n = self.state_dim
        P = np.zeros((n, n))

        for i in range(n):
            k = _subband_of(i)
            same_sb = list(_indices_in_subband(k))
            other_sb = [j for j in range(n) if _subband_of(j) != k]

            # --- Stay 70%: distribute over same subband (excl self for variety) ---
            stay_targets = [j for j in same_sb if j != i]
            if not stay_targets:
                stay_targets = [i]
            nnz_stay = max(1, int(len(stay_targets) * sparsity))
            nnz_stay = min(nnz_stay, len(stay_targets))
            chosen_stay = self._rng.choice(
                stay_targets, size=nnz_stay, replace=False
            )
            probs_stay = self._rng.dirichlet(np.ones(len(chosen_stay)))
            for idx, j in enumerate(chosen_stay):
                P[i, j] = stay_prob * probs_stay[idx]

            # --- Jump 30%: distribute over other subbands with Gaussian ---
            other_subbands = [kk for kk in range(NUM_SUBBANDS) if kk != k]
            dists = [abs(kk - k) for kk in other_subbands]
            weights = np.array(
                [np.exp(-(d**2) / (2 * sigma_subband**2)) for d in dists]
            )
            weights = weights / weights.sum()
            jump_prob_per_sb = 0.30 * weights
            for idx_sb, kk in enumerate(other_subbands):
                sb_indices = list(_indices_in_subband(kk))
                nnz_jump = max(1, int(len(sb_indices) * sparsity))
                nnz_jump = min(nnz_jump, len(sb_indices))
                chosen = self._rng.choice(
                    sb_indices, size=nnz_jump, replace=False
                )
                probs = self._rng.dirichlet(np.ones(len(chosen)))
                for idx, j in enumerate(chosen):
                    P[i, j] += jump_prob_per_sb[idx_sb] * probs[idx]

            row_sum = P[i].sum()
            if row_sum > 0:
                P[i] /= row_sum
            else:
                P[i, i] = 1.0
        return P

    def reset(self, seed: Optional[int] = None) -> None:
        """Reseed RNG (if owned) and reset mode-specific internal state."""
        if seed is not None and self._owns_rng:
            self._rng = np.random.default_rng(seed)
        self._periodic_idx = 0
        self._lcg_state = int(self._rng.integers(0, 2**31))
        # Markov: no position to reset; cumsum stays same

    def next(self, prev_state: Optional[int] = None) -> int:
        """Produce the next frequency index.

        Args:
            prev_state: Previous radar state (for lcg, markov). Can be None
                for uniform/periodic or on first call.

        Returns:
            Index in [0, state_dim).
        """
        if self._mode == "uniform":
            return int(self._rng.integers(0, self.state_dim))
        if self._mode == "periodic":
            idx = self._periodic_sequence[self._periodic_idx % self._periodic_len]
            self._periodic_idx += 1
            return int(idx) % self.state_dim
        if self._mode == "lcg":
            self._lcg_state = (
                self._lcg_a * self._lcg_state + self._lcg_c
            ) % self._lcg_m
            return int(self._lcg_state % self.state_dim)
        if self._mode in ("markov", "markov_subband"):
            i = (
                prev_state
                if prev_state is not None
                else self._start_index
            )
            i = max(0, min(self.state_dim - 1, i))
            u = self._rng.random()
            row = self._markov_cumsum[i]
            j = int(np.searchsorted(row, u))
            return min(j, self.state_dim - 1)
        return int(self._rng.integers(0, self.state_dim))

    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """Return a copy of the transition matrix P for markov/markov_subband."""
        if self._markov_P is not None:
            return self._markov_P.copy()
        return None
