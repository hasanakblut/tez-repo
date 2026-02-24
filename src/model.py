"""
GA-Dueling DQN Network Architecture
====================================
GRU-Attention-based Dueling Deep Q Network for cognitive jamming
decision-making against intra-pulse frequency agile radar.

Paper: Xia et al., "GA-Dueling DQN Jamming Decision-Making Method
       for Intra-Pulse Frequency Agile Radar", Sensors 2024.

Modules:
    NoisyLinear   – Linear layer with factorized Gaussian noise [Fortunato 2017]
    GADuelingDQN  – Full network: GRU → MultiHead Attention → FC → Dueling Heads
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# NoisyLinear — Factorized Gaussian Noise (Paper Section 3.3, Ref [28])
# ---------------------------------------------------------------------------
class NoisyLinear(nn.Module):
    """Linear layer whose weights and biases are perturbed by learned
    factorized Gaussian noise, encouraging exploration without ε-greedy
    alone.

    Uses the *factorized* variant for computational efficiency:
        w = μ_w + σ_w ⊙ (f(ε_i) ⊗ f(ε_j))
        b = μ_b + σ_b ⊙ f(ε_j)
    where f(x) = sgn(x)·√|x|.
    """

    def __init__(self, in_features: int, out_features: int,
                 sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features))
        self.register_buffer(
            "weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(
            self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(
            self.sigma_init / math.sqrt(self.in_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"sigma_init={self.sigma_init}")


# ---------------------------------------------------------------------------
# GA-Dueling DQN — Main Network (Paper Section 3.3, Figures 6-8)
# ---------------------------------------------------------------------------
class GADuelingDQN(nn.Module):
    """GRU-Attention-based Dueling Deep Q Network.

    Data flow (see Paper Figure 7):
        indices (batch, seq_len) ──► Embedding ──► (batch, seq_len, embed_dim)
                                                         │
                                                        GRU
                                                         │
                                              Multi-Head Self-Attention
                                                         │
                                             NoisyLinear FC1 + LayerNorm + ReLU
                                                         │
                                             NoisyLinear FC2 + LayerNorm + ReLU
                                                         │
                                ┌────────────────────────┴────────────────────────┐
                            V(s) stream                              A(s,a) stream
                          NoisyLinear→1                             NoisyLinear→240
                                └────────────────────┬───────────────────────────┘
                                       Q = V + (A − mean(A))

    Args:
        state_dim:  Number of discrete states (default 240).
        action_dim: Number of possible actions (default 240).
        embed_dim:  Embedding dimension; GRU input size (default 64).
        hidden_dim: GRU hidden size (default 128).
        num_heads:  Attention heads (default 8).
        fc_dim:     Width of FC1 / FC2 (default 64).
        sigma_init: Initial noise scale for NoisyLinear layers (default 0.5).
    """

    def __init__(
        self,
        state_dim: int = 240,
        action_dim: int = 240,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 8,
        fc_dim: int = 64,
        sigma_init: float = 0.5,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # --- Embedding: indices → dense vectors (prompts 06, 9) ------------
        self.embedding = nn.Embedding(
            num_embeddings=state_dim,
            embedding_dim=embed_dim,
        )

        # --- Sequence Processor (NN Module, Paper Figure 7) ---------------

        # GRU: input is embedded sequence (batch, seq_len, embed_dim).
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Multi-Head Self-Attention: learns correlations between temporal
        # features produced by the GRU.
        # Paper Section 3.3: "an eight-headed multi-head self-attention module
        # ... leverages the attention mechanism to learn correlations between
        # different input features."
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # --- Fully-Connected Feature Extractor ----------------------------

        # FC1: NoisyLinear 128→64 + LayerNorm  (Paper Table 2, Figure 8)
        self.fc1 = NoisyLinear(hidden_dim, fc_dim, sigma_init=sigma_init)
        self.ln1 = nn.LayerNorm(fc_dim)

        # FC2: NoisyLinear 64→64 + LayerNorm   (Paper Table 2, Figure 8)
        self.fc2 = NoisyLinear(fc_dim, fc_dim, sigma_init=sigma_init)
        self.ln2 = nn.LayerNorm(fc_dim)

        # --- Dueling Heads (Paper Section 3.3, Equation 18, Figure 8) -----

        # State Value stream  V(s):  64 → 1
        self.value_stream = NoisyLinear(fc_dim, 1, sigma_init=sigma_init)

        # Advantage stream    A(s,a): 64 → action_dim
        self.advantage_stream = NoisyLinear(
            fc_dim, action_dim, sigma_init=sigma_init
        )

    # ----- forward --------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:      State index sequence. Shape (batch, seq_len), dtype long.
            hidden: GRU hidden state from the previous time step.
                    Shape (1, batch, hidden_dim) or None.

        Returns:
            q_values: Q-values for every action.  Shape (batch, action_dim).
            hidden:   Updated GRU hidden state.   Shape (1, batch, hidden_dim).
        """
        # 0) Embedding: (batch, seq_len) → (batch, seq_len, embed_dim)
        x = self.embedding(x)
        # 1) GRU — temporal encoding
        gru_out, hidden = self.gru(x, hidden)
        # gru_out: (batch, seq_len, hidden_dim)

        # 2) Multi-Head Self-Attention over the full GRU output sequence
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        # attn_out: (batch, seq_len, hidden_dim)

        # Take the representation at the last time step
        features = attn_out[:, -1, :]  # (batch, hidden_dim)

        # 3) Fully-connected feature extraction with noise + normalization
        features = F.relu(self.ln1(self.fc1(features)))
        features = F.relu(self.ln2(self.fc2(features)))

        # 4) Dueling decomposition (Paper Equation 18)
        value = self.value_stream(features)           # (batch, 1)
        advantage = self.advantage_stream(features)   # (batch, action_dim)

        # Q(s,a) = V(s) + ( A(s,a) - mean_a'(A(s,a')) )
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q_values, hidden

    # ----- utilities ------------------------------------------------------

    def reset_noise(self) -> None:
        """Re-sample noise for every NoisyLinear layer.  Called once per
        training step so that exploration noise varies across updates."""
        for module in (self.fc1, self.fc2,
                       self.value_stream, self.advantage_stream):
            module.reset_noise()

    def init_hidden(self, batch_size: int = 1,
                    device: Optional[torch.device] = None) -> torch.Tensor:
        """Return a zero-initialized GRU hidden state."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)
