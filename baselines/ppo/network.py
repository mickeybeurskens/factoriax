"""Shared actor-critic network for PPO baselines."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class ActorCritic(nn.Module):
    """Shared-trunk MLP with separate policy and value heads.

    A single ``LayerNorm -> tanh`` block is applied after each hidden
    layer, which stabilizes training on the flat FactoriaX observation
    without requiring input normalization to warm up first.

    Attributes:
        hidden_dims: Sizes of shared hidden layers.
        num_actions: Number of discrete actions.
    """

    hidden_dims: tuple[int, ...]
    num_actions: int

    @nn.compact
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute action logits and scalar state value.

        Args:
            obs: Observation array of shape ``(..., obs_dim)``.

        Returns:
            Tuple ``(logits, value)`` with shapes
            ``(..., num_actions)`` and ``(...,)``.
        """
        x = obs.astype(jnp.float32)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
        logits = nn.Dense(self.num_actions)(x)
        value = nn.Dense(1)(x).squeeze(-1)
        return logits, value
