"""Shared actor-critic networks for PPO baselines.

Two network architectures are provided:

- :class:`ActorCritic` — MLP trunk for flat vector observations
  (``local_array`` or ``global_array``).
- :class:`VisionActorCritic` — CNN encoder for pixel observations
  produced by :class:`~factoriax.jax_renderer.JaxRenderer`.

Both produce ``(logits, value)`` and can be used interchangeably
by the training script.
"""

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


class VisionActorCritic(nn.Module):
    """CNN encoder with policy and value heads for pixel observations.

    Three stride-2 convolutional layers downsample the image, followed
    by a dense layer with layer normalization. The same dual-head
    structure as :class:`ActorCritic` produces logits and a value.

    For a 10x10 map at ``tile_px=8`` the input is 80x80x3. After three
    stride-2 convolutions the spatial dimensions reduce to 10x10x32,
    giving 3200 features before the dense layer.

    Attributes:
        num_actions: Number of discrete actions.
        channels: Output channels per conv layer.
        kernel_sizes: Kernel size per conv layer.
        strides: Stride per conv layer.
        dense_dim: Hidden units in the dense layer after flattening.
    """

    num_actions: int
    channels: tuple[int, ...] = (16, 32, 32)
    kernel_sizes: tuple[int, ...] = (3, 3, 3)
    strides: tuple[int, ...] = (2, 2, 2)
    dense_dim: int = 256

    @nn.compact
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute action logits and scalar state value from pixels.

        Args:
            obs: RGB image of shape ``(H, W, 3)`` or batched
                ``(B, H, W, 3)``, uint8 or float32.

        Returns:
            Tuple ``(logits, value)`` with shapes
            ``(..., num_actions)`` and ``(...,)``.
        """
        x = obs.astype(jnp.float32) / 255.0
        for ch, ks, st in zip(self.channels, self.kernel_sizes, self.strides):
            x = nn.Conv(ch, (ks, ks), strides=(st, st))(x)
            x = nn.relu(x)
        x = x.reshape((*x.shape[:-3], -1))
        x = nn.Dense(self.dense_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        logits = nn.Dense(self.num_actions)(x)
        value = nn.Dense(1)(x).squeeze(-1)
        return logits, value
