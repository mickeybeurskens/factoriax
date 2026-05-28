"""Online observation normalization for PPO baselines."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class RunningStats:
    """Welford online mean/variance tracker for observation normalization.

    Attributes:
        mean: Running mean of shape ``(obs_dim,)``.
        var: Running variance of shape ``(obs_dim,)``.
        count: Total number of samples seen.
    """

    mean: jax.Array
    var: jax.Array
    count: jax.Array


def init_running_stats(obs_dim: int) -> RunningStats:
    """Create zero-initialized running statistics.

    Args:
        obs_dim: Dimensionality of the observation vector.

    Returns:
        RunningStats with zero mean, unit variance, and zero count.
    """
    return RunningStats(
        mean=jnp.zeros(obs_dim, dtype=jnp.float32),
        var=jnp.ones(obs_dim, dtype=jnp.float32),
        count=jnp.array(0, dtype=jnp.int32),
    )


def update_running_stats(stats: RunningStats, batch: jax.Array) -> RunningStats:
    """Welford parallel batch update of running mean and variance.

    Args:
        stats: Current running statistics.
        batch: New observations of shape ``(N, obs_dim)``.

    Returns:
        Updated RunningStats.
    """
    n = batch.shape[0]
    batch_mean = batch.mean(axis=0)
    batch_var = batch.var(axis=0)
    total = stats.count + n
    delta = batch_mean - stats.mean
    new_mean = stats.mean + delta * (n / total)
    new_var = (
        stats.var * stats.count + batch_var * n + delta**2 * stats.count * n / total
    ) / total
    return RunningStats(mean=new_mean, var=new_var, count=total)


def normalize_obs(stats: RunningStats, obs: jax.Array) -> jax.Array:
    """Normalize observations with running statistics, clipped to [-10, 10].

    Args:
        stats: Running observation statistics.
        obs: Raw observation array of any batch shape.

    Returns:
        Normalized observation array of the same shape.
    """
    return jnp.clip((obs - stats.mean) / jnp.sqrt(stats.var + 1e-8), -10.0, 10.0)
