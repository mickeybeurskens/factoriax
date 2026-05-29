"""Regression tests for PPO observation-normalization running statistics.

Guards the int32-overflow bug: with JAX x64 disabled an int32 sample
``count`` wraps negative at 2**31 samples (~2.1B env-steps), which
poisons the variance and collapses training to zero reward at exactly
that step. ``count`` must be float so the running total never wraps.
"""

from __future__ import annotations

import jax.numpy as jnp

from baselines.easy_rocket.ppo.normalization import (
    RunningStats,
    init_running_stats,
    normalize_obs,
    update_running_stats,
)


def test_count_initialized_as_float_not_int() -> None:
    """An integer count would wrap at 2**31; the count must be floating."""
    stats = init_running_stats(obs_dim=4)
    assert jnp.issubdtype(stats.count.dtype, jnp.floating)


def test_update_does_not_overflow_past_int32_boundary() -> None:
    """Crossing the 2**31 sample boundary keeps the stats finite and valid."""
    obs_dim = 4
    near_overflow = float(2**31 - 64)  # just below the int32 wrap point
    stats = RunningStats(
        mean=jnp.zeros(obs_dim, dtype=jnp.float32),
        var=jnp.ones(obs_dim, dtype=jnp.float32),
        count=jnp.array(near_overflow, dtype=jnp.float32),
    )
    batch = jnp.ones((1024, obs_dim), dtype=jnp.float32)  # pushes total past 2**31

    updated = update_running_stats(stats, batch)

    # Count stays positive (an int32 count would have wrapped to negative).
    assert float(updated.count) >= near_overflow
    # Variance stays finite and non-negative -> no NaN cascade.
    assert bool(jnp.all(jnp.isfinite(updated.var)))
    assert bool(jnp.all(updated.var >= 0.0))
    # Normalized observations stay finite (NaN obs was the bug's symptom).
    assert bool(jnp.all(jnp.isfinite(normalize_obs(updated, batch))))
