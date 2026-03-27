"""Transition storage and GAE computation for PPO baselines."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Transition(NamedTuple):
    """One (s, a, log_pi, V, r, done) tuple per environment step.

    Attributes:
        obs: Raw (un-normalized) observation of shape ``(obs_dim,)``.
        action: Chosen action index.
        log_prob: Log-probability under the behavior policy.
        value: Value estimate from the value head.
        reward: Received reward.
        done: Whether the episode ended at this step.
    """

    obs: jax.Array
    action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    last_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute generalized advantage estimates (GAE) and value targets.

    Scans backwards over the time dimension so the entire computation
    is a single ``jax.lax.scan`` call.

    Args:
        rewards: Shape ``(T, N)``.
        values: Shape ``(T, N)``.
        dones: Terminal flags, shape ``(T, N)``.
        last_value: Bootstrap value after the rollout, shape ``(N,)``.
        gamma: Discount factor.
        gae_lambda: GAE smoothing parameter.

    Returns:
        Tuple ``(advantages, returns)`` each with shape ``(T, N)``.
    """
    not_done = 1.0 - dones.astype(jnp.float32)
    next_values = jnp.concatenate([values[1:], last_value[None]], axis=0)

    def _step(
        gae: jax.Array,
        xs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        r, v, nd, nv = xs
        delta = r + gamma * nv * nd - v
        gae = delta + gamma * gae_lambda * nd * gae
        return gae, gae

    _, advantages = jax.lax.scan(
        _step,
        jnp.zeros_like(last_value),
        (rewards[::-1], values[::-1], not_done[::-1], next_values[::-1]),
    )
    advantages = advantages[::-1]
    return advantages, advantages + values
