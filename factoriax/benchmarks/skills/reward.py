"""Time-discounted achievement reward for skills training.

Mirrors the benchmark's scoring formula
``solved * (max_timesteps - timesteps_used + 1) / max_timesteps`` —
the policy gets ``1.0`` for unlocking a skill on tick 1 and a small
fraction for unlocking on the final tick. This trains policies that
solve quickly rather than ones that stall and then trigger the
condition right before the budget runs out.

Why this lives next to the benchmark, not in :mod:`factoriax.rewards`:
the reward is *specific* to the skills curriculum's scoring shape.
Rocket and any future benchmark with different scoring should not
share this function. ``factoriax.rewards.achievement_reward`` remains
the generic per-bit-weighted shim for benchmarks that don't care about
solve speed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.state import EnvParams, EnvState


def skills_reward(
    prev_state: EnvState,
    new_state: EnvState,
    params: EnvParams,
    target_bit: int | None = None,
) -> jax.Array:
    """Sparse, time-discounted reward for newly-unlocked skill bits.

    Computes ``(remaining_steps + 1) / max_timesteps`` once per call,
    multiplies by the count of bits newly latched this step, and
    returns the result as a float32 scalar.

    When ``target_bit`` is given, only that single bit's unlock counts
    — needed to keep training reward aligned with
    :meth:`SkillsBenchmark.score`, which credits exclusively the
    level's own achievement. Otherwise the agent harvests free reward
    from incidentally-triggered bits (e.g., place_miner spawns
    auto-mining miners that fire the mine bit) and learns to chase
    those instead of the actual target.

    Args:
        prev_state: EnvState immediately before the step. Provides the
            previous achievement mask for delta computation.
        new_state: EnvState immediately after the step. Provides the
            newly-latched mask and the current ``timestep``.
        params: Environment parameters. Reads ``max_timesteps`` for
            the time-discount denominator.
        target_bit: Index of the achievement bit to credit. When
            ``None``, every newly-unlocked bit is counted.

    Returns:
        Scalar float32 reward. Zero when the relevant bit (or any
        bit, in the unfiltered mode) didn't newly unlock this step.
    """
    newly_unlocked = new_state.achievements_unlocked & ~prev_state.achievements_unlocked
    if target_bit is None:
        count = jnp.sum(newly_unlocked.astype(jnp.float32))
    else:
        count = newly_unlocked[int(target_bit)].astype(jnp.float32)
    max_t = jnp.float32(params.max_timesteps)
    remaining = max_t - jnp.asarray(new_state.timestep, dtype=jnp.float32) + 1.0
    discount = remaining / max_t
    reward: jax.Array = count * discount
    return reward
