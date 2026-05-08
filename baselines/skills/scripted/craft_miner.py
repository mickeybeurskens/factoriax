"""Scripted baseline for the L.3 craft_miner skill.

The level pre-loads ``IRON_PLATE`` + ``WIRE`` into the player's
inventory and exposes ``CRAFT_MINER`` directly under the per-level
mask. The scripted policy presses ``CRAFT_MINER`` on every tick — the
recipe consumes the ingredients on the first successful press and
the achievement (one ``MINER`` in inventory) latches immediately.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.constants import Action
from factoriax.state import EnvParams, EnvState


def craft_miner_policy(state: EnvState, params: EnvParams) -> jax.Array:
    """Always emit ``CRAFT_MINER``.

    Intentionally trivial. Pressing ``CRAFT_MINER`` after the recipe
    has already fired is a NOOP (no ingredients), which is fine — the
    achievement unlocks on the first successful craft tick.

    Args:
        state: Current environment state (unused).
        params: Environment parameters (unused).

    Returns:
        JAX int32 ``CRAFT_MINER`` action.
    """
    del state, params
    out: jax.Array = jnp.asarray(int(Action.CRAFT_MINER), dtype=jnp.int32)
    return out
