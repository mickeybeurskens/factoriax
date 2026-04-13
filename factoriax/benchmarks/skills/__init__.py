"""Benchmark skill wrappers for FactoriaX.

Each skill is a gymnax-compatible environment that wraps
:class:`~factoriax.envs.factoriax_env.FactoriaXEnv`, computes its own
reward from the game state, and exposes the standard ``step()``
interface. Researchers interact with skills exactly like any gymnax
environment.

Shared reward utilities live here so the individual skill modules
stay focused on their wrapper logic and level generators.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.constants import MINEABLE_BLOCKS, MachineType
from factoriax.state import EnvState


def count_miners_on_ore(state: EnvState) -> jax.Array:
    """Count active miners sitting on ore tiles.

    Scans the entity arrays and cross-references each miner's
    position with the terrain map. Inactive entities (``ent_y < 0``)
    are masked out. Fully vectorized, JIT and vmap compatible.

    Args:
        state: Current environment state.

    Returns:
        Scalar int32 count of miners on mineable blocks.
    """
    active = state.ent_y >= 0
    is_miner = state.ent_type == MachineType.MINER
    y = jnp.clip(state.ent_y, 0)
    x = jnp.clip(state.ent_x, 0)
    tile = state.map[y, x]
    on_ore = jnp.isin(tile, MINEABLE_BLOCKS)
    return jnp.sum(active & is_miner & on_ore)


def count_fueled_miners_on_ore(state: EnvState) -> jax.Array:
    """Count active fueled miners sitting on ore tiles.

    Same as :func:`count_miners_on_ore` but additionally requires
    ``ent_fuel > 0``.

    Args:
        state: Current environment state.

    Returns:
        Scalar int32 count of fueled miners on mineable blocks.
    """
    active = state.ent_y >= 0
    is_miner = state.ent_type == MachineType.MINER
    fueled = state.ent_fuel > 0
    y = jnp.clip(state.ent_y, 0)
    x = jnp.clip(state.ent_x, 0)
    tile = state.map[y, x]
    on_ore = jnp.isin(tile, MINEABLE_BLOCKS)
    return jnp.sum(active & is_miner & fueled & on_ore)
