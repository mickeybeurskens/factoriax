"""Regression test for the rocket-benchmark coal-patch resource cap.

The advanced factory agent feeds a coal trunk that smelts every
plate type and refines refractories. A 2520-coal patch (the old
default of 280/tile × 9 tiles) starves the trunk halfway through
the rocket chain. The coal patch carries ~10x the iron/copper
patch capacity so the trunk can run unattended for a full run.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.benchmarks.rocket import build_rocket_level
from factoriax.constants import BLOCK_MAX_RESOURCES, BlockType
from factoriax.levels import build_state
from factoriax.state import EnvParams


def test_coal_patch_holds_ten_times_other_ores() -> None:
    """Coal tiles carry 10x the per-tile resource of every other ore."""
    level = build_rocket_level()
    params = EnvParams(map_width=32, map_height=32, num_players=1)
    state = build_state(level, params)

    block_map = state.map
    resources = state.block_resources

    coal_tiles = block_map == int(BlockType.COAL)
    iron_tiles = block_map == int(BlockType.IRON)

    assert int(coal_tiles.sum()) == 9, "Rocket benchmark expects a 3x3 coal patch."
    assert int(iron_tiles.sum()) == 9, "Rocket benchmark expects a 3x3 iron patch."

    coal_per_tile = int(jnp.unique(resources[coal_tiles])[0])
    iron_per_tile = int(jnp.unique(resources[iron_tiles])[0])

    assert iron_per_tile == 280
    assert coal_per_tile == 2800
    assert coal_per_tile == 10 * iron_per_tile


def test_coal_capacity_fits_within_block_max() -> None:
    """The new per-tile coal value must fit under ``BLOCK_MAX_RESOURCES``.

    Ore-tile observations are normalised by ``BLOCK_MAX_RESOURCES``.
    Letting any patch exceed it would push the obs channel past 1.0
    and break the ``Box[0, 1)`` invariant that
    :class:`FactoriaXEnv.observation_space` declares.
    """
    level = build_rocket_level()
    params = EnvParams(map_width=32, map_height=32, num_players=1)
    state = build_state(level, params)

    assert int(state.block_resources.max()) <= BLOCK_MAX_RESOURCES
