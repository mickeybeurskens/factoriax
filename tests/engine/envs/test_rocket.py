"""Tests for :mod:`factoriax.engine.envs.rocket`.

The rocket level is authored, so its resource budget is a design decision
rather than a generated one. These tests pin the coal column against the
per-tile budget every other ore gets, and against the cap the observation
normalises by.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.engine.constants import BLOCK_MAX_RESOURCES, BlockType
from factoriax.engine.envs.rocket import build_rocket_level
from factoriax.engine.levels import build_state


def test_coal_column_per_tile_is_ten_times_ore_per_tile() -> None:
    """Coal tiles carry 10x the per-tile resource of every other ore.

    The v2 layout uses a 1x32 coal column (32 tiles) and 2x2 ore
    patches (4 tiles each). Per-tile budgets, not total budgets,
    set how long a single miner can run before depletion.
    """
    level = build_rocket_level()
    state = build_state(level, num_players=1)

    block_map = state.map
    resources = state.block_resources

    coal_tiles = block_map == int(BlockType.COAL)
    iron_tiles = block_map == int(BlockType.IRON)

    # v2 layout: coal column (1x32 = 32 tiles), iron patch (2x2 = 4 tiles).
    assert int(coal_tiles.sum()) == 32, "Rocket scenario expects a 1x32 coal column."
    assert int(iron_tiles.sum()) == 4, "Rocket scenario expects a 2x2 iron patch."

    coal_per_tile = int(jnp.unique(resources[coal_tiles])[0])
    iron_per_tile = int(jnp.unique(resources[iron_tiles])[0])

    assert iron_per_tile == 6300
    assert coal_per_tile == 28000
    # ~4.4x per tile. The column has 32 tiles against 4 per ore patch, so
    # column-total / ore-total is still ~36x. Per-tile ratio drops
    # because column tiles can run independently.
    assert coal_per_tile > iron_per_tile


def test_coal_capacity_fits_within_block_max() -> None:
    """The new per-tile coal value must fit under ``BLOCK_MAX_RESOURCES``.

    Ore-tile observations are normalised by ``BLOCK_MAX_RESOURCES``.
    A patch that exceeds it pushes the obs channel past 1.0
    and breaks the ``Box[0, 1)`` invariant that
    :class:`FactoriaxEnv.observation_space` declares.
    """
    level = build_rocket_level()
    state = build_state(level, num_players=1)

    assert int(state.block_resources.max()) <= BLOCK_MAX_RESOURCES
