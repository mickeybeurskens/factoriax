"""Tests for the JAX-native easy-rocket world generator (refactor Phase 1).

Pins the invariants the host ``build_easy_rocket_level`` guaranteed — six
2x2 ore patches, one per block, non-overlapping, off the spawn ring — plus the
JAX properties the host lacked: determinism per key, variation across keys, and
vmap/jit compatibility. JIT-free assertions on small 16x16 arrays.
"""

from __future__ import annotations

import jax
import numpy as np
from jax import random

from factoriax.engine.constants import BlockType
from factoriax.engine.envs.easy_rocket import (
    _INNER_ZONE_MAX,
    _INNER_ZONE_MIN,
    _MAP_SIZE,
    _PATCH_BLOCKS,
    _SPAWN,
    _easy_rocket_terrain,
    easy_rocket,
)
from factoriax.engine.state import EnvParams

_PARAMS = EnvParams(base_resources=3000)
_ENV, _ = easy_rocket()


def _gen_state(key):
    _, state = _ENV.reset_env(key, _PARAMS)
    return state
_ORE_VALUES: tuple[int, ...] = tuple(int(b) for b in _PATCH_BLOCKS)
_TILES_PER_PATCH = 4


def _patch_tiles(world: np.ndarray, block: int) -> np.ndarray:
    """Return ``(n, 2)`` array of ``(row, col)`` tiles holding ``block``."""
    return np.argwhere(world == block)


def test_generator_places_one_2x2_patch_per_block() -> None:
    world = np.asarray(_gen_state(random.PRNGKey(0)).map)
    for block in _ORE_VALUES:
        tiles = _patch_tiles(world, block)
        assert tiles.shape[0] == _TILES_PER_PATCH
        ys, xs = tiles[:, 0], tiles[:, 1]
        assert int(ys.max() - ys.min()) == 1 and int(xs.max() - xs.min()) == 1
    # Exactly 24 ore tiles total => patches do not overlap each other.
    ore_total = sum(int((world == b).sum()) for b in _ORE_VALUES)
    assert ore_total == _TILES_PER_PATCH * len(_ORE_VALUES)


def test_generator_avoids_inner_zone_and_keeps_spawn_dirt() -> None:
    """No ore inside the 4x4 zone (2x2 spawn area + 1-cell inner ring)."""
    world = np.asarray(_gen_state(random.PRNGKey(1)).map)
    sx, sy = _SPAWN
    assert int(world[sy, sx]) == int(BlockType.DIRT)
    for block in _ORE_VALUES:
        for ty, tx in _patch_tiles(world, block):
            assert not (
                _INNER_ZONE_MIN <= int(tx) <= _INNER_ZONE_MAX
                and _INNER_ZONE_MIN <= int(ty) <= _INNER_ZONE_MAX
            )


def test_generator_avoids_outer_ring() -> None:
    """No ore on the outer 1-cell dirt ring (first/last row, first/last col)."""
    for seed in range(20):
        world = np.asarray(
            _gen_state(random.PRNGKey(seed)).map,
        )
        ore_mask = np.isin(world, _ORE_VALUES)
        assert not bool(ore_mask[0, :].any()), f"seed {seed}: ore on top row"
        assert not bool(ore_mask[-1, :].any()), f"seed {seed}: ore on bottom row"
        assert not bool(ore_mask[:, 0].any()), f"seed {seed}: ore on left col"
        assert not bool(ore_mask[:, -1].any()), f"seed {seed}: ore on right col"


def test_generator_resources_on_ore_only() -> None:
    state = _gen_state(random.PRNGKey(2))
    world = np.asarray(state.map)
    resources = np.asarray(state.block_resources)
    ore_mask = np.isin(world, _ORE_VALUES)
    assert np.all(resources[ore_mask] == 3000)
    assert np.all(resources[~ore_mask] == 0)


def test_generator_deterministic() -> None:
    a = _gen_state(random.PRNGKey(7))
    b = _gen_state(random.PRNGKey(7))
    assert np.array_equal(np.asarray(a.map), np.asarray(b.map))


def test_generator_varies_with_key() -> None:
    base = np.asarray(_gen_state(random.PRNGKey(0)).map)
    assert any(
        not np.array_equal(
            base, np.asarray(_gen_state(random.PRNGKey(s)).map)
        )
        for s in (1, 2, 3, 4, 5)
    )


def test_generator_vmaps_and_jits() -> None:
    keys = random.split(random.PRNGKey(0), 8)
    maps = jax.jit(jax.vmap(lambda k: _easy_rocket_terrain(k, _PARAMS)))(keys)
    assert maps.shape == (8, _MAP_SIZE, _MAP_SIZE)
    arr = np.asarray(maps)
    for i in range(8):
        ore_total = sum(int((arr[i] == b).sum()) for b in _ORE_VALUES)
        assert ore_total == _TILES_PER_PATCH * len(_ORE_VALUES)
