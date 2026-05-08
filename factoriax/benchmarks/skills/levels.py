"""Level builders for the skills challenge curriculum.

Each builder is a deterministic function of a ``seed`` argument
returning ``(Level, EnvParams, frozenset[int])`` — level geometry,
runtime parameters, and the per-level action mask. Seed ``0`` is the
canonical evaluation layout used by :class:`SkillsBenchmark`; arbitrary
seeds are available for training-time variation.

Achievement conditions are designed to be **layout-invariant** so the
seed only varies spawn / source positions / distractors, not the target
the agent must hit. See :mod:`factoriax.benchmarks.skills.achievements`.
"""

from __future__ import annotations

import numpy as np

from factoriax.benchmarks.skills.achievements import (
    MINE_BLOCKED_ACTIONS,
    NAVIGATE_BLOCKED_ACTIONS,
)
from factoriax.constants import BlockType
from factoriax.levels import Level, LevelBuilder
from factoriax.state import EnvParams

_NAVIGATE_MAP_SIZE: int = 5
_NAVIGATE_MAX_TIMESTEPS: int = 200

_MINE_MAP_SIZE: int = 5
_MINE_MAX_TIMESTEPS: int = 200
_MINE_ORE_FRACTION: float = 0.4
_MINE_RESOURCES_PER_TILE: int = 3
_MINE_ORE_BLOCKS: tuple[BlockType, ...] = (
    BlockType.COAL,
    BlockType.IRON,
    BlockType.COPPER,
)


def build_navigate_level(
    seed: int = 0,
) -> tuple[Level, EnvParams, frozenset[int]]:
    """L.1 — walk to the bottom-right corner of a 5x5 grass map.

    The goal tile is fixed at ``(map_size - 1, map_size - 1)`` so the
    achievement function (``SKILL_NAVIGATE`` — bit 0) is a pure
    state-only check with no per-level metadata. Only the spawn varies
    with *seed*; that's enough to prevent overfitting to a single
    "always step right then down" trajectory without needing to plumb
    a goal coordinate through ``EnvState``.

    Action mask: only ``MOVE_*`` and ``NOOP`` are exposed. Every other
    action is blocked so the agent cannot mine, place, or craft while
    learning to walk.

    Args:
        seed: Numpy RNG seed for spawn position. Default ``0`` is the
            canonical seed used by :class:`SkillsBenchmark` for
            evaluation. Any other seed produces a deterministic spawn.

    Returns:
        Tuple of ``(level, params, blocked_actions)``. Pass
        ``blocked_actions`` to :class:`ActionMaskWrapper` (or rely on
        :class:`BenchmarkRunner` to do it via the per-level field on
        :class:`BenchmarkLevel`).
    """
    rng = np.random.default_rng(seed)
    builder = LevelBuilder(_NAVIGATE_MAP_SIZE, _NAVIGATE_MAP_SIZE)
    goal = (_NAVIGATE_MAP_SIZE - 1, _NAVIGATE_MAP_SIZE - 1)
    while True:
        spawn_x = int(rng.integers(0, _NAVIGATE_MAP_SIZE))
        spawn_y = int(rng.integers(0, _NAVIGATE_MAP_SIZE))
        if (spawn_x, spawn_y) != goal:
            break
    builder.set_player_position(spawn_x, spawn_y)
    level = builder.build(f"skills_navigate_seed{seed}")
    params = EnvParams(
        map_width=_NAVIGATE_MAP_SIZE,
        map_height=_NAVIGATE_MAP_SIZE,
        num_players=1,
        max_timesteps=_NAVIGATE_MAX_TIMESTEPS,
    )
    return level, params, NAVIGATE_BLOCKED_ACTIONS


def build_mine_level(
    seed: int = 0,
) -> tuple[Level, EnvParams, frozenset[int]]:
    """L.2 — extract ore from a 5x5 map sprinkled with mineable tiles.

    The map gets ~40% ore tiles drawn uniformly from
    ``{COAL, IRON, COPPER}`` (the three ore types the player can hold
    interchangeably for the achievement). Each ore tile carries 3
    resources — generous for a single mine action but small enough
    that the agent can't loiter forever scoring multiple unlocks. The
    player spawns at the centre tile, which is forced to be grass so
    a fresh ``MINE`` from spawn does nothing — the agent must walk
    onto an ore tile first.

    The achievement (``SKILL_MINE`` — bit 1) fires when player 0's
    inventory contains at least one ore item of any mineable type. The
    layout varies with *seed*; the achievement does not depend on
    layout.

    Action mask: ``MOVE_*``, ``MINE``, ``NOOP``. No facing, no
    placement, no crafting — pure walk-and-mine.

    Args:
        seed: Numpy RNG seed for ore layout. Default ``0`` is the
            canonical seed used by :class:`SkillsBenchmark`.

    Returns:
        Tuple of ``(level, params, blocked_actions)``.
    """
    rng = np.random.default_rng(seed)
    map_size = _MINE_MAP_SIZE
    builder = LevelBuilder(map_size, map_size)
    centre = map_size // 2

    n_tiles = map_size * map_size
    n_ore = int(n_tiles * _MINE_ORE_FRACTION)
    ore_indices = rng.choice(n_tiles, size=n_ore, replace=False)
    for idx in ore_indices:
        y, x = divmod(int(idx), map_size)
        if (x, y) == (centre, centre):
            continue
        ore_block = _MINE_ORE_BLOCKS[int(rng.integers(0, len(_MINE_ORE_BLOCKS)))]
        builder.fill_rect(x, y, 1, 1, ore_block, resources=_MINE_RESOURCES_PER_TILE)

    builder.set_player_position(centre, centre)
    level = builder.build(f"skills_mine_seed{seed}")
    params = EnvParams(
        map_width=map_size,
        map_height=map_size,
        num_players=1,
        max_timesteps=_MINE_MAX_TIMESTEPS,
    )
    return level, params, MINE_BLOCKED_ACTIONS
