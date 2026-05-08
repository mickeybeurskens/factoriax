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

from factoriax.benchmarks.skills.achievements import NAVIGATE_BLOCKED_ACTIONS
from factoriax.levels import Level, LevelBuilder
from factoriax.state import EnvParams

_NAVIGATE_MAP_SIZE: int = 5
_NAVIGATE_MAX_TIMESTEPS: int = 200


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
