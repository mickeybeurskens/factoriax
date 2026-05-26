"""Level builders for the skills challenge curriculum.

Each builder is a deterministic function of a ``seed`` argument
returning ``(Level, EnvParams, frozenset[int])`` — level geometry,
runtime parameters, and the per-level action mask. Seed ``0`` is the
canonical evaluation layout used by :class:`SkillsBenchmark`; arbitrary
seeds are available for training-time variation.

Achievement conditions are designed to be **layout-invariant** so the
seed only varies spawn / source positions / distractors, not the target
the agent must hit. See :mod:`factoriax.scenarios.skills.achievements`.
"""

from __future__ import annotations

import numpy as np

from factoriax.engine.constants import BlockType, Direction, ItemType, Machine
from factoriax.engine.levels import Level, LevelBuilder
from factoriax.engine.state import EnvParams
from factoriax.scenarios.skills.achievements import (
    CRAFT_MINER_BLOCKED_ACTIONS,
    MINE_BLOCKED_ACTIONS,
    NAVIGATE_BLOCKED_ACTIONS,
    PLACE_MINER_BLOCKED_ACTIONS,
)

_NAVIGATE_MAP_SIZE: int = 5
_NAVIGATE_MAX_TIMESTEPS: int = 200

_MINE_MAP_SIZE: int = 5
_MINE_MAX_TIMESTEPS: int = 200
_MINE_NUM_ORE_TILES: int = 5
_MINE_RESOURCES_PER_TILE: int = 1
_MINE_ORE_BLOCKS: tuple[BlockType, ...] = (
    BlockType.COAL,
    BlockType.IRON,
    BlockType.COPPER,
)

_CRAFT_MINER_MAP_SIZE: int = 5
_CRAFT_MINER_MAX_TIMESTEPS: int = 400
# Player starts with WIRE in inventory; the lone pallet holds the
# IRON_PLATE the recipe also needs. This shrinks the solve sequence to
# walk → face → WITHDRAW → CRAFT_MINER (4 specific choices) so PPO can
# bootstrap from sparse reward in a reasonable training budget — see
# the trainer notes in baselines/skills/train_ppo.py.
_CRAFT_MINER_INVENTORY_ITEM: int = int(ItemType.WIRE)
_CRAFT_MINER_PALLET_ITEM: int = int(ItemType.IRON_PLATE)

_PLACE_MINER_MAP_SIZE: int = 5
_PLACE_MINER_MAX_TIMESTEPS: int = 300
_PLACE_MINER_NUM_PATCHES: int = 3
_PLACE_MINER_INVENTORY_COUNT: int = 5
# Resources per ore tile must outlast at least one machine tick so each
# placed miner is still on an ore-typed tile when the achievement
# evaluates (the achievement counts miners on mineable tiles, and the
# tile reverts to DIRT once depleted). 100 keeps the patch as ore for
# the full episode budget.
_PLACE_MINER_RESOURCES_PER_TILE: int = 100


def build_navigate_level(
    seed: int = 0,
) -> tuple[Level, EnvParams, frozenset[int]]:
    """L.1 — walk to a single coal patch on a 5x5 grass map.

    A single 1x1 coal tile is placed at a seed-varying position; the
    player spawns elsewhere and must walk onto it. The achievement
    (``SKILL_NAVIGATE`` — bit 0) fires the moment the player stands
    on any mineable tile, so it stays layout-invariant — the level
    can move the coal anywhere without breaking the condition. To
    keep the navigation non-trivial, the spawn and coal tile are
    drawn at manhattan distance >= 2.

    Action mask: only ``MOVE_*`` and ``NOOP`` are exposed. Every other
    action is blocked so the agent cannot mine the patch away
    (``MINE`` would deplete the tile to DIRT and unwind the goal).

    Args:
        seed: Numpy RNG seed for coal + spawn positions. Default
            ``0`` is the canonical seed used by
            :class:`SkillsBenchmark` for evaluation.

    Returns:
        Tuple of ``(level, params, blocked_actions)``.
    """
    rng = np.random.default_rng(seed)
    map_size = _NAVIGATE_MAP_SIZE
    builder = LevelBuilder(map_size, map_size)

    coal_x = int(rng.integers(0, map_size))
    coal_y = int(rng.integers(0, map_size))
    while True:
        spawn_x = int(rng.integers(0, map_size))
        spawn_y = int(rng.integers(0, map_size))
        manhattan = abs(spawn_x - coal_x) + abs(spawn_y - coal_y)
        if manhattan >= 2:
            break

    # Coal tile carries plenty of resources so that even a single
    # accidental MINE (if a future change relaxed the mask) wouldn't
    # deplete it for the whole episode.
    builder.fill_rect(coal_x, coal_y, 1, 1, BlockType.COAL, resources=100)
    builder.set_player_position(spawn_x, spawn_y)
    level = builder.build(f"skills_navigate_seed{seed}")
    params = EnvParams(
        map_width=map_size,
        map_height=map_size,
        num_players=1,
        max_timesteps=_NAVIGATE_MAX_TIMESTEPS,
    )
    return level, params, NAVIGATE_BLOCKED_ACTIONS


def build_mine_level(
    seed: int = 0,
) -> tuple[Level, EnvParams, frozenset[int]]:
    """L.2 — mine all five scattered ore tiles on a 5x5 map.

    Exactly five 1-resource ore tiles are placed at random positions
    (drawn from ``{COAL, IRON, COPPER}``). The player spawns at the
    centre — guaranteed grass — and must walk to each ore tile and
    ``MINE``. With one resource per tile, every successful ``MINE``
    depletes the tile to DIRT and increments ``state.items_mined``
    by 1. The achievement (``SKILL_MINE`` — bit 1) fires when the
    cumulative ore-mined count reaches 5.

    Layout-invariant: the achievement counts mined items, not which
    specific tiles got mined. Different seeds give different ore
    placements but the same target.

    Action mask: ``MOVE_*``, ``MINE``, ``NOOP``. No facing, placement,
    or crafting.

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
    # Pick five distinct tiles excluding the centre (spawn).
    candidates = [
        i for i in range(n_tiles) if (i % map_size, i // map_size) != (centre, centre)
    ]
    chosen = rng.choice(candidates, size=_MINE_NUM_ORE_TILES, replace=False)
    for idx in chosen:
        y, x = divmod(int(idx), map_size)
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


def build_craft_miner_level(
    seed: int = 0,
) -> tuple[Level, EnvParams, frozenset[int]]:
    """L.3 — withdraw the missing ingredient, then craft a miner.

    The miner recipe (``BASE_RECIPE_BOOK``) is
    ``1 IRON_PLATE + 1 WIRE -> 1 MINER``. To keep the solve sequence
    short enough for sparse-reward PPO to bootstrap, the player starts
    with one ingredient (``WIRE``) pre-loaded in inventory and the
    other (``IRON_PLATE``) sits in a single pallet at a seed-varying
    position. The agent has to walk to the pallet, face it,
    ``WITHDRAW``, then ``CRAFT_MINER``.

    Achievement: ``SKILL_CRAFT_MINER`` — bit 2 — fires when player 0
    has at least one ``MINER`` item in inventory.

    Action mask: ``MOVE_*``, ``FACE_*``, ``WITHDRAW``,
    ``CRAFT_MINER``, ``NOOP``. ``MINE`` is blocked; other ``CRAFT_*``
    actions are blocked.

    Args:
        seed: Numpy RNG seed for the pallet's position. Default ``0``
            is the canonical seed used by :class:`SkillsBenchmark`.

    Returns:
        Tuple of ``(level, params, blocked_actions)``.
    """
    rng = np.random.default_rng(seed)
    map_size = _CRAFT_MINER_MAP_SIZE
    builder = LevelBuilder(map_size, map_size)
    centre = map_size // 2

    while True:
        px = int(rng.integers(0, map_size))
        py = int(rng.integers(0, map_size))
        if (px, py) != (centre, centre):
            break
    builder.place_machine(px, py, int(Machine.PALLET), int(Direction.UP))
    builder.set_machine_inventory(px, py, _CRAFT_MINER_PALLET_ITEM, count=1)

    builder.set_player_position(centre, centre)
    builder.set_player_inventory([(_CRAFT_MINER_INVENTORY_ITEM, 1)])
    level = builder.build(f"skills_craft_miner_seed{seed}")
    params = EnvParams(
        map_width=map_size,
        map_height=map_size,
        num_players=1,
        max_timesteps=_CRAFT_MINER_MAX_TIMESTEPS,
    )
    return level, params, CRAFT_MINER_BLOCKED_ACTIONS


def _draw_non_adjacent_tiles(
    rng: np.random.Generator,
    map_size: int,
    n: int,
    exclude: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Pick *n* tile coordinates with no two tiles 4-neighbour adjacent.

    Walks every tile in shuffled order and greedily takes the first
    *n* that don't violate the adjacency constraint or the exclusion
    set. Retries up to ``max_attempts`` times if the greedy walk
    fails to fit *n* tiles. Used by :func:`build_place_miner_level`
    to scatter ore patches per the design rule "no ore should spawn
    next to each other".

    Raises ``RuntimeError`` if no valid layout fits — shouldn't
    happen on a 5x5 map with n=3 and a single excluded tile.
    """
    excluded = exclude or set()
    max_attempts = 5000
    for _ in range(max_attempts):
        chosen: list[tuple[int, int]] = []
        order = rng.permutation(map_size * map_size)
        for idx in order:
            x = int(idx) % map_size
            y = int(idx) // map_size
            if (x, y) in excluded:
                continue
            if any(abs(x - cx) + abs(y - cy) <= 1 for cx, cy in chosen):
                continue
            chosen.append((x, y))
            if len(chosen) == n:
                return chosen
    raise RuntimeError(
        f"Could not place {n} non-adjacent tiles on a {map_size}x{map_size} "
        f"map after {max_attempts} attempts."
    )


def build_place_miner_level(
    seed: int = 0,
) -> tuple[Level, EnvParams, frozenset[int]]:
    """L.4 — place miners on each of three non-adjacent ore patches.

    The map is 5x5 with three 1x1 iron-ore patches scattered such that
    no two patches are 4-neighbour adjacent (the centre spawn is also
    excluded). The player starts with 5 miners in inventory — more
    than enough for three placements. The achievement
    (``SKILL_PLACE_MINER`` — bit 3) fires when at least three placed
    miners are sitting on mineable tiles (via
    :func:`count_miners_on_ore`).

    Distinct from L.2: ``MINE`` is blocked, so the agent cannot
    satisfy the mine-skill condition (ore in inventory). The agent
    must walk adjacent to each patch, face it, and ``PLACE_MINER``.

    Action mask: ``MOVE_*``, ``FACE_*``, ``PLACE_MINER``, ``NOOP``.
    Other ``PLACE_*`` actions are blocked — only the miner placement
    is exposed.

    Args:
        seed: Numpy RNG seed for patch positions. Default ``0`` is
            the canonical seed used by :class:`SkillsBenchmark`.

    Returns:
        Tuple of ``(level, params, blocked_actions)``.
    """
    rng = np.random.default_rng(seed)
    map_size = _PLACE_MINER_MAP_SIZE
    builder = LevelBuilder(map_size, map_size)
    centre = map_size // 2

    patches = _draw_non_adjacent_tiles(
        rng,
        map_size,
        _PLACE_MINER_NUM_PATCHES,
        exclude={(centre, centre)},
    )
    for x, y in patches:
        builder.fill_rect(
            x,
            y,
            1,
            1,
            BlockType.IRON,
            resources=_PLACE_MINER_RESOURCES_PER_TILE,
        )

    builder.set_player_position(centre, centre)
    builder.set_player_inventory([(int(ItemType.MINER), _PLACE_MINER_INVENTORY_COUNT)])
    level = builder.build(f"skills_place_miner_seed{seed}")
    params = EnvParams(
        map_width=map_size,
        map_height=map_size,
        num_players=1,
        max_timesteps=_PLACE_MINER_MAX_TIMESTEPS,
    )
    return level, params, PLACE_MINER_BLOCKED_ACTIONS
