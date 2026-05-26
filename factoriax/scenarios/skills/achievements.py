"""Achievement conditions for the skills challenge scenario.

The curriculum has eight skill levels, each contributing one achievement
bit. Bit ``i`` corresponds to the ``i``-th level returned by
:meth:`SkillsBenchmark.levels` — bit 0 = navigate, bit 1 = mine, …,
bit 7 = mini_factory. The list is grown one entry at a time as Phase L
slices land.

The conditions function returned by :func:`skills_conditions` is a pure
JAX function of :class:`EnvState`, suitable for binding to
:class:`FactoriaXEnv` via its ``achievement_fn`` constructor argument.
The output mask is zero-padded to ``MAX_ACHIEVEMENTS`` so it plugs into
the engine's latching path unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.achievements import AchievementInfo
from factoriax.constants import (
    MAX_ACHIEVEMENTS,
    Action,
    ItemType,
    Machine,
)
from factoriax.state import EnvState
from factoriax.tables import MINEABLE_BLOCKS

#: Per-skill achievement metadata, in curriculum order. Grows by one
#: entry per Phase L slice.
SKILLS_ACHIEVEMENT_INFO: list[AchievementInfo] = [
    AchievementInfo(
        id="skill_navigate",
        name="Skill: Navigate",
        hint="Walk to the coal patch.",
    ),
    AchievementInfo(
        id="skill_mine",
        name="Skill: Mine",
        hint="Mine all five scattered ore tiles.",
    ),
    AchievementInfo(
        id="skill_craft_miner",
        name="Skill: Craft Miner",
        hint="Combine IRON_PLATE and WIRE via CRAFT_MINER.",
    ),
    AchievementInfo(
        id="skill_place_miner",
        name="Skill: Place Miner",
        hint="Place miners on each of the three scattered ore patches.",
    ),
]

#: Number of skills in the curriculum so far. Used by
#: :func:`skills_conditions` to size the boolean output before padding.
NUM_SKILLS: int = len(SKILLS_ACHIEVEMENT_INFO)

# ---------------------------------------------------------------------------
# Action-group helpers (per-level mask construction)
# ---------------------------------------------------------------------------
#
# Each per-level mask is the set complement of "actions this skill needs".
# We derive the masks from the ``Action`` enum so they stay correct when
# new actions are added — no magic numbers, no hand-maintained ranges.

_ALL_ACTIONS: frozenset[int] = frozenset(int(a) for a in Action)

_MOVE_ACTIONS: frozenset[int] = frozenset(
    {int(Action.UP), int(Action.DOWN), int(Action.LEFT), int(Action.RIGHT)}
)
_FACE_ACTIONS: frozenset[int] = frozenset(
    {
        int(Action.FACE_UP),
        int(Action.FACE_DOWN),
        int(Action.FACE_LEFT),
        int(Action.FACE_RIGHT),
    }
)
_ROTATE_ACTIONS: frozenset[int] = frozenset(
    {
        int(Action.ROTATE_LEFT),
        int(Action.ROTATE_RIGHT),
        int(Action.ROTATE_UP),
        int(Action.ROTATE_DOWN),
    }
)
_CRAFT_ACTIONS: frozenset[int] = frozenset(
    int(a) for a in Action if a.name.startswith("CRAFT_")
)


def _block_complement(allowed: set[int]) -> frozenset[int]:
    """Return the mask blocking every action *not* in ``allowed``."""
    return frozenset(_ALL_ACTIONS - allowed)


# Each constant below blocks every action *not* relevant to that skill.
# Comments cite the spec's "Allowed actions" column.

#: Skill 1 — navigate. Allowed: ``MOVE_*``, ``NOOP``.
NAVIGATE_BLOCKED_ACTIONS: frozenset[int] = _block_complement(
    {int(Action.NOOP)} | _MOVE_ACTIONS
)

#: Skill 2 — mine. Allowed: ``MOVE_*``, ``FACE_*``, ``MINE``, ``NOOP``.
#: ``FACE_*`` is required because MINE targets the tile in front of the
#: player; the policy must stand adjacent to ore and rotate to face it
#: without stepping onto the (walkable) ore tile.
MINE_BLOCKED_ACTIONS: frozenset[int] = _block_complement(
    {int(Action.NOOP), int(Action.MINE)} | _MOVE_ACTIONS | _FACE_ACTIONS
)

#: Skill 3 — craft_miner. Allowed: ``MOVE_*``, ``FACE_*``,
#: ``WITHDRAW``, ``CRAFT_MINER``, ``NOOP``. The agent withdraws the
#: recipe ingredients (IRON_PLATE, WIRE) from two pre-placed pallets,
#: then crafts. Only one ``CRAFT_*`` action is exposed; ``MINE`` is
#: blocked (no ore on the map anyway).
CRAFT_MINER_BLOCKED_ACTIONS: frozenset[int] = _block_complement(
    {
        int(Action.NOOP),
        int(Action.WITHDRAW),
        int(Action.CRAFT_MINER),
    }
    | _MOVE_ACTIONS
    | _FACE_ACTIONS
)

#: Skill 4 — place_miner. Allowed: ``MOVE_*``, ``FACE_*``,
#: ``PLACE_MINER``, ``NOOP``. Only the miner placement action is
#: exposed; other ``PLACE_*`` actions are blocked.
PLACE_MINER_BLOCKED_ACTIONS: frozenset[int] = _block_complement(
    {int(Action.NOOP), int(Action.PLACE_MINER)} | _MOVE_ACTIONS | _FACE_ACTIONS
)

#: Skill 5 — fuel_and_collect. Allowed: ``MOVE_*``, ``FACE_*``,
#: ``DEPOSIT_COAL`` (to fuel a miner), ``WITHDRAW`` (to collect ore),
#: ``NOOP``. Hand-mining (``MINE``) is blocked — the agent must collect
#: ore from a placed miner, not by hand.
FUEL_AND_COLLECT_BLOCKED_ACTIONS: frozenset[int] = _block_complement(
    {
        int(Action.NOOP),
        int(Action.DEPOSIT_COAL),
        int(Action.WITHDRAW),
    }
    | _MOVE_ACTIONS
    | _FACE_ACTIONS
)

#: Skill 6 — belt_line. Allowed: ``MOVE_*``, ``FACE_*``, ``PLACE_BELT``,
#: ``ROTATE_*``, ``MINE``, ``NOOP``. The agent places belts and may mine
#: the source if it isn't pre-placed.
BELT_LINE_BLOCKED_ACTIONS: frozenset[int] = _block_complement(
    {
        int(Action.NOOP),
        int(Action.PLACE_CONVEYOR_BELT),
        int(Action.MINE),
    }
    | _MOVE_ACTIONS
    | _FACE_ACTIONS
    | _ROTATE_ACTIONS
)

#: Skill 7 — arm_transfer. Allowed: ``MOVE_*``, ``FACE_*``,
#: ``PLACE_ARM``, ``ROTATE_*``, ``DEPOSIT_COAL`` (fuel the pre-placed
#: miner), ``NOOP``.
ARM_TRANSFER_BLOCKED_ACTIONS: frozenset[int] = _block_complement(
    {
        int(Action.NOOP),
        int(Action.PLACE_ARM),
        int(Action.DEPOSIT_COAL),
    }
    | _MOVE_ACTIONS
    | _FACE_ACTIONS
    | _ROTATE_ACTIONS
)

#: Skill 8 — mini_factory. Every action *except* all ``CRAFT_*``.
#: Mirrors the rocket scenario's "production must flow through
#: machines" stance.
MINI_FACTORY_BLOCKED_ACTIONS: frozenset[int] = _CRAFT_ACTIONS


def count_miners_on_ore(state: EnvState) -> jax.Array:
    """Count active miners sitting on ore tiles.

    Scans the entity arrays and cross-references each miner's
    position with the terrain map. Inactive entities (``ent_y < 0``)
    are masked out. Fully vectorized, JIT and vmap compatible.

    Used by the ``place_miner`` skill condition (Phase L.4) and
    re-exported from :mod:`factoriax.scenarios.skills` for backward
    compatibility with the deprecated per-skill modules
    (:mod:`mining`, :mod:`place_miner`) until those are removed in D.1.

    Args:
        state: Current environment state.

    Returns:
        Scalar int32 count of miners on mineable blocks.
    """
    active = state.ent_y >= 0
    is_miner = state.ent_type == Machine.MINER
    y = jnp.clip(state.ent_y, 0)
    x = jnp.clip(state.ent_x, 0)
    tile = state.map[y, x]
    on_ore = jnp.isin(tile, MINEABLE_BLOCKS)
    return jnp.sum(active & is_miner & on_ore)


_ORE_ITEM_IDS: jax.Array = jnp.array(
    [
        int(ItemType.COAL),
        int(ItemType.IRON_ORE),
        int(ItemType.COPPER_ORE),
        int(ItemType.TIN_ORE),
        int(ItemType.SILICON),
    ],
    dtype=jnp.int32,
)


def _navigate_condition(state: EnvState) -> jax.Array:
    """Bit 0 — player 0 is standing on a mineable tile.

    Layout-invariant: any tile in :data:`MINEABLE_BLOCKS` under the
    player satisfies the condition. The level builder places a single
    coal patch at a seed-varying position; the agent has to walk to
    it. ``MINE`` is masked off, so the player can't accidentally
    mine the tile away — they reach the goal by stepping onto it
    and the bit latches immediately.
    """
    pos = state.player_positions[0]
    px, py = pos[0], pos[1]
    block = state.map[py, px]
    return jnp.any(block == MINEABLE_BLOCKS)


def _mine_condition(state: EnvState) -> jax.Array:
    """Bit 1 — player 0 has mined at least 5 ore items in total.

    Counts the cumulative ``state.items_mined`` for the five mineable
    item types. Each successful manual ``MINE`` action increments by
    1, so this fires after the agent has mined five ore tiles.
    Layout-invariant: doesn't depend on which ore types are placed.
    """
    return jnp.sum(state.items_mined[_ORE_ITEM_IDS]) >= 5


def _craft_miner_condition(state: EnvState) -> jax.Array:
    """Bit 2 — player 0 holds at least one ``MINER`` in inventory.

    Triggered after a successful ``CRAFT_MINER`` action consumes
    ``IRON_PLATE`` + ``WIRE`` and produces a miner item. Layout- and
    seed-invariant.
    """
    return state.player_inventory[0, int(ItemType.MINER)] >= 1


def _place_miner_condition(state: EnvState) -> jax.Array:
    """Bit 3 — at least three placed miners sit on mineable tiles.

    The level builder places three non-adjacent single-tile ore patches;
    success means a miner is sitting on each one. Reuses
    :func:`count_miners_on_ore`. Distinct from bit 1 (``_mine_condition``)
    in two ways: the per-level mask blocks ``MINE`` (so manual mining
    is impossible), and the bit needs a *placed* miner on each ore
    tile — not items in the player inventory.
    """
    return count_miners_on_ore(state) >= 3


def skills_conditions(state: EnvState) -> jax.Array:
    """Compute the per-skill achievement conditions.

    Returns a boolean array of shape ``(MAX_ACHIEVEMENTS,)``. Bit ``i``
    is set when level ``i``'s target condition is satisfied (curriculum
    order). Bits beyond ``NUM_SKILLS`` are zero-padded.

    Each Phase L slice appends one bit by extending the per-skill
    condition list below.

    Args:
        state: Current environment state.

    Returns:
        Boolean array of shape ``(MAX_ACHIEVEMENTS,)``.
    """
    conditions = jnp.array(
        [
            _navigate_condition(state),  # L.1 (bit 0)
            _mine_condition(state),  # L.2 (bit 1)
            _craft_miner_condition(state),  # L.3 (bit 2)
            _place_miner_condition(state),  # L.4 (bit 3)
        ],
        dtype=jnp.bool_,
    )
    return jnp.concatenate(
        [
            conditions,
            jnp.zeros(MAX_ACHIEVEMENTS - NUM_SKILLS, dtype=jnp.bool_),
        ]
    )
