"""Achievement conditions for the skills challenge benchmark.

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
    MINEABLE_BLOCKS,
    Action,
    MachineType,
)
from factoriax.state import EnvState

#: Per-skill achievement metadata, in curriculum order. Grows by one
#: entry per Phase L slice. Empty in the F.2 skeleton.
SKILLS_ACHIEVEMENT_INFO: list[AchievementInfo] = []

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
    range(int(Action.CRAFT_IRON_PLATE), int(Action.CRAFT_CROSSING) + 1)
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

#: Skill 2 — mine. Allowed: ``MOVE_*``, ``MINE``, ``NOOP``.
MINE_BLOCKED_ACTIONS: frozenset[int] = _block_complement(
    {int(Action.NOOP), int(Action.MINE)} | _MOVE_ACTIONS
)

#: Skill 3 — craft_miner. Allowed: ``MOVE_*``, ``MINE``, ``CRAFT_MINER``,
#: ``NOOP``. Only one ``CRAFT_*`` action is exposed.
CRAFT_MINER_BLOCKED_ACTIONS: frozenset[int] = _block_complement(
    {int(Action.NOOP), int(Action.MINE), int(Action.CRAFT_MINER)} | _MOVE_ACTIONS
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
        int(Action.PLACE_BELT),
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
#: Mirrors the rocket benchmark's "production must flow through
#: machines" stance.
MINI_FACTORY_BLOCKED_ACTIONS: frozenset[int] = _CRAFT_ACTIONS


def count_miners_on_ore(state: EnvState) -> jax.Array:
    """Count active miners sitting on ore tiles.

    Scans the entity arrays and cross-references each miner's
    position with the terrain map. Inactive entities (``ent_y < 0``)
    are masked out. Fully vectorized, JIT and vmap compatible.

    Used by the ``place_miner`` skill condition (Phase L.4) and
    re-exported from :mod:`factoriax.benchmarks.skills` for backward
    compatibility with the deprecated per-skill modules
    (:mod:`mining`, :mod:`place_miner`) until those are removed in D.1.

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


def skills_conditions(state: EnvState) -> jax.Array:
    """Compute the per-skill achievement conditions.

    Returns a boolean array of shape ``(MAX_ACHIEVEMENTS,)``. Bit ``i``
    is set when level ``i``'s target condition is satisfied. Bits
    beyond ``NUM_SKILLS`` are zero-padded.

    In the F.2 skeleton no skills are wired up yet, so the function
    returns an all-False mask. Each Phase L slice appends one bit by
    extending the per-skill condition list below.

    Args:
        state: Current environment state.

    Returns:
        Boolean array of shape ``(MAX_ACHIEVEMENTS,)``.
    """
    # Per-skill conditions go here, in curriculum order:
    #   conditions = jnp.array([
    #       _navigate_condition(state),       # L.1
    #       _mine_condition(state),           # L.2
    #       ...
    #   ], dtype=jnp.bool_)
    # The skeleton has no conditions yet — return a zero mask.
    del state
    return jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.bool_)
