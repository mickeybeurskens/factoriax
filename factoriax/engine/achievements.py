"""Achievement system for tracking player progress and awarding rewards.

Achievements form a tutorial progression that guides the player from
hand-mining raw ore to launching a rocket, each one teaching a new
mechanic. :data:`ACHIEVEMENT_INFO` holds the display names and hints;
:func:`core_game_conditions` defines what unlocks each slot.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from factoriax.engine.constants import MAX_ACHIEVEMENTS, ItemType, Machine
from factoriax.engine.state import EnvState


@dataclass(frozen=True)
class AchievementInfo:
    """Metadata for achievements (used for display, not JAX tracing).

    Attributes:
        id: Unique identifier for the achievement
        name: Human-readable name
        hint: Short guidance string shown when selected in the menu
    """

    id: str
    name: str
    hint: str


ACHIEVEMENT_INFO = [
    AchievementInfo(
        id="first_ore",
        name="First Ore",
        hint="Walk onto an ore tile and press SPACE to mine.",
    ),
    AchievementInfo(
        id="stockpile",
        name="Stockpile",
        hint="Keep mining until you have 10 ore total.",
    ),
    AchievementInfo(
        id="apprentice_engineer",
        name="Apprentice Engineer",
        hint="Open inventory (I), select a recipe, and craft it (E).",
    ),
    AchievementInfo(
        id="breaking_ground",
        name="Breaking Ground",
        hint="Select a machine in your hotbar and press E to place it.",
    ),
    AchievementInfo(
        id="coal_gathered",
        name="Coal Gathered",
        hint="Walk onto a coal tile and press SPACE to mine a piece of coal.",
    ),
    AchievementInfo(
        id="automated_mining",
        name="Automated Mining",
        hint="Place a miner on an ore tile and wait for it to produce.",
    ),
    AchievementInfo(
        id="moving_parts",
        name="Moving Parts",
        hint="Craft and place both an arm and a pallet.",
    ),
    AchievementInfo(
        id="first_pipeline",
        name="First Pipeline",
        hint="Use an arm to move miner output into a pallet.",
    ),
    AchievementInfo(
        id="belt_network",
        name="Belt Network",
        hint="Craft and place at least 5 conveyor belts.",
    ),
    AchievementInfo(
        id="scaling_up",
        name="Scaling Up",
        hint="Have 3 miners placed on the map at the same time.",
    ),
    AchievementInfo(
        id="industrialist",
        name="Industrialist",
        hint="Place 10 machines of any type on the map.",
    ),
    AchievementInfo(
        id="assembler_crafted",
        name="Assembler Crafted",
        hint="Open inventory (I), select the Assembler recipe, and craft it.",
    ),
    AchievementInfo(
        id="assembly_line",
        name="Assembly Line",
        hint="Place an assembler on the map.",
    ),
    AchievementInfo(
        id="first_assembly",
        name="First Assembly",
        hint="Set a recipe on your assembler (Q) and feed it inputs.",
    ),
    AchievementInfo(
        id="hull_production",
        name="Hull Production",
        hint="Produce and collect at least 10 hulls.",
    ),
    AchievementInfo(
        id="fuel_production",
        name="Fuel Production",
        hint="Produce and collect at least 10 fuel packs.",
    ),
    AchievementInfo(
        id="rocket_complete",
        name="Rocket Complete",
        hint="Craft a rocket in an assembler and place it on the map.",
    ),
    AchievementInfo(
        id="first_science",
        name="First Science",
        hint="Produce a science pack in an assembler.",
    ),
    AchievementInfo(
        id="advanced_science",
        name="Advanced Science",
        hint="Produce an advanced science pack.",
    ),
]

NUM_ACHIEVEMENTS = len(ACHIEVEMENT_INFO)

#: Per-achievement reward magnitudes for the core game, used by
#: :func:`factoriax.engine.rewards.achievement_reward` as the default weights.
#: Shape ``(MAX_ACHIEVEMENTS,)`` — slots beyond the ``NUM_ACHIEVEMENTS`` core
#: achievements are zero so they contribute no reward.
CORE_ACHIEVEMENT_WEIGHTS = jnp.zeros(MAX_ACHIEVEMENTS, dtype=jnp.float32)
CORE_ACHIEVEMENT_WEIGHTS = CORE_ACHIEVEMENT_WEIGHTS.at[:NUM_ACHIEVEMENTS].set(1.0)

# Miner machine inventory layout (mirrors machines.py constants).
_MINER_FUEL_SLOT: int = 0
_MINER_OUTPUT_SLOT: int = 1


def count_total_items(state: EnvState, item_type: int) -> jax.Array:
    """Count total quantity of an item type across all players' inventories.

    Args:
        state: Current environment state
        item_type: ItemType to count

    Returns:
        Total count of the specified item across all inventories
    """
    total: jax.Array = jnp.sum(state.player_inventory[:, item_type])
    return total


def count_machines(state: EnvState, machine_type: int) -> jax.Array:
    """Count the number of placed machines of a specific type.

    Args:
        state: Current environment state
        machine_type: Machine to count

    Returns:
        Count of machines of the specified type on the map
    """
    count: jax.Array = jnp.sum(state.machine_types == machine_type)
    return count


def _any_miner_has_output(state: EnvState) -> jax.Array:
    """Check whether any placed miner has produced ore in its output slot.

    Args:
        state: Current environment state

    Returns:
        Scalar boolean — True if at least one miner output is non-empty.
    """
    is_miner = state.ent_type == Machine.MINER
    is_active = state.ent_y >= 0
    has_output = state.ent_buf_count > 0
    return jnp.any(is_miner & is_active & has_output)


def _any_pallet_has_items(state: EnvState) -> jax.Array:
    """Check whether any placed pallet contains items.

    Args:
        state: Current environment state

    Returns:
        Scalar boolean — True if at least one pallet slot is non-empty.
    """
    is_pallet = state.ent_type == Machine.PALLET
    is_active = state.ent_y >= 0
    has_items = state.ent_buf_count > 0
    return jnp.any(is_pallet & is_active & has_items)


def _any_assembler_has_output(state: EnvState) -> jax.Array:
    """Check whether any placed assembler has items in its output slot.

    Args:
        state: Current environment state.

    Returns:
        Scalar boolean — True if at least one assembler output is non-empty.
    """
    is_asm = state.ent_type == Machine.ASSEMBLER
    is_active = state.ent_y >= 0
    has_output = state.ent_asm_out_count > 0
    return jnp.any(is_asm & is_active & has_output)


def core_game_conditions(state: EnvState) -> jax.Array:
    """Compute the core game achievement conditions.

    Returns a boolean array of shape ``(MAX_ACHIEVEMENTS,)``. The
    first ``NUM_ACHIEVEMENTS`` slots correspond to the core tutorial
    milestones. Remaining slots are False.

    This is the default condition function used when constructing a
    :class:`~factoriax.engine.envs.factoriax_env.FactoriaXEnv` via
    :func:`factoriax.make`. Benchmarks can provide their own
    function with the same signature.

    Args:
        state: Current environment state.

    Returns:
        Boolean array of shape ``(MAX_ACHIEVEMENTS,)``.
    """
    total_mined = (
        state.items_mined[ItemType.COAL]
        + state.items_mined[ItemType.IRON_ORE]
        + state.items_mined[ItemType.COPPER_ORE]
    )

    total_machines = jnp.sum(state.machine_types != Machine.NONE)

    # Count machine items across all player inventories.
    machine_items_held = (
        count_total_items(state, ItemType.MINER)
        + count_total_items(state, ItemType.PALLET)
        + count_total_items(state, ItemType.CONVEYOR_BELT)
    )

    conditions = jnp.array(
        [
            # 0  First Ore
            total_mined >= 1,
            # 1  Stockpile
            total_mined >= 10,
            # 2  Apprentice Engineer
            machine_items_held >= 1,
            # 3  Breaking Ground
            total_machines >= 1,
            # 4  Coal Gathered
            count_total_items(state, ItemType.COAL) >= 1,
            # 5  Automated Mining
            _any_miner_has_output(state),
            # 6  Moving Parts — checks the pallet only; name/hint also mention an arm
            count_machines(state, Machine.PALLET) >= 1,
            # 7  First Pipeline
            _any_pallet_has_items(state),
            # 8  Belt Network
            count_machines(state, Machine.CONVEYOR_BELT) >= 5,
            # 9  Scaling Up
            count_machines(state, Machine.MINER) >= 3,
            # 10 Industrialist
            total_machines >= 10,
            # 11 Assembler Crafted
            count_total_items(state, ItemType.ASSEMBLER) >= 1,
            # 12 Assembly Line
            count_machines(state, Machine.ASSEMBLER) >= 1,
            # 13 First Assembly
            _any_assembler_has_output(state),
            # 14 Hull Production — placeholder (item removed, always False)
            jnp.bool_(False),
            # 15 Fuel Production — placeholder (item removed, always False)
            jnp.bool_(False),
            # 16 Rocket Complete
            count_machines(state, Machine.ROCKET) >= 1,
            # 17 First Science
            (
                count_total_items(state, ItemType.BASIC_SCIENCE_PACK)
                + count_total_items(state, ItemType.ADVANCED_SCIENCE_PACK)
            )
            >= 1,
            # 18 Advanced Science
            count_total_items(state, ItemType.ADVANCED_SCIENCE_PACK) >= 1,
        ],
        dtype=jnp.bool_,
    )
    # Pad to MAX_ACHIEVEMENTS.
    return jnp.concatenate(
        [conditions, jnp.zeros(MAX_ACHIEVEMENTS - NUM_ACHIEVEMENTS, dtype=jnp.bool_)]
    )
