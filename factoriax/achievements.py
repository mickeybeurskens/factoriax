"""Achievement system for tracking player progress and awarding rewards.

Achievements are ordered as a tutorial progression that guides the player
from hand-mining raw ore all the way to a multi-machine extraction
pipeline.  Each achievement teaches one new concept or mechanic:

 1. First Ore          — mine any ore (teaches mining)
 2. Stockpile          — mine 10 total (build up resources for crafting)
 3. Apprentice Engineer — craft a machine (teaches crafting UI)
 4. Breaking Ground    — place a machine (teaches placement)
 5. Fueled Up          — deliver coal to a miner (teaches machine inspection)
 6. Automated Mining   — miner produces ore (confirms fuel→extraction loop)
 7. Moving Parts       — place an arm and a chest (pipeline building blocks)
 8. First Pipeline     — a chest holds items (full miner→arm→chest flow)
 9. Belt Network       — place 5 belts (transport layer)
10. Scaling Up         — 3 miners on the map (replicate the pattern)
11. Industrialist      — 10 machines total (capstone)
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from factoriax.constants import ItemType, MachineType
from factoriax.state import EnvState


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
        id="fueled_up",
        name="Fueled Up",
        hint="Inspect a miner (F) and transfer coal into its fuel slot (E).",
    ),
    AchievementInfo(
        id="automated_mining",
        name="Automated Mining",
        hint="Place a fueled miner on an ore tile and wait for it to produce.",
    ),
    AchievementInfo(
        id="moving_parts",
        name="Moving Parts",
        hint="Craft and place both an arm and a chest.",
    ),
    AchievementInfo(
        id="first_pipeline",
        name="First Pipeline",
        hint="Use an arm to move miner output into a chest.",
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
]

NUM_ACHIEVEMENTS = len(ACHIEVEMENT_INFO)

#: Per-achievement reward magnitudes used by
#: :func:`factoriax.rewards.achievement_reward`.
ACHIEVEMENT_REWARDS = jnp.ones(NUM_ACHIEVEMENTS, dtype=jnp.float32)

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
    matching_mask = state.inventory_items == item_type
    counts = jnp.where(matching_mask, state.inventory_counts, 0)
    total: jax.Array = jnp.sum(counts)
    return total


def count_machines(state: EnvState, machine_type: int) -> jax.Array:
    """Count the number of placed machines of a specific type.

    Args:
        state: Current environment state
        machine_type: MachineType to count

    Returns:
        Count of machines of the specified type on the map
    """
    count: jax.Array = jnp.sum(state.machine_types == machine_type)
    return count


def _any_miner_has_fuel(state: EnvState) -> jax.Array:
    """Check whether any placed miner has coal in its fuel slot.

    Args:
        state: Current environment state

    Returns:
        Scalar boolean — True if at least one miner is fueled.
    """
    is_miner = state.machine_types == MachineType.MINER
    has_coal = (
        (state.machine_inventory_items[..., _MINER_FUEL_SLOT] == ItemType.COAL)
        & (state.machine_inventory_counts[..., _MINER_FUEL_SLOT] > 0)
    )
    return jnp.any(is_miner & has_coal)


def _any_miner_has_output(state: EnvState) -> jax.Array:
    """Check whether any placed miner has produced ore in its output slot.

    Args:
        state: Current environment state

    Returns:
        Scalar boolean — True if at least one miner output is non-empty.
    """
    is_miner = state.machine_types == MachineType.MINER
    has_output = state.machine_inventory_counts[..., _MINER_OUTPUT_SLOT] > 0
    return jnp.any(is_miner & has_output)


def _any_chest_has_items(state: EnvState) -> jax.Array:
    """Check whether any placed chest contains items.

    Args:
        state: Current environment state

    Returns:
        Scalar boolean — True if at least one chest slot is non-empty.
    """
    is_chest = state.machine_types == MachineType.CHEST
    has_items = jnp.any(state.machine_inventory_counts > 0, axis=-1)
    return jnp.any(is_chest & has_items)


def compute_all_conditions(state: EnvState) -> jax.Array:
    """Compute whether each achievement condition is met.

    This is the JAX-compatible version that computes all conditions
    as a single vectorized operation, avoiding Python list indexing
    with traced values.

    Args:
        state: Current environment state

    Returns:
        Boolean array of shape (NUM_ACHIEVEMENTS,) indicating which
        achievement conditions are currently satisfied
    """
    total_mined = (
        state.items_mined[ItemType.COAL]
        + state.items_mined[ItemType.IRON]
        + state.items_mined[ItemType.COPPER]
    )

    total_machines = jnp.sum(state.machine_types != MachineType.NONE)

    # Count machine items across all player inventories.
    machine_items_held = (
        count_total_items(state, ItemType.MINER)
        + count_total_items(state, ItemType.CHEST)
        + count_total_items(state, ItemType.CONVEYOR_BELT)
        + count_total_items(state, ItemType.ARM)
    )

    conditions = jnp.array(
        [
            # 0  First Ore — mine any ore
            total_mined >= 1,
            # 1  Stockpile — mine 10 total
            total_mined >= 10,
            # 2  Apprentice Engineer — craft any machine
            machine_items_held >= 1,
            # 3  Breaking Ground — place any machine
            total_machines >= 1,
            # 4  Fueled Up — deliver coal to a miner
            _any_miner_has_fuel(state),
            # 5  Automated Mining — miner output slot non-empty
            _any_miner_has_output(state),
            # 6  Moving Parts — place an arm and a chest
            (count_machines(state, MachineType.ARM) >= 1)
            & (count_machines(state, MachineType.CHEST) >= 1),
            # 7  First Pipeline — any chest holds items
            _any_chest_has_items(state),
            # 8  Belt Network — place 5 belts
            count_machines(state, MachineType.CONVEYOR_BELT) >= 5,
            # 9  Scaling Up — 3 miners on the map
            count_machines(state, MachineType.MINER) >= 3,
            # 10 Industrialist — 10 machines total
            total_machines >= 10,
        ],
        dtype=jnp.bool_,
    )
    return conditions


def check_achievements(state: EnvState) -> EnvState:
    """Update ``achievements_unlocked`` in state based on current conditions.

    Uses vectorized JAX operations to check all achievement conditions
    simultaneously.  Reward computation has moved to
    :func:`factoriax.rewards.achievement_reward`, which compares the
    before/after states.

    Args:
        state: Current environment state.

    Returns:
        Updated state with any newly satisfied achievements marked as
        unlocked.
    """
    conditions_met = compute_all_conditions(state)
    new_unlocked = state.achievements_unlocked | conditions_met
    return state.replace(achievements_unlocked=new_unlocked)  # type: ignore[attr-defined, no-any-return]
