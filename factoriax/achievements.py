"""Achievement system for tracking player progress and awarding rewards."""

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
    """

    id: str
    name: str


ACHIEVEMENT_INFO = [
    AchievementInfo(id="collect_coal_1", name="Coal Miner"),
    AchievementInfo(id="collect_coal_2", name="Coal Miner II"),
    AchievementInfo(id="collect_coal_5", name="Coal Miner III"),
    AchievementInfo(id="collect_coal_10", name="Coal Miner IV"),
    AchievementInfo(id="collect_iron_1", name="Iron Age"),
    AchievementInfo(id="collect_iron_2", name="Iron Age II"),
    AchievementInfo(id="collect_iron_5", name="Iron Age III"),
    AchievementInfo(id="collect_iron_10", name="Iron Age IV"),
    AchievementInfo(id="collect_copper_1", name="Copper Collector"),
    AchievementInfo(id="collect_copper_2", name="Copper Collector II"),
    AchievementInfo(id="collect_copper_5", name="Copper Collector III"),
    AchievementInfo(id="collect_copper_10", name="Copper Collector IV"),
    AchievementInfo(id="craft_machine_1", name="Engineer"),
    AchievementInfo(id="place_machine_1", name="Automation"),
]

NUM_ACHIEVEMENTS = len(ACHIEVEMENT_INFO)

#: Per-achievement reward magnitudes used by
#: :func:`factoriax.rewards.achievement_reward`.
ACHIEVEMENT_REWARDS = jnp.ones(NUM_ACHIEVEMENTS, dtype=jnp.float32)


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
    coal = state.items_mined[ItemType.COAL]
    iron = state.items_mined[ItemType.IRON]
    copper = state.items_mined[ItemType.COPPER]
    conditions = jnp.array(
        [
            coal >= 1,
            coal >= 2,
            coal >= 5,
            coal >= 10,
            iron >= 1,
            iron >= 2,
            iron >= 5,
            iron >= 10,
            copper >= 1,
            copper >= 2,
            copper >= 5,
            copper >= 10,
            count_total_items(state, ItemType.MINER) >= 1,
            count_machines(state, MachineType.MINER) >= 1,
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
