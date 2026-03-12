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
    AchievementInfo(id="collect_iron_1", name="Iron Age"),
    AchievementInfo(id="collect_copper_1", name="Copper Collector"),
    AchievementInfo(id="craft_machine_1", name="Engineer"),
    AchievementInfo(id="place_machine_1", name="Automation"),
]

NUM_ACHIEVEMENTS = len(ACHIEVEMENT_INFO)

ACHIEVEMENT_REWARDS = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)


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
    conditions = jnp.array(
        [
            count_total_items(state, ItemType.COAL) >= 1,
            count_total_items(state, ItemType.IRON) >= 1,
            count_total_items(state, ItemType.COPPER) >= 1,
            count_total_items(state, ItemType.MINER) >= 1,
            count_machines(state, MachineType.MINER) >= 1,
        ],
        dtype=jnp.bool_,
    )
    return conditions


def check_achievements(state: EnvState) -> tuple[EnvState, jax.Array]:
    """Check all achievements and return updated state with total reward.

    Uses vectorized JAX operations to check all achievements simultaneously,
    avoiding the need to index Python lists with traced values.

    Args:
        state: Current environment state

    Returns:
        Tuple of (updated_state, total_reward) where total_reward is the sum
        of all newly unlocked achievement rewards
    """
    conditions_met = compute_all_conditions(state)
    already_unlocked = state.achievements_unlocked
    newly_unlocked = conditions_met & ~already_unlocked
    new_unlocked = already_unlocked | newly_unlocked
    total_reward = jnp.sum(jnp.where(newly_unlocked, ACHIEVEMENT_REWARDS, 0.0))
    new_state = state.replace(achievements_unlocked=new_unlocked)
    return new_state, total_reward
