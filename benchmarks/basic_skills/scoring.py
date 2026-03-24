"""Scoring functions for the basic skills benchmark.

Each level has a different metric:
- **mine_resources**: total ore mined (coal + iron).
- **craft_chests**: number of chest items in the player's inventory.
- **fill_chest**: number of full stacks (64 items) across all chest machines.

Per-level scores are normalised to [0, 1] using theoretical maximums before
averaging, so no single level dominates the aggregate.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.constants import (
    MAX_MACHINE_STACK_SIZE,
    ItemType,
    MachineType,
)
from factoriax.state import EnvState

# Theoretical maximums for normalisation.
# Level 1: 9 iron tiles + 9 coal tiles, 1 resource each = 18 max.
MAX_MINE_SCORE: float = 18.0
# Level 2: 30 starting iron + 45 mineable = 75 iron, 15 chests at 5 iron each.
MAX_CRAFT_SCORE: float = 15.0
# Level 3: 8 chest slots, each fillable to 64.
MAX_FILL_SCORE: float = 8.0


def score_mine(items_mined: dict[str, int]) -> float:
    """Score the mining level: total ore extracted.

    Args:
        items_mined: Resources collected, keyed by item name.

    Returns:
        Sum of coal and iron mined.
    """
    return float(items_mined.get("coal", 0) + items_mined.get("iron", 0))


def score_craft(final_state: EnvState) -> float:
    """Score the crafting level: chest items in player inventory.

    Counts the total number of chest items across all inventory slots
    for the first player.

    Args:
        final_state: Environment state at episode end.

    Returns:
        Total chest items in inventory.
    """
    items = final_state.inventory_items[0]
    counts = final_state.inventory_counts[0]
    chest_mask = items == ItemType.CHEST
    return float(jnp.sum(jnp.where(chest_mask, counts, 0)))


def score_fill(final_state: EnvState) -> float:
    """Score the chest-filling level: full stacks in chest machines.

    A "full stack" is any machine inventory slot with count >= 64
    that belongs to a chest-type machine.

    Args:
        final_state: Environment state at episode end.

    Returns:
        Number of full stacks across all chests on the map.
    """
    is_chest = final_state.machine_types == MachineType.CHEST
    full = final_state.machine_inventory_counts >= MAX_MACHINE_STACK_SIZE
    chest_full = full & is_chest[..., None]
    return float(jnp.sum(chest_full))


def aggregate_scores(
    mine_score: float,
    craft_score: float,
    fill_score: float,
) -> float:
    """Aggregate normalised per-level scores into one scalar.

    Each score is divided by its theoretical maximum to produce a [0, 1]
    value, then the three are averaged.

    Args:
        mine_score: Raw mining level score.
        craft_score: Raw crafting level score.
        fill_score: Raw chest-filling level score.

    Returns:
        Mean of the three normalised scores.
    """
    norm_mine = min(mine_score / MAX_MINE_SCORE, 1.0)
    norm_craft = min(craft_score / MAX_CRAFT_SCORE, 1.0)
    norm_fill = min(fill_score / MAX_FILL_SCORE, 1.0)
    return (norm_mine + norm_craft + norm_fill) / 3.0
