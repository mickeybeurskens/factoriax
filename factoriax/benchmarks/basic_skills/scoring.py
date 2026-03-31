"""Scoring functions for the basic skills benchmark.

Each level has a different metric:
- **mine_resources**: total ore mined (coal + iron).
- **craft_chests**: number of chest items in the player's inventory.
- **fill_chest**: number of full stacks (64 items) across all chest machines.
- **craft_miners**: number of miner items in the player's inventory.
- **deploy_miner**: total items in miner output slots.
- **mining_factory**: total ore mined across all types.

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
# Level 4: 180 iron + 180 copper = 36 miners at 10 ore each, but inventory
# caps at 10 slots * 64 stack = practical max ~9 in 300 ticks.
MAX_CRAFT_MINERS_SCORE: float = 9.0
# Level 5: miner output slot caps at 64 items.
MAX_DEPLOY_MINER_SCORE: float = 64.0
# Level 6: reasonable target for combined hand + automated mining in 500 ticks.
MAX_MINING_FACTORY_SCORE: float = 200.0
# Level 7: 3 miners × 64 output cap = 192.
MAX_PLACE_AND_FUEL_SCORE: float = 192.0
# Level 8: 10 + 20 + 30 = 60 pre-loaded ore across 3 miners.
MAX_WITHDRAW_ORE_SCORE: float = 60.0


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


def score_craft_miners(final_state: EnvState) -> float:
    """Score the miner-crafting level: miner items in player inventory.

    Counts the total number of miner items across all inventory slots
    for the first player.

    Args:
        final_state: Environment state at episode end.

    Returns:
        Total miner items in inventory.
    """
    items = final_state.inventory_items[0]
    counts = final_state.inventory_counts[0]
    miner_mask = items == ItemType.MINER
    return float(jnp.sum(jnp.where(miner_mask, counts, 0)))


def score_deploy_miner(final_state: EnvState) -> float:
    """Score the deploy-miner level: items in miner output slots.

    Counts the total item count in slot 1 (OUTPUT) of all miner-type
    machines on the map.

    Args:
        final_state: Environment state at episode end.

    Returns:
        Total items in miner output slots.
    """
    is_miner = final_state.machine_types == MachineType.MINER
    output_counts = final_state.machine_inventory_counts[..., 1]
    return float(jnp.sum(jnp.where(is_miner, output_counts, 0)))


def score_mining_factory(items_mined: dict[str, int]) -> float:
    """Score the mining-factory level: total ore mined across all types.

    Args:
        items_mined: Resources collected, keyed by item name.

    Returns:
        Sum of coal, iron, and copper mined.
    """
    return float(
        items_mined.get("coal", 0)
        + items_mined.get("iron", 0)
        + items_mined.get("copper", 0)
    )


def score_withdraw_ore(final_state: EnvState) -> float:
    """Score the withdraw level: total items in player inventory.

    Counts the total number of items across all inventory slots for the
    first player.

    Args:
        final_state: Environment state at episode end.

    Returns:
        Total item count in player 0's inventory.
    """
    return float(jnp.sum(final_state.inventory_counts[0]))


def aggregate_scores(scores: dict[str, float]) -> float:
    """Aggregate normalised per-level scores into one scalar.

    Each score is divided by its theoretical maximum to produce a [0, 1]
    value, then all are averaged equally.

    Args:
        scores: Raw per-level scores keyed by level name.

    Returns:
        Mean of the normalised scores.
    """
    maximums: dict[str, float] = {
        "mine_resources": MAX_MINE_SCORE,
        "craft_chests": MAX_CRAFT_SCORE,
        "fill_chest": MAX_FILL_SCORE,
        "craft_miners": MAX_CRAFT_MINERS_SCORE,
        "deploy_miner": MAX_DEPLOY_MINER_SCORE,
        "mining_factory": MAX_MINING_FACTORY_SCORE,
        "place_and_fuel": MAX_PLACE_AND_FUEL_SCORE,
        "withdraw_ore": MAX_WITHDRAW_ORE_SCORE,
    }
    total = 0.0
    count = 0
    for name, raw in scores.items():
        cap = maximums.get(name, 1.0)
        total += min(raw / cap, 1.0)
        count += 1
    return total / max(count, 1)
