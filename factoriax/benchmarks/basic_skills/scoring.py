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
    DEFAULT_MACHINE_MAX_HEALTH,
    MAX_MACHINE_STACK_SIZE,
    ItemType,
    MachineType,
)
from factoriax.state import EnvState

# Theoretical maximums for normalisation.
MAX_MINE_SCORE: float = 18.0
MAX_CRAFT_SCORE: float = 15.0
MAX_FILL_SCORE: float = 8.0
MAX_CRAFT_MINERS_SCORE: float = 9.0
MAX_DEPLOY_MINER_SCORE: float = 64.0
MAX_MINING_FACTORY_SCORE: float = 200.0
MAX_PLACE_AND_FUEL_SCORE: float = 192.0
MAX_WITHDRAW_ORE_SCORE: float = 60.0
MAX_DEPOSIT_SCORE: float = 60.0       # 6 slots × 10 iron each
MAX_PICKUP_SCORE: float = 5.0         # 5 chests to pick up
MAX_BELT_SCORE: float = 20.0          # 4 × 5 iron deposited
MAX_ARM_SCORE: float = 20.0           # 20 iron transferred via arm
MAX_FUEL_COLLECT_SCORE: float = 180.0  # ~3 full withdrawals of 64
MAX_ASSEMBLER_SCORE: float = 6.0      # ~6 science packs in 300 ticks
MAX_RESEARCH_SCORE: float = 10.0      # full unlock at 10 progress
MAX_REPAIR_SCORE: float = 3.0         # 3 machines repaired


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


def score_deposit(final_state: EnvState) -> float:
    """Score the deposit level: total items across all chest machines.

    Args:
        final_state: Environment state at episode end.

    Returns:
        Total items stored in chests.
    """
    is_chest = final_state.machine_types == MachineType.CHEST
    return float(
        jnp.sum(
            jnp.where(
                is_chest[..., None],
                final_state.machine_inventory_counts,
                0,
            )
        )
    )


def score_pickup(final_state: EnvState) -> float:
    """Score the pickup level: chest items in player inventory.

    Args:
        final_state: Environment state at episode end.

    Returns:
        Number of chest items held by player 0.
    """
    items = final_state.inventory_items[0]
    counts = final_state.inventory_counts[0]
    return float(jnp.sum(jnp.where(items == ItemType.CHEST, counts, 0)))


def score_belt(final_state: EnvState) -> float:
    """Score the belt level: items in the target chest.

    Args:
        final_state: Environment state at episode end.

    Returns:
        Total items across all chest machine slots.
    """
    return score_deposit(final_state)


def score_arm(final_state: EnvState) -> float:
    """Score the arm level: items in the second chest (x=3).

    Only counts items in the chest at column 3 (the target chest).

    Args:
        final_state: Environment state at episode end.

    Returns:
        Total items in the target chest.
    """
    is_target = (
        (final_state.machine_types == MachineType.CHEST)
        & (jnp.arange(final_state.machine_types.shape[1])[None, :] == 3)
    )
    return float(
        jnp.sum(
            jnp.where(
                is_target[..., None],
                final_state.machine_inventory_counts,
                0,
            )
        )
    )


def score_fuel_collect(final_state: EnvState) -> float:
    """Score the fuel-and-collect level: ore in player inventory.

    Args:
        final_state: Environment state at episode end.

    Returns:
        Total item count in player 0's inventory.
    """
    return float(jnp.sum(final_state.inventory_counts[0]))


def score_assembler(final_state: EnvState) -> float:
    """Score the assembler level: science packs in player inventory.

    Args:
        final_state: Environment state at episode end.

    Returns:
        Total science pack items held by player 0.
    """
    items = final_state.inventory_items[0]
    counts = final_state.inventory_counts[0]
    is_pack = items == ItemType.BASIC_SCIENCE_PACK
    return float(jnp.sum(jnp.where(is_pack, counts, 0)))


def score_research(final_state: EnvState) -> float:
    """Score the research level: progress on tech 0.

    Args:
        final_state: Environment state at episode end.

    Returns:
        Research progress toward Hull technology (0-10).
    """
    return float(final_state.research_progress[0])


def score_repair(final_state: EnvState) -> float:
    """Score the repair level: machines restored to full health.

    Args:
        final_state: Environment state at episode end.

    Returns:
        Number of machines at full health.
    """
    is_miner = final_state.machine_types == MachineType.MINER
    is_full = final_state.machine_health == DEFAULT_MACHINE_MAX_HEALTH
    return float(jnp.sum(is_miner & is_full))


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
        "deposit_into_chests": MAX_DEPOSIT_SCORE,
        "pickup_machines": MAX_PICKUP_SCORE,
        "belt_line": MAX_BELT_SCORE,
        "arm_bridge": MAX_ARM_SCORE,
        "fuel_and_collect": MAX_FUEL_COLLECT_SCORE,
        "assembler_production": MAX_ASSEMBLER_SCORE,
        "research_tech": MAX_RESEARCH_SCORE,
        "repair_machine": MAX_REPAIR_SCORE,
    }
    total = 0.0
    count = 0
    for name, raw in scores.items():
        cap = maximums.get(name, 1.0)
        total += min(raw / cap, 1.0)
        count += 1
    return total / max(count, 1)
