"""Machine update logic for the FactoriaX environment."""

import jax.numpy as jnp

from factoriax.constants import (
    BLOCK_TO_ITEM_ARRAY,
    MACHINE_MINING_RATE,
    MACHINE_POWER_CONSUMPTION,
    MAX_MACHINE_STACK_SIZE,
    POWER_PER_COAL,
    BlockType,
    MachineType,
)
from factoriax.state import EnvState

# Miner slot indices (within machine_inventory_items / machine_inventory_counts).
_MINER_FUEL_SLOT: int = 0   # INPUT — coal fuel consumed by the miner.
_MINER_OUTPUT_SLOT: int = 1  # OUTPUT — mined ore deposited here.


def update_all_machines(state: EnvState) -> EnvState:
    """Update all machines in parallel for one step.

    Processes machine updates in order:
    1. Refuel machines that need power
    2. Run miners to extract resources

    Args:
        state: Current environment state

    Returns:
        Updated environment state with all machines processed
    """
    state = refuel_machines(state)
    state = run_miners(state)
    return state


def refuel_machines(state: EnvState) -> EnvState:
    """Convert coal to power for machines that need it.

    Machines with zero power and coal in slot 0 (the fuel/INPUT slot) will
    consume one coal and gain POWER_PER_COAL units of power.

    Args:
        state: Current environment state

    Returns:
        Updated state with refueled machines
    """
    is_miner = state.machine_types == MachineType.MINER
    needs_power = state.machine_power <= 0
    fuel_count = state.machine_inventory_counts[..., _MINER_FUEL_SLOT]
    has_fuel = fuel_count > 0

    should_refuel = is_miner & needs_power & has_fuel

    new_fuel = fuel_count - should_refuel.astype(jnp.int16)
    new_power = state.machine_power + (should_refuel.astype(jnp.int32) * POWER_PER_COAL)

    new_inv_counts = state.machine_inventory_counts.at[..., _MINER_FUEL_SLOT].set(new_fuel)
    return state.replace(
        machine_inventory_counts=new_inv_counts,
        machine_power=new_power,
    )


def run_miners(state: EnvState) -> EnvState:
    """Execute miner behavior for all miners in parallel.

    Miners with power extract resources from the block below them and
    deposit the items into their output slot (slot index 1). Power is only
    consumed when actually mining.

    Args:
        state: Current environment state

    Returns:
        Updated state with miner operations applied
    """
    is_miner = state.machine_types == MachineType.MINER
    has_power = state.machine_power > 0
    has_resources = state.block_resources > 0

    block_item = BLOCK_TO_ITEM_ARRAY[state.map]

    output_count = state.machine_inventory_counts[..., _MINER_OUTPUT_SLOT]
    output_item = state.machine_inventory_items[..., _MINER_OUTPUT_SLOT]

    output_empty = output_count == 0
    output_matches = output_item == block_item
    has_space = (output_empty | output_matches) & (output_count < MAX_MACHINE_STACK_SIZE)

    can_mine = is_miner & has_power & has_resources & has_space

    mining_rate = MACHINE_MINING_RATE[state.machine_types]
    available_space = MAX_MACHINE_STACK_SIZE - output_count
    mine_amount = jnp.minimum(mining_rate, state.block_resources)
    mine_amount = jnp.minimum(mine_amount, available_space)
    mine_amount = mine_amount * can_mine

    new_resources = state.block_resources - mine_amount.astype(jnp.int16)
    new_output_count = output_count + mine_amount.astype(jnp.int16)
    new_output_item = jnp.where(can_mine, block_item, output_item)

    actually_mined = mine_amount > 0
    power_cost = MACHINE_POWER_CONSUMPTION[state.machine_types] * actually_mined
    new_power = state.machine_power - power_cost

    is_depleted = (new_resources <= 0) & (state.block_resources > 0)
    new_map = jnp.where(is_depleted, BlockType.DIRT, state.map)

    new_inv_counts = state.machine_inventory_counts.at[..., _MINER_OUTPUT_SLOT].set(
        new_output_count
    )
    new_inv_items = state.machine_inventory_items.at[..., _MINER_OUTPUT_SLOT].set(
        new_output_item
    )

    return state.replace(
        block_resources=new_resources,
        machine_inventory_items=new_inv_items,
        machine_inventory_counts=new_inv_counts,
        machine_power=new_power,
        map=new_map,
    )
