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

    Machines with zero power and coal in the fuel slot will consume
    one coal and gain POWER_PER_COAL units of power.

    Args:
        state: Current environment state

    Returns:
        Updated state with refueled machines
    """
    has_machine = state.machine_types != MachineType.NONE
    needs_power = state.machine_power <= 0
    has_fuel = state.machine_fuel_count > 0

    should_refuel = has_machine & needs_power & has_fuel

    new_fuel = state.machine_fuel_count - should_refuel.astype(jnp.int16)
    new_power = state.machine_power + (should_refuel.astype(jnp.int32) * POWER_PER_COAL)

    return state.replace(
        machine_fuel_count=new_fuel,
        machine_power=new_power,
    )


def run_miners(state: EnvState) -> EnvState:
    """Execute miner behavior for all miners in parallel.

    Miners with power extract resources from the block below them and
    deposit the items into their output slot. Power is only consumed
    when actually mining.

    Args:
        state: Current environment state

    Returns:
        Updated state with miner operations applied
    """
    is_miner = state.machine_types == MachineType.MINER
    has_power = state.machine_power > 0
    has_resources = state.block_resources > 0

    block_item = BLOCK_TO_ITEM_ARRAY[state.map]
    output_empty = state.machine_output_count == 0
    output_matches = state.machine_output_item == block_item
    has_space = (output_empty | output_matches) & (
        state.machine_output_count < MAX_MACHINE_STACK_SIZE
    )

    can_mine = is_miner & has_power & has_resources & has_space

    mining_rate = MACHINE_MINING_RATE[state.machine_types]
    available_space = MAX_MACHINE_STACK_SIZE - state.machine_output_count
    mine_amount = jnp.minimum(mining_rate, state.block_resources)
    mine_amount = jnp.minimum(mine_amount, available_space)
    mine_amount = mine_amount * can_mine

    new_resources = state.block_resources - mine_amount.astype(jnp.int16)
    new_output_count = state.machine_output_count + mine_amount.astype(jnp.int16)
    new_output_item = jnp.where(can_mine, block_item, state.machine_output_item)

    actually_mined = mine_amount > 0
    power_cost = MACHINE_POWER_CONSUMPTION[state.machine_types] * actually_mined
    new_power = state.machine_power - power_cost

    is_depleted = (new_resources <= 0) & (state.block_resources > 0)
    new_map = jnp.where(is_depleted, BlockType.DIRT, state.map)

    return state.replace(
        block_resources=new_resources,
        machine_output_item=new_output_item,
        machine_output_count=new_output_count,
        machine_power=new_power,
        map=new_map,
    )
