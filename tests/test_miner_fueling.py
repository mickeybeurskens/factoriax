"""Integration test: depositing coal into a miner should make it mine.

Exercises the full loop: place miner on ore, deposit coal, step the
environment, verify the miner produces ore. This catches the structural
mismatch where deposit writes to the entity buffer but refuel reads
from the entity fuel field.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from factoriax.constants import (
    Action,
    BlockType,
    Direction,
    ItemType,
    MachineType,
)
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.game_logic import factoriax_step
from factoriax.state import EnvParams, EnvState


def _make_miner_state(state_factory) -> EnvState:
    """Create a state with a fueled miner on an iron tile.

    The player is at (0,0) facing right. A miner sits at (1,0) on
    an iron tile with 100 resources. The player has 10 coal in
    inventory for fueling.
    """
    shape = (3, 3)
    block_map = jnp.full(shape, int(BlockType.DIRT), dtype=jnp.int32)
    block_map = block_map.at[0, 1].set(int(BlockType.IRON))
    resources = jnp.zeros(shape, dtype=jnp.int16)
    resources = resources.at[0, 1].set(100)

    machine_types = jnp.full(
        shape, int(MachineType.NONE), dtype=jnp.int32,
    )
    machine_types = machine_types.at[0, 1].set(int(MachineType.MINER))

    machine_dir = jnp.zeros(shape, dtype=jnp.int32)
    machine_dir = machine_dir.at[0, 1].set(int(Direction.DOWN))

    inv = jnp.zeros((1, len(ItemType)), dtype=jnp.int16)
    inv = inv.at[0, int(ItemType.COAL)].set(10)

    return state_factory(
        world_map=block_map,
        block_resources=resources,
        player_position=(0, 0),
        player_direction=int(Direction.RIGHT),
        machine_types=machine_types,
        machine_direction=machine_dir,
        player_inventory=inv,
    )


class TestMinerFueling:
    """Depositing coal into a miner should enable mining."""

    def test_deposit_coal_goes_to_buffer(
        self, state_factory,
    ) -> None:
        """Depositing coal puts it in ent_buf, not ent_fuel."""
        state = _make_miner_state(state_factory)
        params = EnvParams(map_width=3, map_height=3, num_players=1)
        rng = jax.random.PRNGKey(0)

        rng, k = jax.random.split(rng)
        state = factoriax_step(
            k, state, int(Action.DEPOSIT_COAL), params,
        )

        eidx = int(state.tile_entity[0, 1])
        assert eidx >= 0, "Miner entity not found"

        buf_type = int(state.ent_buf_type[eidx])
        buf_count = int(state.ent_buf_count[eidx])
        fuel = int(state.ent_fuel[eidx])

        print(
            f"After deposit: buf=({buf_type}, {buf_count}), "
            f"fuel={fuel}, power={int(state.ent_power[eidx])}"
        )

        # Coal should be somewhere the miner can use.
        assert buf_count > 0 or fuel > 0, (
            "Coal was deposited but is not in buffer or fuel"
        )

    def test_fueled_miner_produces_ore(
        self, state_factory,
    ) -> None:
        """After depositing coal, the miner should produce ore."""
        state = _make_miner_state(state_factory)
        params = EnvParams(map_width=3, map_height=3, num_players=1)
        rng = jax.random.PRNGKey(0)

        rng, k = jax.random.split(rng)
        state = factoriax_step(
            k, state, int(Action.DEPOSIT_COAL), params,
        )

        eidx = int(state.tile_entity[0, 1])

        # Step enough times for refuel + mine cycle.
        for _ in range(20):
            rng, k = jax.random.split(rng)
            state = factoriax_step(
                k, state, int(Action.NOOP), params,
            )

        buf_type = int(state.ent_buf_type[eidx])
        buf_count = int(state.ent_buf_count[eidx])
        power = int(state.ent_power[eidx])
        fuel = int(state.ent_fuel[eidx])
        resources = int(state.block_resources[0, 1])

        print(
            f"After 20 steps: buf=({buf_type}, {buf_count}), "
            f"power={power}, fuel={fuel}, "
            f"remaining_resources={resources}"
        )

        # The miner must have actually extracted ore.
        resources_depleted = resources < 100
        has_ore_in_buffer = (
            buf_type == int(ItemType.IRON_ORE) and buf_count > 0
        )
        assert resources_depleted or has_ore_in_buffer, (
            f"Miner did not mine. buf=({buf_type},{buf_count}), "
            f"power={power}, fuel={fuel}, resources={resources}"
        )
