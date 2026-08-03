"""Tests for pallet storage in :mod:`factoriax.engine.machines`.

A pallet holds one item type in one buffer slot and runs no pass of its own.
It is inert: a tick must leave it exactly as it was, and only a neighbour can
add to it or take from it.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.engine.constants import BlockType, Direction, ItemType, Machine
from factoriax.engine.machines import (
    run_arms,
    run_conveyor_belts,
    run_miners,
    update_all_machines,
)
from factoriax.engine.state import EnvParams
from tests.helpers.states import entity_at as _eid


class TestPalletInventory:
    """Tests for pallet inventory."""

    def test_pallet_initialized_empty(self, state_factory) -> None:
        """A pallet starts with an empty buffer."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            machine_types=jnp.array(
                [[Machine.PALLET]],
                dtype=jnp.int32,
            ),
        )
        eid = _eid(state, 0, 0)
        assert int(state.ent_buf_count[eid]) == 0

    def test_pallet_unaffected_by_update(self, state_factory) -> None:
        """update_all_machines does not modify the pallet contents."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            machine_types=jnp.array(
                [[Machine.PALLET]],
                dtype=jnp.int32,
            ),
            buffer_type=jnp.array(
                [[int(ItemType.IRON_ORE)]],
                dtype=jnp.int8,
            ),
            buffer_count=jnp.array([[10]], dtype=jnp.int16),
        )
        params = EnvParams()
        new = update_all_machines(state, params)
        eid = _eid(new, 0, 0)
        assert int(new.ent_buf_count[eid]) == 10
        assert int(new.ent_buf_type[eid]) == int(ItemType.IRON_ORE)


class TestPushIntoZeroCapacityMachine:
    """Regression: a machine that holds nothing must receive nothing.

    ``MACHINE_MAX_STACK`` is 0 for ``ROCKET``. Every push path tests the
    destination with "empty, or same item with room left", and an empty
    destination passes that test whatever its capacity. The transfer then
    clamps to 0 in :func:`run_miners` and :func:`run_conveyor_belts`, which
    leaves the receiver holding an item type and no items, breaking the rule
    that a zero count means a cleared type. :func:`run_arms` does not clamp
    at all and moves the item outright.
    """

    def test_belt_does_not_stamp_its_item_on_a_rocket(self, state_factory) -> None:
        """A belt facing a rocket keeps its item and leaves the rocket clear."""
        state = state_factory(
            world_map=jnp.array([[BlockType.DIRT, BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.CONVEYOR_BELT, Machine.ROCKET]], dtype=jnp.int32
            ),
            machine_direction=jnp.array(
                [[Direction.RIGHT, Direction.DOWN]], dtype=jnp.int8
            ),
            buffer_type=jnp.array([[int(ItemType.COAL), 0]], dtype=jnp.int8),
            buffer_count=jnp.array([[1, 0]], dtype=jnp.int16),
        )

        state = run_conveyor_belts(state, EnvParams())

        rocket = _eid(state, 0, 1)
        assert state.ent_buf_count[_eid(state, 0, 0)] == 1
        assert state.ent_buf_count[rocket] == 0
        assert state.ent_buf_type[rocket] == 0

    def test_miner_does_not_stamp_its_ore_on_a_rocket(self, state_factory) -> None:
        """A miner facing a rocket holds its ore and leaves the rocket clear."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.COAL, BlockType.DIRT]], dtype=jnp.int32
            ),
            block_resources=jnp.array([[0, 50, 0]], dtype=jnp.int16),
            machine_types=jnp.array(
                [[Machine.NONE, Machine.MINER, Machine.ROCKET]], dtype=jnp.int32
            ),
            machine_direction=jnp.array(
                [[Direction.DOWN, Direction.RIGHT, Direction.DOWN]], dtype=jnp.int8
            ),
        )

        state = run_miners(state, EnvParams())

        rocket = _eid(state, 0, 2)
        assert state.ent_buf_count[_eid(state, 0, 1)] > 0
        assert state.ent_buf_count[rocket] == 0
        assert state.ent_buf_type[rocket] == 0

    def test_arm_does_not_deliver_into_a_rocket(self, state_factory) -> None:
        """An arm facing a rocket leaves the item on its source."""
        state = state_factory(
            world_map=jnp.full((1, 3), int(BlockType.DIRT), dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.PALLET, Machine.ARM, Machine.ROCKET]], dtype=jnp.int32
            ),
            machine_direction=jnp.array(
                [[Direction.DOWN, Direction.RIGHT, Direction.DOWN]], dtype=jnp.int8
            ),
            buffer_type=jnp.array([[int(ItemType.COAL), 0, 0]], dtype=jnp.int8),
            buffer_count=jnp.array([[1, 0, 0]], dtype=jnp.int16),
        )

        state = run_arms(state, EnvParams())

        rocket = _eid(state, 0, 2)
        assert state.ent_buf_count[_eid(state, 0, 0)] == 1
        assert state.ent_buf_count[rocket] == 0
        assert state.ent_buf_type[rocket] == 0
