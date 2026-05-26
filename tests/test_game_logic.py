"""Tests for game logic orchestration (entity-based state).

Covers the action dispatcher ``_handle_player_action``, compound
deposit/withdraw actions, and the top-level ``factoriax_step``.
"""

import jax
import jax.numpy as jnp
import pytest

from factoriax import Action, BlockType, Direction, EnvParams, ItemType
from factoriax.engine.constants import (
    NUM_ITEM_TYPES,
    Machine,
)
from factoriax.engine.game_logic import (
    _handle_player_action,
    deposit_to_adjacent,
    factoriax_step,
    withdraw_from_adjacent,
)

_PARAMS = EnvParams()


class TestHandlePlayerAction:
    """Tests for the action dispatch function."""

    def test_up_moves_north_on_map(self, state_factory) -> None:
        """UP should move the player north (y-1) and face UP."""
        state = state_factory(
            world_map=jnp.full(
                (3, 3),
                BlockType.DIRT,
                dtype=jnp.int32,
            ),
            player_position=(1, 1),
            player_direction=int(Direction.RIGHT),
        )
        new_state = _handle_player_action(state, _PARAMS, Action.UP, 0)
        assert jnp.array_equal(
            new_state.player_positions[0],
            jnp.array([1, 0]),
        )
        # Movement now sets facing to the movement direction.
        assert int(new_state.player_directions[0]) == Direction.UP

    def test_left_moves_west_on_map(self, state_factory) -> None:
        """LEFT should move the player west (x-1)."""
        state = state_factory(
            world_map=jnp.full(
                (3, 3),
                BlockType.DIRT,
                dtype=jnp.int32,
            ),
            player_position=(1, 1),
            player_direction=int(Direction.DOWN),
        )
        new_state = _handle_player_action(state, _PARAMS, Action.LEFT, 0)
        assert jnp.array_equal(
            new_state.player_positions[0],
            jnp.array([0, 1]),
        )

    def test_face_changes_facing_without_moving(
        self,
        state_factory,
    ) -> None:
        """FACE_RIGHT should set facing in place without moving."""
        state = state_factory(
            world_map=jnp.full(
                (3, 3),
                BlockType.DIRT,
                dtype=jnp.int32,
            ),
            player_position=(1, 1),
            player_direction=int(Direction.UP),
        )
        new_state = _handle_player_action(state, _PARAMS, Action.FACE_RIGHT, 0)
        assert jnp.array_equal(
            new_state.player_positions[0],
            jnp.array([1, 1]),
        )
        assert int(new_state.player_directions[0]) == Direction.RIGHT

    @pytest.mark.parametrize(
        "action, expected_dir",
        [
            (Action.FACE_UP, Direction.UP),
            (Action.FACE_DOWN, Direction.DOWN),
            (Action.FACE_LEFT, Direction.LEFT),
            (Action.FACE_RIGHT, Direction.RIGHT),
        ],
    )
    def test_face_action_dispatches(
        self,
        state_factory,
        action: Action,
        expected_dir: Direction,
    ) -> None:
        """FACE_* actions should set facing."""
        state = state_factory(
            world_map=jnp.full(
                (3, 3),
                BlockType.DIRT,
                dtype=jnp.int32,
            ),
            player_position=(1, 1),
            player_direction=int(Direction.UP),
        )
        new_state = _handle_player_action(state, _PARAMS, action, 0)
        assert int(new_state.player_directions[0]) == expected_dir

    def test_mine_decrements_resources(self, state_factory) -> None:
        """MINE should extract a resource from the tile the player faces."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT, BlockType.COAL]],
                dtype=jnp.int32,
            ),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            block_resources=jnp.array([[0, 10]], dtype=jnp.int16),
        )
        new_state = _handle_player_action(state, _PARAMS, Action.MINE, 0)
        assert int(new_state.block_resources[0, 1]) == 9
        assert int(new_state.player_inventory[0, ItemType.COAL]) == 1

    def test_noop_preserves_state(self, state_factory) -> None:
        """NOOP should not change position or inventory."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT]],
                dtype=jnp.int32,
            ),
        )
        new_state = _handle_player_action(state, _PARAMS, Action.NOOP, 0)
        assert jnp.array_equal(
            new_state.player_positions,
            state.player_positions,
        )
        assert jnp.array_equal(
            new_state.player_inventory,
            state.player_inventory,
        )


class TestCompoundDeposit:
    """Tests for typed deposit actions."""

    def test_deposit_into_miner_is_noop(self, state_factory) -> None:
        """Miners have no input slot; depositing any item is a no-op."""
        inv = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int32)
        inv = inv.at[0, ItemType.COAL].set(10)
        state = state_factory(
            world_map=jnp.full(
                (3, 3),
                BlockType.DIRT,
                dtype=jnp.int32,
            ),
            player_position=(1, 0),
            player_direction=int(Direction.DOWN),
            player_inventory=inv,
            machine_types=jnp.full(
                (3, 3),
                Machine.NONE,
                dtype=jnp.int32,
            )
            .at[1, 1]
            .set(Machine.MINER),
        )
        new = deposit_to_adjacent(state, 0, int(ItemType.COAL))
        assert int(new.player_inventory[0, ItemType.COAL]) == 10


class TestCompoundWithdraw:
    """Tests for typed withdraw actions."""

    def test_withdraw_iron_from_miner(self, state_factory) -> None:
        """WITHDRAW drains the whole miner buffer in one action."""
        state = state_factory(
            world_map=jnp.full(
                (3, 3),
                BlockType.DIRT,
                dtype=jnp.int32,
            ),
            player_position=(1, 0),
            player_direction=int(Direction.DOWN),
            machine_types=jnp.full(
                (3, 3),
                Machine.NONE,
                dtype=jnp.int32,
            )
            .at[1, 1]
            .set(Machine.MINER),
            buffer_type=jnp.zeros((3, 3), dtype=jnp.int8)
            .at[1, 1]
            .set(int(ItemType.IRON_ORE)),
            buffer_count=jnp.zeros((3, 3), dtype=jnp.int16).at[1, 1].set(8),
        )
        new = withdraw_from_adjacent(state, 0)
        eid = int(state.tile_entity[1, 1])
        assert int(new.player_inventory[0, ItemType.IRON_ORE]) == 8
        assert int(new.ent_buf_count[eid]) == 0

    def test_withdraw_empty_is_noop(self, state_factory) -> None:
        """Withdrawing an item that isn't there should be a no-op."""
        state = state_factory(
            world_map=jnp.full(
                (3, 3),
                BlockType.DIRT,
                dtype=jnp.int32,
            ),
            player_position=(1, 0),
            player_direction=int(Direction.DOWN),
            machine_types=jnp.full(
                (3, 3),
                Machine.NONE,
                dtype=jnp.int32,
            )
            .at[1, 1]
            .set(Machine.MINER),
        )
        new = withdraw_from_adjacent(state, 0)
        assert int(new.player_inventory[0, ItemType.IRON_ORE]) == 0


class TestFactoriaxStep:
    """Tests for the top-level environment step function."""

    def test_increments_timestep(self, state_factory) -> None:
        """Each step should increment the timestep by 1."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT]],
                dtype=jnp.int32,
            ),
        )
        rng = jax.random.PRNGKey(0)
        params = EnvParams(map_width=1, map_height=1)
        new_state = factoriax_step(rng, state, Action.NOOP, params)
        assert int(new_state.timestep) == int(state.timestep) + 1

    def test_mine_and_machines_in_single_step(
        self,
        state_factory,
    ) -> None:
        """A step should run player action, crafting, and machines."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.COAL, BlockType.DIRT, BlockType.IRON]],
                dtype=jnp.int32,
            ),
            player_position=(1, 0),
            player_direction=int(Direction.LEFT),
            block_resources=jnp.array([[10, 0, 50]], dtype=jnp.int16),
            machine_types=jnp.array(
                [[Machine.NONE, Machine.NONE, Machine.MINER]],
                dtype=jnp.int32,
            ),
            machine_power=jnp.array([[0, 0, 10]], dtype=jnp.int32),
        )
        rng = jax.random.PRNGKey(0)
        params = EnvParams(map_width=3, map_height=1)
        new_state = factoriax_step(rng, state, Action.MINE, params)

        assert int(new_state.block_resources[0, 0]) == 9
        assert int(new_state.block_resources[0, 2]) < 50
