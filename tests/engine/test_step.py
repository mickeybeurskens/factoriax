"""Tests for game logic orchestration (entity-based state).

Covers the action dispatcher ``_handle_player_action``, compound
deposit/withdraw actions, and the top-level ``factoriax_step``.
"""

import jax
import jax.numpy as jnp
import pytest

from factoriax.engine.constants import (
    NUM_ITEM_TYPES,
    Action,
    BlockType,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.step import (
    _handle_player_action,
    deposit_to_adjacent,
    factoriax_step,
    get_block_at,
    is_game_over,
    is_position_in_bounds,
    is_position_walkable,
    move_player,
    withdraw_from_adjacent,
)

_PARAMS = EnvParams()


class TestHandlePlayerAction:
    """Tests for the action dispatch function."""

    def test_up_moves_north_on_map(self, state_factory) -> None:
        """UP moves the player north (y-1) and faces UP."""
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
        """LEFT moves the player west (x-1)."""
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
        """FACE_RIGHT sets the facing in place, with no move."""
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
        """A FACE_* action sets the facing."""
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
        """MINE extracts a resource from the tile that the player faces."""
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
        """NOOP changes neither the position nor the inventory."""
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
        """Miners have no input slot. A deposit of any item is a no-op."""
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
        """A withdraw of an item that is not there is a no-op."""
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
        """Each step increments the timestep by 1."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT]],
                dtype=jnp.int32,
            ),
        )
        rng = jax.random.PRNGKey(0)
        params = EnvParams()
        new_state = factoriax_step(rng, state, Action.NOOP, params)
        assert int(new_state.timestep) == int(state.timestep) + 1

    def test_mine_and_machines_in_single_step(
        self,
        state_factory,
    ) -> None:
        """A step runs the player action, crafting, and the machines."""
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
        params = EnvParams()
        new_state = factoriax_step(rng, state, Action.MINE, params)

        assert int(new_state.block_resources[0, 0]) == 9
        assert int(new_state.block_resources[0, 2]) < 50


# -------------------------------------------------------------------------
# Player movement, facing, and bounds
# -------------------------------------------------------------------------


class TestGameLogic:
    """Tests for game logic."""

    @pytest.fixture
    def simple_state(self, state_factory) -> EnvState:
        """Create a simple state for testing."""
        world_map = jnp.array(
            [
                [BlockType.DIRT, BlockType.DIRT, BlockType.WATER],
                [BlockType.DIRT, BlockType.DIRT, BlockType.DIRT],
                [BlockType.WATER, BlockType.DIRT, BlockType.DIRT],
            ],
            dtype=jnp.int32,
        )
        return state_factory(world_map=world_map, player_position=(1, 1))

    def test_is_position_in_bounds(self) -> None:
        """The position bounds check is correct."""
        assert is_position_in_bounds(jnp.array([0, 0]), 3, 3)
        assert is_position_in_bounds(jnp.array([2, 2]), 3, 3)
        assert not is_position_in_bounds(jnp.array([-1, 0]), 3, 3)
        assert not is_position_in_bounds(jnp.array([3, 0]), 3, 3)
        assert not is_position_in_bounds(jnp.array([0, -1]), 3, 3)
        assert not is_position_in_bounds(jnp.array([0, 3]), 3, 3)

    def test_get_block_at_returns_correct_block(
        self,
        simple_state: EnvState,
    ) -> None:
        """The lookup returns the correct block type at a position."""
        assert get_block_at(simple_state, jnp.array([0, 0])) == BlockType.DIRT
        assert get_block_at(simple_state, jnp.array([2, 0])) == BlockType.WATER

    def test_get_block_at_out_of_bounds(
        self,
        simple_state: EnvState,
    ) -> None:
        """An out-of-bounds position returns OUT_OF_BOUNDS."""
        assert get_block_at(simple_state, jnp.array([-1, 0])) == BlockType.OUT_OF_BOUNDS
        assert get_block_at(simple_state, jnp.array([0, 5])) == BlockType.OUT_OF_BOUNDS

    def test_is_position_walkable_dirt(
        self,
        simple_state: EnvState,
    ) -> None:
        """Dirt is walkable."""
        assert is_position_walkable(simple_state, jnp.array([0, 0]))
        assert is_position_walkable(simple_state, jnp.array([1, 1]))

    def test_is_position_walkable_water(
        self,
        simple_state: EnvState,
    ) -> None:
        """Water is not walkable."""
        assert not is_position_walkable(simple_state, jnp.array([2, 0]))
        assert not is_position_walkable(simple_state, jnp.array([0, 2]))

    def test_is_position_walkable_out_of_bounds(
        self,
        simple_state: EnvState,
    ) -> None:
        """An out-of-bounds tile is not walkable."""
        assert not is_position_walkable(simple_state, jnp.array([-1, 0]))
        assert not is_position_walkable(simple_state, jnp.array([5, 5]))

    def test_is_position_walkable_conveyor_belt(
        self,
        state_factory,
    ) -> None:
        """Conveyor belts are walkable, although they are machines."""
        world_map = jnp.array(
            [[BlockType.DIRT, BlockType.DIRT, BlockType.DIRT]],
            dtype=jnp.int32,
        )
        machine_types = jnp.array(
            [[Machine.NONE, Machine.CONVEYOR_BELT, Machine.MINER]],
            dtype=jnp.int32,
        )
        state = state_factory(
            world_map=world_map,
            player_position=(0, 0),
            machine_types=machine_types,
        )
        # Belt tile is walkable
        assert is_position_walkable(state, jnp.array([1, 0]))
        # Miner tile is not walkable
        assert not is_position_walkable(state, jnp.array([2, 0]))

    def test_up_moves_north(self, state_factory) -> None:
        """UP moves the player north (y-1) on the map."""
        state = state_factory(
            world_map=jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32),
            player_position=(1, 1),
            player_direction=int(Direction.RIGHT),
        )
        new_state = move_player(state, Action.UP, 0)
        assert jnp.array_equal(new_state.player_positions[0], jnp.array([1, 0]))
        # Movement updates facing to the direction of travel.
        assert int(new_state.player_directions[0]) == Direction.UP

    def test_face_does_not_move(self, state_factory) -> None:
        """FACE_LEFT changes the facing without a move."""
        state = state_factory(
            world_map=jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32),
            player_position=(1, 1),
            player_direction=int(Direction.UP),
        )
        new_state = move_player(state, Action.FACE_LEFT, 0)
        assert jnp.array_equal(new_state.player_positions[0], jnp.array([1, 1]))
        assert int(new_state.player_directions[0]) == Direction.LEFT

    @pytest.mark.parametrize(
        "action, expected_dir",
        [
            (Action.FACE_UP, Direction.UP),
            (Action.FACE_DOWN, Direction.DOWN),
            (Action.FACE_LEFT, Direction.LEFT),
            (Action.FACE_RIGHT, Direction.RIGHT),
        ],
    )
    def test_face_sets_direction_without_moving(
        self, state_factory, action: Action, expected_dir: Direction
    ) -> None:
        """A FACE_* action snaps the facing to the target direction."""
        state = state_factory(
            world_map=jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32),
            player_position=(1, 1),
            player_direction=int(Direction.UP),
        )
        new_state = move_player(state, action, 0)
        assert jnp.array_equal(new_state.player_positions[0], jnp.array([1, 1]))
        assert int(new_state.player_directions[0]) == expected_dir

    def test_move_player_blocked_by_water(self, state_factory) -> None:
        """The player does not move into water."""
        world_map = jnp.array(
            [
                [BlockType.DIRT, BlockType.WATER],
                [BlockType.DIRT, BlockType.DIRT],
            ],
            dtype=jnp.int32,
        )
        state = state_factory(
            world_map=world_map,
            player_position=(0, 0),
        )
        # Water is at (1, 0). RIGHT moves x+1.
        new_state = move_player(state, Action.RIGHT, 0)
        assert jnp.array_equal(new_state.player_positions[0], jnp.array([0, 0]))

    def test_move_player_blocked_by_bounds(self, state_factory) -> None:
        """The player does not move out of bounds."""
        state = state_factory(
            world_map=jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32),
            player_position=(0, 0),
        )
        new_state = move_player(state, Action.LEFT, 0)
        assert jnp.array_equal(new_state.player_positions[0], jnp.array([0, 0]))

    def test_noop_does_not_change_position(self, state_factory) -> None:
        """NOOP changes neither the player position nor the direction."""
        state = state_factory(
            world_map=jnp.full((3, 3), BlockType.DIRT, dtype=jnp.int32),
            player_position=(1, 1),
            player_direction=int(Direction.DOWN),
        )
        new_state = move_player(state, Action.NOOP, 0)
        assert jnp.array_equal(new_state.player_positions[0], state.player_positions[0])
        assert int(new_state.player_directions[0]) == Direction.DOWN

    def test_is_game_over_before_max_timesteps(
        self,
        simple_state: EnvState,
    ) -> None:
        """The game is not over before max timesteps."""
        params = EnvParams(max_timesteps=1000)
        assert not is_game_over(simple_state, params)

    def test_is_game_over_at_max_timesteps(
        self,
        simple_state: EnvState,
    ) -> None:
        """The game is over at max timesteps."""
        params = EnvParams(max_timesteps=100)
        state = simple_state.replace(timestep=100)
        assert is_game_over(state, params)
