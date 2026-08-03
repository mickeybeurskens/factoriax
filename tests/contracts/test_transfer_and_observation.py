"""Transfer and observation rules that no single module owns.

This file has the widest import set in the suite: machines, observations,
placement, rewards, step, and tables. Each class pins one defect recorded in
``ISSUES.md``, and they share a theme. A rule the machine passes enforce was
not enforced on a player-facing path or in an observation, so a human and a
policy saw different worlds.
"""

import jax.numpy as jnp
import pytest

from factoriax.engine.constants import (
    NUM_ITEM_TYPES,
    BlockType,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.envs.base import FactoriaxEnv
from factoriax.engine.machines import run_arms
from factoriax.engine.observations import (
    OBSERVATIONS,
    _facing_scalars,
    _reconstruct_machine_direction_grid,
    _reconstruct_slot_grids,
)
from factoriax.engine.placement import pickup_machine
from factoriax.engine.rewards import dense_withdraw_reward
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.step import deposit_to_adjacent
from factoriax.engine.tables import (
    CROSSING_HORIZ_SLOT,
    CROSSING_VERT_SLOT,
    MACHINE_MAX_STACK,
    PLAYER_MAX_STACK,
)

_DIRT = int(BlockType.DIRT)
_COAL = int(ItemType.COAL)


def _eid(state: EnvState, y: int, x: int) -> int:
    """Look up the entity index at grid position (y, x).

    Parameters
    ----------
    state
        Environment state with a ``tile_entity`` grid.
    y
        Row index.
    x
        Column index.

    Returns
    -------
    int
        Entity index at the given tile.
    """
    return int(state.tile_entity[y, x])


def _inv(**items: int) -> jnp.ndarray:
    """Build a one-player inventory array from item id to count.

    Parameters
    ----------
    **items
        Keyword per ``ItemType`` name, lowercased, mapped to a count.

    Returns
    -------
    jnp.ndarray
        Shape ``(1, NUM_ITEM_TYPES)``, int16.
    """
    arr = jnp.zeros((1, NUM_ITEM_TYPES), dtype=jnp.int16)
    for name, count in items.items():
        arr = arr.at[0, int(ItemType[name.upper()])].set(count)
    return arr


class TestFixtureSeedsFullHealth:
    """``state_factory`` must place machines the way the engine does.

    ``place_machine`` always writes ``MACHINE_MAX_HEALTH``, and
    ``pickup_machine`` is gated on full health. A fixture that leaves
    ``ent_health`` at 0 therefore builds machines that can never be picked up,
    so any test asserting on a pickup silently exercises the refused path.
    """

    def test_placed_machines_start_at_full_health(self, state_factory) -> None:
        """Every machine the factory places is at its type's full health."""
        state = state_factory(
            world_map=jnp.full((1, 2), _DIRT, dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.PALLET, Machine.ASSEMBLER]], dtype=jnp.int32
            ),
        )
        for x in (0, 1):
            eid = _eid(state, 0, x)
            assert int(state.ent_health[eid]) > 0

    def test_free_slots_stay_at_zero_health(self, state_factory) -> None:
        """Health is an occupancy marker, so a free slot must read 0."""
        state = state_factory(
            world_map=jnp.full((1, 2), _DIRT, dtype=jnp.int32),
            machine_types=jnp.array([[Machine.PALLET, Machine.NONE]], dtype=jnp.int32),
        )
        free = state.ent_y < 0
        assert bool(jnp.all(state.ent_health[free] == 0))


class TestPickupReturnsEveryStoredItem:
    """A pickup must not destroy what the machine was holding.

    ``pickup_machine`` paid back the machine item and ``ent_buf`` but zeroed
    ``ent_asm_in`` and ``ent_asm_out`` without paying either out, so picking up
    a loaded assembler discarded its inputs and its finished output.
    """

    def _loaded_assembler(self, state_factory) -> EnvState:
        """Build a player facing an assembler holding inputs and an output.

        Parameters
        ----------
        state_factory
            The shared ``state_factory`` fixture.

        Returns
        -------
        EnvState
            Player at (0, 0) facing RIGHT, assembler at (0, 1) holding 5
            IRON_ORE in input slot 0, 2 COAL in slot 1, and 4 IRON_PLATE in
            its output.
        """
        state = state_factory(
            world_map=jnp.full((1, 2), _DIRT, dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.NONE, Machine.ASSEMBLER]], dtype=jnp.int32
            ),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
        )
        e = _eid(state, 0, 1)
        return state.replace(
            ent_asm_in_type=state.ent_asm_in_type.at[e, 0]
            .set(jnp.int8(int(ItemType.IRON_ORE)))
            .at[e, 1]
            .set(jnp.int8(_COAL)),
            ent_asm_in_count=state.ent_asm_in_count.at[e, 0]
            .set(jnp.int16(5))
            .at[e, 1]
            .set(jnp.int16(2)),
            ent_asm_out_type=state.ent_asm_out_type.at[e].set(
                jnp.int8(int(ItemType.IRON_PLATE))
            ),
            ent_asm_out_count=state.ent_asm_out_count.at[e].set(jnp.int16(4)),
        )

    def test_input_and_output_slots_are_paid_out(self, state_factory) -> None:
        """Both input slots and the output slot come back to the player."""
        state = self._loaded_assembler(state_factory)

        after = pickup_machine(state, EnvParams(), 0)

        assert int(after.player_inventory[0, int(ItemType.ASSEMBLER)]) == 1
        assert int(after.player_inventory[0, int(ItemType.IRON_ORE)]) == 5
        assert int(after.player_inventory[0, _COAL]) == 2
        assert int(after.player_inventory[0, int(ItemType.IRON_PLATE)]) == 4

    def test_buffer_contents_still_come_back(self, state_factory) -> None:
        """The pre-existing ``ent_buf`` payout is unchanged."""
        state = state_factory(
            world_map=jnp.full((1, 2), _DIRT, dtype=jnp.int32),
            machine_types=jnp.array([[Machine.NONE, Machine.PALLET]], dtype=jnp.int32),
            buffer_type=jnp.array([[0, _COAL]], dtype=jnp.int8),
            buffer_count=jnp.array([[0, 40]], dtype=jnp.int16),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
        )

        after = pickup_machine(state, EnvParams(), 0)

        assert int(after.player_inventory[0, _COAL]) == 40
        assert int(after.player_inventory[0, int(ItemType.PALLET)]) == 1


class TestPickupRespectsPlayerStack:
    """A pickup must not push a stack past ``PLAYER_MAX_STACK``.

    The old gate checked room for the machine item only, so buffer contents
    were added afterwards with no cap.
    """

    def _full_pallet(self, state_factory, carried: int) -> EnvState:
        """Build a player carrying ``carried`` coal facing a 250-coal pallet.

        Parameters
        ----------
        state_factory
            The shared ``state_factory`` fixture.
        carried
            Coal already in the player's inventory.

        Returns
        -------
        EnvState
            Player at (0, 0) facing RIGHT, pallet at (0, 1).
        """
        return state_factory(
            world_map=jnp.full((1, 2), _DIRT, dtype=jnp.int32),
            machine_types=jnp.array([[Machine.NONE, Machine.PALLET]], dtype=jnp.int32),
            buffer_type=jnp.array([[0, _COAL]], dtype=jnp.int8),
            buffer_count=jnp.array([[0, 250]], dtype=jnp.int16),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            player_inventory=_inv(coal=carried),
        )

    def test_pickup_refused_when_contents_would_overflow(self, state_factory) -> None:
        """Nothing moves when the payout exceeds the stack limit."""
        cap = int(PLAYER_MAX_STACK[_COAL])
        state = self._full_pallet(state_factory, cap - 1)

        after = pickup_machine(state, EnvParams(), 0)

        assert int(after.player_inventory[0, _COAL]) == cap - 1
        assert int(after.player_inventory[0, int(ItemType.PALLET)]) == 0
        assert int(after.machine_types[0, 1]) == int(Machine.PALLET)

    def test_pickup_allowed_when_contents_fit(self, state_factory) -> None:
        """A payout that fits goes through in full."""
        state = self._full_pallet(state_factory, 0)

        after = pickup_machine(state, EnvParams(), 0)

        assert int(after.player_inventory[0, _COAL]) == 250
        assert int(after.player_inventory[0, int(ItemType.PALLET)]) == 1
        assert int(after.machine_types[0, 1]) == int(Machine.NONE)


class TestDepositRespectsMachineCapacity:
    """A player deposit must obey ``MACHINE_MAX_STACK`` like every other path."""

    def _deposit_n(self, state_factory, machine: Machine, n: int) -> int:
        """Deposit coal ``n`` times into ``machine`` and report its buffer.

        Parameters
        ----------
        state_factory
            The shared ``state_factory`` fixture.
        machine
            Machine kind to stand in front of the player.
        n
            How many deposit actions to issue.

        Returns
        -------
        int
            The machine's ``ent_buf_count`` afterwards.
        """
        state = state_factory(
            world_map=jnp.full((1, 2), _DIRT, dtype=jnp.int32),
            machine_types=jnp.array([[Machine.NONE, machine]], dtype=jnp.int32),
            machine_direction=jnp.array(
                [[Direction.DOWN, Direction.DOWN]], dtype=jnp.int8
            ),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            player_inventory=_inv(coal=100),
        )
        for _ in range(n):
            state = deposit_to_adjacent(state, 0, _COAL)
        return int(state.ent_buf_count[_eid(state, 0, 1)])

    @pytest.mark.parametrize(
        "machine",
        [Machine.CONVEYOR_BELT, Machine.SPLITTER, Machine.CROSSING, Machine.ARM],
        ids=["belt", "splitter", "crossing", "arm"],
    )
    def test_deposit_stops_at_the_machine_capacity(
        self, state_factory, machine
    ) -> None:
        """Ten deposits leave the machine at its own limit, not at ten."""
        cap = int(MACHINE_MAX_STACK[int(machine)])
        assert self._deposit_n(state_factory, machine, 10) == cap

    def test_deposit_into_a_zero_capacity_machine_is_refused(
        self, state_factory
    ) -> None:
        """A rocket holds nothing, so nothing lands in it."""
        assert self._deposit_n(state_factory, Machine.ROCKET, 10) == 0

    def test_player_keeps_what_was_refused(self, state_factory) -> None:
        """A refused deposit leaves the item in the inventory."""
        state = state_factory(
            world_map=jnp.full((1, 2), _DIRT, dtype=jnp.int32),
            machine_types=jnp.array([[Machine.NONE, Machine.ROCKET]], dtype=jnp.int32),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            player_inventory=_inv(coal=5),
        )

        after = deposit_to_adjacent(state, 0, _COAL)

        assert int(after.player_inventory[0, _COAL]) == 5


class TestInputSlotCapacity:
    """An ``ent_asm_in`` slot must have a capacity like every other slot."""

    def test_arm_stops_filling_a_full_input_slot(self, state_factory) -> None:
        """An arm feeding a stalled assembler stops at the machine's limit."""
        cap = int(MACHINE_MAX_STACK[int(Machine.ASSEMBLER)])
        state = state_factory(
            world_map=jnp.full((1, 3), _DIRT, dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.PALLET, Machine.ARM, Machine.ASSEMBLER]], dtype=jnp.int32
            ),
            machine_direction=jnp.array(
                [[Direction.DOWN, Direction.RIGHT, Direction.DOWN]], dtype=jnp.int8
            ),
            buffer_type=jnp.array([[_COAL, 0, 0]], dtype=jnp.int8),
            buffer_count=jnp.array([[200, 0, 0]], dtype=jnp.int16),
        )
        asm = _eid(state, 0, 2)
        pallet = _eid(state, 0, 0)
        # Both input slots already full: the arm has nowhere to put an item.
        state = state.replace(
            ent_asm_in_type=state.ent_asm_in_type.at[asm, 0]
            .set(jnp.int8(_COAL))
            .at[asm, 1]
            .set(jnp.int8(_COAL)),
            ent_asm_in_count=state.ent_asm_in_count.at[asm, 0]
            .set(jnp.int16(cap))
            .at[asm, 1]
            .set(jnp.int16(cap)),
        )

        after = run_arms(state, EnvParams())

        assert int(after.ent_asm_in_count[asm, 0]) == cap
        assert int(after.ent_asm_in_count[asm, 1]) == cap
        assert int(after.ent_buf_count[pallet]) == 200

    def test_deposit_stops_filling_a_full_input_slot(self, state_factory) -> None:
        """A player deposit into a full input slot is refused."""
        cap = int(MACHINE_MAX_STACK[int(Machine.ASSEMBLER)])
        state = state_factory(
            world_map=jnp.full((1, 2), _DIRT, dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.NONE, Machine.ASSEMBLER]], dtype=jnp.int32
            ),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
            player_inventory=_inv(coal=10),
        )
        asm = _eid(state, 0, 1)
        state = state.replace(
            ent_asm_in_type=state.ent_asm_in_type.at[asm, 0]
            .set(jnp.int8(_COAL))
            .at[asm, 1]
            .set(jnp.int8(_COAL)),
            ent_asm_in_count=state.ent_asm_in_count.at[asm, 0]
            .set(jnp.int16(cap))
            .at[asm, 1]
            .set(jnp.int16(cap)),
        )

        after = deposit_to_adjacent(state, 0, _COAL)

        assert int(after.ent_asm_in_count[asm, 0]) == cap
        assert int(after.ent_asm_in_count[asm, 1]) == cap
        assert int(after.player_inventory[0, _COAL]) == 10


class TestSlotGridsSeeEveryMachine:
    """The observation's slot channels must not lose a machine to a scatter.

    The grids are built by scattering at clipped entity positions. Every free
    slot clips onto tile (0, 0), so that tile used to collect one real write
    and many stale zero writes, and duplicate scatter indices resolve in an
    unspecified order.
    """

    def test_machine_in_the_corner_is_visible(self, state_factory) -> None:
        """A loaded pallet on tile (0, 0) reports its contents."""
        state = state_factory(
            world_map=jnp.full((2, 2), _DIRT, dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.PALLET, Machine.NONE], [Machine.NONE, Machine.NONE]],
                dtype=jnp.int32,
            ),
            buffer_type=jnp.array([[_COAL, 0], [0, 0]], dtype=jnp.int8),
            buffer_count=jnp.array([[50, 0], [0, 0]], dtype=jnp.int16),
        )

        _, _, _, _, out_t, out_c = _reconstruct_slot_grids(state)

        assert int(out_t[0, 0]) == _COAL
        assert int(out_c[0, 0]) == 50

    def test_direction_in_the_corner_is_visible(self, state_factory) -> None:
        """A belt on tile (0, 0) reports its facing."""
        state = state_factory(
            world_map=jnp.full((2, 2), _DIRT, dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.CONVEYOR_BELT, Machine.NONE], [Machine.NONE, Machine.NONE]],
                dtype=jnp.int32,
            ),
            machine_direction=jnp.array([[Direction.RIGHT, 0], [0, 0]], dtype=jnp.int8),
        )

        grid = _reconstruct_machine_direction_grid(state)

        assert int(grid[0, 0]) == int(Direction.RIGHT)

    def test_science_lab_contents_are_visible(self, state_factory) -> None:
        """A lab keeps its packs in the input slots, so they must show."""
        state = state_factory(
            world_map=jnp.full((1, 2), _DIRT, dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.NONE, Machine.SCIENCE_LAB]], dtype=jnp.int32
            ),
        )
        lab = _eid(state, 0, 1)
        pack = int(ItemType.TIER1_SCIENCE_PACK)
        state = state.replace(
            ent_asm_in_type=state.ent_asm_in_type.at[lab, 0].set(jnp.int8(pack)),
            ent_asm_in_count=state.ent_asm_in_count.at[lab, 0].set(jnp.int16(3)),
        )

        in0_t, in0_c, _, _, _, _ = _reconstruct_slot_grids(state)

        assert int(in0_t[0, 1]) == pack
        assert int(in0_c[0, 1]) == 3

    def test_crossing_axis_contents_are_visible(self, state_factory) -> None:
        """A crossing keeps one buffer per axis in the same two columns."""
        state = state_factory(
            world_map=jnp.full((1, 2), _DIRT, dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.NONE, Machine.CROSSING]], dtype=jnp.int32
            ),
        )
        x = _eid(state, 0, 1)
        state = state.replace(
            ent_asm_in_type=state.ent_asm_in_type.at[x, CROSSING_VERT_SLOT]
            .set(jnp.int8(_COAL))
            .at[x, CROSSING_HORIZ_SLOT]
            .set(jnp.int8(int(ItemType.IRON_ORE))),
            ent_asm_in_count=state.ent_asm_in_count.at[x, CROSSING_VERT_SLOT]
            .set(jnp.int16(2))
            .at[x, CROSSING_HORIZ_SLOT]
            .set(jnp.int16(1)),
        )

        slots = _reconstruct_slot_grids(state)
        types = (slots[0], slots[2])
        counts = (slots[1], slots[3])

        assert int(types[CROSSING_VERT_SLOT][0, 1]) == _COAL
        assert int(counts[CROSSING_VERT_SLOT][0, 1]) == 2
        assert int(types[CROSSING_HORIZ_SLOT][0, 1]) == int(ItemType.IRON_ORE)
        assert int(counts[CROSSING_HORIZ_SLOT][0, 1]) == 1


class TestFacingScalarsNeedAMachine:
    """Facing an empty tile must report zeros, not entity 0's contents."""

    def test_empty_tile_reports_nothing(self, state_factory) -> None:
        """An in-bounds tile with no machine reads as empty on every field."""
        state = state_factory(
            world_map=jnp.full((1, 3), _DIRT, dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.PALLET, Machine.NONE, Machine.NONE]], dtype=jnp.int32
            ),
            buffer_type=jnp.array([[_COAL, 0, 0]], dtype=jnp.int8),
            buffer_count=jnp.array([[50, 0, 0]], dtype=jnp.int16),
            player_position=(1, 0),
            player_direction=int(Direction.RIGHT),
        )

        facing = _facing_scalars(state, 0)

        assert bool(jnp.all(facing == 0.0))

    def test_a_real_machine_still_reports(self, state_factory) -> None:
        """Facing a loaded pallet still reads its buffer."""
        state = state_factory(
            world_map=jnp.full((1, 2), _DIRT, dtype=jnp.int32),
            machine_types=jnp.array([[Machine.NONE, Machine.PALLET]], dtype=jnp.int32),
            buffer_type=jnp.array([[0, _COAL]], dtype=jnp.int8),
            buffer_count=jnp.array([[0, 50]], dtype=jnp.int16),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
        )

        facing = _facing_scalars(state, 0)

        assert float(facing[1]) > 0.0
        assert float(facing[2]) > 0.0


class TestObservationsStayInDeclaredBounds:
    """Every observation value must sit inside the ``Box`` the env declares."""

    @pytest.mark.parametrize("obs", sorted(OBSERVATIONS), ids=sorted(OBSERVATIONS))
    def test_a_full_pallet_does_not_break_the_bound(self, state_factory, obs) -> None:
        """Facing a pallet at capacity keeps every scalar within 0 to 1."""
        env = FactoriaxEnv(map_width=3, map_height=1, obs=obs, obs_radius=1)
        params = EnvParams()
        cap = int(MACHINE_MAX_STACK[int(Machine.PALLET)])
        state = state_factory(
            world_map=jnp.full((1, 3), _DIRT, dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.NONE, Machine.PALLET, Machine.NONE]], dtype=jnp.int32
            ),
            buffer_type=jnp.array([[0, _COAL, 0]], dtype=jnp.int8),
            buffer_count=jnp.array([[0, cap, 0]], dtype=jnp.int16),
            player_position=(0, 0),
            player_direction=int(Direction.RIGHT),
        )

        values = env.get_obs(state, params)
        space = env.observation_space(params)

        assert float(jnp.max(values)) <= float(space.high)
        assert float(jnp.min(values)) >= float(space.low)


class TestWithdrawRewardSeesTheCorner:
    """The proximity mask must not lose a target to the free-slot scatter.

    Free slots hold ``ent_y == ent_x == -1``, which Python indexing wraps to
    the last row and column rather than clipping.
    """

    def test_loaded_miner_in_the_far_corner_is_found(self, state_factory) -> None:
        """A miner with output at (2, 2) scores its true distance."""
        h, w = 3, 3
        state = state_factory(
            world_map=jnp.full((h, w), _DIRT, dtype=jnp.int32),
            machine_types=jnp.zeros((h, w), dtype=jnp.int32)
            .at[h - 1, w - 1]
            .set(int(Machine.MINER)),
            buffer_type=jnp.zeros((h, w), dtype=jnp.int8).at[h - 1, w - 1].set(_COAL),
            buffer_count=jnp.zeros((h, w), dtype=jnp.int16).at[h - 1, w - 1].set(3),
            player_position=(0, 0),
        )

        reward = float(dense_withdraw_reward(state, state, EnvParams()))

        # Manhattan distance 4 gives 1/5; the no-target floor is 1/(1+h+w).
        assert reward == pytest.approx(1.0 / 5.0)
