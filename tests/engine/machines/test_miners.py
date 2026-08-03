"""Tests for the miner pass in :mod:`factoriax.engine.machines`.

``run_miners`` pulls from the tile under each miner and pushes into its output
buffer. The tests cover extraction, the output cap, depletion, several miners
at once, and that a push writes no slot the miner does not own.
"""

import jax.numpy as jnp
import pytest

from factoriax.engine.constants import (
    BlockType,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.machines import (
    MINER_OUTPUT_CAP,
    run_miners,
)
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.tables import MACHINE_MAX_STACK
from tests.helpers.states import entity_at as _eid

_MINER_MAX_STACK = int(MACHINE_MAX_STACK[int(Machine.MINER)])
_MINER_BUF_CAP = _MINER_MAX_STACK

# Room a miner's buffer has, which is far above MINER_OUTPUT_CAP: the cap
# bounds what a miner mines, not what its buffer holds.


class TestMinerOperation:
    """Tests for miner machine behavior."""

    def test_miner_extracts_resources(self, state_factory) -> None:
        """A miner extracts resources from the block under it."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
        )
        params = EnvParams()

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.block_resources[0, 0] == 47  # 50 - 3 (mining rate)
        assert new_state.ent_buf_count[eid] == 3
        assert new_state.ent_buf_type[eid] == ItemType.COAL

    def test_miner_mines_nothing_when_buffer_is_above_the_cap(
        self, state_factory
    ) -> None:
        """A buffer past MINER_OUTPUT_CAP stops mining without going negative.

        A miner only mines up to :data:`MINER_OUTPUT_CAP`, but a belt or an arm
        can push its buffer well past that, up to ``MACHINE_MAX_STACK``. The
        free-space term is then negative and has to clamp at zero rather than
        subtract from the tile.
        """
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
            buffer_type=jnp.array([[int(ItemType.COAL)]], dtype=jnp.int8),
            buffer_count=jnp.array(
                [[_MINER_MAX_STACK]],
                dtype=jnp.int16,
            ),
        )
        params = EnvParams()

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.block_resources[0, 0] == 50
        assert new_state.ent_buf_count[eid] == _MINER_MAX_STACK

    def test_miner_mines_nothing_when_buffer_is_at_the_cap(self, state_factory) -> None:
        """A buffer exactly at MINER_OUTPUT_CAP stops mining."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
            buffer_type=jnp.array([[int(ItemType.COAL)]], dtype=jnp.int8),
            buffer_count=jnp.array([[MINER_OUTPUT_CAP]], dtype=jnp.int16),
        )
        params = EnvParams()

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.block_resources[0, 0] == 50
        assert new_state.ent_buf_count[eid] == MINER_OUTPUT_CAP

    def test_miner_stops_when_no_resources(self, state_factory) -> None:
        """A miner stops when its block has no resources."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[0]], dtype=jnp.int16),
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
        )
        params = EnvParams()

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_buf_count[eid] == 0

    def test_miner_depletes_block_to_dirt(self, state_factory) -> None:
        """A block becomes dirt when a miner fully depletes it."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[2]], dtype=jnp.int16),
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
            max_machines=1,
        )
        params = EnvParams()

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.map[0, 0] == BlockType.DIRT
        assert new_state.block_resources[0, 0] == 0
        assert new_state.ent_buf_count[eid] == 2

    def test_miner_depletes_block_to_dirt_with_free_slots_present(
        self, state_factory
    ) -> None:
        """Depletion converts the tile even when free entity slots exist.

        ``run_miners`` writes the map with ``.at[ey, ex].set(...)``. Free slots
        clip onto tile (0, 0) and write the unchanged block type to that same
        index, so a miner standing on (0, 0) puts one DIRT write and many
        stale-value writes into one scatter. Duplicate scatter indices resolve
        in an unspecified order, so the depletion can be lost. The test above
        misses this because ``max_machines=1`` leaves no free slot.
        """
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[2]], dtype=jnp.int16),
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
        )

        new_state = run_miners(state, EnvParams())

        assert new_state.block_resources[0, 0] == 0
        assert new_state.map[0, 0] == BlockType.DIRT

    def test_miner_mines_partial_when_limited_by_resources(
        self,
        state_factory,
    ) -> None:
        """A miner extracts only the available resources below the rate."""
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[1]], dtype=jnp.int16),
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
        )
        params = EnvParams()

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.block_resources[0, 0] == 0
        assert new_state.ent_buf_count[eid] == 1

    def test_miner_mines_partial_when_limited_by_space(
        self,
        state_factory,
    ) -> None:
        """Miner output slot caps at MINER_OUTPUT_CAP (no buffering).

        Full output slot: next tick mines 0 until a withdraw/arm/belt
        drains it.
        """
        state = state_factory(
            world_map=jnp.array([[BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
            buffer_type=jnp.array([[int(ItemType.COAL)]], dtype=jnp.int8),
            buffer_count=jnp.array([[MINER_OUTPUT_CAP - 1]], dtype=jnp.int16),
        )
        params = EnvParams()

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_buf_count[eid] == MINER_OUTPUT_CAP
        assert new_state.block_resources[0, 0] == 49  # one ore extracted


class TestMinerDifferentOres:
    """Tests for miners on different ore types."""

    @pytest.mark.parametrize(
        "block_type, item_type",
        [
            (BlockType.IRON, ItemType.IRON_ORE),
            (BlockType.COPPER, ItemType.COPPER_ORE),
        ],
        ids=["iron", "copper"],
    )
    def test_miner_on_ore(self, state_factory, block_type, item_type) -> None:
        """A miner mines the given ore type correctly."""
        state = state_factory(
            world_map=jnp.array([[block_type]], dtype=jnp.int32),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array([[Machine.MINER]], dtype=jnp.int32),
        )
        params = EnvParams()

        new_state = run_miners(state, params)

        eid = _eid(new_state, 0, 0)
        assert new_state.ent_buf_count[eid] == 3
        assert new_state.ent_buf_type[eid] == item_type


class TestMultipleMiners:
    """Tests for multiple miners operating in parallel."""

    def test_multiple_miners_update_in_parallel(self, state_factory) -> None:
        """Every miner updates in a single step."""
        state = state_factory(
            world_map=jnp.array(
                [
                    [BlockType.COAL, BlockType.IRON],
                    [BlockType.COPPER, BlockType.DIRT],
                ],
                dtype=jnp.int32,
            ),
            block_resources=jnp.array([[50, 50], [50, 0]], dtype=jnp.int16),
            machine_types=jnp.array(
                [
                    [Machine.MINER, Machine.MINER],
                    [Machine.MINER, Machine.NONE],
                ],
                dtype=jnp.int32,
            ),
        )
        params = EnvParams()

        new_state = run_miners(state, params)

        assert new_state.block_resources[0, 0] == 47
        assert new_state.block_resources[0, 1] == 47
        assert new_state.block_resources[1, 0] == 47
        assert new_state.block_resources[1, 1] == 0

        eid_coal = _eid(new_state, 0, 0)
        eid_iron = _eid(new_state, 0, 1)
        eid_copper = _eid(new_state, 1, 0)
        assert new_state.ent_buf_count[eid_coal] == 3
        assert new_state.ent_buf_type[eid_coal] == ItemType.COAL
        assert new_state.ent_buf_count[eid_iron] == 3
        assert new_state.ent_buf_type[eid_iron] == ItemType.IRON_ORE
        assert new_state.ent_buf_count[eid_copper] == 3
        assert new_state.ent_buf_type[eid_copper] == ItemType.COPPER_ORE


class TestMinerPushDoesNotLeakIntoInactiveSlots:
    """Regression: a miner push must reach only the placed receiver.

    Inactive entity slots carry ``ent_y == ent_x == -1`` and the safety
    clip in :func:`run_miners` maps every one of them onto tile (0, 0).
    A miner standing next to (0, 0) and facing into it must not have its
    ore credited to those phantom slots, both because the slots later
    feed freshly-placed machines and because crediting more than one
    receiver per push mints items from nothing.
    """

    def _push_into_corner_state(self, state_factory) -> EnvState:
        """Build a miner at (0, 1) pushing COAL left into a pallet at (0, 0).

        The pallet occupies the corner tile that every inactive slot
        clips onto, so any ungated gather leaks the miner's ore into the
        inactive slots in lockstep with the real pallet.

        Parameters
        ----------
        state_factory
            The shared ``state_factory`` fixture.

        Returns
        -------
        EnvState
            A state with two active entities (pallet, miner) and the
            remaining slots inactive at (0, 0).
        """
        return state_factory(
            world_map=jnp.array([[BlockType.DIRT, BlockType.COAL]], dtype=jnp.int32),
            block_resources=jnp.array([[0, 50]], dtype=jnp.int16),
            machine_types=jnp.array([[Machine.PALLET, Machine.MINER]], dtype=jnp.int32),
            machine_direction=jnp.array(
                [[Direction.DOWN, Direction.LEFT]], dtype=jnp.int8
            ),
        )

    def test_inactive_slots_stay_empty_under_neighbour_push(
        self, state_factory
    ) -> None:
        """Inactive slots keep an empty buffer while the pallet fills."""
        state = self._push_into_corner_state(state_factory)
        params = EnvParams()

        for _ in range(5):
            state = run_miners(state, params)

        pallet_eid = _eid(state, 0, 0)
        assert state.ent_buf_count[pallet_eid] > 0
        assert state.ent_buf_type[pallet_eid] == ItemType.COAL

        inactive = state.ent_y < 0
        assert bool(jnp.all(state.ent_buf_count[inactive] == 0))
        assert bool(jnp.all(state.ent_buf_type[inactive] == 0))

    def test_push_conserves_items(self, state_factory) -> None:
        """Total buffered ore equals total mined ore (no duplication)."""
        state = self._push_into_corner_state(state_factory)
        params = EnvParams()

        for _ in range(5):
            state = run_miners(state, params)

        total_buffered = int(jnp.sum(state.ent_buf_count.astype(jnp.int32)))
        total_mined = int(jnp.sum(state.items_mined))
        assert total_buffered == total_mined


# -------------------------------------------------------------------------
# Miner buffer state
# -------------------------------------------------------------------------


class TestMinerInventory:
    """Tests for miner inventory operations using entity buffers."""

    def test_run_miners_deposits_ore(self, state_factory) -> None:
        """A miner deposits ore into its entity buffer."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.IRON]],
                dtype=jnp.int32,
            ),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array(
                [[Machine.MINER]],
                dtype=jnp.int32,
            ),
        )
        params = EnvParams()
        new = run_miners(state, params)
        eid = _eid(new, 0, 0)
        assert int(new.ent_buf_count[eid]) > 0
        assert int(new.ent_buf_type[eid]) == int(ItemType.IRON_ORE)

    def test_output_full_blocks_mining(self, state_factory) -> None:
        """A miner does not mine when its buffer is at max stack."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.IRON]],
                dtype=jnp.int32,
            ),
            block_resources=jnp.array([[50]], dtype=jnp.int16),
            machine_types=jnp.array(
                [[Machine.MINER]],
                dtype=jnp.int32,
            ),
            buffer_type=jnp.array(
                [[int(ItemType.IRON_ORE)]],
                dtype=jnp.int8,
            ),
            buffer_count=jnp.array(
                [[_MINER_BUF_CAP]],
                dtype=jnp.int16,
            ),
        )
        params = EnvParams()
        new = run_miners(state, params)
        eid = _eid(new, 0, 0)
        assert int(new.ent_buf_count[eid]) == _MINER_BUF_CAP
        assert int(new.block_resources[0, 0]) == 50
