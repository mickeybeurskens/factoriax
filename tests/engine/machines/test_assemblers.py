"""Tests for the assembler machine system.

Uses the entity-based state model where assembler inputs live in
ent_asm_in_type/count and outputs in ent_asm_out_type/count.
"""

import jax.numpy as jnp

from factoriax.engine.constants import (
    BlockType,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.machines import run_assemblers, update_all_machines
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.step import deposit_to_adjacent

_PARAMS = EnvParams()


def _eid(state: EnvState, y: int, x: int) -> int:
    """Return entity index at grid position (y, x).

    Parameters
    ----------
    state
        Current environment state.
    y
        Row position.
    x
        Column position.

    Returns
    -------
    int
        Entity index, asserted to be >= 0.
    """
    eid = int(state.tile_entity[y, x])
    assert eid >= 0, f"No entity at ({y}, {x})"
    return eid


def _make_assembler_state(
    state_factory,
    *,
    power: int = 0,
    asm_in_type: list[int] | None = None,
    asm_in_count: list[int] | None = None,
    asm_out_type: int = 0,
    asm_out_count: int = 0,
    buf_type: int = 0,
    buf_count: int = 0,
    machine_type: int = int(Machine.ASSEMBLER),
) -> EnvState:
    """Create a 3x3 world with an assembler or furnace at (0, 0).

    Parameters
    ----------
    state_factory
        Conftest fixture for building states.
    power
        Initial ``ent_power`` for the machine.
    asm_in_type
        Input slot types, ``[slot0, slot1]``.
    asm_in_count
        Input slot counts, ``[slot0, slot1]``.
    asm_out_type
        Output item type.
    asm_out_count
        Output item count.
    buf_type
        Buffer item type.
    buf_count
        Buffer item count.
    machine_type
        Machine to place. It defaults to ASSEMBLER. Use FURNACE for
        smelting-recipe tests.

    Returns
    -------
    EnvState
        Configured state.
    """
    shape = (3, 3)
    world_map = jnp.full(shape, int(BlockType.DIRT), dtype=jnp.int32)
    mt = jnp.full(shape, int(Machine.NONE), dtype=jnp.int32)
    mt = mt.at[0, 0].set(machine_type)

    in_types = asm_in_type or [0, 0]
    in_counts = asm_in_count or [0, 0]
    ait = jnp.zeros((*shape, 2), dtype=jnp.int32)
    ait = ait.at[0, 0, 0].set(in_types[0])
    ait = ait.at[0, 0, 1].set(in_types[1])
    aic = jnp.zeros((*shape, 2), dtype=jnp.int32)
    aic = aic.at[0, 0, 0].set(in_counts[0])
    aic = aic.at[0, 0, 1].set(in_counts[1])

    aot = jnp.zeros(shape, dtype=jnp.int32)
    aot = aot.at[0, 0].set(asm_out_type)
    aoc = jnp.zeros(shape, dtype=jnp.int32)
    aoc = aoc.at[0, 0].set(asm_out_count)

    mp = jnp.zeros(shape, dtype=jnp.int32)
    mp = mp.at[0, 0].set(power)

    bt = jnp.zeros(shape, dtype=jnp.int32)
    bt = bt.at[0, 0].set(buf_type)
    bc = jnp.zeros(shape, dtype=jnp.int32)
    bc = bc.at[0, 0].set(buf_count)

    return state_factory(
        world_map=world_map,
        machine_types=mt,
        machine_power=mp,
        asm_in_type=ait,
        asm_in_count=aic,
        asm_out_type=aot,
        asm_out_count=aoc,
        buffer_type=bt,
        buffer_count=bc,
    )


class TestAssemblerStartsCraft:
    """The assembler consumes the inputs and starts a countdown."""

    def test_iron_plate_recipe_starts(self, state_factory) -> None:
        """The iron plate recipe needs IRON_ORE and COAL. It consumes both."""
        state = _make_assembler_state(
            state_factory,
            machine_type=int(Machine.FURNACE),
            asm_in_type=[int(ItemType.IRON_ORE), int(ItemType.COAL)],
            asm_in_count=[5, 3],
        )
        new = run_assemblers(state, _PARAMS)
        eid = _eid(new, 0, 0)

        assert int(new.ent_power[eid]) == 2
        assert int(new.ent_asm_in_count[eid, 0]) == 0
        assert int(new.ent_asm_in_count[eid, 1]) == 0

    def test_assembler_rejects_smelting_recipe(self, state_factory) -> None:
        """Assemblers cannot run smelting recipes, which are gated to FURNACE."""
        state = _make_assembler_state(
            state_factory,
            asm_in_type=[int(ItemType.IRON_ORE), int(ItemType.COAL)],
            asm_in_count=[5, 3],
        )
        new = run_assemblers(state, _PARAMS)
        eid = _eid(new, 0, 0)

        assert int(new.ent_power[eid]) == 0
        assert int(new.ent_asm_in_count[eid, 0]) == 5

    def test_no_start_without_coal(self, state_factory) -> None:
        """Smelting needs BOTH ore and coal. Ore alone keeps the
        furnace idle. After the LIMESTONE addition every furnace
        recipe is two-input (no coal-alone refractory shortcut), so
        a single populated slot never fires."""
        state = _make_assembler_state(
            state_factory,
            machine_type=int(Machine.FURNACE),
            asm_in_type=[int(ItemType.IRON_ORE), 0],
            asm_in_count=[5, 0],
        )
        new = run_assemblers(state, _PARAMS)
        eid = _eid(new, 0, 0)

        assert int(new.ent_power[eid]) == 0
        assert int(new.ent_asm_in_count[eid, 0]) == 5

    def test_refractory_fires_on_limestone_plus_coal(self, state_factory) -> None:
        """REFRACTORY is now LIMESTONE + COAL in two slots. The
        slot-emptiness gate that used to fire it on coal-alone is no
        longer reachable for any shipped recipe. Every furnace
        recipe needs both slots populated."""
        state = _make_assembler_state(
            state_factory,
            machine_type=int(Machine.FURNACE),
            asm_in_type=[int(ItemType.LIMESTONE), int(ItemType.COAL)],
            asm_in_count=[1, 1],
        )
        new = run_assemblers(state, _PARAMS)
        eid = _eid(new, 0, 0)

        assert int(new.ent_power[eid]) == 4  # REFRACTORY ticks
        assert int(new.ent_asm_in_count[eid, 0]) == 0
        assert int(new.ent_asm_in_count[eid, 1]) == 0

    def test_refractory_does_not_fire_on_coal_alone(self, state_factory) -> None:
        """Coal-alone used to fire the old 1-input REFRACTORY recipe.
        After the LIMESTONE addition the recipe needs both inputs;
        coal-alone keeps the furnace idle."""
        state = _make_assembler_state(
            state_factory,
            machine_type=int(Machine.FURNACE),
            asm_in_type=[int(ItemType.COAL), 0],
            asm_in_count=[1, 0],
        )
        new = run_assemblers(state, _PARAMS)
        eid = _eid(new, 0, 0)

        assert int(new.ent_power[eid]) == 0
        assert int(new.ent_asm_in_count[eid, 0]) == 1

    def test_no_start_without_ore(self, state_factory) -> None:
        """Empty input slot means nothing to smelt."""
        state = _make_assembler_state(
            state_factory,
            machine_type=int(Machine.FURNACE),
            asm_in_type=[0, 0],
            asm_in_count=[0, 0],
        )
        new = run_assemblers(state, _PARAMS)
        eid = _eid(new, 0, 0)

        assert int(new.ent_power[eid]) == 0


class TestAssemblerCompletesCraft:
    """An assembler at power == 1 produces output."""

    def test_iron_plate_output_produced(self, state_factory) -> None:
        """Power == 1 completes the craft. The output lands in asm_out.

        With Phase 4 removed, the output stays in ``ent_asm_out``.
        It does NOT drain into ``ent_buf``. That drain is now the
        caller's responsibility (withdraw action, arm, or belt).
        """
        state = _make_assembler_state(
            state_factory,
            power=1,
            asm_out_type=int(ItemType.IRON_PLATE),
        )
        new = run_assemblers(state, _PARAMS)
        eid = _eid(new, 0, 0)

        assert int(new.ent_power[eid]) == 0
        assert int(new.ent_asm_out_count[eid]) == 1
        assert int(new.ent_asm_out_type[eid]) == int(ItemType.IRON_PLATE)
        # Buffer is unrelated and stays untouched.
        assert int(new.ent_buf_count[eid]) == 0


class TestAssemblerStallsOutputFull:
    """A machine with an occupied output slot cannot start a new cycle."""

    def test_idle_gate_blocks_new_cycle_while_output_pending(
        self,
        state_factory,
    ) -> None:
        """Even with valid inputs in the slots, the machine does not
        start a new craft while ``ent_asm_out_count > 0``. Pressure to
        withdraw builds naturally."""
        state = _make_assembler_state(
            state_factory,
            machine_type=int(Machine.FURNACE),
            asm_in_type=[int(ItemType.IRON_ORE), 0],
            asm_in_count=[5, 0],
            asm_out_type=int(ItemType.IRON_PLATE),
            asm_out_count=1,
        )
        new = run_assemblers(state, _PARAMS)
        eid = _eid(new, 0, 0)

        # Output still parked in asm_out. Power never ticked up.
        assert int(new.ent_asm_out_count[eid]) == 1
        assert int(new.ent_power[eid]) == 0
        assert int(new.ent_asm_in_count[eid, 0]) == 5


class TestAssemblerTwoInputRecipe:
    """Frame recipe requires two distinct inputs."""

    def test_frame_recipe_starts(self, state_factory) -> None:
        """Frame: 1 iron_plate + 1 tin_plate -> power set, consumed."""
        state = _make_assembler_state(
            state_factory,
            asm_in_type=[int(ItemType.IRON_PLATE), int(ItemType.TIN_PLATE)],
            asm_in_count=[3, 2],
        )
        new = run_assemblers(state, _PARAMS)
        eid = _eid(new, 0, 0)

        assert int(new.ent_power[eid]) == 4
        assert int(new.ent_asm_in_count[eid, 0]) == 0
        assert int(new.ent_asm_in_count[eid, 1]) == 0

    def test_frame_missing_second_input(self, state_factory) -> None:
        """A missing tin_plate prevents the craft from starting."""
        state = _make_assembler_state(
            state_factory,
            asm_in_type=[int(ItemType.IRON_PLATE), 0],
            asm_in_count=[3, 0],
        )
        new = run_assemblers(state, _PARAMS)
        eid = _eid(new, 0, 0)

        assert int(new.ent_power[eid]) == 0


class TestAssemblerRecipeChangeBlocked:
    """Recipe change conditions based on crafting state."""

    def test_power_indicates_mid_craft(self, state_factory) -> None:
        """An assembler mid-craft (power > 0) is not idle."""
        state = _make_assembler_state(state_factory, power=3)
        eid = _eid(state, 0, 0)

        is_idle = int(state.ent_power[eid]) == 0
        assert not is_idle

    def test_items_in_input_slots(self, state_factory) -> None:
        """Items in the input slots are detectable."""
        state = _make_assembler_state(
            state_factory,
            asm_in_type=[int(ItemType.IRON_ORE), 0],
            asm_in_count=[5, 0],
        )
        eid = _eid(state, 0, 0)

        has_inputs = int(state.ent_asm_in_count[eid, 0]) > 0
        assert has_inputs


class TestAssemblerDepositFiltering:
    """Deposits into assembler go to asm_in slots."""

    def test_deposit_goes_to_asm_in(self, state_factory) -> None:
        """Depositing iron_ore into assembler fills asm_in slot."""
        state = _make_assembler_state(state_factory)
        state = state.replace(
            player_positions=jnp.array([[1, 0]], dtype=jnp.int16),
            player_directions=jnp.array(
                [int(Direction.LEFT)],
                dtype=jnp.int8,
            ),
            player_inventory=state.player_inventory.at[0, int(ItemType.IRON_ORE)].set(
                10
            ),
        )
        new = deposit_to_adjacent(state, 0, int(ItemType.IRON_ORE))
        eid = _eid(new, 0, 0)

        in_c0 = int(new.ent_asm_in_count[eid, 0])
        in_c1 = int(new.ent_asm_in_count[eid, 1])
        assert in_c0 + in_c1 == 1

    def test_second_item_type_goes_to_slot1(self, state_factory) -> None:
        """A different item type fills the second input slot."""
        state = _make_assembler_state(
            state_factory,
            asm_in_type=[int(ItemType.IRON_ORE), 0],
            asm_in_count=[2, 0],
        )
        state = state.replace(
            player_positions=jnp.array([[1, 0]], dtype=jnp.int16),
            player_directions=jnp.array(
                [int(Direction.LEFT)],
                dtype=jnp.int8,
            ),
            player_inventory=state.player_inventory.at[0, int(ItemType.COPPER_ORE)].set(
                10
            ),
        )
        new = deposit_to_adjacent(state, 0, int(ItemType.COPPER_ORE))
        eid = _eid(new, 0, 0)

        assert int(new.ent_asm_in_count[eid, 0]) == 2
        assert int(new.ent_asm_in_count[eid, 1]) == 1
        assert int(new.ent_asm_in_type[eid, 1]) == int(ItemType.COPPER_ORE)


class TestAssemblerPlacementAndPickup:
    """The assembler is placed and queryable in the entity state."""

    def test_assembler_exists_in_state(self, state_factory) -> None:
        """Placing an assembler sets the correct machine type."""
        state = _make_assembler_state(state_factory)
        assert int(state.machine_types[0, 0]) == int(Machine.ASSEMBLER)

    def test_progress_decrements(self, state_factory) -> None:
        """Power > 1 decrements by 1 on each tick."""
        state = _make_assembler_state(state_factory, power=5)
        new = run_assemblers(state, _PARAMS)
        eid = _eid(new, 0, 0)
        assert int(new.ent_power[eid]) == 4


# -------------------------------------------------------------------------
# Assembler input slots and pull conservation
# -------------------------------------------------------------------------


class TestAssemblerInventory:
    """Tests for assembler inventory."""

    def test_assembler_initialized_empty(
        self,
        state_factory,
    ) -> None:
        """An assembler starts with empty buffers."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            machine_types=jnp.array(
                [[Machine.ASSEMBLER]],
                dtype=jnp.int32,
            ),
        )
        eid = _eid(state, 0, 0)
        assert int(state.ent_buf_count[eid]) == 0
        assert jnp.all(state.ent_asm_in_count[eid] == 0)
        assert int(state.ent_asm_out_count[eid]) == 0

    def test_assembler_idle_without_inputs(
        self,
        state_factory,
    ) -> None:
        """An assembler without inputs stays idle."""
        state = state_factory(
            world_map=jnp.array(
                [[BlockType.DIRT]],
                dtype=jnp.int32,
            ),
            machine_types=jnp.array(
                [[Machine.ASSEMBLER]],
                dtype=jnp.int32,
            ),
        )
        params = EnvParams()
        new = update_all_machines(state, params)
        eid = _eid(new, 0, 0)
        assert int(new.ent_buf_count[eid]) == 0
        assert jnp.all(new.ent_asm_in_count[eid] == 0)
        assert int(new.ent_asm_out_count[eid]) == 0


class TestAssemblerPullDoesNotLeakIntoInactiveSlots:
    """Regression: the belt pull must debit only the placed belt.

    ``run_assemblers`` opens with a directional pull: the machine takes one
    item from each adjacent belt facing it. The pull side is gated on
    ``is_assembler_or_furnace``, which carries ``active``, but the side that pays for it
    is not. Every inactive slot clips onto tile (0, 0), so a machine on
    (0, 1) pulling leftwards reads as pulling from all of them at once and
    drives each to ``-1``.
    """

    def _belt_feeding_assembler(self, state_factory) -> EnvState:
        """Build a belt on (0, 0) facing RIGHT into an assembler on (0, 1).

        The belt occupies the corner tile that every inactive slot clips
        onto, which is what puts the inactive slots on the paying side of
        the pull.

        Parameters
        ----------
        state_factory
            The shared ``state_factory`` fixture.

        Returns
        -------
        EnvState
            A state with two active entities and the remaining slots
            inactive at (0, 0). The belt holds one IRON_ORE.
        """
        return state_factory(
            world_map=jnp.array([[BlockType.DIRT, BlockType.DIRT]], dtype=jnp.int32),
            machine_types=jnp.array(
                [[Machine.CONVEYOR_BELT, Machine.ASSEMBLER]], dtype=jnp.int32
            ),
            machine_direction=jnp.array(
                [[Direction.RIGHT, Direction.DOWN]], dtype=jnp.int8
            ),
            buffer_type=jnp.array([[int(ItemType.IRON_ORE), 0]], dtype=jnp.int8),
            buffer_count=jnp.array([[1, 0]], dtype=jnp.int16),
        )

    def test_inactive_slots_keep_empty_buffers_under_pull(self, state_factory) -> None:
        """The belt pays for the pull and no inactive slot goes negative."""
        state = self._belt_feeding_assembler(state_factory)

        state = run_assemblers(state, EnvParams())

        assert state.ent_buf_count[_eid(state, 0, 0)] == 0
        assert state.ent_asm_in_count[_eid(state, 0, 1), 0] == 1

        inactive = state.ent_y < 0
        assert bool(jnp.all(state.ent_buf_count[inactive] == 0))

    def test_pull_conserves_items(self, state_factory) -> None:
        """The IRON_ORE moves from belt buffer to input slot, once."""
        state = self._belt_feeding_assembler(state_factory)

        before = int(jnp.sum(state.ent_buf_count.astype(jnp.int32))) + int(
            jnp.sum(state.ent_asm_in_count.astype(jnp.int32))
        )
        state = run_assemblers(state, EnvParams())
        after = int(jnp.sum(state.ent_buf_count.astype(jnp.int32))) + int(
            jnp.sum(state.ent_asm_in_count.astype(jnp.int32))
        )

        assert after == before
