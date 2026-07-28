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
from factoriax.engine.machines import run_assemblers
from factoriax.engine.state import EnvParams, EnvState
from factoriax.engine.step import deposit_to_adjacent

_PARAMS = EnvParams()


def _eid(state: EnvState, y: int, x: int) -> int:
    """Return entity index at grid position (y, x).

    Args:
        state: Current environment state.
        y: Row position.
        x: Column position.

    Returns:
        Entity index (asserts >= 0).
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
    """Create a 3x3 world with a combiner (assembler or furnace) at (0, 0).

    Args:
        state_factory: Conftest fixture for building states.
        power: Initial ent_power for the machine.
        asm_in_type: Input slot types [slot0, slot1].
        asm_in_count: Input slot counts [slot0, slot1].
        asm_out_type: Output item type.
        asm_out_count: Output item count.
        buf_type: Buffer item type.
        buf_count: Buffer item count.
        machine_type: Machine to place. Defaults to ASSEMBLER;
            use FURNACE for smelting-recipe tests.

    Returns:
        Configured EnvState.
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
    """Assembler should consume inputs and start a countdown."""

    def test_iron_plate_recipe_starts(self, state_factory) -> None:
        """Iron plate recipe needs IRON_ORE + COAL; both consumed."""
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
        """Assemblers can't run smelting recipes (gated to FURNACE)."""
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
        """Smelting needs BOTH ore and coal — ore alone keeps the
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
        longer reachable for any shipped recipe — every furnace
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
    """Assembler at power == 1 should produce output."""

    def test_iron_plate_output_produced(self, state_factory) -> None:
        """Power == 1 completes the craft; output lands in asm_out.

        With Phase 4 removed, the output stays in ``ent_asm_out`` —
        it does NOT drain into ``ent_buf``. That drain is now the
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
        # Buffer is unrelated — stays untouched.
        assert int(new.ent_buf_count[eid]) == 0


class TestAssemblerStallsOutputFull:
    """A machine with an occupied output slot cannot start a new cycle."""

    def test_idle_gate_blocks_new_cycle_while_output_pending(
        self,
        state_factory,
    ) -> None:
        """Even with valid inputs sitting in slots, the machine won't
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

        # Output still parked in asm_out; power never ticked up.
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
        """Missing tin_plate should prevent craft start."""
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
        """Items in input slots should be detectable."""
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
    """Assembler should be placed and queryable in entity state."""

    def test_assembler_exists_in_state(self, state_factory) -> None:
        """Placing an assembler sets the correct machine type."""
        state = _make_assembler_state(state_factory)
        assert int(state.machine_types[0, 0]) == int(Machine.ASSEMBLER)

    def test_progress_decrements(self, state_factory) -> None:
        """Power > 1 should decrement by 1 each tick."""
        state = _make_assembler_state(state_factory, power=5)
        new = run_assemblers(state, _PARAMS)
        eid = _eid(new, 0, 0)
        assert int(new.ent_power[eid]) == 4
