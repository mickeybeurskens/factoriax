"""Tests for ``build_state`` in :mod:`factoriax.engine.levels`.

``build_state`` turns an authored :class:`Level` into the flat entity
arrays the engine steps. The machine inventories a level ships must land
in the slots their machine kind reads, and no machine may exceed the
capacity the engine allocates.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.engine.constants import (
    BLOCK_MAX_RESOURCES,
    NUM_ITEM_TYPES,
    BlockType,
    Direction,
    ItemType,
    Machine,
)
from factoriax.engine.levels import (
    Level,
    LevelBuilder,
    build_state,
)
from factoriax.engine.state import EnvParams

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_PARAMS = EnvParams()


def _dirt_level(w: int = 8, h: int = 8, name: str = "dirt") -> Level:
    return LevelBuilder(w, h).build(name)


# ---------------------------------------------------------------------------
# Level validation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LevelBuilder
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# default_resources helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _place_players helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# build_state
# ---------------------------------------------------------------------------


class TestBuildState:
    """build_state produces correctly-shaped, zero-initialised JAX states."""

    def test_map_shape(self) -> None:
        state = build_state(_dirt_level(), num_players=1)
        assert state.map.shape == (8, 8)

    def test_player_count_matches_params(self) -> None:
        state2 = build_state(_dirt_level(), num_players=2)
        assert state2.player_positions.shape == (2, 2)
        assert state2.player_directions.shape == (2,)

    def test_inventory_zero_initialised(self) -> None:
        state = build_state(_dirt_level(), num_players=1)
        assert jnp.all(state.player_inventory == 0)

    def test_timestep_zero(self) -> None:
        state = build_state(_dirt_level(), num_players=1)
        assert int(state.timestep) == 0

    def test_auto_fill_resources_from_ore(self) -> None:
        level = LevelBuilder(4, 4).fill_rect(0, 0, 2, 2, BlockType.COAL).build("t")
        state = build_state(level, num_players=1)
        resources = np.array(state.block_resources)
        assert (resources[:2, :2] == BLOCK_MAX_RESOURCES).all()
        assert (resources[2:, :] == 0).all()

    def test_explicit_resources_respected(self) -> None:
        custom = np.zeros((4, 4), dtype=np.int32)
        custom[0, 0] = 42
        level = Level(
            name="t",
            map_width=4,
            map_height=4,
            block_map=np.full((4, 4), int(BlockType.DIRT), dtype=np.int32),
            block_resources=custom,
        )
        state = build_state(level, num_players=1)
        assert int(state.block_resources[0, 0]) == 42

    def test_player_directions_default_down(self) -> None:
        state = build_state(_dirt_level(), num_players=1)
        assert int(state.player_directions[0]) == int(Direction.DOWN)

    def test_machine_types_none_by_default(self) -> None:
        state = build_state(_dirt_level(), num_players=1)
        assert jnp.all(state.machine_types == int(Machine.NONE))

    def test_machine_directions_zero_by_default(self) -> None:
        """Without directions in the level, all entity directions default to zero."""
        state = build_state(_dirt_level(), num_players=1)
        assert jnp.all(state.ent_direction == 0)

    def test_machine_directions_preserved(self) -> None:
        """Directions set in the Level must appear in the built state."""
        dirs = np.zeros((8, 8), dtype=np.int32)
        dirs[3, 3] = int(Direction.RIGHT)
        machines = np.full((8, 8), int(Machine.NONE), dtype=np.int32)
        machines[3, 3] = int(Machine.CONVEYOR_BELT)
        level = Level(
            name="dir_test",
            map_width=8,
            map_height=8,
            block_map=np.full((8, 8), int(BlockType.DIRT), dtype=np.int32),
            machine_types=machines,
            machine_directions=dirs,
        )
        state = build_state(level, num_players=1)
        eid = int(state.tile_entity[3, 3])
        assert eid >= 0, "Expected an entity at tile (3, 3)"
        assert int(state.ent_direction[eid]) == int(Direction.RIGHT)


class TestAssemblerInventoryLoading:
    """Stored items load into the slot the machine actually reads.

    ``Level.machine_inventory`` is item-indexed and carries no slot, so
    ``build_state`` decides per item whether it is an input or the finished
    output. An item the machine's own recipes produce belongs in
    ``ent_asm_out``; anything else is an input.
    """

    @staticmethod
    def _furnace_level(contents: dict[int, int]) -> Level:
        """Build an 8x8 level with one furnace at (3, 3) holding *contents*."""
        machines = np.full((8, 8), int(Machine.NONE), dtype=np.int32)
        machines[3, 3] = int(Machine.FURNACE)
        inventory = np.zeros((8, 8, NUM_ITEM_TYPES), dtype=np.int32)
        for item, count in contents.items():
            inventory[3, 3, item] = count
        return Level(
            name="furnace_inv",
            map_width=8,
            map_height=8,
            block_map=np.full((8, 8), int(BlockType.DIRT), dtype=np.int32),
            machine_types=machines,
            machine_inventory=inventory,
        )

    def test_finished_output_loads_into_the_output_slot(self) -> None:
        """A plate held by a furnace lands in ``ent_asm_out``, not an input."""
        level = self._furnace_level({int(ItemType.IRON_PLATE): 5})
        state = build_state(level, num_players=1)
        eid = int(state.tile_entity[3, 3])

        assert int(state.ent_asm_out_type[eid]) == int(ItemType.IRON_PLATE)
        assert int(state.ent_asm_out_count[eid]) == 5

    def test_finished_output_does_not_occupy_an_input_slot(self) -> None:
        """The output item is absent from ``ent_asm_in``, which feeds crafting."""
        level = self._furnace_level({int(ItemType.IRON_PLATE): 5})
        state = build_state(level, num_players=1)
        eid = int(state.tile_entity[3, 3])

        assert int(state.ent_asm_in_count[eid, 0]) == 0
        assert int(state.ent_asm_in_count[eid, 1]) == 0

    def test_two_inputs_and_an_output_all_survive(self) -> None:
        """A full furnace keeps both inputs and its output.

        Three item types exceed the two input columns, so before the output
        was routed separately the third item was dropped on load.
        """
        level = self._furnace_level(
            {
                int(ItemType.IRON_ORE): 3,
                int(ItemType.COAL): 2,
                int(ItemType.IRON_PLATE): 1,
            }
        )
        state = build_state(level, num_players=1)
        eid = int(state.tile_entity[3, 3])

        loaded_inputs = {
            int(state.ent_asm_in_type[eid, s]): int(state.ent_asm_in_count[eid, s])
            for s in range(2)
        }
        assert loaded_inputs == {
            int(ItemType.IRON_ORE): 3,
            int(ItemType.COAL): 2,
        }
        assert int(state.ent_asm_out_type[eid]) == int(ItemType.IRON_PLATE)
        assert int(state.ent_asm_out_count[eid]) == 1

    def test_input_items_still_load_into_input_slots(self) -> None:
        """Ore and coal remain inputs. Only recipe outputs move."""
        level = self._furnace_level({int(ItemType.IRON_ORE): 4, int(ItemType.COAL): 6})
        state = build_state(level, num_players=1)
        eid = int(state.tile_entity[3, 3])

        loaded = {
            int(state.ent_asm_in_type[eid, s]): int(state.ent_asm_in_count[eid, s])
            for s in range(2)
        }
        assert loaded == {int(ItemType.IRON_ORE): 4, int(ItemType.COAL): 6}
        assert int(state.ent_asm_out_count[eid]) == 0

    def test_output_of_a_different_machine_is_treated_as_an_input(self) -> None:
        """A furnace holding an assembler-made item keeps it as an input.

        ``FRAME`` is produced by an assembler, so a furnace cannot have made
        it. It is stored as an input rather than claimed as this machine's
        finished output.
        """
        level = self._furnace_level({int(ItemType.FRAME): 2})
        state = build_state(level, num_players=1)
        eid = int(state.tile_entity[3, 3])

        assert int(state.ent_asm_in_type[eid, 0]) == int(ItemType.FRAME)
        assert int(state.ent_asm_in_count[eid, 0]) == 2
        assert int(state.ent_asm_out_count[eid]) == 0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Built-in level registry
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Procedural generation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FactoriaxEnv(level=...) + reset_env
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Regressions for the defects recorded in ISSUES.md
# ---------------------------------------------------------------------------


class TestMachineCapacity:
    """A level that overflows the entity table must not build a broken state."""

    def test_machines_beyond_capacity_raise(self) -> None:
        builder = LevelBuilder(10, 10)
        for y in range(10):
            for x in range(10):
                builder.place_machine(x, y, int(Machine.PALLET))
        with pytest.raises(ValueError, match="max_machines"):
            build_state(builder.build("many"), num_players=1, max_machines=8)

    def test_machines_within_capacity_all_get_entities(self) -> None:
        builder = LevelBuilder(4, 4)
        for x in range(3):
            builder.place_machine(x, 0, int(Machine.PALLET))
        state = build_state(builder.build("few"), num_players=1, max_machines=8)
        machine_tiles = np.asarray(state.machine_types) != int(Machine.NONE)
        backed = np.asarray(state.tile_entity) >= 0
        assert np.array_equal(machine_tiles, backed)


class TestMachineInventoryPreserved:
    """Contents a level records must survive the build, not be truncated."""

    def test_pallet_keeps_every_item(self) -> None:
        builder = LevelBuilder(3, 3).place_machine(1, 1, int(Machine.PALLET))
        builder.set_machine_inventory(1, 1, int(ItemType.COAL), 5)
        builder.set_machine_inventory(1, 1, int(ItemType.IRON_ORE), 7)
        with pytest.raises(ValueError, match="one item"):
            build_state(builder.build("pallet"), num_players=1)

    def test_pallet_with_one_item_builds(self) -> None:
        builder = LevelBuilder(3, 3).place_machine(1, 1, int(Machine.PALLET))
        builder.set_machine_inventory(1, 1, int(ItemType.COAL), 5)
        state = build_state(builder.build("pallet"), num_players=1)
        assert int(np.asarray(state.ent_buf_type)[0]) == int(ItemType.COAL)
        assert int(np.asarray(state.ent_buf_count)[0]) == 5


# -------------------------------------------------------------------------
# World generation and the env constructor's bound level
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# The env constructor's bound level
# -------------------------------------------------------------------------
