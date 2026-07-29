"""Tests for factoriax.engine.levels: Level, LevelBuilder, build_state, serialization,
procedural generation, and the built-in level registry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
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
    LEVELS,
    Level,
    LevelBuilder,
    _place_players,
    build_state,
    default_resources,
    generate_state,
    get_level,
    load_level,
    save_level,
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


class TestLevelValidation:
    """Level.__post_init__ catches mismatched array shapes."""

    def test_block_map_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="block_map shape"):
            Level(
                name="bad",
                map_width=8,
                map_height=8,
                block_map=np.zeros((3, 4), dtype=np.int32),
            )

    def test_block_resources_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="block_resources shape"):
            Level(
                name="bad",
                map_width=4,
                map_height=4,
                block_map=np.zeros((4, 4), dtype=np.int32),
                block_resources=np.zeros((3, 4), dtype=np.int32),
            )

    def test_machine_types_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="machine_types shape"):
            Level(
                name="bad",
                map_width=4,
                map_height=4,
                block_map=np.zeros((4, 4), dtype=np.int32),
                machine_types=np.zeros((4, 5), dtype=np.int32),
            )

    def test_valid_level_no_error(self) -> None:
        level = _dirt_level()
        assert level.map_width == 8
        assert level.map_height == 8


# ---------------------------------------------------------------------------
# LevelBuilder
# ---------------------------------------------------------------------------


class TestLevelBuilder:
    """LevelBuilder constructs Level objects correctly."""

    def test_default_fill_is_dirt(self) -> None:
        level = LevelBuilder(4, 4).build("test")
        assert (level.block_map == int(BlockType.DIRT)).all()

    def test_custom_default_block(self) -> None:
        level = LevelBuilder(4, 4, default_block=BlockType.WATER).build("test")
        assert (level.block_map == int(BlockType.WATER)).all()

    def test_fill_rect_sets_region(self) -> None:
        level = LevelBuilder(8, 8).fill_rect(0, 0, 3, 3, BlockType.COAL).build("t")
        assert (level.block_map[:3, :3] == int(BlockType.COAL)).all()
        assert (level.block_map[3:, :] == int(BlockType.DIRT)).all()
        assert (level.block_map[:, 3:] == int(BlockType.DIRT)).all()

    def test_fill_rect_clips_to_boundary(self) -> None:
        """fill_rect must not raise when rectangle exceeds map bounds."""
        level = LevelBuilder(4, 4).fill_rect(-1, -1, 10, 10, BlockType.WATER).build("t")
        assert (level.block_map == int(BlockType.WATER)).all()

    def test_multiple_fill_rects_compose(self) -> None:
        level = (
            LevelBuilder(8, 8)
            .fill_rect(0, 0, 4, 4, BlockType.COAL)
            .fill_rect(4, 4, 4, 4, BlockType.IRON)
            .build("t")
        )
        assert (level.block_map[:4, :4] == int(BlockType.COAL)).all()
        assert (level.block_map[4:, 4:] == int(BlockType.IRON)).all()
        # Non-filled quadrants remain DIRT.
        assert (level.block_map[:4, 4:] == int(BlockType.DIRT)).all()

    def test_set_resources_single_tile(self) -> None:
        level = (
            LevelBuilder(4, 4)
            .fill_rect(0, 0, 4, 4, BlockType.COAL)
            .set_resources(1, 1, 50)
            .build("t")
        )
        assert level.block_resources is not None
        assert level.block_resources[1, 1] == 50

    def test_set_resources_out_of_bounds_raises(self) -> None:
        builder = LevelBuilder(4, 4)
        with pytest.raises(IndexError):
            builder.set_resources(10, 0, 50)

    def test_build_returns_independent_copy(self) -> None:
        """Mutating the builder after build must not affect the built Level."""
        builder = LevelBuilder(4, 4)
        level = builder.build("t")
        builder.fill_rect(0, 0, 4, 4, BlockType.WATER)
        assert (level.block_map == int(BlockType.DIRT)).all()

    def test_fluent_chaining(self) -> None:
        """All builder methods return self."""
        builder = LevelBuilder(4, 4)
        assert builder.fill_rect(0, 0, 1, 1, BlockType.COAL) is builder
        assert builder.set_resources(0, 0, 10) is builder


# ---------------------------------------------------------------------------
# default_resources helper
# ---------------------------------------------------------------------------


class TestDefaultResources:
    """default_resources fills ore tiles correctly."""

    def test_ore_tiles_get_max_resources(self) -> None:
        block_map = np.array(
            [[int(BlockType.COAL), int(BlockType.IRON), int(BlockType.COPPER)]],
            dtype=np.int32,
        )
        res = default_resources(block_map)
        assert (res == BLOCK_MAX_RESOURCES).all()

    def test_non_ore_tiles_get_zero(self) -> None:
        block_map = np.array(
            [[int(BlockType.DIRT), int(BlockType.WATER)]],
            dtype=np.int32,
        )
        res = default_resources(block_map)
        assert (res == 0).all()


# ---------------------------------------------------------------------------
# _place_players helper
# ---------------------------------------------------------------------------


class TestPlacePlayers:
    """_place_players centres players and forces spawn tiles to DIRT."""

    def test_single_player_at_centre(self) -> None:
        block_map = np.full((8, 8), int(BlockType.DIRT), dtype=np.int32)
        _, positions = _place_players(block_map, 1)
        assert positions.shape == (1, 2)
        cx, cy = 8 // 2, 8 // 2
        assert positions[0, 0] == cx
        assert positions[0, 1] == cy

    def test_spawn_tile_forced_to_dirt(self) -> None:
        """Even if the spawn tile was COAL, it becomes DIRT."""
        block_map = np.full((8, 8), int(BlockType.COAL), dtype=np.int32)
        new_map, positions = _place_players(block_map, 1)
        px, py = positions[0, 0], positions[0, 1]
        assert new_map[py, px] == int(BlockType.DIRT)

    def test_original_map_not_mutated(self) -> None:
        block_map = np.full((8, 8), int(BlockType.COAL), dtype=np.int32)
        _place_players(block_map, 1)
        assert (block_map == int(BlockType.COAL)).all()

    def test_multiple_players_spread_horizontally(self) -> None:
        block_map = np.full((8, 8), int(BlockType.DIRT), dtype=np.int32)
        _, positions = _place_players(block_map, 2)
        assert positions.shape == (2, 2)
        # All players share the same y.
        assert (positions[:, 1] == positions[0, 1]).all()
        # x-coordinates are distinct (players are offset).
        assert positions[0, 0] != positions[1, 0]


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


class TestCombinerInventoryLoading:
    """A combiner's stored items load into the slot that holds that item.

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
        """Ore and coal remain inputs; only recipe outputs move."""
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


class TestSerialization:
    """save_level and load_level round-trip correctly."""

    def test_roundtrip_minimal_level(self) -> None:
        level = _dirt_level()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "level.json"
            save_level(level, path)
            loaded = load_level(path)
        assert loaded.name == level.name
        assert loaded.map_width == level.map_width
        assert loaded.map_height == level.map_height
        np.testing.assert_array_equal(loaded.block_map, level.block_map)
        assert loaded.block_resources is None
        assert loaded.machine_types is None

    def test_roundtrip_with_resources(self) -> None:
        level = (
            LevelBuilder(4, 4)
            .fill_rect(0, 0, 2, 2, BlockType.COAL)
            .set_resources(0, 0, 77)
            .build("coal_res")
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "level.json"
            save_level(level, path)
            loaded = load_level(path)
        assert loaded.block_resources is not None
        assert loaded.block_resources[0, 0] == 77

    def test_file_is_json(self) -> None:
        """Saved file must be valid JSON (human-readable)."""
        import orjson

        level = _dirt_level()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "level.json"
            save_level(level, path)
            data = orjson.loads(path.read_bytes())
        assert "block_map" in data
        assert "name" in data

    def test_parent_dirs_created(self) -> None:
        level = _dirt_level()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nested" / "dir" / "level.json"
            save_level(level, path)
            assert path.exists()

    def test_roundtrip_with_machine_directions(self) -> None:
        """Machine directions must survive save/load."""
        machines = np.full((4, 4), int(Machine.NONE), dtype=np.int32)
        machines[1, 2] = int(Machine.CONVEYOR_BELT)
        dirs = np.zeros((4, 4), dtype=np.int32)
        dirs[1, 2] = int(Direction.LEFT)
        level = Level(
            name="dir",
            map_width=4,
            map_height=4,
            block_map=np.full((4, 4), int(BlockType.DIRT), dtype=np.int32),
            machine_types=machines,
            machine_directions=dirs,
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "level.json"
            save_level(level, path)
            loaded = load_level(path)
        assert loaded.machine_directions is not None
        np.testing.assert_array_equal(loaded.machine_directions, dirs)

    def test_roundtrip_no_directions_stays_none(self) -> None:
        """Levels without directions must load as None."""
        level = _dirt_level()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "level.json"
            save_level(level, path)
            loaded = load_level(path)
        assert loaded.machine_directions is None

    def test_load_legacy_file_without_directions(self) -> None:
        """Files saved before directions existed must load fine."""
        import orjson

        payload = {
            "name": "legacy",
            "map_width": 2,
            "map_height": 2,
            "block_map": [[2, 2], [2, 2]],
            "block_resources": None,
            "machine_types": None,
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "level.json"
            path.write_bytes(orjson.dumps(payload))
            loaded = load_level(path)
        assert loaded.machine_directions is None

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_level(Path("/nonexistent/path/level.json"))


# ---------------------------------------------------------------------------
# Built-in level registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """LEVELS registry and get_level."""

    def test_levels_dict_nonempty(self) -> None:
        assert len(LEVELS) > 0

    def test_get_level_returns_level(self) -> None:
        level = get_level("15x15_resources")
        assert isinstance(level, Level)

    def test_get_level_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown level"):
            get_level("does_not_exist")

    def test_15x15_resources_dimensions(self) -> None:
        level = get_level("15x15_resources")
        assert level.map_width == 15
        assert level.map_height == 15

    def test_15x15_coal_top_left(self) -> None:
        m = get_level("15x15_resources").block_map
        assert (m[:4, :4] == int(BlockType.COAL)).all()

    def test_15x15_copper_top_right(self) -> None:
        m = get_level("15x15_resources").block_map
        assert (m[:4, 11:] == int(BlockType.COPPER)).all()

    def test_15x15_iron_bottom_left(self) -> None:
        m = get_level("15x15_resources").block_map
        assert (m[11:, :4] == int(BlockType.IRON)).all()

    def test_15x15_remaining_tiles_dirt(self) -> None:
        m = get_level("15x15_resources").block_map
        # Centre region (away from all corners) must be DIRT.
        assert (m[4:11, 4:11] == int(BlockType.DIRT)).all()

    def test_15x15_buildable_for_1_player(self) -> None:
        level = get_level("15x15_resources")
        num_players = 1
        state = build_state(level, num_players=num_players)
        assert state.player_positions.shape == (num_players, 2)


# ---------------------------------------------------------------------------
# Procedural generation
# ---------------------------------------------------------------------------


class TestGenerateState:
    """generate_state produces correctly-shaped states."""

    def test_map_shape_matches_params(self) -> None:
        state = generate_state(
            jax.random.PRNGKey(0), _PARAMS, map_height=16, map_width=16
        )
        assert state.map.shape == (16, 16)

    def test_player_count_matches_params(self) -> None:
        num_players = 3
        state = generate_state(jax.random.PRNGKey(1), _PARAMS, num_players=num_players)
        assert state.player_positions.shape == (num_players, 2)

    def test_spawn_tiles_are_dirt(self) -> None:
        """Every player spawn tile must be DIRT after generation."""
        num_players = 2
        state = generate_state(jax.random.PRNGKey(2), _PARAMS, num_players=num_players)
        positions = np.array(state.player_positions)
        world_map = np.array(state.map)
        for px, py in positions:
            assert world_map[py, px] == int(BlockType.DIRT)

    def test_different_seeds_differ(self) -> None:
        s0 = generate_state(jax.random.PRNGKey(0), _PARAMS)
        s1 = generate_state(jax.random.PRNGKey(99), _PARAMS)
        assert not jnp.array_equal(s0.map, s1.map)

    def test_same_seed_deterministic(self) -> None:
        s0 = generate_state(jax.random.PRNGKey(7), _PARAMS)
        s1 = generate_state(jax.random.PRNGKey(7), _PARAMS)
        np.testing.assert_array_equal(np.array(s0.map), np.array(s1.map))

    def test_ore_tiles_get_base_resources(self) -> None:
        """All ore tiles should have exactly base_resources resources."""
        import numpy as np

        from factoriax.engine.tables import MINEABLE_BLOCKS

        params = EnvParams(base_resources=3)
        state = generate_state(jax.random.PRNGKey(5), params)
        world_map = np.array(state.map)
        resources = np.array(state.block_resources)
        is_ore = np.isin(world_map, [int(b) for b in MINEABLE_BLOCKS])
        assert np.all(resources[is_ore] == 3)
        assert np.all(resources[~is_ore] == 0)

    def test_custom_base_resources(self) -> None:
        """base_resources param controls starting resources on ore tiles."""
        import numpy as np

        from factoriax.engine.tables import MINEABLE_BLOCKS

        for count in (1, 5, 10):
            params = EnvParams(base_resources=count)
            state = generate_state(jax.random.PRNGKey(0), params)
            world_map = np.array(state.map)
            resources = np.array(state.block_resources)
            is_ore = np.isin(world_map, [int(b) for b in MINEABLE_BLOCKS])
            if is_ore.any():
                assert np.all(resources[is_ore] == count)


# ---------------------------------------------------------------------------
# FactoriaxEnv(level=...) + reset_env
# ---------------------------------------------------------------------------


class TestResetWithBoundLevel:
    """FactoriaxEnv constructed with ``level=`` resets to that level."""

    def test_obs_and_state_returned(self) -> None:
        from factoriax.engine.envs.base import FactoriaxEnv

        level = get_level("15x15_resources")
        env = FactoriaxEnv(level=level)
        params = EnvParams()
        obs, state = env.reset_env(jax.random.PRNGKey(0), params)
        assert obs.ndim == 1
        assert state.map.shape == (15, 15)

    def test_obs_shape_matches_observation_space(self) -> None:
        from factoriax.engine.envs.base import FactoriaxEnv

        level = get_level("15x15_resources")
        env = FactoriaxEnv(level=level)
        params = EnvParams()
        obs, _ = env.reset_env(jax.random.PRNGKey(0), params)
        expected = env.observation_space(params).shape[0]
        assert obs.shape == (expected,)

    def test_deterministic_no_key_needed(self) -> None:
        from factoriax.engine.envs.base import FactoriaxEnv

        level = get_level("15x15_resources")
        env = FactoriaxEnv(level=level)
        params = EnvParams()
        _, s1 = env.reset_env(jax.random.PRNGKey(0), params)
        _, s2 = env.reset_env(jax.random.PRNGKey(123), params)
        np.testing.assert_array_equal(np.array(s1.map), np.array(s2.map))
        np.testing.assert_array_equal(
            np.array(s1.player_positions), np.array(s2.player_positions)
        )
