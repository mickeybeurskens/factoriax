"""Per-level builder tests: shape, determinism, and seed variation.

Each Phase L slice adds one ``test_<skill>_*`` block here. Tests are
fast — they just construct levels and check fields, no JIT or env
stepping.
"""

from __future__ import annotations

import numpy as np

from factoriax.benchmarks.skills import (
    build_craft_miner_level,
    build_mine_level,
    build_navigate_level,
    build_place_miner_level,
)
from factoriax.benchmarks.skills.achievements import (
    CRAFT_MINER_BLOCKED_ACTIONS,
    MINE_BLOCKED_ACTIONS,
    NAVIGATE_BLOCKED_ACTIONS,
    PLACE_MINER_BLOCKED_ACTIONS,
)
from factoriax.constants import BlockType, ItemType


class TestNavigateLevel:
    """L.1 navigate: 5x5 grass, fixed-corner goal, varying spawn."""

    def test_shape_is_5x5(self) -> None:
        level, params, _ = build_navigate_level(seed=0)
        assert level.map_width == 5
        assert level.map_height == 5
        assert params.map_width == 5
        assert params.map_height == 5

    def test_max_timesteps_is_200(self) -> None:
        _, params, _ = build_navigate_level(seed=0)
        assert params.max_timesteps == 200

    def test_returns_navigate_blocked_actions(self) -> None:
        _, _, blocked = build_navigate_level(seed=0)
        assert blocked is NAVIGATE_BLOCKED_ACTIONS

    def test_canonical_seed_is_deterministic(self) -> None:
        """Same seed must produce byte-identical layout across calls."""
        l1, _, _ = build_navigate_level(seed=0)
        l2, _, _ = build_navigate_level(seed=0)
        assert l1.player_positions == l2.player_positions
        assert l1.name == l2.name

    def test_different_seeds_produce_different_spawns(self) -> None:
        """Seeds 0 and 1 must give different spawn positions."""
        spawns = set()
        for seed in range(8):
            level, _, _ = build_navigate_level(seed=seed)
            assert level.player_positions is not None
            spawns.add(tuple(level.player_positions[0]))
        # 24 valid spawn tiles (5×5 minus the goal); seeing ≥ 4 unique
        # positions across 8 seeds rules out a constant-spawn bug.
        assert len(spawns) >= 4

    def test_spawn_never_on_goal(self) -> None:
        """The goal tile (4, 4) must not be a spawn for any seed."""
        for seed in range(50):
            level, _, _ = build_navigate_level(seed=seed)
            assert level.player_positions is not None
            assert tuple(level.player_positions[0]) != (4, 4)

    def test_level_name_includes_seed(self) -> None:
        """Per-seed levels are name-distinguishable for caching/debug."""
        l0, _, _ = build_navigate_level(seed=0)
        l7, _, _ = build_navigate_level(seed=7)
        assert l0.name != l7.name
        assert "seed0" in l0.name
        assert "seed7" in l7.name


_MINE_BLOCK_VALUES = (
    int(BlockType.COAL),
    int(BlockType.IRON),
    int(BlockType.COPPER),
)


class TestMineLevel:
    """L.2 mine: 5x5 with ~40% mineable tiles, spawn at centre."""

    def test_shape_is_5x5(self) -> None:
        level, params, _ = build_mine_level(seed=0)
        assert level.map_width == 5
        assert level.map_height == 5
        assert params.map_width == 5
        assert params.map_height == 5

    def test_max_timesteps_is_200(self) -> None:
        _, params, _ = build_mine_level(seed=0)
        assert params.max_timesteps == 200

    def test_returns_mine_blocked_actions(self) -> None:
        _, _, blocked = build_mine_level(seed=0)
        assert blocked is MINE_BLOCKED_ACTIONS

    def test_canonical_seed_is_deterministic(self) -> None:
        l1, _, _ = build_mine_level(seed=0)
        l2, _, _ = build_mine_level(seed=0)
        np.testing.assert_array_equal(l1.block_map, l2.block_map)
        assert l1.name == l2.name

    def test_at_least_five_ore_tiles_per_seed(self) -> None:
        """Level builder should produce >= 5 ore tiles for any of the first 10 seeds."""
        for seed in range(10):
            level, _, _ = build_mine_level(seed=seed)
            ore_count = int(np.isin(level.block_map, _MINE_BLOCK_VALUES).sum())
            assert ore_count >= 5, f"seed={seed}: only {ore_count} ore tiles"

    def test_centre_tile_never_ore(self) -> None:
        """Spawn tile (2, 2) must be grass — agent can't MINE from spawn."""
        for seed in range(20):
            level, _, _ = build_mine_level(seed=seed)
            centre = level.map_width // 2
            assert int(level.block_map[centre, centre]) not in _MINE_BLOCK_VALUES

    def test_uses_three_ore_types(self) -> None:
        """Across enough seeds, at least two distinct ore types appear."""
        observed_types: set[int] = set()
        for seed in range(20):
            level, _, _ = build_mine_level(seed=seed)
            for v in _MINE_BLOCK_VALUES:
                if int((level.block_map == v).sum()) > 0:
                    observed_types.add(v)
        assert len(observed_types) >= 2

    def test_level_name_includes_seed(self) -> None:
        l0, _, _ = build_mine_level(seed=0)
        l7, _, _ = build_mine_level(seed=7)
        assert l0.name != l7.name
        assert "seed0" in l0.name
        assert "seed7" in l7.name


class TestCraftMinerLevel:
    """L.3 craft_miner: pre-loaded ingredients, one-shot CRAFT_MINER."""

    def test_shape_is_5x5(self) -> None:
        level, params, _ = build_craft_miner_level(seed=0)
        assert level.map_width == 5
        assert level.map_height == 5
        assert params.map_width == 5
        assert params.map_height == 5

    def test_max_timesteps_is_400(self) -> None:
        _, params, _ = build_craft_miner_level(seed=0)
        assert params.max_timesteps == 400

    def test_returns_craft_miner_blocked_actions(self) -> None:
        _, _, blocked = build_craft_miner_level(seed=0)
        assert blocked is CRAFT_MINER_BLOCKED_ACTIONS

    def test_starting_inventory_has_iron_plate_and_wire(self) -> None:
        """Player must start with the immediate ingredients."""
        level, _, _ = build_craft_miner_level(seed=0)
        assert level.player_inventory is not None
        items = dict(level.player_inventory)
        assert items.get(int(ItemType.IRON_PLATE), 0) >= 1
        assert items.get(int(ItemType.WIRE), 0) >= 1

    def test_canonical_seed_is_deterministic(self) -> None:
        l1, _, _ = build_craft_miner_level(seed=0)
        l2, _, _ = build_craft_miner_level(seed=0)
        np.testing.assert_array_equal(l1.block_map, l2.block_map)
        assert l1.player_positions == l2.player_positions

    def test_spawn_never_on_coal_tile(self) -> None:
        """Spawn always lands on grass so MINE-from-spawn does nothing."""
        for seed in range(20):
            level, _, _ = build_craft_miner_level(seed=seed)
            assert level.player_positions is not None
            sx, sy = level.player_positions[0]
            assert int(level.block_map[sy, sx]) != int(BlockType.COAL)

    def test_has_coal_tiles_for_mine_action(self) -> None:
        """The MINE action in the mask must have somewhere to do work."""
        for seed in range(10):
            level, _, _ = build_craft_miner_level(seed=seed)
            coal_count = int((level.block_map == int(BlockType.COAL)).sum())
            assert coal_count >= 1

    def test_level_name_includes_seed(self) -> None:
        l0, _, _ = build_craft_miner_level(seed=0)
        l7, _, _ = build_craft_miner_level(seed=7)
        assert l0.name != l7.name
        assert "seed0" in l0.name
        assert "seed7" in l7.name


class TestPlaceMinerLevel:
    """L.4 place_miner: 5x5 with 2x2 ore patch in a varying corner."""

    def test_shape_is_5x5(self) -> None:
        level, params, _ = build_place_miner_level(seed=0)
        assert level.map_width == 5
        assert level.map_height == 5

    def test_max_timesteps_is_300(self) -> None:
        _, params, _ = build_place_miner_level(seed=0)
        assert params.max_timesteps == 300

    def test_returns_place_miner_blocked_actions(self) -> None:
        _, _, blocked = build_place_miner_level(seed=0)
        assert blocked is PLACE_MINER_BLOCKED_ACTIONS

    def test_starting_inventory_has_five_miners(self) -> None:
        level, _, _ = build_place_miner_level(seed=0)
        assert level.player_inventory is not None
        items = dict(level.player_inventory)
        assert items.get(int(ItemType.MINER), 0) >= 5

    def test_canonical_seed_is_deterministic(self) -> None:
        l1, _, _ = build_place_miner_level(seed=0)
        l2, _, _ = build_place_miner_level(seed=0)
        np.testing.assert_array_equal(l1.block_map, l2.block_map)
        assert l1.player_positions == l2.player_positions

    def test_spawn_at_centre(self) -> None:
        """Spawn is fixed at centre (2, 2) — patch corners avoid it."""
        for seed in range(20):
            level, _, _ = build_place_miner_level(seed=seed)
            assert level.player_positions == [(2, 2)]

    def test_has_2x2_ore_patch(self) -> None:
        """Each level has exactly 4 iron tiles arranged as a 2x2 patch."""
        for seed in range(20):
            level, _, _ = build_place_miner_level(seed=seed)
            iron_count = int((level.block_map == int(BlockType.IRON)).sum())
            assert iron_count == 4, f"seed={seed}: {iron_count} iron tiles (want 4)"

    def test_centre_tile_never_ore(self) -> None:
        """The centre (spawn tile) must always be grass."""
        for seed in range(20):
            level, _, _ = build_place_miner_level(seed=seed)
            assert int(level.block_map[2, 2]) != int(BlockType.IRON)

    def test_patch_corner_varies_with_seed(self) -> None:
        """Different seeds should land the patch in different corners."""
        corners_seen: set[tuple[int, int]] = set()
        for seed in range(20):
            level, _, _ = build_place_miner_level(seed=seed)
            ys, xs = np.where(level.block_map == int(BlockType.IRON))
            corners_seen.add((int(xs.min()), int(ys.min())))
        # At least two distinct top-left corners over 20 seeds.
        assert len(corners_seen) >= 2

    def test_level_name_includes_seed(self) -> None:
        l0, _, _ = build_place_miner_level(seed=0)
        l7, _, _ = build_place_miner_level(seed=7)
        assert l0.name != l7.name
        assert "seed0" in l0.name
        assert "seed7" in l7.name
