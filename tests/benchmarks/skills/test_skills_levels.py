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
    """L.1 navigate: 5x5 grass, single coal patch, varying spawn + coal."""

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

    def test_has_exactly_one_coal_tile(self) -> None:
        """Each seed places one coal tile — not zero, not multiple."""
        for seed in range(20):
            level, _, _ = build_navigate_level(seed=seed)
            coal_count = int((level.block_map == int(BlockType.COAL)).sum())
            assert coal_count == 1, f"seed={seed}: {coal_count} coal tiles (want 1)"

    def test_canonical_seed_is_deterministic(self) -> None:
        """Same seed must produce byte-identical layout across calls."""
        l1, _, _ = build_navigate_level(seed=0)
        l2, _, _ = build_navigate_level(seed=0)
        assert l1.player_positions == l2.player_positions
        np.testing.assert_array_equal(l1.block_map, l2.block_map)
        assert l1.name == l2.name

    def test_different_seeds_produce_different_layouts(self) -> None:
        """Seeds 0..7 should give multiple distinct (spawn, coal) pairs."""
        layouts: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for seed in range(8):
            level, _, _ = build_navigate_level(seed=seed)
            assert level.player_positions is not None
            ys, xs = np.where(level.block_map == int(BlockType.COAL))
            spawn = tuple(level.player_positions[0])
            coal = (int(xs[0]), int(ys[0]))
            layouts.add((spawn, coal))
        assert len(layouts) >= 4

    def test_spawn_at_distance_at_least_two_from_coal(self) -> None:
        """Spawn never adjacent to coal — keeps the navigation non-trivial."""
        for seed in range(50):
            level, _, _ = build_navigate_level(seed=seed)
            assert level.player_positions is not None
            sx, sy = level.player_positions[0]
            ys, xs = np.where(level.block_map == int(BlockType.COAL))
            cx, cy = int(xs[0]), int(ys[0])
            assert abs(sx - cx) + abs(sy - cy) >= 2

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
    """L.2 mine: 5x5 with exactly five 1-resource ore tiles, spawn at centre."""

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

    def test_exactly_five_ore_tiles_per_seed(self) -> None:
        """The level must have exactly 5 mineable tiles — not more, not fewer."""
        for seed in range(20):
            level, _, _ = build_mine_level(seed=seed)
            ore_count = int(np.isin(level.block_map, _MINE_BLOCK_VALUES).sum())
            assert ore_count == 5, f"seed={seed}: {ore_count} ore tiles (want 5)"

    def test_each_ore_tile_has_one_resource(self) -> None:
        """Each ore tile carries exactly 1 resource so a single MINE depletes it."""
        for seed in range(10):
            level, _, _ = build_mine_level(seed=seed)
            assert level.block_resources is not None
            ore_mask = np.isin(level.block_map, _MINE_BLOCK_VALUES)
            ore_resources = level.block_resources[ore_mask]
            assert (ore_resources == 1).all()

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
    """L.3 craft_miner: empty inventory, two non-adjacent pallets."""

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

    def test_player_starts_with_empty_inventory(self) -> None:
        """No ingredients pre-loaded — agent must withdraw from pallets."""
        level, _, _ = build_craft_miner_level(seed=0)
        assert level.player_inventory in (None, [])

    def test_has_two_pallets(self) -> None:
        """Each level pre-places exactly two pallets."""
        from factoriax.constants import MachineType

        for seed in range(20):
            level, _, _ = build_craft_miner_level(seed=seed)
            assert level.machine_types is not None
            pallet_count = int((level.machine_types == int(MachineType.PALLET)).sum())
            assert pallet_count == 2, f"seed={seed}: {pallet_count} pallets (want 2)"

    def test_pallets_not_adjacent(self) -> None:
        """Per design rule: the two pallets are not 4-neighbour adjacent."""
        from factoriax.constants import MachineType

        for seed in range(50):
            level, _, _ = build_craft_miner_level(seed=seed)
            assert level.machine_types is not None
            ys, xs = np.where(level.machine_types == int(MachineType.PALLET))
            tiles = list(zip(xs.tolist(), ys.tolist(), strict=True))
            assert len(tiles) == 2
            (xi, yi), (xj, yj) = tiles
            assert abs(xi - xj) + abs(yi - yj) >= 2, (
                f"seed={seed}: pallets at {tiles[0]} and {tiles[1]} are adjacent"
            )

    def test_pallets_pre_filled_with_iron_plate_and_wire(self) -> None:
        """One pallet holds IRON_PLATE, the other holds WIRE."""
        from factoriax.constants import MachineType

        level, _, _ = build_craft_miner_level(seed=0)
        assert level.machine_types is not None
        assert level.machine_inventory is not None
        ys, xs = np.where(level.machine_types == int(MachineType.PALLET))
        items_held: list[int] = []
        for x, y in zip(xs.tolist(), ys.tolist(), strict=True):
            inv = level.machine_inventory[y, x]
            for it in range(1, len(inv)):
                if int(inv[it]) > 0:
                    items_held.append(it)
        assert sorted(items_held) == sorted(
            [int(ItemType.IRON_PLATE), int(ItemType.WIRE)]
        )

    def test_canonical_seed_is_deterministic(self) -> None:
        l1, _, _ = build_craft_miner_level(seed=0)
        l2, _, _ = build_craft_miner_level(seed=0)
        assert l1.machine_types is not None
        assert l2.machine_types is not None
        np.testing.assert_array_equal(l1.machine_types, l2.machine_types)
        assert l1.player_positions == l2.player_positions

    def test_spawn_at_centre(self) -> None:
        """Spawn fixed at centre (2, 2); pallets always avoid that tile."""
        for seed in range(20):
            level, _, _ = build_craft_miner_level(seed=seed)
            assert level.player_positions == [(2, 2)]

    def test_pallet_layout_varies_with_seed(self) -> None:
        """Different seeds produce distinct pallet layouts."""
        from factoriax.constants import MachineType

        layouts: set[tuple[tuple[int, int], ...]] = set()
        for seed in range(20):
            level, _, _ = build_craft_miner_level(seed=seed)
            assert level.machine_types is not None
            ys, xs = np.where(level.machine_types == int(MachineType.PALLET))
            tiles = tuple(sorted(zip(xs.tolist(), ys.tolist(), strict=True)))
            layouts.add(tiles)
        assert len(layouts) >= 4

    def test_level_name_includes_seed(self) -> None:
        l0, _, _ = build_craft_miner_level(seed=0)
        l7, _, _ = build_craft_miner_level(seed=7)
        assert l0.name != l7.name
        assert "seed0" in l0.name
        assert "seed7" in l7.name


class TestPlaceMinerLevel:
    """L.4 place_miner: 5x5 with three non-adjacent 1x1 ore patches."""

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
        """Spawn is fixed at centre (2, 2) — patches always avoid it."""
        for seed in range(20):
            level, _, _ = build_place_miner_level(seed=seed)
            assert level.player_positions == [(2, 2)]

    def test_has_three_ore_tiles(self) -> None:
        """Each level has exactly 3 single-tile iron patches."""
        for seed in range(20):
            level, _, _ = build_place_miner_level(seed=seed)
            iron_count = int((level.block_map == int(BlockType.IRON)).sum())
            assert iron_count == 3, f"seed={seed}: {iron_count} iron tiles (want 3)"

    def test_no_two_patches_adjacent(self) -> None:
        """Per design rule: no two ore tiles are 4-neighbour adjacent."""
        for seed in range(50):
            level, _, _ = build_place_miner_level(seed=seed)
            ys, xs = np.where(level.block_map == int(BlockType.IRON))
            tiles = list(zip(xs.tolist(), ys.tolist(), strict=True))
            for i, (xi, yi) in enumerate(tiles):
                for xj, yj in tiles[i + 1 :]:
                    assert abs(xi - xj) + abs(yi - yj) >= 2, (
                        f"seed={seed}: tiles {(xi, yi)} and {(xj, yj)} are adjacent"
                    )

    def test_centre_tile_never_ore(self) -> None:
        """The centre (spawn tile) must always be grass."""
        for seed in range(20):
            level, _, _ = build_place_miner_level(seed=seed)
            assert int(level.block_map[2, 2]) != int(BlockType.IRON)

    def test_patch_layout_varies_with_seed(self) -> None:
        """Different seeds should produce distinct ore-tile sets."""
        layouts: set[tuple[tuple[int, int], ...]] = set()
        for seed in range(20):
            level, _, _ = build_place_miner_level(seed=seed)
            ys, xs = np.where(level.block_map == int(BlockType.IRON))
            tiles = tuple(sorted(zip(xs.tolist(), ys.tolist(), strict=True)))
            layouts.add(tiles)
        assert len(layouts) >= 4

    def test_level_name_includes_seed(self) -> None:
        l0, _, _ = build_place_miner_level(seed=0)
        l7, _, _ = build_place_miner_level(seed=7)
        assert l0.name != l7.name
        assert "seed0" in l0.name
        assert "seed7" in l7.name
