"""Tests for balanced gathering benchmark level definitions."""

from __future__ import annotations

import numpy as np

from factoriax.benchmarks.balanced_gathering.levels import BALANCED_LEVELS
from factoriax.constants import BlockType
from factoriax.levels import build_state


class TestLevelDefinitions:
    """Validate that all levels are well-formed and buildable."""

    def test_five_levels(self) -> None:
        assert len(BALANCED_LEVELS) == 5

    def test_unique_names(self) -> None:
        names = [bl.name for bl in BALANCED_LEVELS]
        assert len(names) == len(set(names))

    def test_all_levels_buildable(self) -> None:
        for bl in BALANCED_LEVELS:
            state = build_state(bl.level, bl.env_params)
            h, w = state.map.shape
            assert w == bl.env_params.map_width
            assert h == bl.env_params.map_height

    def test_dimensions_match_params(self) -> None:
        for bl in BALANCED_LEVELS:
            assert bl.level.map_width == bl.env_params.map_width
            assert bl.level.map_height == bl.env_params.map_height

    def test_single_player(self) -> None:
        for bl in BALANCED_LEVELS:
            assert bl.env_params.num_players == 1

    def test_all_three_ore_types_present(self) -> None:
        for bl in BALANCED_LEVELS:
            block_map = bl.level.block_map
            has_coal = np.any(block_map == int(BlockType.COAL))
            has_iron = np.any(block_map == int(BlockType.IRON))
            has_copper = np.any(block_map == int(BlockType.COPPER))
            assert has_coal, f"{bl.name} missing coal"
            assert has_iron, f"{bl.name} missing iron"
            assert has_copper, f"{bl.name} missing copper"

    def test_obstacle_level_has_water(self) -> None:
        obstacle = BALANCED_LEVELS[2]
        assert obstacle.name == "obstacle"
        has_water = np.any(obstacle.level.block_map == int(BlockType.WATER))
        assert has_water

    def test_scarce_level_has_unequal_resources(self) -> None:
        scarce = BALANCED_LEVELS[3]
        assert scarce.name == "scarce_one"
        bmap = scarce.level.block_map
        coal_count = int(np.sum(bmap == int(BlockType.COAL)))
        copper_count = int(np.sum(bmap == int(BlockType.COPPER)))
        assert copper_count < coal_count

    def test_difficulty_increases(self) -> None:
        sizes = [
            bl.env_params.map_width * bl.env_params.map_height for bl in BALANCED_LEVELS
        ]
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1], (
                f"Level {i} ({BALANCED_LEVELS[i].name}) is smaller than "
                f"level {i - 1} ({BALANCED_LEVELS[i - 1].name})"
            )
