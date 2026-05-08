"""Per-level builder tests: shape, determinism, and seed variation.

Each Phase L slice adds one ``test_<skill>_*`` block here. Tests are
fast — they just construct levels and check fields, no JIT or env
stepping.
"""

from __future__ import annotations

from factoriax.benchmarks.skills import build_navigate_level
from factoriax.benchmarks.skills.achievements import NAVIGATE_BLOCKED_ACTIONS


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
