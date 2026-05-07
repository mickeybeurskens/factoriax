"""Unit tests for benchmark skill wrappers and level generators."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from factoriax.benchmarks.skills.mining import MiningSkill, mining_level
from factoriax.benchmarks.skills.place_miner import (
    PlaceMinerSkill,
    place_miner_level,
)
from factoriax.constants import Action, BlockType, ItemType
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.levels import build_state

# -----------------------------------------------------------------------
# Level generator tests
# -----------------------------------------------------------------------


class TestMiningLevel:
    """Verify mining_level produces valid, parameterized levels."""

    def test_dimensions(self) -> None:
        """Level and params should match the requested map size."""
        level, params = mining_level(map_size=5)
        assert level.map_width == 5
        assert level.map_height == 5
        assert params.map_width == 5
        assert params.map_height == 5

    def test_ore_tiles_present(self) -> None:
        """At least some tiles should be ore."""
        level, params = mining_level(map_size=5, ore_fraction=0.5)
        state = build_state(level, params)
        ore_mask = np.isin(
            np.array(state.map),
            [BlockType.IRON, BlockType.COPPER, BlockType.COAL],
        )
        assert np.sum(ore_mask) > 0

    def test_resources_in_range(self) -> None:
        """Ore tiles should have resources in [min_res, max_res]."""
        level, params = mining_level(min_res=1, max_res=3)
        state = build_state(level, params)
        resources = np.array(state.block_resources)
        ore_mask = np.isin(
            np.array(state.map),
            [BlockType.IRON, BlockType.COPPER, BlockType.COAL],
        )
        ore_resources = resources[ore_mask]
        assert np.all(ore_resources >= 1)
        assert np.all(ore_resources <= 3)

    def test_seed_determinism(self) -> None:
        """Same seed should produce identical levels."""
        l1, _ = mining_level(seed=42)
        l2, _ = mining_level(seed=42)
        assert np.array_equal(l1.block_map, l2.block_map)


class TestPlaceMinerLevel:
    """Verify place_miner_level produces valid levels."""

    def test_miners_in_inventory(self) -> None:
        """Player should start with the requested number of miners."""
        level, params = place_miner_level(num_miners=5)
        state = build_state(level, params)
        miner_count = int(state.player_inventory[0, int(ItemType.MINER)])
        assert miner_count == 5

    def test_ore_patches_present(self) -> None:
        """Map should have ore tiles for placement."""
        level, params = place_miner_level(num_patches=5)
        state = build_state(level, params)
        ore_mask = np.isin(
            np.array(state.map),
            [BlockType.IRON, BlockType.COPPER, BlockType.COAL],
        )
        assert np.sum(ore_mask) > 0

    def test_max_machines_sized(self) -> None:
        """max_machines should be num_miners + 4."""
        _, params = place_miner_level(num_miners=5)
        assert params.max_machines == 9


# -----------------------------------------------------------------------
# Reward tests
# -----------------------------------------------------------------------


@pytest.mark.slow
class TestMiningReward:
    """Verify mining skill reward computation."""

    def test_noop_gives_zero(self) -> None:
        """NOOP should not mine anything."""
        level, params = mining_level()
        env = MiningSkill(inner=FactoriaXEnv(level=level))
        _, state = env.reset_env(jax.random.PRNGKey(0), params)
        step_fn = jax.jit(env.step_env)
        key = jax.random.PRNGKey(0)
        _, _, reward, _, _ = step_fn(key, state, int(Action.NOOP), params)
        assert float(reward) == 0.0


@pytest.mark.slow
class TestPlaceMinerReward:
    """Verify place miner skill reward computation."""

    def test_no_miners_gives_zero(self) -> None:
        """With no miners placed, reward should be zero."""
        level, params = place_miner_level()
        env = PlaceMinerSkill(inner=FactoriaXEnv(level=level))
        _, state = env.reset_env(jax.random.PRNGKey(0), params)
        step_fn = jax.jit(env.step_env)
        key = jax.random.PRNGKey(0)
        _, _, reward, _, _ = step_fn(key, state, int(Action.NOOP), params)
        assert float(reward) == 0.0
