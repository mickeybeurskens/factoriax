"""Unit tests for benchmark skill wrappers and level generators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from factoriax.benchmarks.skills.mining import MiningSkill, mining_level
from factoriax.benchmarks.skills.place_miner import (
    PlaceMinerSkill,
    place_miner_level,
)
from factoriax.constants import Action, BlockType, ItemType
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


class _StubInner:
    """Stand-in inner env: ``step_env`` returns a post-step state controlled
    by an injected mutator. Lets the skill-wrapper reward tests below
    exercise the wrappers without paying real ``FactoriaXEnv.step_env``
    compile cost.
    """

    def __init__(self, mutate=lambda state: state) -> None:
        self._mutate = mutate

    @property
    def default_params(self):
        return None

    def step_env(self, key, state, action, params):
        return (
            jnp.zeros(1),
            self._mutate(state),
            jnp.float32(0.0),
            jnp.bool_(False),
            {},
        )

    def reset_env(self, key, params):
        return jnp.zeros(1), object()

    def get_obs(self, state, params):
        return jnp.zeros(1)

    def is_terminal(self, state, params):
        return jnp.bool_(False)

    def action_space(self, params):
        return None

    def observation_space(self, params):
        return None


class TestMiningReward:
    """``MiningSkill.step_env`` reward = sum(new.items_mined - prev.items_mined).

    Stub-based unit tests; no XLA compile. Replaces the previous
    integration that built MiningSkill(inner=FactoriaXEnv(level=...))
    and paid ~3.3s per call for the unique wrapper compile.
    """

    def test_noop_gives_zero(self, canonical_env_8x8_1p) -> None:
        _, params, _, state = canonical_env_8x8_1p
        state = state.replace(items_mined=jnp.zeros_like(state.items_mined))
        skill = MiningSkill(inner=_StubInner(mutate=lambda s: s))
        _, _, reward, _, _ = skill.step_env(
            jax.random.PRNGKey(0), state, int(Action.NOOP), params
        )
        assert float(reward) == 0.0

    def test_items_mined_delta_becomes_reward(self, canonical_env_8x8_1p) -> None:
        _, params, _, state = canonical_env_8x8_1p
        state = state.replace(items_mined=jnp.zeros_like(state.items_mined))

        def _add_three_coal(s):
            return s.replace(items_mined=s.items_mined.at[int(ItemType.COAL)].set(3))

        skill = MiningSkill(inner=_StubInner(mutate=_add_three_coal))
        _, _, reward, _, _ = skill.step_env(
            jax.random.PRNGKey(0), state, int(Action.MINE), params
        )
        assert float(reward) == 3.0


class TestPlaceMinerReward:
    """``PlaceMinerSkill.step_env`` reward = count_miners_on_ore(new_state).

    Stub-based unit tests; no XLA compile.
    """

    def test_no_miners_gives_zero(self, canonical_env_8x8_1p) -> None:
        _, params, _, state = canonical_env_8x8_1p
        # canonical state has no machines placed → reward is zero.
        skill = PlaceMinerSkill(inner=_StubInner(mutate=lambda s: s))
        _, _, reward, _, _ = skill.step_env(
            jax.random.PRNGKey(0), state, int(Action.NOOP), params
        )
        assert float(reward) == 0.0
