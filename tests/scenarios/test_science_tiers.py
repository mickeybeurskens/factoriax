"""Tests for the ScienceTiers-v1 scenario.

Contracts: registration and the cross-scenario obs-shape invariant,
start-state (six-patch world, pre-placed lab, empty inventory), the
tier recipe book's proportional-output economics, and the reward wiring
(reward == packs consumed by labs that tick, any tier, weight 1).

The escalating-tier economics oracles (tier-2 setups overtaking tier-1
hand play inside the horizon) live in their own module once the
balance numbers are pinned.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax import random

from factoriax.engine.constants import (
    Action,
    ItemType,
    Machine,
)
from factoriax.engine.envs.easy_rocket import easy_rocket
from factoriax.engine.envs.science_tiers import (
    SCIENCE_TIERS_MAX_TIMESTEPS,
    SCIENCE_TIERS_RECIPES,
    TIER_ORE_COSTS,
    TIER_OUTPUTS,
    TIER_PACKS,
    science_tiers,
)
from factoriax.make import env_from_name

_BLOCK_COUNT_PER_PATCH = 4  # six 2x2 patches, one per ore block


@pytest.fixture(scope="module")
def science_env():
    return science_tiers()


@pytest.fixture(scope="module")
def science_step(science_env):
    env, _ = science_env
    return jax.jit(env.step_env)


def test_science_tiers_registered() -> None:
    env, params = env_from_name("ScienceTiers-v1")
    obs, _ = env.reset_env(random.PRNGKey(0), params)
    assert obs.shape == env.observation_space(params).shape


def test_obs_shape_matches_easy_rocket(science_env) -> None:
    """Transfer invariant: same obs surface as the curriculum family."""
    env, params = science_env
    er_env, er_params = easy_rocket(obs=env.obs, obs_radius=env.obs_radius)
    assert (
        env.observation_space(params).shape == er_env.observation_space(er_params).shape
    )


def test_start_state_has_lab_and_empty_inventory(science_env) -> None:
    """One pre-placed lab the player is facing; nothing in inventory."""
    env, params = science_env
    assert int(params.max_timesteps) == SCIENCE_TIERS_MAX_TIMESTEPS
    for seed in range(3):
        _, state = env.reset_env(random.PRNGKey(seed), params)
        machines = np.asarray(state.machine_types)
        assert int((machines == int(Machine.SCIENCE_LAB)).sum()) == 1
        assert int((machines != int(Machine.NONE)).sum()) == 1
        assert not np.asarray(state.player_inventory).any()


def test_tier_economics_double_per_ore() -> None:
    """Ore costs double per tier while science/ore doubles too.

    cost 2/4/8 ore with outputs 1/4/16 -> 0.5 / 1.0 / 2.0 science per
    ore. The reward weighs every pack at 1, so the tier ladder lives
    entirely in the recipe book.
    """
    assert TIER_ORE_COSTS == (2, 4, 8)
    assert TIER_OUTPUTS == (1, 4, 16)
    by_output = {r.output: r for r in SCIENCE_TIERS_RECIPES}
    for pack, out in zip(TIER_PACKS, TIER_OUTPUTS, strict=True):
        assert by_output[pack].output_count == out
    # Tiers chain: each pack above tier 1 consumes the pack below it.
    assert int(ItemType.TIER1_SCIENCE_PACK) in {
        item for item, _ in by_output[int(ItemType.TIER2_SCIENCE_PACK)].inputs
    }
    assert int(ItemType.TIER2_SCIENCE_PACK) in {
        item for item, _ in by_output[int(ItemType.TIER3_SCIENCE_PACK)].inputs
    }


def test_deposited_packs_pay_one_each(science_env, science_step) -> None:
    """DEPOSIT a pack -> lab consumes it the same tick -> +1 reward."""
    env, params = science_env
    key = random.PRNGKey(0)
    _, state = env.reset_env(key, params)

    inv = state.player_inventory.at[0, int(ItemType.TIER1_SCIENCE_PACK)].set(3)
    state = state.replace(player_inventory=inv)

    total = 0.0
    for _ in range(3):
        key, ks = random.split(key)
        _, state, reward, _, _ = science_step(
            ks, state, int(Action.DEPOSIT_TIER1_SCIENCE_PACK), params
        )
        total += float(reward)
    assert total == 3.0
    assert int(state.player_inventory[0, int(ItemType.TIER1_SCIENCE_PACK)]) == 0


def test_higher_tier_packs_also_pay_one_each(science_env, science_step) -> None:
    """A tier-3 pack pays the same per unit — value is volume, not weight."""
    env, params = science_env
    key = random.PRNGKey(1)
    _, state = env.reset_env(key, params)

    inv = state.player_inventory.at[0, int(ItemType.TIER3_SCIENCE_PACK)].set(1)
    state = state.replace(player_inventory=inv)

    key, ks = random.split(key)
    _, state, reward, _, _ = science_step(
        ks, state, int(Action.DEPOSIT_TIER3_SCIENCE_PACK), params
    )
    assert float(reward) == 1.0


def test_hand_crafting_tier1_pack_works(science_env, science_step) -> None:
    """1 coal + 1 limestone hand-crafts into one tier-1 pack."""
    env, params = science_env
    key = random.PRNGKey(2)
    _, state = env.reset_env(key, params)

    inv = state.player_inventory
    inv = inv.at[0, int(ItemType.COAL)].set(1)
    inv = inv.at[0, int(ItemType.LIMESTONE)].set(1)
    state = state.replace(player_inventory=inv)

    key, ks = random.split(key)
    _, state, reward, _, _ = science_step(
        ks, state, int(Action.CRAFT_TIER1_SCIENCE_PACK), params
    )
    assert int(state.player_inventory[0, int(ItemType.TIER1_SCIENCE_PACK)]) == 1
    assert int(state.player_inventory[0, int(ItemType.COAL)]) == 0
    assert float(reward) == 0.0  # crafting itself pays nothing


def test_no_early_termination(science_env, science_step) -> None:
    """Only the timestep budget ends the episode."""
    env, params = science_env
    key = random.PRNGKey(3)
    _, state = env.reset_env(key, params)
    key, ks = random.split(key)
    _, state, _, done, _ = science_step(ks, state, int(Action.NOOP), params)
    assert not bool(done)
