"""Economics oracles for ScienceTiers-v1: the balance spec as tests.

Three invariants pin the tier ladder (tasks/plan_science_tiers.md in
the outer repo). Any recipe rebalance that breaks one fails CI:

1. **Baseline**: a scripted tier-1 hand loop (mine coal+limestone,
   hand-craft, deposit) earns at least ``TIER1_BASELINE_FLOOR`` by the
   1000-tick horizon.
2. **Dominance**: the tier-2 hand loop out-earns tier-1 by at least
   ``TIER2_DOMINANCE_RATIO`` at the horizon: climbing the ladder pays.
3. **Trap**: at ``EARLY_WINDOW_END`` the tier-1 loop still leads.
   climbing costs reward up front, so tier-1 is a genuine local
   optimum for short-sighted play.

Constants come from a 3-seed probe of the scripted loops (2026-07-13,
batch size 5 of each ore per trip): tier-1's first 5-pack batch lands
at t≈42-45 and totals 100-110; tier-2 visits four ore patches before
its first 20-pack batch lands at t≈72-84 and totals 188-220 (1.88-2.0x
tier-1, matching the doubled science-per-ore design target).

Both oracles are phase machines (MINE -> CRAFT -> DEPOSIT, repeat)
over the shared navigation helpers in ``oracle_utils``:

  tier 1: mine 5 coal + 5 limestone -> craft 5 t1 packs -> deposit 5.
  tier 2: mine 5 each of coal/limestone/iron/tin -> craft 5 t1 packs,
          5 wires, 5 t2 packs (each yields 4) -> deposit 20.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax import random

from factoriax.engine.constants import Action, BlockType, ItemType, Machine
from factoriax.engine.envs.science_tiers import (
    SCIENCE_TIERS_MAX_TIMESTEPS,
    science_tiers,
)
from tests.scenarios.oracle_utils import goto_and_act

SEEDS = (0, 1, 2)

#: Probe-pinned balance constants: the numbers under test.
TIER1_BASELINE_FLOOR = 100
TIER2_DOMINANCE_RATIO = 1.5
EARLY_WINDOW_END = 70

#: Packs-worth of ore mined per trip. Travel amortizes across the batch.
_BATCH = 5

_ORE_BLOCK = {
    int(ItemType.COAL): int(BlockType.COAL),
    int(ItemType.LIMESTONE): int(BlockType.LIMESTONE),
    int(ItemType.IRON_ORE): int(BlockType.IRON),
    int(ItemType.TIN_ORE): int(BlockType.TIN),
}

_TIER_ORES = {
    1: (int(ItemType.COAL), int(ItemType.LIMESTONE)),
    2: (
        int(ItemType.COAL),
        int(ItemType.LIMESTONE),
        int(ItemType.IRON_ORE),
        int(ItemType.TIN_ORE),
    ),
}
_TIER_PACK = {
    1: int(ItemType.TIER1_SCIENCE_PACK),
    2: int(ItemType.TIER2_SCIENCE_PACK),
}
_TIER_DEPOSIT = {
    1: int(Action.DEPOSIT_TIER1_SCIENCE_PACK),
    2: int(Action.DEPOSIT_TIER2_SCIENCE_PACK),
}


class _TierOracle:
    """Hand loop for one tier: MINE -> CRAFT -> DEPOSIT, repeat."""

    def __init__(self, tier: int) -> None:
        self.tier = tier
        self.phase = "mine"

    def _craft_action(self, inv: np.ndarray) -> int | None:
        """Next hand-craft toward this tier's pack, or None when done."""
        if self.tier == 2:
            if inv[int(ItemType.TIER1_SCIENCE_PACK)] and inv[int(ItemType.WIRE)]:
                return int(Action.CRAFT_TIER2_SCIENCE_PACK)
            if inv[int(ItemType.IRON_ORE)] and inv[int(ItemType.TIN_ORE)]:
                return int(Action.CRAFT_WIRE)
        if inv[int(ItemType.COAL)] and inv[int(ItemType.LIMESTONE)]:
            return int(Action.CRAFT_TIER1_SCIENCE_PACK)
        return None

    def __call__(self, state) -> int:
        inv = np.asarray(state.player_inventory[0])
        m = np.asarray(state.map)

        if self.phase == "mine":
            for ore in _TIER_ORES[self.tier]:
                if int(inv[ore]) < _BATCH:
                    return goto_and_act(state, m == _ORE_BLOCK[ore], int(Action.MINE))
            self.phase = "craft"

        if self.phase == "craft":
            action = self._craft_action(inv)
            if action is not None:
                return action
            self.phase = "deposit"

        if int(inv[_TIER_PACK[self.tier]]) > 0:
            labs = np.asarray(state.machine_types) == int(Machine.SCIENCE_LAB)
            return goto_and_act(state, labs, _TIER_DEPOSIT[self.tier])
        self.phase = "mine"
        return self(state)


def _rollout(env, params, step_fn, tier: int, seed: int) -> np.ndarray:
    """Cumulative reward per tick for one tier oracle, full horizon."""
    oracle = _TierOracle(tier)
    key = random.PRNGKey(seed)
    _, state = env.reset_env(key, params)
    cum = np.zeros(SCIENCE_TIERS_MAX_TIMESTEPS)
    total = 0.0
    for t in range(SCIENCE_TIERS_MAX_TIMESTEPS):
        action = oracle(state)
        key, ks = random.split(key)
        _, state, reward, _, _ = step_fn(ks, state, action, params)
        total += float(reward)
        cum[t] = total
    return cum


@pytest.fixture(scope="module")
def curves():
    """Cumulative-reward curve per (tier, seed), shared by all tests."""
    env, params = science_tiers()
    step_fn = jax.jit(env.step_env)
    return {
        (tier, seed): _rollout(env, params, step_fn, tier, seed)
        for tier in (1, 2)
        for seed in SEEDS
    }


@pytest.mark.parametrize("seed", SEEDS)
def test_hand_tier1_oracle_earns_baseline(curves, seed) -> None:
    """Invariant 1: the tier-1 hand loop reaches the reward floor."""
    total = curves[(1, seed)][-1]
    assert total >= TIER1_BASELINE_FLOOR, (
        f"tier-1 oracle earned {total}, floor is {TIER1_BASELINE_FLOOR}"
    )
    print(f"\nScienceTiers tier-1 oracle seed {seed}: {total:.0f} by horizon")


@pytest.mark.parametrize("seed", SEEDS)
def test_tier2_overtakes_tier1_within_horizon(curves, seed) -> None:
    """Invariant 2: climbing to tier 2 pays off by the horizon."""
    t1 = curves[(1, seed)][-1]
    t2 = curves[(2, seed)][-1]
    assert t2 >= TIER2_DOMINANCE_RATIO * t1, (
        f"tier-2 earned {t2} vs tier-1 {t1} "
        f"(ratio {t2 / t1:.2f} < {TIER2_DOMINANCE_RATIO})"
    )
    print(f"\nScienceTiers seed {seed}: T2/T1 ratio {t2 / t1:.2f}")


@pytest.mark.parametrize("seed", SEEDS)
def test_tier1_leads_in_early_window(curves, seed) -> None:
    """Invariant 3: greedy tier-1 play still leads at the early bound."""
    t1 = curves[(1, seed)][EARLY_WINDOW_END - 1]
    t2 = curves[(2, seed)][EARLY_WINDOW_END - 1]
    assert t1 > 0, f"tier-1 oracle earned nothing by t={EARLY_WINDOW_END}"
    assert t1 >= t2, (
        f"tier-2 already leads at t={EARLY_WINDOW_END}: {t2} > {t1}. "
        "The tier-1 trap window is gone"
    )
