"""Each scripted baseline solves its level under the per-level mask.

Drives the env step-by-step rather than going through
:class:`BenchmarkRunner` — scripted policies read state directly, so
the obs-only ``BenchmarkRunner.run`` interface doesn't fit. Each test
builds the level, wraps the env in :class:`ActionMaskWrapper` with the
level's blocked_actions, and runs the policy until the target
achievement bit unlocks (or the budget runs out).

JIT-heavy: marked ``@pytest.mark.slow``. Each Phase L slice appends one
``test_<skill>`` block.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from baselines.skills.scripted import SCRIPTED_POLICIES, ScriptedPolicy
from factoriax.benchmarks.skills import SkillsBenchmark, skills_conditions
from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
from factoriax.envs.factoriax_env import FactoriaXEnv
from factoriax.levels import build_state
from factoriax.state import EnvState

pytestmark = pytest.mark.slow


def _run_scripted(
    level_idx: int,
    policy: ScriptedPolicy,
    seed: int = 0,
) -> tuple[bool, int]:
    """Drive ``policy`` through ``SkillsBenchmark.levels()[level_idx]``.

    Builds env + state, wraps in :class:`ActionMaskWrapper` with the
    level's mask, JIT-compiles ``step_env`` once, then loops calling
    ``policy(state, params)`` until the target achievement bit
    unlocks or the step budget runs out.

    Args:
        level_idx: Index into ``SkillsBenchmark().levels()``. Bit
            ``level_idx`` of ``state.achievements_unlocked`` is the
            unlock to watch.
        policy: Scripted policy with signature
            ``(state, params) -> action``.
        seed: PRNG seed for ``step_env`` (deterministic episode).

    Returns:
        ``(solved, timesteps_used)`` — whether the bit unlocked, and
        how many steps the policy took to do it.
    """
    bench = SkillsBenchmark()
    bench_level = bench.levels()[level_idx]
    params = bench_level.env_params

    inner = FactoriaXEnv(achievement_fn=skills_conditions)
    blocked = bench_level.blocked_actions or frozenset()
    env = ActionMaskWrapper(inner, tuple(blocked)) if blocked else inner
    jit_step = jax.jit(env.step_env)

    state: EnvState = build_state(bench_level.level, params)
    rng = jax.random.PRNGKey(seed)

    for t in range(params.max_timesteps):
        action = policy(state, params)
        rng, subkey = jax.random.split(rng)
        _obs, state, _r, _done, _info = jit_step(subkey, state, action, params)
        if bool(jnp.asarray(state.achievements_unlocked)[level_idx]):
            return True, t + 1
    return False, params.max_timesteps


# ---------------------------------------------------------------------------
# L.1 — navigate
# ---------------------------------------------------------------------------


class TestNavigateScripted:
    """The greedy manhattan policy reaches (4, 4) within a tight budget."""

    def test_solves_canonical_seed(self) -> None:
        """Seed 0 solves; budget is generous, but we want efficiency."""
        solved, t = _run_scripted(
            level_idx=0,
            policy=SCRIPTED_POLICIES["navigate"],
            seed=0,
        )
        assert solved, f"navigate scripted failed to solve canonical seed (t={t})"
        # Worst case from any 5x5 spawn that isn't the goal: manhattan
        # distance 8 (spawn (0, 0) to goal (4, 4)). Canonical seed
        # spawn varies but should never need more than ~12 ticks
        # including the action being NOOP'd one tick.
        assert t <= 12, f"navigate solved but took {t} ticks (budget hint: ≤ 12)"

    @pytest.mark.parametrize("seed", [1, 2, 3, 7, 13, 42])
    def test_solves_other_seeds(self, seed: int) -> None:
        """Layout variation: scripted policy solves regardless of spawn."""
        solved, t = _run_scripted(
            level_idx=0,
            policy=SCRIPTED_POLICIES["navigate"],
            seed=seed,
        )
        assert solved, f"navigate scripted failed seed {seed} (t={t})"
        assert t <= 12

    def test_level_index_zero_is_navigate(self) -> None:
        """Bit 0 / level 0 contract: navigate is at index 0."""
        bench = SkillsBenchmark()
        assert bench.levels()[0].name == "navigate"


# ---------------------------------------------------------------------------
# L.2 — mine
# ---------------------------------------------------------------------------


class TestMineScripted:
    """The walk-to-nearest-ore policy mines within a tight budget."""

    def test_solves_canonical_seed(self) -> None:
        solved, t = _run_scripted(
            level_idx=1,
            policy=SCRIPTED_POLICIES["mine"],
            seed=0,
        )
        assert solved, f"mine scripted failed canonical seed (t={t})"
        # On a 5x5 with 40% ore and centre spawn, the nearest ore is
        # almost always 1-2 tiles away. 30 ticks is the plan's
        # acceptance ceiling; we want noticeably better.
        assert t <= 10, f"mine solved but took {t} ticks (target: <= 10)"

    @pytest.mark.parametrize("seed", [1, 2, 3, 7, 13, 42])
    def test_solves_other_seeds(self, seed: int) -> None:
        solved, t = _run_scripted(
            level_idx=1,
            policy=SCRIPTED_POLICIES["mine"],
            seed=seed,
        )
        assert solved, f"mine scripted failed seed {seed} (t={t})"
        assert t <= 15

    def test_level_index_one_is_mine(self) -> None:
        """Bit 1 / level 1 contract: mine is at index 1."""
        bench = SkillsBenchmark()
        assert bench.levels()[1].name == "mine"


# ---------------------------------------------------------------------------
# L.3 — craft_miner
# ---------------------------------------------------------------------------


class TestCraftMinerScripted:
    """The always-CRAFT_MINER policy unlocks bit 2 within a tick or two."""

    def test_solves_canonical_seed(self) -> None:
        solved, t = _run_scripted(
            level_idx=2,
            policy=SCRIPTED_POLICIES["craft_miner"],
            seed=0,
        )
        assert solved, f"craft_miner scripted failed canonical seed (t={t})"
        # Pre-loaded ingredients + one CRAFT_MINER action = solved on tick 1.
        assert t <= 3, f"craft_miner solved but took {t} ticks (target: <= 3)"

    @pytest.mark.parametrize("seed", [1, 2, 3, 7, 13, 42])
    def test_solves_other_seeds(self, seed: int) -> None:
        solved, t = _run_scripted(
            level_idx=2,
            policy=SCRIPTED_POLICIES["craft_miner"],
            seed=seed,
        )
        assert solved, f"craft_miner scripted failed seed {seed} (t={t})"
        assert t <= 3

    def test_level_index_two_is_craft_miner(self) -> None:
        """Bit 2 / level 2 contract: craft_miner is at index 2."""
        bench = SkillsBenchmark()
        assert bench.levels()[2].name == "craft_miner"


# ---------------------------------------------------------------------------
# L.4 — place_miner
# ---------------------------------------------------------------------------


class TestPlaceMinerScripted:
    """The walk-adjacent + face + place policy lands a miner on ore."""

    def test_solves_canonical_seed(self) -> None:
        solved, t = _run_scripted(
            level_idx=3,
            policy=SCRIPTED_POLICIES["place_miner"],
            seed=0,
        )
        assert solved, f"place_miner scripted failed canonical seed (t={t})"
        # Centre to nearest patch tile is <=2 manhattan steps; +1 to face,
        # +1 to place → ~5 ticks worst case.
        assert t <= 8, f"place_miner solved but took {t} ticks (target: <= 8)"

    @pytest.mark.parametrize("seed", [1, 2, 3, 7, 13, 42])
    def test_solves_other_seeds(self, seed: int) -> None:
        solved, t = _run_scripted(
            level_idx=3,
            policy=SCRIPTED_POLICIES["place_miner"],
            seed=seed,
        )
        assert solved, f"place_miner scripted failed seed {seed} (t={t})"
        assert t <= 8

    def test_level_index_three_is_place_miner(self) -> None:
        """Bit 3 / level 3 contract: place_miner is at index 3."""
        bench = SkillsBenchmark()
        assert bench.levels()[3].name == "place_miner"

    def test_does_not_unlock_mine_bit(self) -> None:
        """Distinct from L.2: placing a miner doesn't put ore in inventory."""
        from factoriax.envs.action_mask_wrapper import ActionMaskWrapper
        from factoriax.envs.factoriax_env import FactoriaXEnv
        from factoriax.levels import build_state

        bench = SkillsBenchmark()
        bench_level = bench.levels()[3]
        params = bench_level.env_params
        inner = FactoriaXEnv(achievement_fn=skills_conditions)
        env = ActionMaskWrapper(inner, tuple(bench_level.blocked_actions or ()))
        jit_step = jax.jit(env.step_env)

        state = build_state(bench_level.level, params)
        rng = jax.random.PRNGKey(0)
        policy = SCRIPTED_POLICIES["place_miner"]
        for _ in range(params.max_timesteps):
            action = policy(state, params)
            rng, subkey = jax.random.split(rng)
            _o, state, _r, _d, _i = jit_step(subkey, state, action, params)
            mask = jnp.asarray(state.achievements_unlocked)
            if bool(mask[3]):
                # Solve fired; verify mine bit (1) did NOT also fire.
                assert not bool(mask[1]), (
                    "place_miner accidentally unlocked the mine bit"
                )
                return
        raise AssertionError("place_miner scripted failed to solve in budget")


# ---------------------------------------------------------------------------
# Aggregate: scripted policies score well above zero on SkillsBenchmark
# ---------------------------------------------------------------------------


class TestScriptedAggregate:
    """Scripted policies as a group score well on the curriculum so far."""

    def test_curriculum_aggregate_above_floor(self) -> None:
        """Each scripted policy solves its level; aggregate well above 0.9."""
        bench = SkillsBenchmark()
        scores: list[float] = []
        for i, bench_level in enumerate(bench.levels()):
            policy = SCRIPTED_POLICIES[bench_level.name]
            solved, t = _run_scripted(level_idx=i, policy=policy, seed=0)
            assert solved, f"{bench_level.name!r} scripted failed (t={t})"
            max_t = bench_level.env_params.max_timesteps
            scores.append((max_t - t + 1) / max_t)
        agg = sum(scores) / len(scores)
        assert agg > 0.9, f"curriculum aggregate {agg:.3f} below 0.9 floor"
