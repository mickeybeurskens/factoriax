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
    """The greedy manhattan policy reaches the coal patch."""

    def test_solves_canonical_seed(self) -> None:
        """Seed 0 solves; the scripted policy walks straight to the coal tile."""
        solved, t = _run_scripted(
            level_idx=0,
            policy=SCRIPTED_POLICIES["navigate"],
            seed=0,
        )
        assert solved, f"navigate scripted failed to solve canonical seed (t={t})"
        # Worst-case manhattan on 5x5 is 8 (corner-to-corner). Add a
        # tick of slack and cap at 10.
        assert t <= 10, f"navigate solved but took {t} ticks (budget hint: ≤ 10)"

    @pytest.mark.parametrize("seed", [1, 2, 3, 7, 13, 42])
    def test_solves_other_seeds(self, seed: int) -> None:
        """Layout variation: scripted policy solves regardless of spawn."""
        solved, t = _run_scripted(
            level_idx=0,
            policy=SCRIPTED_POLICIES["navigate"],
            seed=seed,
        )
        assert solved, f"navigate scripted failed seed {seed} (t={t})"
        assert t <= 10

    def test_level_index_zero_is_navigate(self) -> None:
        """Bit 0 / level 0 contract: navigate is at index 0."""
        bench = SkillsBenchmark()
        assert bench.levels()[0].name == "navigate"


# ---------------------------------------------------------------------------
# L.2 — mine
# ---------------------------------------------------------------------------


class TestMineScripted:
    """Walk-to-nearest-ore + MINE clears all five tiles."""

    def test_solves_canonical_seed(self) -> None:
        solved, t = _run_scripted(
            level_idx=1,
            policy=SCRIPTED_POLICIES["mine"],
            seed=0,
        )
        assert solved, f"mine scripted failed canonical seed (t={t})"
        # Five tiles to clear, each ≈ 1 walk + 1 mine. Adjacent tiles
        # let the policy hop without retreating, so ~10-15 ticks is
        # typical; cap at 30 with slack.
        assert t <= 30, f"mine solved but took {t} ticks (target: <= 30)"

    @pytest.mark.parametrize("seed", [1, 2, 3, 7, 13, 42])
    def test_solves_other_seeds(self, seed: int) -> None:
        solved, t = _run_scripted(
            level_idx=1,
            policy=SCRIPTED_POLICIES["mine"],
            seed=seed,
        )
        assert solved, f"mine scripted failed seed {seed} (t={t})"
        assert t <= 40

    def test_level_index_one_is_mine(self) -> None:
        """Bit 1 / level 1 contract: mine is at index 1."""
        bench = SkillsBenchmark()
        assert bench.levels()[1].name == "mine"


# ---------------------------------------------------------------------------
# L.3 — craft_miner
# ---------------------------------------------------------------------------


class TestCraftMinerScripted:
    """Walk → face → WITHDRAW twice, then CRAFT_MINER."""

    def test_solves_canonical_seed(self) -> None:
        solved, t = _run_scripted(
            level_idx=2,
            policy=SCRIPTED_POLICIES["craft_miner"],
            seed=0,
        )
        assert solved, f"craft_miner scripted failed canonical seed (t={t})"
        # Two pallets to visit: each ≈ walk (≤4) + face (1) + withdraw (1)
        # ≈ 6 ticks; +1 for the final CRAFT_MINER. Total ≤ 15 typical.
        assert t <= 30, f"craft_miner solved but took {t} ticks (target: <= 30)"

    @pytest.mark.parametrize("seed", [1, 2, 3, 7, 13, 42])
    def test_solves_other_seeds(self, seed: int) -> None:
        solved, t = _run_scripted(
            level_idx=2,
            policy=SCRIPTED_POLICIES["craft_miner"],
            seed=seed,
        )
        assert solved, f"craft_miner scripted failed seed {seed} (t={t})"
        assert t <= 40

    def test_level_index_two_is_craft_miner(self) -> None:
        """Bit 2 / level 2 contract: craft_miner is at index 2."""
        bench = SkillsBenchmark()
        assert bench.levels()[2].name == "craft_miner"


# ---------------------------------------------------------------------------
# L.4 — place_miner
# ---------------------------------------------------------------------------


class TestPlaceMinerScripted:
    """Walk-adjacent + face + place lands miners on each of three patches."""

    def test_solves_canonical_seed(self) -> None:
        solved, t = _run_scripted(
            level_idx=3,
            policy=SCRIPTED_POLICIES["place_miner"],
            seed=0,
        )
        assert solved, f"place_miner scripted failed canonical seed (t={t})"
        # Three placements: each ≈ walk (≤4) + face (1) + place (1) =
        # ≤6 ticks; total ≤18. Allow slack to 30.
        assert t <= 30, f"place_miner solved but took {t} ticks (target: <= 30)"

    @pytest.mark.parametrize("seed", [1, 2, 3, 7, 13, 42])
    def test_solves_other_seeds(self, seed: int) -> None:
        solved, t = _run_scripted(
            level_idx=3,
            policy=SCRIPTED_POLICIES["place_miner"],
            seed=seed,
        )
        assert solved, f"place_miner scripted failed seed {seed} (t={t})"
        assert t <= 40

    def test_level_index_three_is_place_miner(self) -> None:
        """Bit 3 / level 3 contract: place_miner is at index 3."""
        bench = SkillsBenchmark()
        assert bench.levels()[3].name == "place_miner"

    # Note: under the harder mine condition (``items_mined.sum() >= 5``,
    # which counts ore from BOTH manual MINEs and automated placed
    # miners), the place_miner rollout will incidentally unlock the
    # mine bit too — three placed miners on ore tiles produce enough
    # ore over the time-to-place to exceed the threshold. This is by
    # design and doesn't affect scoring: each level's aggregate score
    # reads only its own achievement bit.


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
