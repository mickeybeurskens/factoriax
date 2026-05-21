"""Skeleton + protocol conformance tests for :class:`SkillsBenchmark`.

Phase F.2 ships the scenario class shell with no levels yet. These
tests pin the public shape so subsequent Phase L slices can append
levels without breaking consumers. Tests are fast — no JIT, no env
construction.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.constants import MAX_ACHIEVEMENTS
from factoriax.scenarios import (
    LevelResult,
    Scenario,
    SkillsBenchmark,
    skills_conditions,
    skills_reward,
)
from factoriax.scenarios.skills import NUM_SKILLS, SKILLS_ACHIEVEMENT_INFO
from factoriax.state import EnvParams, EnvState


class TestSkillsBenchmarkProtocol:
    """SkillsBenchmark satisfies the Scenario protocol."""

    def test_is_benchmark(self) -> None:
        assert isinstance(SkillsBenchmark(), Scenario)

    def test_name_is_skills(self) -> None:
        assert SkillsBenchmark().name == "skills"

    def test_num_players_is_one(self) -> None:
        assert SkillsBenchmark().num_players == 1

    def test_advertises_achievement_fn(self) -> None:
        assert SkillsBenchmark.achievement_fn is skills_conditions


class TestSkillsBenchmarkCurriculum:
    """Curriculum length tracks the number of Phase L slices that have landed."""

    def test_levels_count_matches_num_skills(self) -> None:
        """Every entry in ``levels()`` corresponds to one bit in ``NUM_SKILLS``."""
        assert len(SkillsBenchmark().levels()) == NUM_SKILLS

    def test_achievement_info_count_matches_num_skills(self) -> None:
        assert len(SKILLS_ACHIEVEMENT_INFO) == NUM_SKILLS

    def test_level_names_unique(self) -> None:
        names = [scenario_level.name for scenario_level in SkillsBenchmark().levels()]
        assert len(names) == len(set(names))

    def test_first_level_is_navigate(self) -> None:
        """Bit 0 / level 0 contract: navigate is at index 0 (L.1)."""
        levels = SkillsBenchmark().levels()
        assert levels[0].name == "navigate"
        assert levels[0].blocked_actions is not None

    def test_navigate_bit_is_zero(self) -> None:
        """SKILLS_ACHIEVEMENT_INFO[0] documents bit 0 = navigate."""
        assert SKILLS_ACHIEVEMENT_INFO[0].id == "skill_navigate"


class TestSkillsBenchmarkScoring:
    """``score()`` is well-defined on edge cases the runner produces."""

    def test_empty_results_returns_zero(self) -> None:
        assert SkillsBenchmark().score([]) == 0.0

    def test_results_without_mask_score_zero(self) -> None:
        """A LevelResult missing achievements_unlocked counts as unsolved."""
        result = LevelResult(
            level_name="anything",
            items_mined={"coal": 0, "iron": 0, "copper": 0},
            weighted_score=0.0,
            timesteps_used=10,
            actions=np.zeros(10, dtype=np.int32),
            achievements_unlocked=None,
        )
        assert SkillsBenchmark().score([result]) == 0.0

    def test_score_returns_float(self) -> None:
        """Aggregate is a Python float, not a numpy / jax scalar."""
        assert isinstance(SkillsBenchmark().score([]), float)


class TestSkillsConditions:
    """Condition function returns a correctly-shaped padded mask."""

    def _navigate_state(self) -> EnvState:
        """Build a real state from the canonical navigate level."""
        from factoriax.levels import build_state
        from factoriax.scenarios.skills import build_navigate_level

        level, params, _ = build_navigate_level(seed=0)
        return build_state(level, params)

    def test_returns_bool_array_shape(self) -> None:
        mask = skills_conditions(self._navigate_state())
        assert mask.shape == (MAX_ACHIEVEMENTS,)
        assert mask.dtype == jnp.bool_

    def test_navigate_bit_zero_at_initial_state(self) -> None:
        """Spawn isn't on the goal corner, so bit 0 starts as False."""
        mask = skills_conditions(self._navigate_state())
        assert not bool(mask[0])

    def test_padding_is_all_false(self) -> None:
        """Bits beyond ``NUM_SKILLS`` are zero-padded regardless of state."""
        mask = skills_conditions(self._navigate_state())
        assert not bool(jnp.any(mask[NUM_SKILLS:]))


class TestSkillsReward:
    """skills_reward returns a float32 scalar shaped to time-discount."""

    def _state_with_unlocked(self, mask_bits: list[int], timestep: int = 0) -> EnvState:
        """Stub EnvState with only the fields skills_reward reads."""

        # Minimal duck-typed object — skills_reward only accesses
        # achievements_unlocked and timestep on the state. Keeps the
        # test hermetic without standing up the full env.
        class _Stub:
            def __init__(self, bits: list[int], t: int) -> None:
                m = np.zeros(MAX_ACHIEVEMENTS, dtype=bool)
                for b in bits:
                    m[b] = True
                self.achievements_unlocked = jnp.asarray(m)
                self.timestep = jnp.int32(t)

        return _Stub(mask_bits, timestep)  # type: ignore[return-value]

    def test_zero_when_nothing_new_unlocks(self) -> None:
        prev = self._state_with_unlocked([0])
        new = self._state_with_unlocked([0], timestep=1)
        params = EnvParams(map_width=5, map_height=5, num_players=1, max_timesteps=100)
        assert float(skills_reward(prev, new, params)) == pytest.approx(0.0)

    def test_unlock_on_first_step_yields_near_one(self) -> None:
        prev = self._state_with_unlocked([])
        new = self._state_with_unlocked([0], timestep=1)
        params = EnvParams(map_width=5, map_height=5, num_players=1, max_timesteps=100)
        # remaining = 100 - 1 + 1 = 100; discount = 1.0
        assert float(skills_reward(prev, new, params)) == pytest.approx(1.0)

    def test_unlock_on_last_step_yields_smallest_positive(self) -> None:
        prev = self._state_with_unlocked([])
        new = self._state_with_unlocked([0], timestep=100)
        params = EnvParams(map_width=5, map_height=5, num_players=1, max_timesteps=100)
        # remaining = 100 - 100 + 1 = 1; discount = 0.01
        assert float(skills_reward(prev, new, params)) == pytest.approx(0.01)

    def test_two_bits_unlock_same_step_count_both(self) -> None:
        prev = self._state_with_unlocked([])
        new = self._state_with_unlocked([0, 1], timestep=1)
        params = EnvParams(map_width=5, map_height=5, num_players=1, max_timesteps=100)
        # 2 newly-unlocked × discount 1.0 = 2.0
        assert float(skills_reward(prev, new, params)) == pytest.approx(2.0)

    def test_target_bit_only_credits_that_bit(self) -> None:
        """target_bit=2 ignores incidental bit-0/bit-1 unlocks.

        Locks training reward to the level's own achievement so the
        agent can't farm free score from auto-mining or pass-through
        navigate that other levels accidentally trigger.
        """
        prev = self._state_with_unlocked([])
        new = self._state_with_unlocked([0, 1], timestep=1)
        params = EnvParams(map_width=5, map_height=5, num_players=1, max_timesteps=100)
        assert float(skills_reward(prev, new, params, target_bit=2)) == pytest.approx(
            0.0
        )

    def test_target_bit_credits_when_target_unlocks(self) -> None:
        """target_bit=2 fires reward only when bit 2 latches."""
        prev = self._state_with_unlocked([0])
        new = self._state_with_unlocked([0, 2], timestep=1)
        params = EnvParams(map_width=5, map_height=5, num_players=1, max_timesteps=100)
        assert float(skills_reward(prev, new, params, target_bit=2)) == pytest.approx(
            1.0
        )
