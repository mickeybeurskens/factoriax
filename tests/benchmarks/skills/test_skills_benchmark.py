"""Skeleton + protocol conformance tests for :class:`SkillsBenchmark`.

Phase F.2 ships the benchmark class shell with no levels yet. These
tests pin the public shape so subsequent Phase L slices can append
levels without breaking consumers. Tests are fast — no JIT, no env
construction.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from factoriax.benchmarks import (
    Benchmark,
    LevelResult,
    SkillsBenchmark,
    skills_conditions,
    skills_reward,
)
from factoriax.benchmarks.skills import NUM_SKILLS, SKILLS_ACHIEVEMENT_INFO
from factoriax.constants import MAX_ACHIEVEMENTS
from factoriax.state import EnvParams, EnvState


class TestSkillsBenchmarkProtocol:
    """SkillsBenchmark satisfies the Benchmark protocol."""

    def test_is_benchmark(self) -> None:
        assert isinstance(SkillsBenchmark(), Benchmark)

    def test_name_is_skills(self) -> None:
        assert SkillsBenchmark().name == "skills"

    def test_num_players_is_one(self) -> None:
        assert SkillsBenchmark().num_players == 1

    def test_advertises_achievement_fn(self) -> None:
        assert SkillsBenchmark.achievement_fn is skills_conditions


class TestSkillsBenchmarkSkeleton:
    """The F.2 skeleton ships with zero levels and zero curriculum bits."""

    def test_levels_empty(self) -> None:
        assert SkillsBenchmark().levels() == []

    def test_achievement_info_empty(self) -> None:
        assert SKILLS_ACHIEVEMENT_INFO == []

    def test_num_skills_zero(self) -> None:
        assert NUM_SKILLS == 0


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
    """Skeleton condition function returns a properly-shaped zero mask."""

    def _stub_state(self) -> EnvState:
        """Build a tiny EnvState — fields used by skills_conditions only."""
        # skills_conditions in the skeleton ignores its argument entirely
        # (del state). Pass an EnvState built minimally so the function
        # signature is exercised without depending on field internals.
        # We can't construct EnvState by hand cheaply, so call with a
        # mock: the skeleton's `del state` keeps this safe.
        return None  # type: ignore[return-value]

    def test_returns_bool_array_shape(self) -> None:
        mask = skills_conditions(self._stub_state())
        assert mask.shape == (MAX_ACHIEVEMENTS,)
        assert mask.dtype == jnp.bool_

    def test_all_false_in_skeleton(self) -> None:
        mask = skills_conditions(self._stub_state())
        assert not bool(jnp.any(mask))


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
