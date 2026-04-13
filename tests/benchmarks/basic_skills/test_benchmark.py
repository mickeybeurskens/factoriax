"""Tests for the redesigned 5-level basic skills benchmark."""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.benchmarks.basic_skills import BasicSkillsBenchmark
from factoriax.benchmarks.basic_skills.levels import (
    ACHIEVEMENT_COUNTS,
    BASIC_SKILLS_LEVELS,
)
from factoriax.benchmarks.basic_skills.scoring import (
    aggregate_score,
    score_level_achievements,
)
from factoriax.benchmarks.core import Benchmark, LevelResult
from factoriax.constants import MachineType
from factoriax.levels import build_state

# -----------------------------------------------------------------------
# Level construction
# -----------------------------------------------------------------------


class TestLevelConstruction:
    """Verify that all 5 levels build into valid states."""

    def test_all_levels_build(self) -> None:
        """Every level should produce a valid EnvState."""
        for bl in BASIC_SKILLS_LEVELS:
            state = build_state(bl.level, bl.env_params)
            assert state.map.shape == (
                bl.env_params.map_height,
                bl.env_params.map_width,
            )

    def test_level_count(self) -> None:
        """Should have exactly 5 levels."""
        assert len(BASIC_SKILLS_LEVELS) == 5

    def test_level_names(self) -> None:
        """Level names should match the 5 essential skills."""
        names = [bl.name for bl in BASIC_SKILLS_LEVELS]
        assert names == [
            "mine_ores",
            "craft_all",
            "fuel_miner",
            "deploy_miners",
            "assembler_science",
        ]

    def test_fuel_miner_has_prebuilt_miner(self) -> None:
        """The fuel_miner level should have a pre-placed miner."""
        bl = next(bl for bl in BASIC_SKILLS_LEVELS if bl.name == "fuel_miner")
        state = build_state(bl.level, bl.env_params)
        has_miner = int(jnp.sum(state.machine_types == MachineType.MINER))
        assert has_miner >= 1

    def test_assembler_has_prebuilt_assembler(self) -> None:
        """The assembler_science level should have a pre-placed assembler."""
        bl = next(
            bl for bl in BASIC_SKILLS_LEVELS if bl.name == "assembler_science"
        )
        state = build_state(bl.level, bl.env_params)
        has_asm = int(jnp.sum(state.machine_types == MachineType.ASSEMBLER))
        assert has_asm >= 1


# -----------------------------------------------------------------------
# Achievement functions
# -----------------------------------------------------------------------


class TestAchievementFunctions:
    """Verify standalone achievement condition functions produce valid output.

    Achievement functions were moved out of BenchmarkLevel during the
    wrapper refactor. These tests verify the functions still exist as
    pure state predicates in the levels module.
    """

    def test_achievement_functions_exist(self) -> None:
        """The levels module should export achievement functions."""
        from factoriax.benchmarks.basic_skills import levels

        for name in ACHIEVEMENT_COUNTS:
            fn_name = f"_{name}_achievements"
            assert hasattr(levels, fn_name), f"Missing {fn_name}"


# -----------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------


class TestScoring:
    """Tests for achievement-based scoring."""

    def test_score_zero_on_fresh_state(self) -> None:
        """Fresh state should yield score near 0."""
        for bl in BASIC_SKILLS_LEVELS:
            state = build_state(bl.level, bl.env_params)
            result = LevelResult(
                level_name=bl.name,
                items_mined={"coal": 0, "iron": 0, "copper": 0},
                weighted_score=0.0,
                timesteps_used=0,
                actions=jnp.zeros(1, dtype=jnp.int32),
                final_state=state,
            )
            score = score_level_achievements(result)
            n = ACHIEVEMENT_COUNTS[bl.name]
            assert 0.0 <= score <= 1.0 / n + 0.01, (
                f"{bl.name}: expected near-zero score, got {score}"
            )

    def test_aggregate_range(self) -> None:
        """Aggregate score should be in [0, 1]."""
        results = []
        for bl in BASIC_SKILLS_LEVELS:
            state = build_state(bl.level, bl.env_params)
            results.append(
                LevelResult(
                    level_name=bl.name,
                    items_mined={"coal": 0, "iron": 0, "copper": 0},
                    weighted_score=0.0,
                    timesteps_used=0,
                    actions=jnp.zeros(1, dtype=jnp.int32),
                    final_state=state,
                )
            )
        agg = aggregate_score(results)
        assert 0.0 <= agg <= 1.0


# -----------------------------------------------------------------------
# Benchmark class
# -----------------------------------------------------------------------


class TestBasicSkillsBenchmark:
    """Tests for the BasicSkillsBenchmark class."""

    def test_protocol_compliance(self) -> None:
        """Benchmark should satisfy the Benchmark protocol."""
        b = BasicSkillsBenchmark()
        assert isinstance(b, Benchmark)

    def test_name(self) -> None:
        """Name should be 'basic_skills'."""
        assert BasicSkillsBenchmark().name == "basic_skills"

    def test_num_players(self) -> None:
        """Should require 1 player."""
        assert BasicSkillsBenchmark().num_players == 1

    def test_level_count(self) -> None:
        """Should have exactly 5 levels."""
        assert len(BasicSkillsBenchmark().levels()) == 5

    def test_vector_obs_fn(self) -> None:
        """Vector obs function should return a flat array."""
        bl = BASIC_SKILLS_LEVELS[0]
        state = build_state(bl.level, bl.env_params)
        obs = BasicSkillsBenchmark.vector_obs_fn(state, bl.env_params, 0)
        assert obs.ndim == 1
        assert obs.dtype == jnp.float32
