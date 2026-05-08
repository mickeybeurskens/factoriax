"""Skills challenge benchmark — curriculum of foundational mechanics.

Eight levels in the order navigate, mine, craft_miner, place_miner,
fuel_and_collect, belt_line, arm_transfer, mini_factory. Each level
exercises a different subset of the action space (via per-level
``blocked_actions``) and unlocks a single achievement bit on solve.
Score is time-weighted:
``solved * (max_timesteps - timesteps_used + 1) / max_timesteps``,
mean across levels.

The :class:`SkillsBenchmark` implements the
:class:`~factoriax.benchmarks.core.Benchmark` protocol the same way
:class:`RocketBenchmark` does, so it plugs into
:class:`BenchmarkRunner` unchanged.

Phase F ships only the skeleton — :meth:`SkillsBenchmark.levels`
returns an empty list and :func:`skills_conditions` returns an
all-False mask. Phase L appends one bit + one level + one scripted
baseline per slice.
"""

from __future__ import annotations

from factoriax.benchmarks.core import Benchmark, BenchmarkLevel, LevelResult
from factoriax.benchmarks.skills.achievements import (
    NUM_SKILLS,
    SKILLS_ACHIEVEMENT_INFO,
    count_miners_on_ore,
    skills_conditions,
)
from factoriax.benchmarks.skills.reward import skills_reward


class SkillsBenchmark:
    """Curriculum of eight skills, evaluated as one time-weighted score.

    Implements the :class:`Benchmark` protocol. Bit ``i`` of
    ``state.achievements_unlocked`` corresponds to level ``i``'s
    target condition (curriculum order). Aggregate score in ``[0, 1]``.

    The skeleton ships with zero levels; Phase L grows the curriculum
    one slice at a time.
    """

    name: str = "skills"
    num_players: int = 1
    achievement_fn = staticmethod(skills_conditions)

    def levels(self) -> list[BenchmarkLevel]:
        """Curriculum levels in solve order (canonical seed = 0).

        Returns:
            Empty list in the F.2 skeleton. Phase L slices append one
            ``BenchmarkLevel`` at a time using their per-level
            ``build_*_level`` helper and the level-specific
            ``blocked_actions`` mask.
        """
        return []

    def score_level(
        self, bench_level: BenchmarkLevel, items_mined: dict[str, int]
    ) -> float:
        """Per-level placeholder — real scoring happens in :meth:`score`.

        The protocol hands ``score_level`` only ``items_mined``, which
        isn't enough to compute the time-weighted score (we need the
        achievement mask and the step budget). Returns zero so the
        protocol is satisfied; aggregate callers should use
        :meth:`score`.
        """
        del bench_level, items_mined
        return 0.0

    def score(self, level_results: list[LevelResult]) -> float:
        """Mean time-weighted score across all curriculum levels.

        For each ``LevelResult``:

        - Look up the matching ``BenchmarkLevel`` by index (curriculum
          order). The achievement bit for level ``i`` is bit ``i``.
        - If the bit is unlocked, contribute
          ``(max_timesteps - timesteps_used + 1) / max_timesteps``;
          otherwise contribute zero.

        Returns the mean across contributions. Zero on empty input.

        Args:
            level_results: Per-level outcomes from
                :class:`BenchmarkRunner`, in evaluation order.

        Returns:
            Aggregate score in ``[0, 1]``.
        """
        if not level_results:
            return 0.0
        levels = self.levels()
        scores: list[float] = []
        for i, result in enumerate(level_results):
            mask = result.achievements_unlocked
            if mask is None or i >= len(levels) or i >= NUM_SKILLS:
                scores.append(0.0)
                continue
            solved = bool(mask[i])
            if not solved:
                scores.append(0.0)
                continue
            max_t = levels[i].env_params.max_timesteps
            scores.append(
                (max_t - result.timesteps_used + 1) / max_t,
            )
        if not scores:
            return 0.0
        return sum(scores) / len(scores)


__all__ = [
    "NUM_SKILLS",
    "SKILLS_ACHIEVEMENT_INFO",
    "SkillsBenchmark",
    "count_miners_on_ore",
    "skills_conditions",
    "skills_reward",
]


# Runtime-checkable conformance: SkillsBenchmark satisfies the protocol.
assert isinstance(SkillsBenchmark(), Benchmark), (
    "SkillsBenchmark must implement the Benchmark protocol — "
    "check name, num_players, levels(), score_level(), score()."
)
