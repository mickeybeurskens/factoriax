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

The curriculum grows one slice at a time per the implementation plan
(``tasks/skills_benchmark_plan.md``). The current set of wired-up
levels is reflected by ``len(SkillsBenchmark().levels())`` and
``NUM_SKILLS``.
"""

from __future__ import annotations

from factoriax.benchmarks.core import Benchmark, BenchmarkLevel, LevelResult
from factoriax.benchmarks.skills.achievements import (
    NUM_SKILLS,
    SKILLS_ACHIEVEMENT_INFO,
    count_miners_on_ore,
    skills_conditions,
)
from factoriax.benchmarks.skills.levels import (
    build_craft_miner_level,
    build_mine_level,
    build_navigate_level,
)
from factoriax.benchmarks.skills.reward import skills_reward


class SkillsBenchmark:
    """Curriculum of skills, evaluated as one time-weighted score.

    Implements the :class:`Benchmark` protocol. Bit ``i`` of
    ``state.achievements_unlocked`` corresponds to level ``i``'s
    target condition (curriculum order). Aggregate score in ``[0, 1]``.

    Final curriculum target is eight levels (navigate through
    mini_factory). Phase L grows the curriculum one slice at a time;
    inspect ``levels()`` for what is currently wired up.
    """

    name: str = "skills"
    num_players: int = 1
    achievement_fn = staticmethod(skills_conditions)

    def levels(self) -> list[BenchmarkLevel]:
        """Curriculum levels in solve order (canonical seed = 0).

        Levels are constructed lazily on each call. Each entry uses
        the matching ``build_*_level(seed=0)`` helper and the
        skill-specific ``blocked_actions`` mask. Phase L slices append
        one entry at a time.

        Returns:
            ``BenchmarkLevel`` list, currently of length 1 (navigate).
        """
        navigate_level, navigate_params, navigate_blocked = build_navigate_level(seed=0)
        mine_level, mine_params, mine_blocked = build_mine_level(seed=0)
        craft_level, craft_params, craft_blocked = build_craft_miner_level(seed=0)
        return [
            BenchmarkLevel(
                name="navigate",
                description=(
                    "Walk to the bottom-right corner of a 5x5 grass map. "
                    "Action space is restricted to movement and NOOP."
                ),
                level=navigate_level,
                env_params=navigate_params,
                blocked_actions=navigate_blocked,
            ),
            BenchmarkLevel(
                name="mine",
                description=(
                    "Extract at least one ore from a 5x5 map sprinkled "
                    "with coal/iron/copper. Stand on an ore tile and MINE."
                ),
                level=mine_level,
                env_params=mine_params,
                blocked_actions=mine_blocked,
            ),
            BenchmarkLevel(
                name="craft_miner",
                description=(
                    "Combine 1 IRON_PLATE + 1 WIRE (pre-loaded in "
                    "inventory) into a miner via CRAFT_MINER. Other "
                    "CRAFT_* actions are blocked."
                ),
                level=craft_level,
                env_params=craft_params,
                blocked_actions=craft_blocked,
            ),
        ]

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
    "build_craft_miner_level",
    "build_mine_level",
    "build_navigate_level",
    "count_miners_on_ore",
    "skills_conditions",
    "skills_reward",
]


# Runtime-checkable conformance: SkillsBenchmark satisfies the protocol.
assert isinstance(SkillsBenchmark(), Benchmark), (
    "SkillsBenchmark must implement the Benchmark protocol — "
    "check name, num_players, levels(), score_level(), score()."
)
