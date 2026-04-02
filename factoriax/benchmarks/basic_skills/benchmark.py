"""BasicSkillsBenchmark: five levels from raw mining to science production.

The benchmark evaluates five progressively harder RL skills:

1. **mine_ores** -- navigate and extract ore from three patch types.
2. **craft_all** -- mine two resources and craft all five recipe types.
3. **fuel_miner** -- mine coal and fuel a pre-placed miner.
4. **deploy_miners** -- full deployment loop: mine, craft, place, fuel.
5. **assembler_science** -- feed an assembler to produce science packs.

Scoring is achievement-based: each level defines custom milestones and
the score is the fraction of milestones unlocked at episode end. The
aggregate score is the mean across all five levels, giving a single
number in [0, 1].

Typical usage::

    from factoriax.benchmarks.basic_skills import BasicSkillsBenchmark
    from factoriax.benchmarks.runner import BenchmarkRunner

    benchmark = BasicSkillsBenchmark()
    runner = BenchmarkRunner(seed=0)
    result = runner.run(benchmark, policies=[my_policy])
    print(result.aggregate_score)
"""

from __future__ import annotations

from collections.abc import Callable

import jax

from factoriax.benchmarks.basic_skills.levels import (
    ACHIEVEMENT_WEIGHTS,
    BASIC_SKILLS_LEVELS,
)
from factoriax.benchmarks.basic_skills.scoring import (
    aggregate_score,
    score_level_items,
)
from factoriax.benchmarks.core import BenchmarkLevel, LevelResult
from factoriax.observations import global_array
from factoriax.rewards import mining_reward
from factoriax.state import EnvParams, EnvState

# Per-level reward functions: dense shaping only.
# mining_reward gives proximity-to-ore + 5.0 per ore mined per step,
# providing continuous signal at every timestep.
REWARD_FNS: dict[
    str, Callable[[EnvState, EnvState, EnvParams], jax.Array]
] = {name: mining_reward for name in ACHIEVEMENT_WEIGHTS}


class BasicSkillsBenchmark:
    """Five-level benchmark from basic mining to science production.

    Each level targets a specific skill progression. Scoring uses custom
    per-level achievements evaluated on the final game state, making it
    robust against reward hacking. The aggregate score normalises each
    level to [0, 1] before averaging.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this benchmark.

        Returns:
            ``"basic_skills"``.
        """
        return "basic_skills"

    @property
    def num_players(self) -> int:
        """Number of simultaneous agents this benchmark requires.

        Returns:
            ``1``.
        """
        return 1

    def levels(self) -> list[BenchmarkLevel]:
        """Return all skill levels in order of increasing difficulty.

        Returns:
            List of five ``BenchmarkLevel`` objects.
        """
        return list(BASIC_SKILLS_LEVELS)

    def score_level(
        self,
        bench_level: BenchmarkLevel,
        items_mined: dict[str, int],
    ) -> float:
        """Compute a rough score from items_mined for one level.

        This is called by the runner before ``final_state`` is
        available. It returns total ore mined as a simple progress
        indicator. The true achievement-based score is computed in
        :meth:`score` using the full ``final_state``.

        Args:
            bench_level: The level that was evaluated.
            items_mined: Resources collected, keyed by item name.

        Returns:
            Total ore mined.
        """
        return score_level_items(bench_level, items_mined)

    def score(self, level_results: list[LevelResult]) -> float:
        """Compute the aggregate score from all level results.

        Uses ``final_state.achievements_unlocked`` on each result to
        compute the fraction of level-specific achievements unlocked,
        then averages across all levels.

        Args:
            level_results: Per-level results from the runner.

        Returns:
            Mean achievement fraction in [0.0, 1.0].
        """
        return aggregate_score(level_results)

    @staticmethod
    def vector_obs_fn(
        state: EnvState,
        params: EnvParams,
        player_idx: int | jax.Array,
    ) -> jax.Array:
        """Full-map vector observation for one player.

        Wraps :func:`~factoriax.observations.global_array` for
        convenient use with the benchmark runner.

        Args:
            state: Current environment state.
            params: Environment parameters.
            player_idx: Index of the observing player.

        Returns:
            Flat float32 observation array.
        """
        return global_array(state, params, player_idx)
