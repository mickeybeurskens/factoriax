"""SingleAgentMiningBenchmark: five progressively harder mining challenges.

The benchmark asks a single agent to collect as many resources as possible
in 200 ticks across five levels. Resources are scored by value
(copper > iron > coal). The final benchmark score is the mean weighted
score across all five levels.

Typical usage::

    from factoriax.benchmarks.single_agent_mining import SingleAgentMiningBenchmark
    from factoriax.benchmarks.runner import BenchmarkRunner

    def my_policy(obs):
        ...  # returns a JAX integer action scalar

    benchmark = SingleAgentMiningBenchmark()
    runner = BenchmarkRunner(seed=0)
    result = runner.run(benchmark, policies=[my_policy])
    print(result.aggregate_score)
"""

from __future__ import annotations

from factoriax.benchmarks.core import BenchmarkLevel, BenchmarkResult, LevelResult
from factoriax.benchmarks.single_agent_mining.levels import MINING_LEVELS
from factoriax.benchmarks.single_agent_mining.scoring import aggregate_scores, score_items


class SingleAgentMiningBenchmark:
    """Five-level single-agent mining benchmark.

    Levels progress from a trivial adjacent-coal scenario to a large map
    where the agent must choose between two equidistant but differently
    valued deposits. The aggregate score is the mean weighted score across
    all five levels, preventing strong early-level performance from hiding
    weak late-level performance.

    Resource weights: coal=1, iron=2, copper=3.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this benchmark.

        Returns:
            ``"single_agent_mining"``.
        """
        return "single_agent_mining"

    @property
    def num_players(self) -> int:
        """Number of simultaneous agents this benchmark requires.

        Returns:
            ``1``.
        """
        return 1

    def levels(self) -> list[BenchmarkLevel]:
        """Return the five mining levels in order of increasing difficulty.

        Returns:
            List of five ``BenchmarkLevel`` objects.
        """
        return list(MINING_LEVELS)

    def score_level(
        self,
        bench_level: BenchmarkLevel,
        items_mined: dict[str, int],
    ) -> float:
        """Compute the weighted score for a single completed level.

        Args:
            bench_level: The level that was evaluated (unused here; the
                score depends only on what was mined, not which level it
                came from).
            items_mined: Resources collected, keyed by item name.

        Returns:
            Weighted sum: ``coal * 1 + iron * 2 + copper * 3``.
        """
        return score_items(items_mined)

    def score(self, level_results: list[LevelResult]) -> float:
        """Compute the aggregate benchmark score.

        Args:
            level_results: Per-level results from the runner, in level order.

        Returns:
            Mean of per-level weighted scores.
        """
        return aggregate_scores([r.weighted_score for r in level_results])
