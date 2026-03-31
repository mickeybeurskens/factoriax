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

from collections.abc import Callable

import jax

from factoriax.benchmarks.core import BenchmarkLevel, LevelResult
from factoriax.benchmarks.single_agent_mining.levels import MINING_LEVELS
from factoriax.benchmarks.single_agent_mining.scoring import (
    aggregate_scores,
    score_items,
)
from factoriax.rewards import sparse_mining_reward
from factoriax.state import EnvParams, EnvState


class SingleAgentMiningBenchmark:
    """Five-level single-agent mining benchmark.

    Levels progress from a trivial adjacent-coal scenario to a large map
    where the agent must choose between two equidistant but differently
    valued deposits. The aggregate score is the mean weighted score across
    all five levels, preventing strong early-level performance from hiding
    weak late-level performance.

    Every ore type scores 1 point, matching the sparse reward signal.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this benchmark.

        Returns:
            ``"single_agent_mining"``.
        """
        return "single_agent_mining"

    @property
    def reward_fn(
        self,
    ) -> Callable[[EnvState, EnvState, EnvParams], jax.Array]:
        """Reward function for training agents on this benchmark.

        Returns one reward unit per ore item extracted during a step —
        coal, iron, or copper each count equally.  The sparse signal
        encourages direct mining without the proximity shaping of
        :func:`~factoriax.rewards.mining_reward`.

        Returns:
            :func:`~factoriax.rewards.sparse_mining_reward`.
        """
        return sparse_mining_reward

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
            Total ore count: ``coal + iron + copper``.
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
