"""BalancedGatheringBenchmark: five levels testing constraint transfer.

Research question (RQ3)
-----------------------
Given a sequence of constrained MDPs with shared structure but increasing
complexity, how should a curriculum be designed so that safety constraints
satisfied on simpler levels transfer to harder levels with provable
guarantees?

The constraint is *resource balance*: the pairwise ratio between mined
ore types should stay below a threshold.  The reward is total ore mined.
These two signals are stored separately and never blended by the
benchmark — researchers choose how to combine them.

The five levels progressively test whether a constraint-aware rotation
strategy transfers across layouts with asymmetric distances, obstacles,
scarce resources, and conflicting reward incentives.

Typical usage::

    import functools
    from benchmarks.balanced_gathering import BalancedGatheringBenchmark
    from benchmarks.runner import BenchmarkRunner
    from factoriax.constraints import balance_cost

    benchmark = BalancedGatheringBenchmark()
    runner = BenchmarkRunner(seed=0)
    constraint_fn = functools.partial(balance_cost, threshold=3.0)
    result = runner.run(
        benchmark,
        policies=[my_policy],
        constraint_fn=constraint_fn,
    )
    for lr in result.level_results:
        print(lr.level_name, lr.weighted_score, lr.constraint_costs.sum())
"""

from __future__ import annotations

from collections.abc import Callable

import jax

from benchmarks.balanced_gathering.levels import BALANCED_LEVELS
from benchmarks.balanced_gathering.scoring import (
    aggregate_scores,
    score_items,
)
from benchmarks.core import BenchmarkLevel, LevelResult
from factoriax.rewards import sparse_mining_reward
from factoriax.state import EnvParams, EnvState


class BalancedGatheringBenchmark:
    """Five-level benchmark for curriculum-based constraint transfer.

    Levels progress from a symmetric layout where balance is trivial to
    an asymmetric layout where reward maximisation conflicts with
    constraint satisfaction.  Each level isolates one factor that
    distinguishes layout-dependent from constraint-aware behaviour.

    The reward function is ``sparse_mining_reward`` (1 per ore mined).
    The constraint function is *not* built into the benchmark — pass it
    to :meth:`BenchmarkRunner.run` via the ``constraint_fn`` parameter.
    This keeps the benchmark scoring pure (reward only) and lets
    researchers choose their own constraint formulation and threshold.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this benchmark.

        Returns:
            ``"balanced_gathering"``.
        """
        return "balanced_gathering"

    @property
    def reward_fn(
        self,
    ) -> Callable[[EnvState, EnvState, EnvParams], jax.Array]:
        """Reward function for training agents on this benchmark.

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
        """Return the five levels in curriculum order.

        Returns:
            List of five ``BenchmarkLevel`` objects.
        """
        return list(BALANCED_LEVELS)

    def score_level(
        self,
        bench_level: BenchmarkLevel,
        items_mined: dict[str, int],
    ) -> float:
        """Compute the reward-side score for one level.

        This is purely the total ore mined.  Constraint costs are
        stored separately in ``LevelResult.constraint_costs`` and
        are not reflected here.

        Args:
            bench_level: The level that was evaluated.
            items_mined: Resources collected, keyed by item name.

        Returns:
            Total ore count.
        """
        return score_items(items_mined)

    def score(self, level_results: list[LevelResult]) -> float:
        """Compute the aggregate reward-side score.

        Args:
            level_results: Per-level results from the runner.

        Returns:
            Mean of per-level scores.
        """
        return aggregate_scores([r.weighted_score for r in level_results])
