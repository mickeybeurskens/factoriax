"""BasicSkillsBenchmark: six levels from basic mining to automated factories.

The benchmark evaluates six progressively harder RL skills:

1. **mine_resources** — navigate and extract ore (sparse mining reward).
2. **craft_chests** — select a recipe and craft items (sparse crafting reward).
3. **fill_chest** — mine, navigate, and deposit into a machine (chest filling
   reward).
4. **craft_miners** — mine two resource types and craft miner machines.
5. **deploy_miner** — place a miner on ore and fuel it with coal.
6. **mining_factory** — full automation loop: mine, craft, place, fuel, collect.

The aggregate score normalises each level to [0, 1] before averaging, so no
single skill dominates the overall number.

Typical usage::

    from benchmarks.basic_skills import BasicSkillsBenchmark
    from benchmarks.runner import BenchmarkRunner

    benchmark = BasicSkillsBenchmark()
    runner = BenchmarkRunner(seed=0)
    result = runner.run(benchmark, policies=[my_policy])
    print(result.aggregate_score)
"""

from __future__ import annotations

from collections.abc import Callable

import jax

from benchmarks.basic_skills.levels import BASIC_SKILLS_LEVELS
from benchmarks.basic_skills.scoring import (
    aggregate_scores,
    score_craft,
    score_craft_miners,
    score_deploy_miner,
    score_fill,
    score_mine,
    score_mining_factory,
)
from benchmarks.core import BenchmarkLevel, LevelResult
from factoriax.rewards import (
    chest_filling_reward,
    miner_output_reward,
    miner_throughput_reward,
    sparse_chest_crafting_reward,
    sparse_miner_crafting_reward,
    sparse_mining_reward,
)
from factoriax.state import EnvParams, EnvState

# Map level names to their reward functions. Training scripts should select
# the appropriate reward function per level using this mapping.
REWARD_FNS: dict[str, Callable[[EnvState, EnvState, EnvParams], jax.Array]] = {
    "mine_resources": sparse_mining_reward,
    "craft_chests": sparse_chest_crafting_reward,
    "fill_chest": chest_filling_reward,
    "craft_miners": sparse_miner_crafting_reward,
    "deploy_miner": miner_output_reward,
    "mining_factory": miner_throughput_reward,
}


class BasicSkillsBenchmark:
    """Six-level benchmark from basic mining to automated factories.

    Each level targets a specific skill. The aggregate score normalises
    per-level metrics to [0, 1] before averaging, ensuring equal weight
    across all skills.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this benchmark.

        Returns:
            ``"basic_skills"``.
        """
        return "basic_skills"

    @property
    def reward_fn(
        self,
    ) -> Callable[[EnvState, EnvState, EnvParams], jax.Array]:
        """Default reward function for training.

        Returns the sparse mining reward as a sensible default. For
        per-level reward functions, use :data:`REWARD_FNS`.

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
        """Return all skill levels in order of increasing difficulty.

        Returns:
            List of ``BenchmarkLevel`` objects.
        """
        return list(BASIC_SKILLS_LEVELS)

    def score_level(
        self,
        bench_level: BenchmarkLevel,
        items_mined: dict[str, int],
    ) -> float:
        """Compute the score for a single completed level.

        For levels scored by ``items_mined`` the score is returned
        directly. For levels that require the final state (crafting,
        filling, deploying), this returns 0.0 because accurate scoring
        happens in :meth:`score` via ``LevelResult.final_state``.

        Args:
            bench_level: The level that was evaluated.
            items_mined: Resources collected, keyed by item name.

        Returns:
            Level score or 0.0 when final state is needed.
        """
        if bench_level.name in ("mine_resources", "mining_factory"):
            return score_mine(items_mined)
        return 0.0

    def score(self, level_results: list[LevelResult]) -> float:
        """Compute the aggregate benchmark score using final states.

        Rescores levels that need the final environment state rather
        than relying solely on ``items_mined``.

        Args:
            level_results: Per-level results from the runner.

        Returns:
            Mean of normalised per-level scores on [0, 1].
        """
        scores: dict[str, float] = {}

        for r in level_results:
            if r.level_name == "mine_resources":
                scores["mine_resources"] = score_mine(r.items_mined)
            elif r.level_name == "craft_chests" and r.final_state is not None:
                scores["craft_chests"] = score_craft(r.final_state)
            elif r.level_name == "fill_chest" and r.final_state is not None:
                scores["fill_chest"] = score_fill(r.final_state)
            elif r.level_name == "craft_miners" and r.final_state is not None:
                scores["craft_miners"] = score_craft_miners(r.final_state)
            elif r.level_name == "deploy_miner" and r.final_state is not None:
                scores["deploy_miner"] = score_deploy_miner(r.final_state)
            elif r.level_name == "mining_factory":
                scores["mining_factory"] = score_mining_factory(r.items_mined)

        return aggregate_scores(scores)
