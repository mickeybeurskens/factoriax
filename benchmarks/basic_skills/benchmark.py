"""BasicSkillsBenchmark: three levels testing mining, crafting, and depositing.

The benchmark evaluates three foundational RL skills in isolation:

1. **mine_resources** — navigate and extract ore (sparse mining reward).
2. **craft_chests** — select a recipe and craft items (sparse crafting reward).
3. **fill_chest** — mine, navigate, and deposit into a machine (chest filling
   reward).

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
    score_fill,
    score_mine,
)
from benchmarks.core import BenchmarkLevel, LevelResult
from factoriax.rewards import (
    chest_filling_reward,
    sparse_crafting_reward,
    sparse_mining_reward,
)
from factoriax.state import EnvParams, EnvState

# Map level names to their reward functions. The runner does not currently
# support per-level reward functions, so reward_fn returns the mining reward
# as a sensible default. Training scripts should select the appropriate
# reward function per level using this mapping.
REWARD_FNS: dict[str, Callable[[EnvState, EnvState, EnvParams], jax.Array]] = {
    "mine_resources": sparse_mining_reward,
    "craft_chests": sparse_crafting_reward,
    "fill_chest": chest_filling_reward,
}


class BasicSkillsBenchmark:
    """Three-level benchmark testing mining, crafting, and depositing.

    Each level isolates a single skill. The aggregate score normalises
    per-level metrics to [0, 1] before averaging, ensuring equal weight
    across the three skills.
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
        """Return the three skill levels in order.

        Returns:
            List of three ``BenchmarkLevel`` objects.
        """
        return list(BASIC_SKILLS_LEVELS)

    def score_level(
        self,
        bench_level: BenchmarkLevel,
        items_mined: dict[str, int],
    ) -> float:
        """Compute the score for a single completed level.

        For the mining level, the score is total ore mined. For the
        crafting and chest-filling levels, this returns 0.0 because
        accurate scoring requires the final state (available in
        :meth:`score` via ``LevelResult.final_state``).

        Args:
            bench_level: The level that was evaluated.
            items_mined: Resources collected, keyed by item name.

        Returns:
            Level score (mining) or 0.0 (crafting/fill — scored in
            :meth:`score`).
        """
        if bench_level.name == "mine_resources":
            return score_mine(items_mined)
        return 0.0

    def score(self, level_results: list[LevelResult]) -> float:
        """Compute the aggregate benchmark score using final states.

        Rescores the crafting and chest-filling levels from their final
        environment state rather than relying on ``items_mined``.

        Args:
            level_results: Per-level results from the runner.

        Returns:
            Mean of normalised per-level scores on [0, 1].
        """
        mine = 0.0
        craft = 0.0
        fill = 0.0

        for r in level_results:
            if r.level_name == "mine_resources":
                mine = score_mine(r.items_mined)
            elif r.level_name == "craft_chests" and r.final_state is not None:
                craft = score_craft(r.final_state)
            elif r.level_name == "fill_chest" and r.final_state is not None:
                fill = score_fill(r.final_state)

        return aggregate_scores(mine, craft, fill)
