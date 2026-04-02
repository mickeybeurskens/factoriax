"""Scoring functions for the basic skills benchmark.

Scoring is achievement-based: each level defines custom achievements as
binary state predicates, and the score is the fraction of achievements
unlocked at episode end. This approach is robust against reward hacking
because achievements test actual game state rather than accumulated
reward signals.

The ``score_level`` function uses ``items_mined`` for a rough per-level
metric (called by the runner before ``final_state`` is available on the
result). The ``aggregate_score`` function uses ``final_state`` to compute
the achievement-based score that matters.
"""

from __future__ import annotations

import jax.numpy as jnp

from factoriax.benchmarks.basic_skills.levels import ACHIEVEMENT_COUNTS
from factoriax.benchmarks.core import BenchmarkLevel, LevelResult


def score_level_items(
    bench_level: BenchmarkLevel,
    items_mined: dict[str, int],
) -> float:
    """Compute a rough score from items_mined for a single level.

    This is called by the runner's ``score_level`` before the full
    ``final_state`` is attached to the result. It provides a basic
    metric based solely on what was mined.

    Args:
        bench_level: The level definition that was evaluated.
        items_mined: Resources collected, keyed by item name.

    Returns:
        Total ore mined across all resource types.
    """
    return float(
        items_mined.get("coal", 0)
        + items_mined.get("iron", 0)
        + items_mined.get("copper", 0)
    )


def score_level_achievements(result: LevelResult) -> float:
    """Compute the achievement-based score for a single level result.

    Returns the fraction of achievements unlocked at episode end,
    ranging from 0.0 (no progress) to 1.0 (all milestones reached).
    Falls back to 0.0 if the result has no ``final_state``.

    Args:
        result: Completed level result with ``final_state`` attached.

    Returns:
        Achievement fraction in [0.0, 1.0].
    """
    if result.final_state is None:
        return 0.0
    total = ACHIEVEMENT_COUNTS.get(result.level_name, 0)
    if total == 0:
        return 0.0
    unlocked = int(jnp.sum(result.final_state.achievements_unlocked[:total]))
    return unlocked / total


def aggregate_score(level_results: list[LevelResult]) -> float:
    """Compute the aggregate benchmark score from all level results.

    Takes the mean of per-level achievement fractions. Each level
    contributes equally regardless of its number of achievements.

    Args:
        level_results: Per-level results in evaluation order.

    Returns:
        Mean achievement fraction across all levels, in [0.0, 1.0].
    """
    if not level_results:
        return 0.0
    total = 0.0
    for result in level_results:
        total += score_level_achievements(result)
    return total / len(level_results)
