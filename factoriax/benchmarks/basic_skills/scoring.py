"""Scoring functions for the basic skills benchmark.

Scoring uses ``items_mined`` as a rough per-level metric. Achievement-
based scoring is temporarily disabled while the benchmark system is
being redesigned around gymnax wrappers.
"""

from __future__ import annotations

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
    """Placeholder for achievement-based scoring.

    Returns the items-based score until the benchmark is redesigned
    with gymnax skill wrappers that track their own progression.

    Args:
        result: Completed level result.

    Returns:
        Items-based score as a float.
    """
    return result.weighted_score


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
