"""Scoring functions for the single-agent mining benchmark.

Every ore type is worth exactly one point, matching the sparse reward signal
(1.0 per item mined).  This makes the benchmark score directly interpretable
as total ore extracted — no conversion factor needed when comparing reward
curves to evaluation scores.

The aggregate benchmark score is the unweighted mean across all five levels,
so strong performance on easy levels cannot mask poor performance on hard ones.
"""

from __future__ import annotations

# All ore types worth 1 point, matching the sparse per-item reward signal.
RESOURCE_WEIGHTS: dict[str, int] = {
    "coal": 1,
    "iron": 1,
    "copper": 1,
}


def score_items(items_mined: dict[str, int]) -> float:
    """Compute the weighted score for a set of mined resources.

    Unrecognised item names are silently ignored, which allows the function
    to accept any superset of the standard resource dict without error.

    Args:
        items_mined: Resources collected, keyed by item name.

    Returns:
        Weighted sum of resources collected.
    """
    return float(
        sum(
            RESOURCE_WEIGHTS.get(name, 0) * count for name, count in items_mined.items()
        )
    )


def aggregate_scores(level_scores: list[float]) -> float:
    """Compute the aggregate benchmark score as the mean of per-level scores.

    Using the mean rather than the sum means the aggregate score is on the
    same scale as per-level scores, making it easier to interpret and
    compare across benchmarks with different numbers of levels.

    Args:
        level_scores: Per-level weighted scores in evaluation order.

    Returns:
        Mean of level scores.

    Raises:
        ValueError: If ``level_scores`` is empty.
    """
    if not level_scores:
        raise ValueError("level_scores must not be empty.")
    return sum(level_scores) / len(level_scores)
