"""Scoring and constraint summary functions for the balanced gathering benchmark.

The benchmark measures two things independently:

1. **Reward**: total ore mined (same as the mining benchmark).
2. **Constraint costs**: per-step violation vectors, stored raw in
   ``LevelResult.constraint_costs``.

This module does *not* blend reward and constraint into a single number.
That decision belongs to the researcher — Lagrangian penalties, hard
filters, or separate CMDP channels are all valid approaches.  The
functions here provide summary statistics that make constraint costs
easy to inspect and compare across levels.
"""

from __future__ import annotations

import numpy as np

RESOURCE_WEIGHTS: dict[str, int] = {
    "coal": 1,
    "iron": 1,
    "copper": 1,
}


def score_items(items_mined: dict[str, int]) -> float:
    """Compute the reward-side score: total weighted ore mined.

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

    Args:
        level_scores: Per-level weighted scores.

    Returns:
        Mean of level scores.

    Raises:
        ValueError: If ``level_scores`` is empty.
    """
    if not level_scores:
        raise ValueError("level_scores must not be empty.")
    return sum(level_scores) / len(level_scores)


def constraint_summary(constraint_costs: np.ndarray) -> dict[str, float]:
    """Aggregate per-step constraint costs into summary statistics.

    The input is the raw ``(T, K)`` cost matrix from
    ``LevelResult.constraint_costs``.  Returns a flat dictionary of
    statistics that are easy to log, plot, or compare.

    Args:
        constraint_costs: Array of shape ``(T, K)`` where *T* is the
            number of timesteps and *K* is the number of constraint
            dimensions.

    Returns:
        Dictionary with keys:

        - ``total_cost``: sum over all steps and dimensions.
        - ``max_step_cost``: largest per-step total (sum over K).
        - ``steps_violated``: number of steps where any cost > 0.
        - ``fraction_violated``: ``steps_violated / T``.
    """
    per_step_total = constraint_costs.sum(axis=1)
    steps_violated = int(np.count_nonzero(per_step_total > 0))
    total_steps = constraint_costs.shape[0]
    return {
        "total_cost": float(constraint_costs.sum()),
        "max_step_cost": float(per_step_total.max()),
        "steps_violated": float(steps_violated),
        "fraction_violated": steps_violated / total_steps if total_steps > 0 else 0.0,
    }
