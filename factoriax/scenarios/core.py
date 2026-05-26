"""Core types and protocols for the factoriax scenario system.

The central abstraction is the separation between *what* a scenario is
(levels, scoring rules) and *how* it is executed (stepping, observation
extraction, PRNG management). A ``Scenario`` declares both; a
``ScenarioRunner`` handles execution.

This module contains only pure data types and the ``Scenario`` protocol.
It does not import anything from ``scenarios.runner`` and has no side
effects on import.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import jax
import numpy as np

from factoriax.engine.levels import Level
from factoriax.engine.state import EnvParams, EnvState

# A policy maps a JAX float32 observation array to a JAX integer action scalar.
# This is intentionally minimal: any callable with this signature works,
# whether it wraps a trained network, a rule-based heuristic, or random noise.
Policy = Callable[[jax.Array], jax.Array]


@dataclasses.dataclass
class ScenarioLevel:
    """A level definition bundled with its environment parameters.

    A ``ScenarioLevel`` is the atomic unit of a scenario: one challenge
    to solve, one score to earn. The ``Level`` defines the world geometry
    and the ``EnvParams`` define the runtime constraints (episode budget,
    player count, map dimensions).

    ``env_params.map_width`` and ``env_params.map_height`` must match the
    level's declared dimensions.

    Attributes:
        name: Unique identifier for this level within its scenario.
        description: Human-readable description of the challenge and what
            makes it difficult.
        level: Factoriax ``Level`` describing the world layout.
        env_params: Runtime constraints. Map dimensions must match the level.
        blocked_actions: Optional set of action ids to block on this
            level. When ``None`` (default), the runner falls back to the
            scenario's class-level ``blocked_actions`` attribute (if
            any). When set — including to an empty ``frozenset()`` — it
            overrides the class-level mask for this level only. Used by
            curriculum scenarios where each level exercises a
            different subset of the action space.
    """

    name: str
    description: str
    level: Level
    env_params: EnvParams
    blocked_actions: frozenset[int] | None = None


@dataclasses.dataclass
class LevelResult:
    """Raw outcome of running a policy through one ``ScenarioLevel``.

    Attributes:
        level_name: Name of the level this result corresponds to.
        items_mined: Resources collected, keyed by item name
            (``"coal"``, ``"iron"``, ``"copper"``).
        weighted_score: Score computed by the scenario's per-level scoring
            function. The weighting scheme is scenario-specific.
        timesteps_used: Environment steps taken before the episode ended,
            at most ``env_params.max_timesteps``.
        actions: Integer action sequence of shape ``(T,)`` for player 0.
            For single-agent scenarios this covers the full episode.
        constraint_costs: Per-step constraint cost vectors of shape
            ``(T, K)`` where *K* is the number of constraint
            dimensions, or ``None`` when no constraint function was
            supplied.  Each row holds the cost vector returned by the
            constraint function after that tick.  Researchers can
            aggregate these however they like (sum, max, threshold).
        achievements_unlocked: Latched achievement mask at episode end,
            shape ``(MAX_ACHIEVEMENTS,)`` bool. Populated when the
            scenario supplies an ``achievement_fn`` (or one is passed
            to the runner). ``None`` for scenarios that do not track
            achievements.
    """

    level_name: str
    items_mined: dict[str, int]
    weighted_score: float
    timesteps_used: int
    actions: np.ndarray
    constraint_costs: np.ndarray | None = None
    final_state: EnvState | None = None
    achievements_unlocked: np.ndarray | None = None


@dataclasses.dataclass
class ScenarioResult:
    """Aggregated results across all levels in a scenario.

    Attributes:
        scenario_name: Name of the scenario that produced these results.
        level_results: Per-level results in evaluation order.
        aggregate_score: Single scalar summarising overall performance.
            The aggregation method is defined by the scenario (typically
            the mean of per-level weighted scores).
    """

    scenario_name: str
    level_results: list[LevelResult]
    aggregate_score: float


@runtime_checkable
class Scenario(Protocol):
    """Protocol that all scenario implementations must satisfy.

    A ``Scenario`` declares three things: the levels to evaluate on, how
    to score a single completed level, and how to aggregate per-level scores
    into one number. Execution is entirely the runner's concern.

    Using ``@runtime_checkable`` allows ``isinstance(obj, Scenario)``
    checks, which is useful for validating scenario arguments without
    importing concrete scenario classes.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this scenario."""
        ...

    @property
    def num_players(self) -> int:
        """Number of simultaneous agents this scenario expects."""
        ...

    def levels(self) -> list[ScenarioLevel]:
        """Ordered list of levels to evaluate against.

        Results are reported in the same order as this list.

        Returns:
            List of ``ScenarioLevel`` in evaluation order.
        """
        ...

    def score_level(
        self,
        scenario_level: ScenarioLevel,
        items_mined: dict[str, int],
    ) -> float:
        """Compute the score for a single completed level.

        Called by the runner once per level after the episode ends.

        Args:
            scenario_level: The level definition that was evaluated.
            items_mined: Resources collected, keyed by item name.

        Returns:
            Scalar score for this level.
        """
        ...

    def score(self, level_results: list[LevelResult]) -> float:
        """Compute the aggregate score from all level results.

        Called by the runner after all levels have been evaluated.

        Args:
            level_results: Per-level results in evaluation order.

        Returns:
            Single scalar representing overall scenario performance.
        """
        ...
