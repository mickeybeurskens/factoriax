"""Scenario registry — the catalog behind ``factoriax.make("<id>")``.

A scenario is a factory ``() -> (env, params)``; the registry maps a gymnax-style
version id to a :class:`ScenarioSpec` carrying display metadata plus the factory.
:func:`make` resolves an id and applies the requested observation / auto-reset
wrappers; :func:`list_scenarios` is the catalog the playground iterates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from factoriax.engine.envs.auto_reset_wrapper import AutoResetWrapper
from factoriax.engine.envs.local_observation_wrapper import LocalObservationWrapper
from factoriax.engine.scenarios.easy_rocket import easy_rocket
from factoriax.engine.scenarios.rocket import rocket
from factoriax.engine.state import EnvParams


@dataclass(frozen=True)
class ScenarioSpec:
    """A registered scenario: display metadata plus its env factory.

    Attributes:
        name: Human-readable display name for the playground picker.
        description: Description-panel text.
        build: Factory returning ``(env, params)`` for the scenario.
        resample: Whether the scenario's reset is keyed (a fresh layout per
            episode). Controls ``AutoResetWrapper(resample=...)`` so fixed-level
            scenarios use the cheap cached restore.
    """

    name: str
    description: str
    build: Callable[[], tuple[Any, EnvParams]]
    resample: bool = False


SCENARIOS: dict[str, ScenarioSpec] = {
    "EasyRocket-v1": ScenarioSpec(
        name="Easy Rocket",
        description=(
            "Build a rocket from raw ore on a 16x16 procgen map. "
            "Each episode draws a fresh layout; 2000-step budget."
        ),
        build=easy_rocket,
        resample=True,
    ),
    "Rocket-v1": ScenarioSpec(
        name="Rocket",
        description=(
            "Build a rocket on a fixed 32x32 map. Furnace and assembler are "
            "pre-placed; hand-craft actions are masked; 8000-step budget."
        ),
        build=rocket,
        resample=False,
    ),
}


def make(
    env_id: str,
    *,
    obs: str = "global",
    obs_radius: int = 7,
    auto_reset: bool = False,
    resample: bool | None = None,
) -> tuple[Any, EnvParams]:
    """Resolve a scenario id to ``(env, params)``, applying optional wrappers.

    Args:
        env_id: Registered scenario id (e.g. ``"EasyRocket-v1"``).
        obs: ``"global"`` (default) or ``"local"`` for a windowed observation.
        obs_radius: Local-observation half-width (ignored for global obs).
        auto_reset: Wrap in :class:`AutoResetWrapper`.
        resample: Auto-reset mode. ``None`` (default) uses the scenario's
            ``resample`` setting; pass ``True``/``False`` to override — e.g.
            ``False`` for the cheap cached restore even on a keyed scenario.

    Returns:
        ``(env, params)``.

    Raises:
        KeyError: If ``env_id`` is not registered.
    """
    spec = SCENARIOS[env_id]
    env, params = spec.build()
    if obs == "local":
        env = LocalObservationWrapper(env, radius=obs_radius)
    if auto_reset:
        use_resample = spec.resample if resample is None else resample
        env = AutoResetWrapper(env, resample=use_resample)
    return env, params


def list_scenarios() -> tuple[tuple[str, ScenarioSpec], ...]:
    """Return ``(id, spec)`` pairs for every registered scenario."""
    return tuple(SCENARIOS.items())
