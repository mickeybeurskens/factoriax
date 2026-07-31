"""The scenario registry, which stands behind ``factoriax.make("<id>")``.

A scenario is a factory of the form ``() -> (env, params)``. The registry maps
a gymnax-style version id to a :class:`ScenarioSpec`. That spec holds the
display name and description of the scenario, and the factory itself.

:func:`make` resolves an id and applies the observation settings and the
auto-reset wrapper that the caller asked for. :func:`list_scenarios` returns
the list that the playground walks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from factoriax.engine.envs.easy_rocket import easy_rocket
from factoriax.engine.envs.miner_curriculum import miner_bootstrap
from factoriax.engine.envs.mining import mining
from factoriax.engine.envs.rocket import rocket
from factoriax.engine.envs.science_tiers import science_tiers
from factoriax.engine.envs.wrappers import AutoResetWrapper
from factoriax.engine.state import EnvParams


@dataclass(frozen=True)
class ScenarioSpec:
    """A registered scenario: its display name, its description, and its factory."""

    name: str
    description: str
    build: Callable[[], tuple[Any, EnvParams]]
    resample: bool = False


SCENARIOS: dict[str, ScenarioSpec] = {
    "Mining-v1": ScenarioSpec(
        name="Mining",
        description=(
            "Hand-mine ore on a randomly generated 8×8 map. "
            "Ten iron-ore tiles with 3 resources each; 100-step budget."
        ),
        build=mining,
        resample=True,
    ),
    "MinerBootstrap-v1": ScenarioSpec(
        name="Miner Bootstrap",
        description=(
            "From an empty inventory, mine limestone and silicon, craft "
            "six miners, and get all six producing. Completion-only "
            "reward: 1.0 once when all six produce, else 0; max score 1; "
            "300-step budget, early exit on completion."
        ),
        build=miner_bootstrap,
        resample=True,
    ),
    "ScienceTiers-v1": ScenarioSpec(
        name="Science Tiers",
        description=(
            "Three-tier science economy on the 16x16 six-patch map with "
            "one pre-placed lab. Labs pay 1 per pack consumed, any tier; "
            "ore costs double per tier while output quadruples, so "
            "science/ore doubles: 0.5 / 1.0 / 2.0. Dense throughput "
            "reward, unbounded score; 1000-step budget."
        ),
        build=science_tiers,
        resample=True,
    ),
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
    obs: str | None = None,
    obs_radius: int | None = None,
    auto_reset: bool = True,
    resample: bool | None = None,
) -> tuple[Any, EnvParams]:
    """Resolve a scenario id to an ``(env, params)`` pair.

    Parameters
    ----------
    env_id :
        Registered scenario id, for example ``"EasyRocket-v1"``.
    obs :
        Name of an observation variant, which must be a key in
        :data:`~factoriax.engine.observations.OBSERVATIONS`. ``None`` takes the
        default of the scenario.
    obs_radius :
        Half-width of a local window. ``None`` takes the default of the
        scenario. A ``_global`` observation variant ignores this argument.
    resample :
        Auto-reset mode. ``None``, the default, takes the ``resample`` setting
        of the scenario. Pass ``True`` or ``False`` to override it. Pass
        ``False`` for the cheap cached restore, even on a keyed scenario.
    auto_reset :
        Whether to wrap the environment in :class:`AutoResetWrapper`. The
        default is ``True``, because most callers train inside a ``lax.scan``
        rollout of fixed length and need an episode to restart on ``done``.
        Pass ``False`` to control the episodes yourself, as a scripted
        rollout, interactive play, or a training loop with its own
        reset-on-done code does.

    Returns
    -------
    tuple
        The pair ``(env, params)``.

    Raises
    ------
    KeyError
        If no scenario carries the id ``env_id``.
    """
    spec = SCENARIOS[env_id]
    overrides: dict[str, Any] = {}
    if obs is not None:
        overrides["obs"] = obs
    if obs_radius is not None:
        overrides["obs_radius"] = obs_radius
    env, params = spec.build(**overrides)
    if auto_reset:
        use_resample = spec.resample if resample is None else resample
        env = AutoResetWrapper(env, resample=use_resample)
    return env, params


def list_scenarios() -> tuple[tuple[str, ScenarioSpec], ...]:
    """Return every registered scenario id, with its spec."""
    return tuple(SCENARIOS.items())
