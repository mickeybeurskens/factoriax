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

from factoriax.engine.envs.easy_rocket import easy_rocket
from factoriax.engine.envs.miner_curriculum import (
    craft_miners,
    mine_ores,
    place_miners,
)
from factoriax.engine.envs.mining import mining
from factoriax.engine.envs.rocket import rocket
from factoriax.engine.envs.wrappers import AutoResetWrapper
from factoriax.engine.state import EnvParams


@dataclass(frozen=True)
class ScenarioSpec:
    """A registered scenario: display metadata plus its env factory."""

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
    "MineOres-v1": ScenarioSpec(
        name="Mine Ores",
        description=(
            "Curriculum stage 1: mine 5 of every ore type on the 16x16 "
            "six-patch map. Graded latched bits, max score 30; "
            "300-step budget."
        ),
        build=mine_ores,
        resample=True,
    ),
    "CraftMiners-v1": ScenarioSpec(
        name="Craft Miners",
        description=(
            "Curriculum stage 2: craft six miners from a pre-stocked "
            "inventory (6 limestone + 6 silicon). Graded latched bits, "
            "max score 6; 300-step budget."
        ),
        build=craft_miners,
        resample=True,
    ),
    "PlaceMiners-v1": ScenarioSpec(
        name="Place Miners",
        description=(
            "Curriculum stage 3: six miners pre-stocked; get all six "
            "producing on ore patches. Graded latched bits, max score 6; "
            "300-step budget, early exit when all six produce."
        ),
        build=place_miners,
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
    auto_reset: bool = False,
    resample: bool | None = None,
) -> tuple[Any, EnvParams]:
    """Resolve a scenario id to ``(env, params)``.

    Parameters
    ----------
    env_id :
        Registered scenario id (e.g. ``"EasyRocket-v1"``).
    obs :
        Observation variant name (one of the keys in
        :data:`~factoriax.engine.observations.OBSERVATIONS`). ``None``
        uses the scenario's opinionated default.
    obs_radius :
        Local-window half-width. ``None`` uses the scenario's
        opinionated default; ignored for ``_global`` obs variants.
    auto_reset :
        Wrap in :class:`AutoResetWrapper`.
    resample :
        Auto-reset mode. ``None`` (default) uses the scenario's
        ``resample`` setting; pass ``True``/``False`` to override — e.g.
        ``False`` for the cheap cached restore even on a keyed scenario.
    env_id: str :
        
    * :
        
    obs: str | None :
         (Default value = None)
    obs_radius: int | None :
         (Default value = None)
    auto_reset: bool :
         (Default value = False)
    resample: bool | None :
         (Default value = None)

    Returns
    -------
    
        ``(env, params)``.

    Raises
    ------
    KeyError
        If ``env_id`` is not registered.

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
    """Return all registered scenario ids and their specs."""
    return tuple(SCENARIOS.items())
